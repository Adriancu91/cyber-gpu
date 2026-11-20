//! SIMULARE: a whole Kronos model on the simulated work channel, laid out the way the OS lays it out for
//! `i14-simbol` (one blob = every weight, 16-aligned, then the activation scratch and the final vector + logits,
//! both started as NaN; two caches per layer; one launch slot of 512 bytes per launch; one records buffer), and the
//! three operations `kronos_diag::TokenIo` asks for. Shared by `tests/sim_i14_diag.rs` and the host tool
//! `examples/analiza_i14.rs`. The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here says anything
//! about the board.
#![allow(dead_code)]

use gpu_core::diag::Journal;
use gpu_core::error::{GpuError, GpuResult};
use gpu_sim::cmdproc::{
    ATTENTION_NEEDS, LAYERNORM_NEEDS, Needs, ProgramFn, Q8_FUSED_NEEDS, Q8_MATVEC_NEEDS,
    attention_on_cpu, copy_b128_on_cpu, embed_on_cpu, gelu_on_cpu, layernorm_on_cpu,
    q8_fused_on_cpu, q8_matvec_on_cpu, vadd_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, Way};
use nvidia_ga104::kronos_layer::{self as kl, Launch, LayerVas, PROGRAMS_FUSED, Prog, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas, OwnedModel};
use nvidia_ga104::mmu::SysMapping;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer;

use super::{
    Rig, dma_pages, get_bytes, gpu_runs, put_bytes, rig_with_push, span, sys_read, sys_write,
};

/// Rows the rig's journal keeps (the longest diagnosis of the tests fits several times).
pub const JOURNAL_ROWS: usize = 16384;

pub const NONE: Needs = Needs {
    shared_bytes: 0,
    barriers: 0,
};

/// The eight programs with their models (what `gpu-sim` runs for each; C-147: + the glued `q8_fused`).
pub const MODELS: [(Prog, ProgramFn, Needs); 8] = [
    (Prog::Copy, copy_b128_on_cpu, NONE),
    (Prog::Q8Matvec, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
    (Prog::Vadd, vadd_on_cpu, NONE),
    (Prog::LayerNorm, layernorm_on_cpu, LAYERNORM_NEEDS),
    (Prog::Attention, attention_on_cpu, ATTENTION_NEEDS),
    (Prog::Gelu, gelu_on_cpu, NONE),
    (Prog::Embed, embed_on_cpu, NONE),
    (Prog::Q8Fused, q8_fused_on_cpu, Q8_FUSED_NEEDS),
];

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A model uploaded on a rig, ready for tokens.
pub struct TokenSim<'a> {
    pub r: Rig,
    pub m: &'a OwnedModel,
    pub vas: ModelVas,
    pub progs: Vec<(Prog, Mapping)>,
    pub blob: Mapping,
    pub caches: Vec<(Mapping, Mapping)>,
    pub launches: Mapping,
    pub records: Mapping,
    /// The OS's RAM buffer the logits are copied down into (host RAM model, DMA pages).
    pub host: SysMapping,
    /// What the GPU model runs for each program (a test may plant a wrong one before the first doorbell).
    pub models: Vec<(Prog, ProgramFn, Needs)>,
    next_payload: u32,
    /// Doorbells so far, and the launches of each.
    pub submissions: Vec<usize>,
    /// Where the diagnosis writes its rows (on the board: the stick journal).
    pub journal: Journal,
    /// C-142: the upload pieces through the arena (bytes, copy-launch ns from the records), empty when the blob
    /// was written straight into VRAM.
    pub upload: Vec<(u64, u64)>,
    /// C-142: for every doorbell, the board clock (model's) at record 0 and at the last record, and the sum of the
    /// launch durations.
    pub times: Vec<(u64, u64, u64)>,
    /// C-145: when set, the launch slots are written through this shadow (only the changed words).
    pub shadow: Option<nvidia_ga104::kronos_fast::WriteShadow>,
    /// C-145: bytes written into the launch slots so far.
    pub launch_bytes: u64,
    /// C-147: every doorbell's launch durations (ns, from its records), in order.
    pub durations: Vec<Vec<u64>>,
    /// C-147 point 7 (the driver's diet): when set, the pushbuffer words go through this shadow (only the changed
    /// words are written).
    pub push_shadow: Option<nvidia_ga104::kronos_fast::WriteShadow>,
    /// C-147 point 7: `false` = the timestamp records are not zeroed before a submission (the payloads are unique
    /// per submission, so a stale record cannot pass `analyze_sequence`).
    pub zero_records: bool,
    /// C-147 point 7: for every doorbell, (pushbuffer + ring words written, words left out by the shadow, record
    /// bytes zeroed).
    pub driver_words: Vec<(u64, u64, u64)>,
}

impl<'a> TokenSim<'a> {
    /// `stale_line`: bytes per line of the GPU model's non-coherent read cache (0 = coherent memory).
    pub fn new(m: &'a OwnedModel, stale_line: u64) -> Self {
        Self::with(m, stale_line, 0x4000, None).unwrap()
    }

    /// C-142: the rig with a pushbuffer area of `push` bytes; with `upload = Some(piece)` the blob reaches VRAM as
    /// the OS sends it — written into a host-RAM arena `piece` bytes at a time and copied up by one `copy_b128`
    /// launch per piece (I11) — instead of being written straight into VRAM. A piece that does not arrive whole is
    /// an error.
    pub fn with(
        m: &'a OwnedModel,
        stale_line: u64,
        push: u64,
        upload: Option<u64>,
    ) -> GpuResult<Self> {
        let e = m.shape.e;
        let n = km::launches_per_token(m.layers.len()) + 1;
        let mut r = rig_with_push(push);
        r.stale_line = stale_line;
        let progs: Vec<(Prog, Mapping)> = PROGRAMS_FUSED
            .iter()
            .map(|&p| {
                let mm = r.mem.alloc_mapped(p.code().len() as u64, 0x1000).unwrap();
                put_bytes(r.mem.writer_mut(), mm.pa, p.code());
                (p, mm)
            })
            .collect();
        // The blob: every part 16-aligned, in the order of the KGR1 tensors.
        let mut b: Vec<u8> = Vec::new();
        let part = |b: &mut Vec<u8>, bytes: &[u8]| {
            let at = b.len() as u64;
            b.extend_from_slice(bytes);
            b.resize(b.len().next_multiple_of(16), 0);
            at
        };
        let emb_tok = part(&mut b, &m.emb_tok);
        let emb_pos = part(&mut b, &m.emb_pos);
        let ln_g = part(&mut b, &f32s(&m.ln_g));
        let ln_b = part(&mut b, &f32s(&m.ln_b));
        let weights: Vec<[u64; 12]> = m
            .layers
            .iter()
            .map(|l| {
                [
                    part(&mut b, &f32s(&l.ln1_g)),
                    part(&mut b, &f32s(&l.ln1_b)),
                    part(&mut b, &l.wqkv),
                    part(&mut b, &f32s(&l.bqkv)),
                    part(&mut b, &l.wout),
                    part(&mut b, &f32s(&l.bout)),
                    part(&mut b, &f32s(&l.ln2_g)),
                    part(&mut b, &f32s(&l.ln2_b)),
                    part(&mut b, &l.wfc),
                    part(&mut b, &f32s(&l.bfc)),
                    part(&mut b, &l.wproj),
                    part(&mut b, &f32s(&l.bproj)),
                ]
            })
            .collect();
        let scratch = b.len() as u64;
        b.resize((scratch + kl::scratch_bytes(e)) as usize, 0xff);
        let fin = b.len() as u64;
        b.resize(
            (fin + 4 * (e + m.vocab) as u64).next_multiple_of(16) as usize,
            0xff,
        );
        let blob = r.mem.alloc_mapped(b.len() as u64, 0x1000).unwrap();
        let mut pieces = Vec::new();
        match upload {
            None => put_bytes(r.mem.writer_mut(), blob.pa, &b),
            Some(piece) => pieces = upload_blob(&mut r, &progs, &blob, &b, piece)?,
        }
        let at = |o: u64| blob.va + o;
        let mut caches = Vec::new();
        let mut layers = Vec::new();
        for o in &weights {
            let kc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
            let vc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
            let wv = WeightVas {
                ln1_g: at(o[0]),
                ln1_b: at(o[1]),
                wqkv: at(o[2]),
                bqkv: at(o[3]),
                wout: at(o[4]),
                bout: at(o[5]),
                ln2_g: at(o[6]),
                ln2_b: at(o[7]),
                wfc: at(o[8]),
                bfc: at(o[9]),
                wproj: at(o[10]),
                bproj: at(o[11]),
            };
            layers.push(LayerVas::with_scratch(e, at(scratch), kc.va, vc.va, wv).unwrap());
            caches.push((kc, vc));
        }
        let launches = r.mem.alloc_mapped(512 * n as u64, 0x1000).unwrap();
        let records = r.mem.alloc_mapped(16 * (n as u64 + 1), 0x1000).unwrap();
        let logit_bytes = (4 * m.vocab as u64).next_multiple_of(16);
        let host = r
            .mem
            .alloc_sys_mapped(&dma_pages(3, logit_bytes.div_ceil(0x1000) as usize))
            .unwrap();
        let vas = ModelVas {
            emb_tok: at(emb_tok),
            emb_pos: at(emb_pos),
            ln_g: at(ln_g),
            ln_b: at(ln_b),
            layers,
            h_final: at(fin),
            logits: at(fin) + 4 * e as u64,
        };
        Ok(TokenSim {
            r,
            m,
            vas,
            progs,
            blob,
            caches,
            launches,
            records,
            host,
            models: MODELS.to_vec(),
            next_payload: 0x1_0000,
            submissions: Vec::new(),
            journal: Journal::new(JOURNAL_ROWS),
            upload: pieces,
            times: Vec::new(),
            shadow: None,
            launch_bytes: 0,
            durations: Vec::new(),
            push_shadow: None,
            zero_records: true,
            driver_words: Vec::new(),
        })
    }

    fn mappings(&self) -> Vec<Mapping> {
        let mut v = vec![self.blob, self.launches, self.records];
        v.extend(self.caches.iter().flat_map(|(k, c)| [*k, *c]));
        v.extend(self.progs.iter().map(|(_, m)| *m));
        v
    }

    /// The whole token in `way` (a `SeqMode` = that mode with the program's embedding), sent with
    /// `kronos_diag::send_token`; the logits read back from VRAM.
    pub fn run_token(
        &mut self,
        token: usize,
        pos: usize,
        way: impl Into<Way>,
    ) -> GpuResult<Vec<f32>> {
        let (m, vas) = (self.m, self.vas.clone());
        kd::send_token(self, &m.weights(), &vas, token, pos, way.into(), &[])?;
        let got = self.read(&[(vas.logits, m.vocab)])?;
        Ok(got[0].iter().map(|&b| f32::from_bits(b)).collect())
    }

    /// The whole token as the OS runs it for the texts: the list, then the copy of the logits down into RAM
    /// (filled with 0xEE before), sent in `way`; the logits read from RAM.
    pub fn run_token_ram(
        &mut self,
        token: usize,
        pos: usize,
        way: impl Into<Way>,
    ) -> GpuResult<Vec<f32>> {
        let (m, vas) = (self.m, self.vas.clone());
        let bytes = (4 * m.vocab as u64).next_multiple_of(16);
        let down = kl::copy_launch(vas.logits, self.host.va, bytes)?;
        sys_write(&mut self.r.sys, &self.host, &vec![0xee; bytes as usize]);
        kd::send_token(self, &m.weights(), &vas, token, pos, way.into(), &[down])?;
        Ok(self.ram_logits())
    }

    /// What is in the RAM buffer of the logits now.
    pub fn ram_logits(&self) -> Vec<f32> {
        sys_read(&self.r.sys, &self.host, 4 * self.m.vocab as u64)
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    /// The rows written so far, as the OS prints them on the stick; the journal starts again empty.
    pub fn take_journal_text(&mut self) -> String {
        let j = std::mem::replace(&mut self.journal, Journal::new(JOURNAL_ROWS));
        j.events()
            .enumerate()
            .map(|(i, e)| format!("[  126.{i:06}] gpu: {e}\n"))
            .collect()
    }

    /// Bytes the stale-read model served that differed from memory (0 when the model is off or changed nothing).
    pub fn stale_bytes(&self) -> u64 {
        self.r.cp.as_ref().map_or(0, |c| c.stale_bytes_served())
    }
}

/// The blob `b` into VRAM `blob` the way the OS uploads it (`gpu_i14::run_model` step 1): each piece of at most
/// `piece` bytes written into a host-RAM arena, then one `copy_b128` launch (I11) copies it up, one doorbell per
/// piece; its two records give the copy's time. The GPU model sees only the arena, the launch, the program and the
/// piece's stretch of VRAM.
fn upload_blob(
    r: &mut Rig,
    progs: &[(Prog, Mapping)],
    blob: &Mapping,
    b: &[u8],
    piece: u64,
) -> GpuResult<Vec<(u64, u64)>> {
    let copy = progs.iter().find(|(p, _)| *p == Prog::Copy).unwrap().1;
    let piece = piece.min(b.len() as u64);
    let host = r
        .mem
        .alloc_sys_mapped(&dma_pages(4, piece.div_ceil(0x1000) as usize))?;
    let cb = r.mem.alloc_mapped(0x1000, 0x1000)?;
    let q = r.mem.alloc_mapped(0x1000, 0x1000)?;
    let rec = r.mem.alloc_mapped(0x1000, 0x1000)?;
    let mut out = Vec::new();
    for (i, chunk) in b.chunks(piece as usize).enumerate() {
        let len = (chunk.len() as u64).next_multiple_of(16);
        let dst = blob.va + i as u64 * piece;
        sys_write(&mut r.sys, &host, chunk);
        let w = r.mem.writer_mut();
        put_bytes(w, cb.pa, &transfer::copy_cbuf(host.va, dst, len)?);
        put_bytes(
            w,
            q.pa,
            &transfer::copy_qmd(copy.va, cb.va, len)?.to_bytes()?,
        );
        put_bytes(w, rec.pa, &[0; 32]);
        let first = 0xc000 + 0x100 * i as u32;
        let mut words = vec![0u32; transfer::sequence_words(1)];
        let mut pb = PushBuf::new(&mut words);
        transfer::encode_sequence(&mut pb, &[q.va], rec.va, first)?;
        let nw = pb.len();
        let (fence, done) = (rec.va + 16, first as u64 + 1);
        r.wq.submit_fenced(r.mem.writer_mut(), &words[..nw], fence, done)?;
        let programs: Vec<(&[u8], ProgramFn, Needs)> = MODELS
            .iter()
            .map(|&(p, f, needs)| (p.code(), f, needs))
            .collect();
        let loads = [
            (host.va, host.size),
            span(&cb),
            span(&q),
            span(&rec),
            span(&copy),
            (dst, len),
        ];
        gpu_runs(r, &programs, &loads, &[(dst, len), span(&rec)])?;
        let recs = get_bytes(r.mem.writer_mut(), rec.pa, 32);
        let ns = transfer::analyze_sequence(&recs, 1, first)?[0];
        r.wq.retire(fence, done)?;
        out.push((chunk.len() as u64, ns));
    }
    Ok(out)
}

impl TokenIo for TokenSim<'_> {
    fn submit(&mut self, launches: &[Launch], first_slot: usize, mode: SeqMode) -> GpuResult<()> {
        let n = launches.len();
        if 512 * (first_slot + n) as u64 > self.launches.size {
            return Err(GpuError::InvalidArgument(
                "SIM: prea multe lansari pentru zona lor",
            ));
        }
        let va = |p: Prog| self.progs.iter().find(|(q, _)| *q == p).unwrap().1.va;
        let mut qmds = Vec::with_capacity(n);
        for (i, l) in launches.iter().enumerate() {
            let off = 512 * (first_slot + i) as u64;
            let (cb_va, cb_pa) = (self.launches.va + off, self.launches.pa + off);
            let qmd = l.qmd(va(l.prog), cb_va)?;
            let w = self.r.mem.writer_mut();
            for (pa, bytes) in [(cb_pa, &l.cbuf[..]), (cb_pa + 256, &qmd[..])] {
                match self.shadow.as_mut() {
                    None => {
                        put_bytes(w, pa, bytes);
                        self.launch_bytes += bytes.len() as u64;
                    }
                    Some(sh) => {
                        for (at, r) in sh.changes(pa, bytes)? {
                            put_bytes(w, at, &bytes[r.clone()]);
                            self.launch_bytes += r.len() as u64;
                        }
                        sh.remember(pa, bytes)?;
                    }
                }
            }
            qmds.push(cb_va + 256);
        }
        let zeroed = if self.zero_records {
            put_bytes(
                self.r.mem.writer_mut(),
                self.records.pa,
                &vec![0; 16 * (n + 1)],
            );
            16 * (n as u64 + 1)
        } else {
            0
        };
        let first = self.next_payload;
        self.next_payload += 0x100;
        let mut words = vec![0u32; kd::words(mode, n)];
        let mut pb = PushBuf::new(&mut words);
        kd::encode(mode, &mut pb, &qmds, self.records.va, first)?;
        assert_eq!(pb.len(), kd::words(mode, n));
        let fence = self.records.va + 16 * n as u64;
        let done = (first + n as u32) as u64;
        let counted = match self.push_shadow.as_mut() {
            None => {
                self.r
                    .wq
                    .submit_fenced(self.r.mem.writer_mut(), &words, fence, done)?;
                (words.len() as u64 + 3, 0)
            }
            Some(sh) => {
                let mut m =
                    nvidia_ga104::kronos_fast::ShadowedMemory::new(self.r.mem.writer_mut(), sh);
                let r = self.r.wq.submit_fenced(&mut m, &words, fence, done);
                let c = (m.written, m.skipped);
                if r.is_err() {
                    sh.forget();
                }
                r?;
                c
            }
        };
        self.driver_words.push((counted.0, counted.1, zeroed));
        let programs: Vec<(&[u8], ProgramFn, Needs)> = self
            .models
            .iter()
            .map(|&(p, f, needs)| (p.code(), f, needs))
            .collect();
        let maps = self.mappings();
        let mut loads: Vec<(u64, u64)> = maps.iter().map(span).collect();
        loads.push((self.host.va, self.host.size));
        let mut stores = vec![
            span(&self.blob),
            span(&self.records),
            (self.host.va, self.host.size),
        ];
        stores.extend(self.caches.iter().flat_map(|(k, c)| [span(k), span(c)]));
        gpu_runs(&mut self.r, &programs, &loads, &stores)?;
        let rec = get_bytes(
            self.r.mem.writer_mut(),
            self.records.pa,
            16 * (n as u64 + 1),
        );
        let each = transfer::analyze_sequence(&rec, n, first)?;
        let busy: u64 = each.iter().sum();
        self.durations.push(each);
        let ts = |k: usize| u64::from_le_bytes(rec[16 * k + 8..16 * k + 16].try_into().unwrap());
        self.times.push((ts(0), ts(n), busy));
        self.r.wq.retire(fence, done)?;
        self.submissions.push(n);
        Ok(())
    }

    fn read(&mut self, ranges: &[(u64, usize)]) -> GpuResult<Vec<Vec<u32>>> {
        let maps = self.mappings();
        ranges
            .iter()
            .map(|&(va, words)| {
                let m = maps
                    .iter()
                    .find(|m| va >= m.va && va + 4 * words as u64 <= m.va + m.size)
                    .ok_or(GpuError::InvalidArgument("SIM: citire in afara bufferelor"))?;
                let bytes = get_bytes(
                    self.r.mem.writer_mut(),
                    m.pa + (va - m.va),
                    4 * words as u64,
                );
                Ok(bytes
                    .chunks(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect())
            })
            .collect()
    }

    fn write(&mut self, va: u64, words: &[u32]) -> GpuResult<()> {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let m = self
            .mappings()
            .into_iter()
            .find(|m| va >= m.va && va + bytes.len() as u64 <= m.va + m.size)
            .ok_or(GpuError::InvalidArgument(
                "SIM: scriere in afara bufferelor",
            ))?;
        put_bytes(self.r.mem.writer_mut(), m.pa + (va - m.va), &bytes);
        Ok(())
    }

    fn journal(&mut self) -> &mut Journal {
        &mut self.journal
    }
}
