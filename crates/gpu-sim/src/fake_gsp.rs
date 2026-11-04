//! SIMULARE: a register-level stand-in for the two falcons that boot GSP-RM
//! on GA102 (the GSP falcon at `0x110000` and SEC2 at `0x840000`), the
//! WPR2 registers and the boot scratch registers.
//!
//! Idea from tinygrad's mock GPUs (`test/mockgpu/am/amgpu.py`: small state
//! machines answering a driver on registers; MIT, idea only — nothing
//! translated). Register offsets come from the generated NVIDIA headers in
//! `nvidia-rm570`; the *behaviour* (how many polls until a falcon halts, what
//! a failure looks like) is ours, marked SIM. It answers the question "does
//! our driver follow the sequence we read in the sources, and does it notice
//! every failure?" — never "does the RTX 3070 Ti do this".
//!
//! The firmware itself is not modelled at all: a test says which program is
//! "loaded" with [`FakeGsp::sim_load`], or (session 12) registers a synthetic
//! image with [`FakeGsp::sim_expect`] and lets the driver's DMA loader put it
//! in place: the falcon DMA engine below copies 256-byte blocks from a SIM
//! system memory ([`FakeGsp::sysmem`]) into IMEM/DMEM, and at `STARTCPU` the
//! "boot ROM" accepts the program only if the bytes, the secure flag of the
//! code blocks, the BROM registers and BOOTVEC are all exactly as expected
//! (SIM stand-in for the signature check; no real signature is checked).
//! Register positions: `nvidia-rm570` (NVIDIA MIT headers); behaviour: ours.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use gpu_core::hal::Mmio;
use nvidia_rm570::regs::ga102_dev_falcon_second_pri as f2;
use nvidia_rm570::regs::ga102_dev_falcon_v4 as fv4;
use nvidia_rm570::regs::ga102_dev_fbif_v4 as fbif;
use nvidia_rm570::regs::ga102_dev_gc6_island as gc6;
use nvidia_rm570::regs::ga102_dev_gsp as gsp;
use nvidia_rm570::regs::ga102_dev_riscv_pri as riscv;
use nvidia_rm570::regs::tu102_dev_bus as bus;
use nvidia_rm570::regs::tu102_dev_fb as fb;

/// Base of the GSP falcon (`NV_PGSP`).
pub const GSP_BASE: u32 = 0x0011_0000;
/// Base of the SEC2 falcon (`NV_PSEC`).
pub const SEC2_BASE: u32 = 0x0084_0000;

/// Which signed program the test pretends was loaded into a falcon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimUcode {
    /// FWSEC from the VBIOS with the FRTS command (I6), on the GSP falcon.
    FwsecFrts,
    /// `booter_load` (I7), on SEC2.
    BooterLoad,
}

/// SIM: what the "boot ROM" expects to find in a falcon before it accepts a
/// program (synthetic bytes, never NVIDIA firmware).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimHsImage {
    /// `GSP_BASE` or `SEC2_BASE`.
    pub falcon: u32,
    pub ucode: SimUcode,
    pub imem_pa: u32,
    pub code: Vec<u8>,
    pub dmem_pa: u32,
    pub data: Vec<u8>,
    pub sig_dmem_addr: u32,
    pub ucode_id: u32,
    pub engine_id_mask: u32,
    pub bootvec: u32,
}

/// SIM sizes of IMEM and DMEM (64 KiB each; the real sizes are read from
/// HWCFG by the driver, and the fake reports this one).
pub const SIM_MEM: usize = 64 << 10;

#[derive(Clone, Debug)]
struct FalconMem {
    imem: Vec<u8>,
    /// Secure flag per 256-byte IMEM block.
    imem_secure: Vec<bool>,
    dmem: Vec<u8>,
    /// DMATRFCMD reads left until IDLE.
    busy: u32,
}

impl FalconMem {
    fn new() -> Self {
        FalconMem {
            imem: alloc::vec![0; SIM_MEM],
            imem_secure: alloc::vec![false; SIM_MEM / 256],
            dmem: alloc::vec![0; SIM_MEM],
            busy: 0,
        }
    }
}

/// What the fake should do wrong (one at a time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SimFault {
    #[default]
    None,
    /// Memory scrubbing never finishes after a reset.
    ScrubStuck,
    /// FWSEC halts with this error code in `NV_PBUS_SW_SCRATCH(0xe)[31:16]`.
    FrtsError(u16),
    /// FWSEC halts without error but WPR2 stays 0.
    NoWpr2,
    /// A falcon never halts after `STARTCPU`.
    NeverHalts,
    /// Booter halts but the RISC-V core never becomes active.
    RiscvStaysOff,
    /// The falcon DMA request queue stays full.
    DmaQueueFull,
    /// The falcon DMA engine never reports idle.
    DmaNeverIdle,
}

#[derive(Clone, Copy, Debug, Default)]
struct Falcon {
    in_reset: bool,
    /// HWCFG2 reads left until scrubbing is reported done.
    scrub_left: u32,
    /// `None` = not running; `Some(n)` = halts after n more CPUCTL reads.
    run_left: Option<u32>,
    halted: bool,
    loaded: Option<SimUcode>,
    core_riscv: bool,
}

/// SIM: a sparse 16 MiB BAR0 with two falcons behind it.
pub struct FakeGsp {
    regs: BTreeMap<u32, u32>,
    gsp: Falcon,
    sec2: Falcon,
    pub fault: SimFault,
    /// The FRTS region FWSEC "creates" (byte addresses in VRAM, SIM).
    pub frts: (u64, u64),
    riscv_active: bool,
    /// Every write the driver made: (offset, value).
    pub writes: Vec<(u32, u32)>,
    /// Number of `NV_PGSP_QUEUE_HEAD(0)` writes (the doorbell).
    pub doorbells: u32,
    /// The address booter received in SEC2 mailbox 0/1 (the WPR meta).
    pub booter_arg: Option<u64>,
    /// SIM system memory the falcon DMA reads: (physical base, bytes).
    pub sysmem: Option<(u64, Vec<u8>)>,
    /// Programs the "boot ROM" accepts (see the module comment).
    pub expected: Vec<SimHsImage>,
    /// DMA commands refused by the model (bad setup, out of range).
    pub dma_faults: u32,
    /// DMA blocks copied.
    pub dma_blocks: u32,
    /// SIM: when set, the "FWSEC" program reads its command from its DMEM
    /// mapper at this DMEM offset (positions written by hand from RM's
    /// `FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3` / `FWSECLIC_FRTS_CMD`)
    /// and creates WPR2 where the command says, instead of at `frts`.
    pub fwsec_mapper_at: Option<u32>,
    /// SIM, from the board (3 Nov 2026): how FWSEC writes WPR2 HI. `true` =
    /// the last 4 KiB page of the region (inclusive), the most likely reading
    /// of the board (LO = start, HI != exclusive end); `false` = the exclusive
    /// end. Still to be confirmed with the raw HI in `gpu-jurnal-002.txt`.
    pub wpr2_hi_last_page: bool,
    /// SIM, C-66, from the board (4 Nov 2026, os-222): HI = the start of the
    /// last block of this many bytes of the region (board: 128 KiB, HI
    /// `0x1ffee0` for FRTS `0x1ffe00000..0x1fff00000`). When set it wins
    /// over `wpr2_hi_last_page`; `None` = the older forms above.
    pub wpr2_hi_back: Option<u64>,
    mem: [FalconMem; 2],
}

/// SIM: polls of HWCFG2 until scrubbing is done, polls of CPUCTL until halt.
const SCRUB_POLLS: u32 = 3;
const RUN_POLLS: u32 = 5;

impl Default for FakeGsp {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeGsp {
    pub fn new() -> Self {
        let mut regs = BTreeMap::new();
        regs.insert(0, 0x1740_00a1); // NV_PMC_BOOT_0, SIM revision bits
        // After the VBIOS ran (GFW_BOOT completed), falcons are idle, out of
        // reset, scrubbing done. SIM.
        regs.insert(
            gc6::NV_PGC6_AON_SECURE_SCRATCH_GROUP_05(0),
            0xff, // GFW_BOOT_PROGRESS_COMPLETED
        );
        FakeGsp {
            regs,
            gsp: Falcon::default(),
            sec2: Falcon::default(),
            fault: SimFault::None,
            // SIM: 1 MiB just under a 1 MiB VGA area at the end of 8 GiB.
            frts: ((8 << 30) - (2 << 20), (8 << 30) - (1 << 20)),
            riscv_active: false,
            writes: Vec::new(),
            doorbells: 0,
            booter_arg: None,
            sysmem: None,
            expected: Vec::new(),
            dma_faults: 0,
            dma_blocks: 0,
            fwsec_mapper_at: None,
            wpr2_hi_last_page: true,
            wpr2_hi_back: Some(128 << 10),
            mem: [FalconMem::new(), FalconMem::new()],
        }
    }

    /// The address FWSEC puts in WPR2 HI for `frts` (see `wpr2_hi_back`,
    /// then `wpr2_hi_last_page`).
    pub fn wpr2_hi_end(&self) -> u64 {
        if let Some(n) = self.wpr2_hi_back {
            self.frts.1 - n
        } else if self.wpr2_hi_last_page {
            self.frts.1 - 0x1000
        } else {
            self.frts.1
        }
    }

    /// Registers a program the "boot ROM" will accept (SIM).
    pub fn sim_expect(&mut self, img: SimHsImage) {
        self.expected.push(img);
    }

    /// IMEM / DMEM of a falcon, for tests.
    pub fn imem(&self, base: u32) -> &[u8] {
        &self.mem[Self::idx(base)].imem
    }
    pub fn dmem(&self, base: u32) -> &[u8] {
        &self.mem[Self::idx(base)].dmem
    }

    fn idx(base: u32) -> usize {
        usize::from(base != GSP_BASE)
    }

    fn f2base(base: u32) -> u32 {
        if base == GSP_BASE {
            f2::NV_FALCON2_GSP_BASE
        } else {
            f2::NV_FALCON2_SEC_BASE
        }
    }

    /// One DMATRFCMD: copy 256 bytes from SIM system memory, if the driver
    /// set the engine up the way RM does (otherwise nothing is copied).
    fn dma_cmd(&mut self, base: u32, cmd: u32) {
        let fb = base + 0x600;
        let ctx = fv4::NV_PFALCON_FALCON_DMATRFCMD_CTXDMA.get(cmd);
        let tc = self.raw(fb + fbif::NV_PFALCON_FBIF_TRANSCFG(ctx));
        let setup_ok = fbif::NV_PFALCON_FBIF_CTL_ALLOW_PHYS_NO_CTX
            .get(self.raw(fb + fbif::NV_PFALCON_FBIF_CTL))
            == 1
            && fv4::NV_PFALCON_FALCON_DMACTL_REQUIRE_CTX.get(self.dmactl(base)) == 0
            && fbif::NV_PFALCON_FBIF_TRANSCFG_TARGET.get(tc) == 1
            && fbif::NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE.get(tc) == 1
            && fv4::NV_PFALCON_FALCON_DMATRFCMD_SIZE.get(cmd) == 6
            && fv4::NV_PFALCON_FALCON_DMATRFCMD_WRITE.get(cmd) == 0;
        let addr = (u64::from(self.raw(base + fv4::NV_PFALCON_FALCON_DMATRFBASE))
            | u64::from(self.raw(base + fv4::NV_PFALCON_FALCON_DMATRFBASE1) & 0x1ff) << 32)
            << 8;
        let src = addr + u64::from(self.raw(base + fv4::NV_PFALCON_FALCON_DMATRFFBOFFS));
        let dst = fv4::NV_PFALCON_FALCON_DMATRFMOFFS_OFFS
            .get(self.raw(base + fv4::NV_PFALCON_FALCON_DMATRFMOFFS)) as usize;
        let imem = fv4::NV_PFALCON_FALCON_DMATRFCMD_IMEM.get(cmd) == 1;
        let secure = fv4::NV_PFALCON_FALCON_DMATRFCMD_SEC.get(cmd) == 1;
        let block = self.sysmem.as_ref().and_then(|(b, bytes)| {
            let o = usize::try_from(src.checked_sub(*b)?).ok()?;
            bytes.get(o..o.checked_add(256)?).map(Vec::from)
        });
        let m = &mut self.mem[Self::idx(base)];
        m.busy = 2;
        let fits = dst.is_multiple_of(256) && dst + 256 <= SIM_MEM;
        match block {
            Some(b) if setup_ok && fits => {
                if imem {
                    m.imem[dst..dst + 256].copy_from_slice(&b);
                    m.imem_secure[dst / 256] = secure;
                } else {
                    m.dmem[dst..dst + 256].copy_from_slice(&b);
                }
                self.dma_blocks += 1;
            }
            _ => self.dma_faults += 1,
        }
    }

    /// SIM "FWSEC": init_cmd must be FRTS (0x15) and the FRTS command well
    /// formed; returns the region it asks for.
    fn sim_fwsec_command(&self, m: usize) -> Option<(u64, u64)> {
        let d = &self.mem[0].dmem;
        let w = |o: usize| -> Option<u32> {
            Some(u32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
        };
        if w(m + 44)? != 0x15 {
            return None;
        }
        let c = w(m + 8)? as usize;
        let ok = w(c)? == 1
            && w(c + 4)? == 24
            && w(c + 20)? == 2
            && w(c + 24)? == 1
            && w(c + 28)? == 20
            && w(c + 40)? == 2;
        let start = u64::from(w(c + 32)?) << 12;
        let size = u64::from(w(c + 36)?) << 12;
        (ok && start != 0 && size != 0).then_some((start, start + size))
    }

    /// SIM reset value: REQUIRE_CTX set until the driver writes DMACTL.
    fn dmactl(&self, base: u32) -> u32 {
        *self
            .regs
            .get(&(base + fv4::NV_PFALCON_FALCON_DMACTL))
            .unwrap_or(&1)
    }

    /// At STARTCPU: does the falcon hold one of the expected programs?
    fn brom_check(&mut self, base: u32) -> Option<SimUcode> {
        let r = Self::f2base(base);
        let m = &self.mem[Self::idx(base)];
        self.expected.iter().find_map(|x| {
            let code = m
                .imem
                .get(x.imem_pa as usize..x.imem_pa as usize + x.code.len())?;
            let data = m
                .dmem
                .get(x.dmem_pa as usize..x.dmem_pa as usize + x.data.len())?;
            let first = x.imem_pa as usize / 256;
            let blocks = x.code.len().div_ceil(256);
            let ok = x.falcon == base
                && code == x.code.as_slice()
                && data == x.data.as_slice()
                && m.imem_secure[first..first + blocks].iter().all(|&s| s)
                && self.raw(r + f2::NV_PFALCON2_FALCON_BROM_PARAADDR(0)) == x.sig_dmem_addr
                && self.raw(r + f2::NV_PFALCON2_FALCON_BROM_ENGIDMASK) == x.engine_id_mask
                && self.raw(r + f2::NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID) == x.ucode_id
                && self.raw(r + f2::NV_PFALCON2_FALCON_MOD_SEL) == 1
                && self.raw(base + fv4::NV_PFALCON_FALCON_BOOTVEC) == x.bootvec;
            ok.then_some(x.ucode)
        })
    }

    /// The test's stand-in for loading a signed program over DMA.
    pub fn sim_load(&mut self, base: u32, ucode: SimUcode) {
        self.falcon_mut(base).loaded = Some(ucode);
    }

    pub fn riscv_active(&self) -> bool {
        self.riscv_active
    }

    fn falcon_mut(&mut self, base: u32) -> &mut Falcon {
        if base == GSP_BASE {
            &mut self.gsp
        } else {
            &mut self.sec2
        }
    }

    fn falcon_of(off: u32) -> Option<(u32, u32)> {
        for base in [GSP_BASE, SEC2_BASE] {
            if (base..base + 0x1000).contains(&off) {
                return Some((base, off - base));
            }
        }
        None
    }

    fn raw(&self, off: u32) -> u32 {
        *self.regs.get(&off).unwrap_or(&0)
    }

    fn read_falcon(&mut self, base: u32, rel: u32) -> u32 {
        let fault = self.fault;
        match rel {
            fv4::NV_PFALCON_FALCON_HWCFG2 => {
                let f = self.falcon_mut(base);
                let mut v = fv4::NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING
                    .set(0, u32::from(f.scrub_left > 0 || f.in_reset));
                v |= 1 << 31; // RESET_READY (addendum), always ready. SIM
                if f.scrub_left > 0 && !f.in_reset && fault != SimFault::ScrubStuck {
                    f.scrub_left -= 1;
                }
                v
            }
            fv4::NV_PFALCON_FALCON_CPUCTL => {
                let mut halt_now = false;
                let f = self.falcon_mut(base);
                if let Some(n) = f.run_left {
                    if n == 0 {
                        f.run_left = None;
                        f.halted = true;
                        halt_now = true;
                    } else if fault != SimFault::NeverHalts {
                        f.run_left = Some(n - 1);
                    }
                }
                if halt_now {
                    self.halt_effects(base);
                }
                let halted = self.falcon_mut(base).halted;
                fv4::NV_PFALCON_FALCON_CPUCTL_HALTED.set(0, u32::from(halted))
            }
            fv4::NV_PFALCON_FALCON_HWCFG => {
                fv4::NV_PFALCON_FALCON_HWCFG_IMEM_SIZE.set(0, (SIM_MEM / 256) as u32)
            }
            fv4::NV_PFALCON_FALCON_DMATRFCMD => {
                let m = &mut self.mem[Self::idx(base)];
                let idle = m.busy == 0 && fault != SimFault::DmaNeverIdle;
                m.busy = m.busy.saturating_sub(1);
                fv4::NV_PFALCON_FALCON_DMATRFCMD_FULL
                    .set(0, u32::from(fault == SimFault::DmaQueueFull))
                    | fv4::NV_PFALCON_FALCON_DMATRFCMD_IDLE.set(0, u32::from(idle))
            }
            fv4::NV_PFALCON_FALCON_DMACTL => self.dmactl(base),
            _ => self.raw(base + rel),
        }
    }

    /// What the "program" did when it halts. SIM error codes: 0xdead =
    /// nothing loaded / wrong core, 0xbad1 = booter found no WPR2.
    fn halt_effects(&mut self, base: u32) {
        let f = *self.falcon_mut(base);
        let mb0 = base + fv4::NV_PFALCON_FALCON_MAILBOX0;
        let ok_core = !f.core_riscv;
        match (base, f.loaded, ok_core) {
            (GSP_BASE, Some(SimUcode::FwsecFrts), true) => match self.fault {
                SimFault::FrtsError(code) => {
                    self.regs
                        .insert(bus::NV_PBUS_SW_SCRATCH(0xe), u32::from(code) << 16);
                }
                SimFault::NoWpr2 => {
                    self.regs.insert(bus::NV_PBUS_SW_SCRATCH(0xe), 0);
                }
                _ => {
                    if let Some(m) = self.fwsec_mapper_at {
                        match self.sim_fwsec_command(m as usize) {
                            Some(r) => self.frts = r,
                            None => {
                                // SIM error code: command missing or malformed.
                                self.regs.insert(bus::NV_PBUS_SW_SCRATCH(0xe), 0x00e1 << 16);
                                return;
                            }
                        }
                    }
                    self.regs.insert(bus::NV_PBUS_SW_SCRATCH(0xe), 0);
                    let enc = |a: u64| ((a >> 12) << 4) as u32;
                    self.regs
                        .insert(fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO, enc(self.frts.0));
                    self.regs
                        .insert(fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI, enc(self.wpr2_hi_end()));
                }
            },
            (SEC2_BASE, Some(SimUcode::BooterLoad), true) => {
                let wpr2 = self.raw(fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI) != 0;
                let mut arg = u64::from(self.raw(mb0))
                    | u64::from(self.raw(base + fv4::NV_PFALCON_FALCON_MAILBOX1)) << 32;
                if arg == 0 {
                    // Resume (sequencer CORE_RESUME): Booter re-runs with the
                    // address it got at boot. SIM.
                    arg = self.booter_arg.unwrap_or(0);
                }
                if !wpr2 || arg == 0 {
                    self.regs.insert(mb0, 0xbad1);
                    return;
                }
                self.booter_arg = Some(arg);
                self.regs.insert(mb0, 0);
                if self.fault != SimFault::RiscvStaysOff {
                    self.riscv_active = true;
                    let h = gc6::NV_PGC6_BSI_SECURE_SCRATCH_14;
                    let v = self.raw(h) | 1 << 26; // BOOT_STAGE_3_HANDOFF
                    self.regs.insert(h, v);
                }
            }
            _ => {
                self.regs.insert(mb0, 0xdead);
            }
        }
    }

    fn write_falcon(&mut self, base: u32, rel: u32, v: u32) {
        match rel {
            0x3c0 => {
                // NV_P{GSP,SEC}_FALCON_ENGINE: RESET bit 0.
                let fault = self.fault;
                let f = self.falcon_mut(base);
                if v & 1 == 1 {
                    *f = Falcon {
                        in_reset: true,
                        ..Falcon::default()
                    };
                } else if f.in_reset {
                    f.in_reset = false;
                    f.scrub_left = if fault == SimFault::ScrubStuck {
                        1
                    } else {
                        SCRUB_POLLS
                    };
                }
                if v & 1 == 1 {
                    // Reset + scrubbing clears IMEM/DMEM; DMACTL back to
                    // its reset value. SIM
                    self.mem[Self::idx(base)] = FalconMem::new();
                    self.regs.remove(&(base + fv4::NV_PFALCON_FALCON_DMACTL));
                }
                if base == GSP_BASE && v & 1 == 1 {
                    // GSP reset stops GSP-RM and clears the stage-3 hand-off.
                    self.riscv_active = false;
                    let h = gc6::NV_PGC6_BSI_SECURE_SCRATCH_14;
                    let cleared = self.raw(h) & !(1 << 26);
                    self.regs.insert(h, cleared);
                }
                self.regs.insert(base + rel, v);
            }
            fv4::NV_PFALCON_FALCON_CPUCTL | fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS => {
                let start = fv4::NV_PFALCON_FALCON_CPUCTL_STARTCPU.get(v) == 1
                    && !self.falcon_mut(base).in_reset;
                if start {
                    if self.falcon_mut(base).loaded.is_none() {
                        let accepted = self.brom_check(base);
                        self.falcon_mut(base).loaded = accepted;
                    }
                    let f = self.falcon_mut(base);
                    f.halted = false;
                    f.run_left = Some(RUN_POLLS);
                }
            }
            fv4::NV_PFALCON_FALCON_DMATRFCMD => {
                if self.fault != SimFault::DmaQueueFull {
                    self.dma_cmd(base, v);
                } else {
                    self.dma_faults += 1;
                }
            }
            _ => {
                self.regs.insert(base + rel, v);
            }
        }
    }

    fn finish_sec2(&mut self) {
        if self.sec2.run_left.is_some() && self.fault != SimFault::NeverHalts {
            self.sec2.run_left = None;
            self.sec2.halted = true;
            self.halt_effects(SEC2_BASE);
        }
    }

    fn write_riscv_bcr(&mut self, base: u32, v: u32) {
        // NV_PRISCV_RISCV_BCR_CTRL: CORE_SELECT bit 4; VALID (bit 0) reads 1.
        let f = self.falcon_mut(base);
        f.core_riscv = riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT.get(v) == 1;
        let off = if base == GSP_BASE {
            f2::NV_FALCON2_GSP_BASE
        } else {
            f2::NV_FALCON2_SEC_BASE
        } + riscv::NV_PRISCV_RISCV_BCR_CTRL;
        self.regs.insert(off, v | 1);
    }
}

impl Mmio for FakeGsp {
    fn len(&self) -> u64 {
        16 << 20
    }
    fn is_simulation(&self) -> bool {
        true
    }
    fn read32_raw(&mut self, offset: u64) -> u32 {
        let off = offset as u32;
        if off == gc6::NV_PGC6_BSI_SECURE_SCRATCH_14
            || off == f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_CPUCTL
        {
            // Polling the hand-off lets a started SEC2 finish (resume). SIM
            self.finish_sec2();
        }
        if off == f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_CPUCTL {
            return riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT.set(0, u32::from(self.riscv_active));
        }
        match Self::falcon_of(off) {
            Some((base, rel)) => self.read_falcon(base, rel),
            None => self.raw(off),
        }
    }
    fn write32_raw(&mut self, offset: u64, value: u32) {
        let off = offset as u32;
        self.writes.push((off, value));
        if off == gsp::NV_PGSP_QUEUE_HEAD(0) {
            self.doorbells += 1;
        }
        for (base, f2base) in [
            (GSP_BASE, f2::NV_FALCON2_GSP_BASE),
            (SEC2_BASE, f2::NV_FALCON2_SEC_BASE),
        ] {
            if off == f2base + riscv::NV_PRISCV_RISCV_BCR_CTRL {
                self.write_riscv_bcr(base, value);
                return;
            }
        }
        match Self::falcon_of(off) {
            Some((base, rel)) => self.write_falcon(base, rel, value),
            None => {
                self.regs.insert(off, value);
            }
        }
    }
}
