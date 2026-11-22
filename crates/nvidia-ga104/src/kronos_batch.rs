//! C-157 point 7 (etapa 5 / P9, first step) — the chosen token stays on the board: several positions on ONE doorbell.
//!
//! Today every position is its own submission: the driver writes the ~89 launches, rings, waits for the last record,
//! reads the 16 bytes of the argmax ([`crate::argmax`]) and only then knows the token to put into the next position's
//! embedding launch (`Embed`: the row address `emb_tok + token * row` sits in its cbuf). Nothing else in a position's
//! list depends on the token before it: the cache addresses and the attention length depend only on the position,
//! which the driver knows in advance.
//!
//! Here the embedding of a position whose token comes from the board is the program `embed_dev`
//! (`tools/nak/embed_dev.comp`, Mesa NAK 25.0.7): it reads the token from word 0 of the argmax result of the position
//! before, in the results buffer **one slot of 16 bytes per position** (`results + 16 * pos`). The rest of the list is
//! the list of today ([`TokenList`]), byte for byte; every position ends with its argmax into its own slot. So a whole
//! text (or any number of positions) is one submission and the driver reads `16 * positions` bytes once.
//!
//! Honesty: a token the argmax did not find (`0xffffffff`, or any index not in the vocabulary) makes `embed_dev` read
//! nothing from the table and write NaN into x — every later argmax of the batch then says "no token", and
//! [`crate::kronos_gen::Generation::accept_batch`] stops at the first bad slot. Never a guessed token.
//!
//! Tested **only in SIMULARE** (`gpu-sim`, `tests/sim_c157_p7_batch.rs`); nothing here has run on the board.

use alloc::vec::Vec;

use crate::argmax::{self as am, ArgmaxResult};
use crate::kronos_layer::{Launch, Prog, blocks, cbuf};
use crate::kronos_model::{self as km, ModelVas};
use gpu_compute::layer::Shape;
use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// `tools/nak/out/embed_dev.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const EMBED_DEV: &[u8] = include_bytes!("../../../tools/nak/out/embed_dev.sm86.sass");
/// SHA-256 of [`EMBED_DEV`] (checked by a test; written in the note for the OS).
pub const EMBED_DEV_SHA256: &str =
    "5eb883c68fb91fd8e0919d51357d171b62455f20041eab731da46b3cbaddfce0";
/// Registers per thread (NAK dump `embed_dev.sm86.txt`: 68 instructions, 1 088 bytes; `embed` 55).
pub const REGISTERS: u32 = 19;
/// Threads per block (`local_size_x`), as `embed`.
pub const BLOCK: u32 = 256;
/// Where the arguments sit in the cbuf (NVK 25.0.7 push constants from 0x28).
pub const CBUF_TABLE: usize = 0x28;
pub const CBUF_RESULT: usize = 0x30;
pub const CBUF_POS_ROW: usize = 0x38;
pub const CBUF_X: usize = 0x40;
pub const CBUF_N: usize = 0x48;
pub const CBUF_VOCAB: usize = 0x4c;
pub const CBUF_STRIDE: usize = 0x50;
/// Bytes of one argmax slot in the results buffer.
pub const SLOT: u64 = am::RESULT_BYTES;

/// Bytes of one Q8_0 row of `e` values (the stride of the token table).
pub fn row_bytes(e: usize) -> u64 {
    (e / q8::BLOCK * q8::BLOCK_BYTES) as u64
}

/// One launch of `embed_dev`: x (`e` values) = row `token` of the table at `table` (`vocab` rows of `row_bytes(e)`)
/// + the position row at `pos_row`, the token read from word 0 at `result` (a 16-aligned argmax slot).
pub fn embed_dev_launch(
    table: u64,
    vocab: usize,
    e: usize,
    result: u64,
    pos_row: u64,
    x: u64,
) -> GpuResult<Launch> {
    if e == 0 || !e.is_multiple_of(q8::BLOCK) || vocab == 0 || vocab > u32::MAX as usize {
        return Err(bad("embed_dev: e nemultiplu de 32 sau vocabular gol"));
    }
    if !result.is_multiple_of(SLOT)
        || !table.is_multiple_of(4)
        || !pos_row.is_multiple_of(4)
        || !x.is_multiple_of(4)
    {
        return Err(bad("embed_dev: adrese nealiniate"));
    }
    let stride = row_bytes(e);
    if stride > u32::MAX as u64 {
        return Err(bad("embed_dev: rand prea lat"));
    }
    Ok(Launch {
        prog: Prog::EmbedDev,
        cbuf: cbuf(
            &[table, result, pos_row, x],
            &[e as u32, vocab as u32, stride as u32],
        ),
        grid: blocks(e, BLOCK),
    })
}

/// Where the token of a position comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// Known by the driver (a prompt token): today's `Embed`, the row address in the cbuf.
    Cpu(usize),
    /// The argmax of the position before, read on the board by `embed_dev` from its slot.
    Board,
}

/// Which per-position list the batch uses (the same lists as one position per doorbell).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenList {
    /// [`km::token_launches_fused_ln`] (`PROGRAMS_ARGMAX`).
    FusedLn,
    /// [`km::token_launches_fused_ln32`] (`lipita-ln-32`, the default on the board since os-346).
    FusedLn32,
}

impl TokenList {
    fn launches(
        self,
        s: &Shape,
        vocab: usize,
        context: usize,
        v: &ModelVas,
        tok: usize,
        pos: usize,
    ) -> GpuResult<Vec<Launch>> {
        match self {
            TokenList::FusedLn => km::token_launches_fused_ln(s, vocab, context, v, tok, pos),
            TokenList::FusedLn32 => km::token_launches_fused_ln32(s, vocab, context, v, tok, pos),
        }
    }
}

/// The launches of positions `first_pos ..` (one per entry of `inputs`), in order, as ONE list: for each position its
/// list of today (with `embed_dev` instead of `Embed` when the token comes from the board), then its argmax into slot
/// `results + 16 * pos`. A [`Input::Board`] at position 0 is refused (there is no position before it).
#[allow(clippy::too_many_arguments)]
pub fn batch_launches(
    list: TokenList,
    s: &Shape,
    vocab: usize,
    context: usize,
    v: &ModelVas,
    results: u64,
    first_pos: usize,
    inputs: &[Input],
) -> GpuResult<Vec<Launch>> {
    if inputs.is_empty() {
        return Err(bad("lot: nicio pozitie"));
    }
    if !results.is_multiple_of(SLOT) {
        return Err(bad("lot: rezultatele nealiniate la 16"));
    }
    if first_pos + inputs.len() > context {
        return Err(bad("lot: pozitiile trec de context"));
    }
    let row = row_bytes(s.e);
    let mut out = Vec::new();
    for (k, &inp) in inputs.iter().enumerate() {
        let pos = first_pos + k;
        let tok = match inp {
            Input::Cpu(t) => t,
            Input::Board => 0,
        };
        let mut l = list.launches(s, vocab, context, v, tok, pos)?;
        if inp == Input::Board {
            if pos == 0 {
                return Err(bad("lot: pozitia 0 nu are simbol de pe placa"));
            }
            if l.first().map(|x| x.prog) != Some(Prog::Embed) {
                return Err(bad("lot: lista nu incepe cu incorporarea"));
            }
            l[0] = embed_dev_launch(
                v.emb_tok,
                vocab,
                s.e,
                results + SLOT * (pos as u64 - 1),
                v.emb_pos + pos as u64 * row,
                v.layers[0].x,
            )?;
        }
        out.extend(l);
        out.push(am::launch(v.logits, results + SLOT * pos as u64, vocab)?);
    }
    Ok(out)
}

/// Launches per position in a batch: the list's + the argmax.
pub fn launches_per_position(layers: usize) -> usize {
    km::launches_per_token_fused(layers) + 1
}

/// The argmax results of positions `first_pos .. first_pos + n` from the bytes of the whole results buffer.
pub fn read_results(buf: &[u8], first_pos: usize, n: usize) -> GpuResult<Vec<ArgmaxResult>> {
    let end = (first_pos + n) * SLOT as usize;
    if buf.len() < end {
        return Err(bad("lot: bufferul de rezultate e prea scurt"));
    }
    (first_pos..first_pos + n)
        .map(|p| ArgmaxResult::from_bytes(&buf[p * SLOT as usize..]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256;

    #[test]
    fn program_bytes_and_sha() {
        assert_eq!(EMBED_DEV.len() % 16, 0);
        let hex: alloc::string::String = sha256::digest(EMBED_DEV)
            .iter()
            .map(|b| alloc::format!("{b:02x}"))
            .collect();
        assert_eq!(hex, EMBED_DEV_SHA256);
        assert_eq!(Prog::EmbedDev.code(), EMBED_DEV);
        assert_eq!(Prog::EmbedDev.block(), BLOCK);
        assert_eq!(Prog::EmbedDev.registers(), REGISTERS);
        assert!(Prog::EmbedDev.shared().is_none());
    }

    #[test]
    fn launch_cbuf_and_checks() {
        let l = embed_dev_launch(
            0x1_0000_0000,
            256,
            64,
            0x2_0000_0030,
            0x1_0000_4000,
            0x3_0000_0000,
        )
        .unwrap();
        assert_eq!(l.prog, Prog::EmbedDev);
        assert_eq!(l.grid, 1);
        let u64at = |o: usize| u64::from_le_bytes(l.cbuf[o..o + 8].try_into().unwrap());
        let u32at = |o: usize| u32::from_le_bytes(l.cbuf[o..o + 4].try_into().unwrap());
        assert_eq!(u64at(CBUF_TABLE), 0x1_0000_0000);
        assert_eq!(u64at(CBUF_RESULT), 0x2_0000_0030);
        assert_eq!(u64at(CBUF_POS_ROW), 0x1_0000_4000);
        assert_eq!(u64at(CBUF_X), 0x3_0000_0000);
        assert_eq!(u32at(CBUF_N), 64);
        assert_eq!(u32at(CBUF_VOCAB), 256);
        assert_eq!(u32at(CBUF_STRIDE), 2 * 36);
        assert!(l.cbuf[..CBUF_TABLE].iter().all(|&b| b == 0));
        assert!(l.cbuf[CBUF_STRIDE + 4..].iter().all(|&b| b == 0));
        assert!(embed_dev_launch(0, 256, 64, 0x2008, 0, 0).is_err());
        assert!(embed_dev_launch(0, 256, 63, 0x2000, 0, 0).is_err());
        assert!(embed_dev_launch(0, 0, 64, 0x2000, 0, 0).is_err());
        assert_eq!(embed_dev_launch(0, 9, 768, 0, 0, 0).unwrap().grid, 3);
    }

    #[test]
    fn results_read_by_slot() {
        let mut b = alloc::vec![0u8; 16 * 4];
        for p in 0..4u32 {
            for w in 0..4u32 {
                let at = (16 * p + 4 * w) as usize;
                b[at..at + 4].copy_from_slice(&(10 * p + w).to_le_bytes());
            }
        }
        let r = read_results(&b, 1, 3).unwrap();
        assert_eq!(r[0].words, [10, 11, 12, 13]);
        assert_eq!(r[2].words, [30, 31, 32, 33]);
        assert!(read_results(&b, 2, 3).is_err());
    }
}
