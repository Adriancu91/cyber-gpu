//! One GSP message = queue-element header + RPC header + payload (r570).
//!
//! ```text
//! +0   authTagBuffer[16]  aadBuffer[16]   (zero: no Confidential Compute)
//! +32  checkSum   u32     makes the XOR checksum of the message zero
//! +36  seqNum     u32     per-queue sequence number
//! +40  elemCount  u32     4 KiB elements this message occupies (1..=16)
//! +44  (padding to 8)
//! +48  rpc_message_header_v03_00:
//!        header_version (0x03000000), signature ('VRPC' = 0x43505256),
//!        length (RPC header + payload), function, rpc_result,
//!        rpc_result_private, sequence, spare/cpuRmGfid
//! +80  payload
//! ```
//!
//! Checksum: XOR of the message as little-endian u64 words (zero-padded to 8
//! bytes), folded as high ^ low 32 bits; the `checkSum` field is chosen so the
//! result is 0. Sources (MIT, open-gpu-kernel-modules@570.144):
//! `message_queue_priv.h`, `g_rpc-message-header.h`, `rpc_headers.h`,
//! `rpc_global_enums.h`, `message_queue_cpu.c` — read, re-written.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use super::msgq::{ELEMENT_SIZE, MAX_ELEMENTS};
use super::{get_u32, put_u32};

/// `GSP_MSG_QUEUE_ELEMENT_HDR_SIZE` (offset of the RPC header, 8-aligned).
pub const ELEMENT_HDR_SIZE: usize = 48;
/// Size of `rpc_message_header_v03_00` without the flexible payload.
pub const RPC_HDR_SIZE: usize = 32;
pub const HEADER_VERSION: u32 = 0x0300_0000;
pub const SIGNATURE: u32 = 0x4350_5256;
/// Largest payload that fits in 16 elements.
pub const MAX_PAYLOAD: usize =
    ELEMENT_SIZE * MAX_ELEMENTS as usize - ELEMENT_HDR_SIZE - RPC_HDR_SIZE;

/// Function numbers (`rpc_global_enums.h`, 570.144). Only the ones the boot
/// sequence and the first compute work need.
pub mod function {
    pub const NOP: u32 = 0;
    pub const GET_GSP_STATIC_INFO: u32 = 65;
    pub const GSP_SET_SYSTEM_INFO: u32 = 72;
    pub const SET_REGISTRY: u32 = 73;
    /// Carries the rest of an RPC larger than one message (`rpc.c`
    /// `_issueRpcLarge`: same header, function replaced, payload continues).
    pub const CONTINUATION_RECORD: u32 = 71;
    pub const GSP_RM_CONTROL: u32 = 76;
    pub const GSP_RM_ALLOC: u32 = 103;
}

/// Events GSP-RM sends on its own (`rpc_global_enums.h`, 570.144).
pub mod event {
    pub const GSP_INIT_DONE: u32 = 0x1001;
    pub const GSP_RUN_CPU_SEQUENCER: u32 = 0x1002;
    pub const POST_EVENT: u32 = 0x1003;
    pub const RC_TRIGGERED: u32 = 0x1004;
    pub const MMU_FAULT_QUEUED: u32 = 0x1005;
    pub const OS_ERROR_LOG: u32 = 0x1006;
    pub const UCODE_LIBOS_PRINT: u32 = 0x100c;
    pub const GSP_LOCKDOWN_NOTICE: u32 = 0x101c;
}

mod off {
    pub const CHECKSUM: usize = 32;
    pub const SEQ: usize = 36;
    pub const ELEM_COUNT: usize = 40;
    pub const R: usize = super::ELEMENT_HDR_SIZE;
    pub const HEADER_VERSION: usize = R;
    pub const SIGNATURE: usize = R + 4;
    pub const LENGTH: usize = R + 8;
    pub const FUNCTION: usize = R + 12;
    pub const RESULT: usize = R + 16;
    pub const RESULT_PRIVATE: usize = R + 20;
    pub const SEQUENCE: usize = R + 24;
    pub const PAYLOAD: usize = R + super::RPC_HDR_SIZE;
}

/// XOR checksum over `data` (length padded with zeros to 8 bytes).
pub fn checksum32(data: &[u8]) -> u32 {
    let mut x = 0u64;
    for c in data.chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        x ^= u64::from_le_bytes(w);
    }
    (x >> 32) as u32 ^ x as u32
}

/// Elements a message needs.
pub fn element_count(first_element: &[u8]) -> u32 {
    get_u32(first_element, off::ELEM_COUNT)
}

/// Builds one message (not padded to 4 KiB; the queue pads).
pub fn encode(function: u32, queue_seq: u32, rpc_seq: u32, payload: &[u8]) -> GpuResult<Vec<u8>> {
    if payload.len() > MAX_PAYLOAD {
        return Err(GpuError::InvalidArgument(
            "RPC: mesaj mai mare de 16 elemente de 4 KiB",
        ));
    }
    let len = off::PAYLOAD + payload.len();
    let mut m = alloc::vec![0u8; len];
    put_u32(&mut m, off::SEQ, queue_seq);
    put_u32(&mut m, off::ELEM_COUNT, len.div_ceil(ELEMENT_SIZE) as u32);
    put_u32(&mut m, off::HEADER_VERSION, HEADER_VERSION);
    put_u32(&mut m, off::SIGNATURE, SIGNATURE);
    put_u32(&mut m, off::LENGTH, (RPC_HDR_SIZE + payload.len()) as u32);
    put_u32(&mut m, off::FUNCTION, function);
    put_u32(&mut m, off::SEQUENCE, rpc_seq);
    m[off::PAYLOAD..].copy_from_slice(payload);
    let c = checksum32(&m);
    put_u32(&mut m, off::CHECKSUM, c);
    debug_assert_eq!(checksum32(&m), 0);
    Ok(m)
}

/// Largest RPC payload [`encode_split`] accepts (our limit, not NVIDIA's):
/// the boot RPCs are a few KiB; anything near this is a bug on our side.
pub const MAX_SPLIT_PAYLOAD: usize = 1 << 20;

/// Builds an RPC that may need more than one queue message: the first message
/// has `function` and the first [`MAX_PAYLOAD`] bytes, each following one has
/// [`function::CONTINUATION_RECORD`], the same `rpc_seq` and the next part.
/// Queue sequence numbers go `first_queue_seq`, +1, +2, … (one per message).
/// Rule read in open-gpu-kernel-modules@570.144 `src/nvidia/src/kernel/vgpu/rpc.c`
/// (MIT), `_issueRpcLarge`; re-written. Also what tinygrad does (study, lesson 9).
pub fn encode_split(
    function: u32,
    first_queue_seq: u32,
    rpc_seq: u32,
    payload: &[u8],
) -> GpuResult<Vec<Vec<u8>>> {
    if payload.len() > MAX_SPLIT_PAYLOAD {
        return Err(GpuError::InvalidArgument("RPC: mesaj mai mare de 1 MiB"));
    }
    let mut out = Vec::new();
    let mut parts = payload.chunks(MAX_PAYLOAD);
    let first = parts.next().unwrap_or(&[]);
    out.push(encode(function, first_queue_seq, rpc_seq, first)?);
    for (i, part) in parts.enumerate() {
        let qseq = first_queue_seq.wrapping_add(i as u32 + 1);
        out.push(encode(function::CONTINUATION_RECORD, qseq, rpc_seq, part)?);
    }
    Ok(out)
}

/// Puts back together an RPC received as several messages (inverse of
/// [`encode_split`]). The first must not be a continuation; every following
/// one must be a continuation with the same RPC sequence and consecutive
/// queue sequence numbers; otherwise the whole RPC is refused.
pub fn join_split(parts: &[Message]) -> GpuResult<Message> {
    let (first, rest) = parts
        .split_first()
        .ok_or(GpuError::InvalidArgument("RPC: niciun mesaj de unit"))?;
    if first.function == function::CONTINUATION_RECORD {
        return Err(GpuError::InvalidArgument(
            "RPC: o continuare fara mesajul de inceput",
        ));
    }
    let mut whole = first.clone();
    for (i, m) in rest.iter().enumerate() {
        if m.function != function::CONTINUATION_RECORD
            || m.sequence != first.sequence
            || m.queue_seq != first.queue_seq.wrapping_add(i as u32 + 1)
        {
            return Err(GpuError::InvalidArgument(
                "RPC: continuare nepotrivita (functie, secventa sau ordine)",
            ));
        }
        if whole.payload.len() + m.payload.len() > MAX_SPLIT_PAYLOAD {
            return Err(GpuError::InvalidArgument(
                "RPC: mesaj unit mai mare de 1 MiB",
            ));
        }
        whole.payload.extend_from_slice(&m.payload);
        whole.elem_count += m.elem_count;
    }
    Ok(whole)
}

/// A received message, checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub function: u32,
    pub rpc_result: u32,
    pub rpc_result_private: u32,
    pub sequence: u32,
    pub queue_seq: u32,
    pub elem_count: u32,
    pub payload: Vec<u8>,
}

impl Message {
    pub fn is_event(&self) -> bool {
        self.function >= 0x1000
    }
}

/// Checks and splits a message read from the status queue (`raw` = whole
/// elements). Bad signature, version, length or checksum is an error: the
/// driver never acts on a message it cannot trust.
pub fn decode(raw: &[u8]) -> GpuResult<Message> {
    if raw.len() < off::PAYLOAD {
        return Err(GpuError::InvalidArgument("RPC: mesaj trunchiat"));
    }
    let elem_count = get_u32(raw, off::ELEM_COUNT);
    if get_u32(raw, off::SIGNATURE) != SIGNATURE {
        return Err(GpuError::InvalidArgument("RPC: semnatura 'VRPC' lipsa"));
    }
    if get_u32(raw, off::HEADER_VERSION) >> 24 != HEADER_VERSION >> 24 {
        return Err(GpuError::InvalidArgument(
            "RPC: versiunea antetului nu e 3 (alt ABI decat 570.144)",
        ));
    }
    let length = get_u32(raw, off::LENGTH) as usize;
    let total = ELEMENT_HDR_SIZE + length;
    if length < RPC_HDR_SIZE
        || total > raw.len()
        || total.div_ceil(ELEMENT_SIZE) as u32 != elem_count
    {
        return Err(GpuError::InvalidArgument(
            "RPC: lungime nepotrivita cu numarul de elemente",
        ));
    }
    if checksum32(&raw[..total]) != 0 {
        return Err(GpuError::InvalidArgument("RPC: suma de control gresita"));
    }
    Ok(Message {
        function: get_u32(raw, off::FUNCTION),
        rpc_result: get_u32(raw, off::RESULT),
        rpc_result_private: get_u32(raw, off::RESULT_PRIVATE),
        sequence: get_u32(raw, off::SEQUENCE),
        queue_seq: get_u32(raw, off::SEQ),
        elem_count,
        payload: raw[off::PAYLOAD..total].to_vec(),
    })
}

/// Header of `GSP_RUN_CPU_SEQUENCER` (`rpc_run_cpu_sequencer_v17_00`):
/// buffer size in u32 words, current command index, 8 saved registers, then
/// the command words. Decoding only: *executing* the commands means register
/// writes and stays behind the hardware permit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequencerHeader {
    pub buffer_size_dwords: u32,
    pub cmd_index: u32,
    pub reg_save: [u32; 8],
    pub commands: Vec<u32>,
}

pub fn decode_sequencer(payload: &[u8]) -> GpuResult<SequencerHeader> {
    if payload.len() < 40 || !payload.len().is_multiple_of(4) {
        return Err(GpuError::InvalidArgument("secventiator: antet trunchiat"));
    }
    let words = (payload.len() - 40) / 4;
    let size = get_u32(payload, 0);
    let index = get_u32(payload, 4);
    if size as usize > words || index > size {
        return Err(GpuError::InvalidArgument(
            "secventiator: marime sau index in afara bufferului",
        ));
    }
    let mut reg_save = [0u32; 8];
    for (i, r) in reg_save.iter_mut().enumerate() {
        *r = get_u32(payload, 8 + 4 * i);
    }
    let commands = (0..size as usize)
        .map(|i| get_u32(payload, 40 + 4 * i))
        .collect();
    Ok(SequencerHeader {
        buffer_size_dwords: size,
        cmd_index: index,
        reg_save,
        commands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_shape_matches_570_144() {
        // authTag 16 + aad 16 + 3 x u32 = 44 -> rpc aligned to 8 = 48.
        assert_eq!(ELEMENT_HDR_SIZE, 48);
        // 8 x u32 in rpc_message_header_v03_00 before rpc_message_data[].
        assert_eq!(RPC_HDR_SIZE, 32);
        assert_eq!(&SIGNATURE.to_le_bytes(), b"VRPC");
        assert_eq!(MAX_PAYLOAD, 65_536 - 80);
    }

    #[test]
    fn encode_decode_roundtrip_and_zero_checksum() {
        for len in [0usize, 1, 7, 8, 4016, 4017, 10_000, MAX_PAYLOAD] {
            let p: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let m = encode(function::SET_REGISTRY, 5, 9, &p).unwrap();
            assert_eq!(checksum32(&m), 0);
            let mut raw = m.clone();
            raw.resize(m.len().div_ceil(ELEMENT_SIZE) * ELEMENT_SIZE, 0);
            let d = decode(&raw).unwrap();
            assert_eq!(d.function, function::SET_REGISTRY);
            assert_eq!((d.queue_seq, d.sequence), (5, 9));
            assert_eq!(d.payload, p);
            assert_eq!(d.elem_count as usize, raw.len() / ELEMENT_SIZE);
        }
        assert!(encode(0, 0, 0, &alloc::vec![0; MAX_PAYLOAD + 1]).is_err());
    }

    #[test]
    fn corrupted_messages_are_refused() {
        let m = encode(function::NOP, 0, 0, b"abcdefgh12345").unwrap();
        let mut raw = m.clone();
        raw.resize(ELEMENT_SIZE, 0);
        let mut c = raw.clone();
        c[85] ^= 0x40; // one payload bit
        assert!(matches!(decode(&c), Err(GpuError::InvalidArgument(w)) if w.contains("suma")));
        let mut c = raw.clone();
        c[off::SIGNATURE] = b'X';
        assert!(decode(&c).is_err());
        let mut c = raw.clone();
        put_u32(&mut c, off::HEADER_VERSION, 0x0200_0000);
        assert!(decode(&c).is_err());
        let mut c = raw.clone();
        put_u32(&mut c, off::LENGTH, 9000); // claims 3 elements, has 1
        assert!(decode(&c).is_err());
        assert!(decode(&raw[..40]).is_err());
    }

    #[test]
    fn sequencer_header_decodes_without_executing() {
        let mut p = alloc::vec![0u8; 40 + 4 * 6];
        put_u32(&mut p, 0, 4);
        put_u32(&mut p, 4, 0);
        put_u32(&mut p, 8, 0xdead);
        for i in 0..6 {
            put_u32(&mut p, 40 + 4 * i, 100 + i as u32);
        }
        let s = decode_sequencer(&p).unwrap();
        assert_eq!(s.commands, [100, 101, 102, 103]);
        assert_eq!(s.reg_save[0], 0xdead);
        put_u32(&mut p, 0, 7); // more words than present
        assert!(decode_sequencer(&p).is_err());
    }

    #[test]
    fn large_rpc_splits_into_continuation_records_and_joins_back() {
        // 20 elements' worth of payload: 16 in the first message, rest after.
        let payload: Vec<u8> = (0..20 * ELEMENT_SIZE).map(|i| (i * 7 + 3) as u8).collect();
        let msgs = encode_split(function::GSP_SET_SYSTEM_INFO, 40, 9, &payload).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(element_count(&msgs[0]), 16);
        let decoded: Vec<Message> = msgs.iter().map(|m| decode(m).unwrap()).collect();
        assert_eq!(decoded[0].function, function::GSP_SET_SYSTEM_INFO);
        assert_eq!(decoded[0].payload.len(), MAX_PAYLOAD);
        assert_eq!(decoded[1].function, function::CONTINUATION_RECORD);
        assert_eq!((decoded[1].sequence, decoded[1].queue_seq), (9, 41));
        let whole = join_split(&decoded).unwrap();
        assert_eq!(whole.payload, payload);
        assert_eq!(whole.function, function::GSP_SET_SYSTEM_INFO);

        // A small RPC stays one message.
        assert_eq!(
            encode_split(function::NOP, 0, 0, &[1, 2, 3]).unwrap().len(),
            1
        );
        assert!(encode_split(function::NOP, 0, 0, &vec![0; MAX_SPLIT_PAYLOAD + 1]).is_err());

        // Wrong order, a foreign sequence or a lone continuation are refused.
        let swapped = [decoded[1].clone(), decoded[0].clone()];
        assert!(join_split(&swapped).is_err());
        let mut other = decoded.clone();
        other[1].sequence = 10;
        assert!(join_split(&other).is_err());
        let mut gap = decoded.clone();
        gap[1].queue_seq = 43;
        assert!(join_split(&gap).is_err());
        assert!(join_split(&[]).is_err());
    }
}
