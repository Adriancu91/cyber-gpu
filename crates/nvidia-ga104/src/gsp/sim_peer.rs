//! SIMULARE: a stand-in for GSP-RM on the other side of the queues, written
//! independently of `msgq`/`rpc` (raw offsets straight from the 570.144
//! headers), so the tests catch asymmetric mistakes. It proves nothing about
//! the real GSP-RM; it only checks that both sides of *our reading* of the
//! ABI agree.

use alloc::vec::Vec;

use super::msgq::SharedMem;

pub struct SimGsp {
    stat: usize,
    cmd: usize,
    count: u32,
    write: u32,
    read: u32,
    seq: u32,
    pub served: Vec<(u32, u32)>,
}

fn rd(mem: &dyn SharedMem, o: usize) -> u32 {
    mem.read_u32(o)
}

impl SimGsp {
    /// What GSP-RM does when it starts: write the status-queue header.
    pub fn start(mem: &mut dyn SharedMem, cmd: usize, stat: usize) -> Self {
        let count = (0x40000 - 0x1000) / 0x1000;
        for (o, v) in [
            (0, 0),
            (4, 0x40000),
            (8, 0x1000),
            (12, count),
            (16, 0),
            (20, 1),
            (24, 32),
            (28, 0x1000),
        ] {
            mem.write_u32(stat + o, v);
        }
        SimGsp {
            stat,
            cmd,
            count,
            write: 0,
            read: 0,
            seq: 0,
            served: Vec::new(),
        }
    }

    /// Posts one message (function, result, payload) into the status queue.
    pub fn post(&mut self, mem: &mut dyn SharedMem, function: u32, result: u32, payload: &[u8]) {
        let len = 80 + payload.len();
        let n = len.div_ceil(0x1000) as u32;
        let mut m = alloc::vec![0u8; n as usize * 0x1000];
        let put = |m: &mut [u8], o: usize, v: u32| m[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put(&mut m, 36, self.seq);
        put(&mut m, 40, n);
        put(&mut m, 48, 0x0300_0000);
        put(&mut m, 52, 0x4350_5256);
        put(&mut m, 56, (32 + payload.len()) as u32);
        put(&mut m, 60, function);
        put(&mut m, 64, result);
        put(&mut m, 72, self.seq);
        m[80..len].copy_from_slice(payload);
        let mut x = 0u64;
        for w in m[..len.div_ceil(8) * 8].chunks(8) {
            x ^= u64::from_le_bytes(w.try_into().unwrap());
        }
        put(&mut m, 32, (x >> 32) as u32 ^ x as u32);
        for i in 0..n {
            let slot = (self.write + i) % self.count;
            let off = self.stat + 0x1000 + slot as usize * 0x1000;
            mem.write(off, &m[i as usize * 0x1000..(i as usize + 1) * 0x1000]);
        }
        self.write = (self.write + n) % self.count;
        mem.write_u32(self.stat + 16, self.write);
        self.seq += 1;
    }

    /// Consumes every command and answers each with the same function, result 0.
    pub fn service(&mut self, mem: &mut dyn SharedMem) {
        loop {
            let w = rd(mem, self.cmd + 16);
            if w == self.read {
                break;
            }
            let base = self.cmd + 0x1000 + self.read as usize * 0x1000;
            let n = rd(mem, base + 40);
            let function = rd(mem, base + 60);
            let seq = rd(mem, base + 72);
            self.served.push((function, seq));
            self.read = (self.read + n) % self.count;
            // GSP's read pointer of the command queue: status queue rx header.
            mem.write_u32(self.stat + 32, self.read);
            self.post(mem, function, 0, &seq.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::msgq::{CpuQueues, QueueLayout};
    use crate::gsp::rpc::{self, event, function};
    use gpu_core::error::GpuError;

    fn setup() -> (Vec<u8>, CpuQueues, SimGsp) {
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        let g = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        (mem, q, g)
    }

    #[test]
    fn boot_handshake_simulated() {
        let (mut mem, mut q, mut g) = setup();
        q.link_rx(&mem).unwrap();
        g.post(&mut mem, event::GSP_INIT_DONE, 0, &[]);
        let m = q.recv_rpc(&mut mem).unwrap().unwrap();
        assert!(m.is_event());
        assert_eq!(m.function, event::GSP_INIT_DONE);
        assert_eq!(q.recv_rpc(&mut mem).unwrap(), None);
        let s0 = q
            .send_rpc(&mut mem, function::GSP_SET_SYSTEM_INFO, &[1; 900])
            .unwrap();
        let s1 = q
            .send_rpc(&mut mem, function::SET_REGISTRY, &[2; 5000])
            .unwrap();
        g.service(&mut mem);
        assert_eq!(
            g.served,
            [
                (function::GSP_SET_SYSTEM_INFO, s0),
                (function::SET_REGISTRY, s1)
            ]
        );
        for (f, s) in [
            (function::GSP_SET_SYSTEM_INFO, s0),
            (function::SET_REGISTRY, s1),
        ] {
            let m = q.recv_rpc(&mut mem).unwrap().unwrap();
            assert_eq!((m.function, m.rpc_result), (f, 0));
            assert_eq!(m.payload, s.to_le_bytes());
        }
    }

    #[test]
    fn many_messages_wrap_around_both_rings() {
        let (mut mem, mut q, mut g) = setup();
        q.link_rx(&mem).unwrap();
        for i in 0..300u32 {
            // 1..=3 elements, so the 63-slot rings wrap at odd places.
            let len = [10usize, 5000, 9000][(i % 3) as usize];
            let p: Vec<u8> = (0..len).map(|k| (k as u32 ^ i) as u8).collect();
            let s = q.send_rpc(&mut mem, function::GSP_RM_CONTROL, &p).unwrap();
            g.service(&mut mem);
            let m = q.recv_rpc(&mut mem).unwrap().unwrap();
            assert_eq!(m.payload, s.to_le_bytes());
        }
        assert_eq!(g.served.len(), 300);
    }

    #[test]
    fn full_queue_is_an_error_not_an_overwrite() {
        let (mut mem, mut q, _g) = setup();
        // 62 free elements; 2-element messages: 31 fit, the 32nd must fail.
        for _ in 0..31 {
            q.send_rpc(&mut mem, function::NOP, &[0; 5000]).unwrap();
        }
        assert!(matches!(
            q.send_rpc(&mut mem, function::NOP, &[0; 5000]),
            Err(GpuError::OutOfMemory { .. })
        ));
        assert_eq!(q.tx_free(&mem).unwrap(), 0);
    }

    #[test]
    fn corrupted_or_out_of_order_replies_are_refused() {
        let (mut mem, mut q, mut g) = setup();
        q.link_rx(&mem).unwrap();
        g.post(&mut mem, event::OS_ERROR_LOG, 0, b"x");
        // Flip a payload byte in the status ring.
        let at = q.layout.stat_queue_offset + 0x1000 + 80;
        mem[at] ^= 1;
        assert!(q.recv_rpc(&mut mem).is_err());
        // A message with the wrong queue sequence number.
        let (mut mem, mut q, mut g) = setup();
        q.link_rx(&mem).unwrap();
        g.seq = 7;
        g.post(&mut mem, event::POST_EVENT, 0, &[]);
        assert!(
            matches!(q.recv_rpc(&mut mem), Err(GpuError::InvalidArgument(w)) if w.contains("secventa"))
        );
        // A write pointer past the ring is a fault, not a read.
        let (mut mem, mut q, _g) = setup();
        q.link_rx(&mem).unwrap();
        mem.write_u32(q.layout.stat_queue_offset + 16, 63);
        assert!(matches!(
            q.recv_raw(&mut mem),
            Err(GpuError::HardwareFault { .. })
        ));
        let _ = rpc::checksum32(&[]);
    }
}
