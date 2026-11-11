//! The shared memory between the driver and GSP-RM: a page table, the command
//! queue (driver → GSP) and the status queue (GSP → driver). I5 allocates it,
//! I8 talks through it.
//!
//! Layout (open-gpu-kernel-modules@570.144, MIT: `message_queue_priv.h`,
//! `msgq_priv.h`, `msgq.c`, `message_queue_cpu.c` — read, re-written):
//!
//! ```text
//! 0x00000  page table: one u64 DMA address per 4 KiB page of the whole area
//!          (itself included), 129 entries -> padded to one page
//! 0x01000  command queue, 0x40000 bytes:
//!            +0x000 msgqTxHeader (8 x u32)  written by the driver
//!            +0x020 msgqRxHeader (readPtr)  the driver's read pointer of the
//!                                           STATUS queue (MSGQ_FLAGS_SWAP_RX)
//!            +0x1000 63 elements of 4 KiB
//! 0x41000  status queue, 0x40000 bytes, same shape, tx header written by GSP;
//!            its +0x020 holds GSP's read pointer of the COMMAND queue
//! ```
//!
//! A message occupies 1..=16 consecutive elements (they may wrap). Pointers are
//! element indices modulo `msgCount`. Free space = read + count − write − 1.
//!
//! Memory ordering: [`SharedMem::fence`] must be a store/full fence on real
//! memory (the element bytes must be visible before the write pointer). The
//! in-memory implementation used by the tests is SIMULARE.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use super::rpc;

pub const PAGE: usize = 4096;
/// Default size of each queue in the 570.144 driver (`message_queue_cpu.c`).
pub const QUEUE_SIZE: usize = 0x40000;
/// One element (`GSP_MSG_QUEUE_ELEMENT_SIZE_MIN`).
pub const ELEMENT_SIZE: usize = PAGE;
/// A message may span at most 16 elements (`..._SIZE_MAX`).
pub const MAX_ELEMENTS: u32 = 16;
pub const MSGQ_VERSION: u32 = 0;
pub const MSGQ_FLAGS_SWAP_RX: u32 = 1;

/// Offsets inside one queue.
pub mod qoff {
    pub const VERSION: usize = 0;
    pub const SIZE: usize = 4;
    pub const MSG_SIZE: usize = 8;
    pub const MSG_COUNT: usize = 12;
    pub const WRITE_PTR: usize = 16;
    pub const FLAGS: usize = 20;
    pub const RX_HDR_OFF: usize = 24;
    pub const ENTRY_OFF: usize = 28;
    /// msgqTxHeader is 32 bytes; header alignment 2^4 keeps the rx header at 32.
    pub const RX_HDR: usize = 32;
    /// Entries aligned to 2^12.
    pub const ENTRIES: usize = 4096;
}

/// Where everything is inside the shared area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueLayout {
    pub cmd_size: usize,
    pub stat_size: usize,
    pub page_table_entries: usize,
    pub page_table_size: usize,
    pub cmd_queue_offset: usize,
    pub stat_queue_offset: usize,
    pub total_size: usize,
}

impl QueueLayout {
    /// 0x40000 + 0x40000, as the 570.144 driver uses by default.
    pub fn standard() -> Self {
        Self::new(QUEUE_SIZE, QUEUE_SIZE).expect("marimile standard sunt valide")
    }

    pub fn new(cmd_size: usize, stat_size: usize) -> GpuResult<Self> {
        let min = qoff::ENTRIES + ELEMENT_SIZE * MAX_ELEMENTS as usize;
        if !cmd_size.is_multiple_of(PAGE)
            || !stat_size.is_multiple_of(PAGE)
            || cmd_size < min
            || stat_size < min
        {
            return Err(GpuError::InvalidArgument(
                "cozi GSP: marime nealiniata la 4 KiB sau prea mica pentru un mesaj de 16 elemente",
            ));
        }
        let queue_pages = (cmd_size + stat_size) / PAGE;
        // The page table describes itself too.
        let entries = queue_pages + (queue_pages * 8).div_ceil(PAGE);
        let page_table_size = (entries * 8).div_ceil(PAGE) * PAGE;
        Ok(QueueLayout {
            cmd_size,
            stat_size,
            page_table_entries: entries,
            page_table_size,
            cmd_queue_offset: page_table_size,
            stat_queue_offset: page_table_size + cmd_size,
            total_size: page_table_size + cmd_size + stat_size,
        })
    }

    pub fn msg_count(queue_size: usize) -> u32 {
        ((queue_size - qoff::ENTRIES) / ELEMENT_SIZE) as u32
    }
}

/// The shared area as the driver sees it. The kernel implements it over the
/// DMA buffer with volatile accesses; tests implement it over a byte vector.
pub trait SharedMem {
    fn size(&self) -> usize;
    fn read(&self, off: usize, buf: &mut [u8]);
    fn write(&mut self, off: usize, data: &[u8]);
    /// Order all previous writes before the next ones (store fence).
    fn fence(&mut self) {}

    fn read_u32(&self, off: usize) -> u32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        u32::from_le_bytes(b)
    }
    fn write_u32(&mut self, off: usize, v: u32) {
        self.write(off, &v.to_le_bytes());
    }
}

/// SIMULARE: plain memory.
impl SharedMem for Vec<u8> {
    fn size(&self) -> usize {
        self.len()
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self[off..off + buf.len()]);
    }
    fn write(&mut self, off: usize, data: &[u8]) {
        self[off..off + data.len()].copy_from_slice(data);
    }
}

/// Driver-side state of both queues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuQueues {
    pub layout: QueueLayout,
    tx_count: u32,
    tx_write: u32,
    rx_count: Option<u32>,
    rx_read: u32,
    tx_seq: u32,
    rx_seq: u32,
}

impl CpuQueues {
    /// I5: writes the page table and the command-queue header into a zeroed
    /// area. `page_addrs` = DMA address of every page of the area, in order.
    pub fn init(
        mem: &mut dyn SharedMem,
        layout: QueueLayout,
        page_addrs: &[u64],
    ) -> GpuResult<Self> {
        if mem.size() < layout.total_size {
            return Err(GpuError::InvalidArgument(
                "cozi GSP: zona partajata mai mica decat aranjamentul",
            ));
        }
        if page_addrs.len() != layout.page_table_entries
            || page_addrs
                .iter()
                .any(|&a| a == 0 || !a.is_multiple_of(PAGE as u64))
        {
            return Err(GpuError::InvalidArgument(
                "cozi GSP: tabela de pagini incompleta sau adrese nealiniate",
            ));
        }
        for (i, a) in page_addrs.iter().enumerate() {
            mem.write(i * 8, &a.to_le_bytes());
        }
        let q = layout.cmd_queue_offset;
        let count = QueueLayout::msg_count(layout.cmd_size);
        let hdr = [
            (qoff::VERSION, MSGQ_VERSION),
            (qoff::SIZE, layout.cmd_size as u32),
            (qoff::MSG_SIZE, ELEMENT_SIZE as u32),
            (qoff::MSG_COUNT, count),
            (qoff::WRITE_PTR, 0),
            (qoff::FLAGS, MSGQ_FLAGS_SWAP_RX),
            (qoff::RX_HDR_OFF, qoff::RX_HDR as u32),
            (qoff::ENTRY_OFF, qoff::ENTRIES as u32),
        ];
        for (o, v) in hdr {
            mem.write_u32(q + o, v);
        }
        mem.write_u32(q + qoff::RX_HDR, 0);
        mem.fence();
        Ok(CpuQueues {
            layout,
            tx_count: count,
            tx_write: 0,
            rx_count: None,
            rx_read: 0,
            tx_seq: 0,
            rx_seq: 0,
        })
    }

    /// I8: after GSP-RM started, checks the status-queue header it wrote and
    /// links the receive side. Refuses a header that is not the 570.144 shape.
    pub fn link_rx(&mut self, mem: &dyn SharedMem) -> GpuResult<()> {
        let q = self.layout.stat_queue_offset;
        let r = |o| mem.read_u32(q + o);
        let count = QueueLayout::msg_count(self.layout.stat_size);
        let ok = r(qoff::VERSION) == MSGQ_VERSION
            && r(qoff::SIZE) as usize == self.layout.stat_size
            && r(qoff::MSG_SIZE) as usize == ELEMENT_SIZE
            && r(qoff::MSG_COUNT) == count
            && r(qoff::RX_HDR_OFF) as usize == qoff::RX_HDR
            && r(qoff::ENTRY_OFF) as usize == qoff::ENTRIES
            && r(qoff::FLAGS) & MSGQ_FLAGS_SWAP_RX != 0
            && r(qoff::WRITE_PTR) < count;
        if !ok {
            return Err(GpuError::Blocked(
                "cozi GSP: antetul cozii de stare nu are forma 570.144 (GSP nu a pornit sau alta versiune)",
            ));
        }
        self.rx_count = Some(count);
        Ok(())
    }

    /// Free elements in the command queue (GSP's read pointer lives in the
    /// status queue's rx header: MSGQ_FLAGS_SWAP_RX).
    pub fn tx_free(&self, mem: &dyn SharedMem) -> GpuResult<u32> {
        let read = mem.read_u32(self.layout.stat_queue_offset + qoff::RX_HDR);
        if read >= self.tx_count {
            return Err(GpuError::HardwareFault { code: read });
        }
        Ok((read + self.tx_count - self.tx_write - 1) % self.tx_count)
    }

    /// Writes one already-encoded element sequence (see [`rpc`]) and publishes
    /// it by moving the write pointer. The doorbell (a GSP register write) is
    /// NOT rung here: that is hardware, behind `HardwareWritePermit`.
    pub fn send_raw(&mut self, mem: &mut dyn SharedMem, msg: &[u8]) -> GpuResult<()> {
        let n = msg.len().div_ceil(ELEMENT_SIZE);
        if n == 0 || n > MAX_ELEMENTS as usize {
            return Err(GpuError::InvalidArgument(
                "cozi GSP: mesaj gol sau mai mare de 16 elemente",
            ));
        }
        if (n as u32) > self.tx_free(mem)? {
            return Err(GpuError::OutOfMemory {
                requested: (n * ELEMENT_SIZE) as u64,
            });
        }
        let base = self.layout.cmd_queue_offset + qoff::ENTRIES;
        let mut page = [0u8; ELEMENT_SIZE];
        for (i, chunk) in msg.chunks(ELEMENT_SIZE).enumerate() {
            page.fill(0);
            page[..chunk.len()].copy_from_slice(chunk);
            let slot = (self.tx_write as usize + i) % self.tx_count as usize;
            mem.write(base + slot * ELEMENT_SIZE, &page);
        }
        mem.fence();
        self.tx_write = (self.tx_write + n as u32) % self.tx_count;
        mem.write_u32(
            self.layout.cmd_queue_offset + qoff::WRITE_PTR,
            self.tx_write,
        );
        mem.fence();
        Ok(())
    }

    /// Reads the next whole message from the status queue, if any, and
    /// releases its elements. Returns the raw elements (n × 4 KiB).
    pub fn recv_raw(&mut self, mem: &mut dyn SharedMem) -> GpuResult<Option<Vec<u8>>> {
        let count = self.rx_count.ok_or(GpuError::Blocked(
            "cozi GSP: coada de stare nelegata (link_rx inainte)",
        ))?;
        let q = self.layout.stat_queue_offset;
        let write = mem.read_u32(q + qoff::WRITE_PTR);
        if write >= count {
            return Err(GpuError::HardwareFault { code: write });
        }
        let avail = (write + count - self.rx_read) % count;
        if avail == 0 {
            return Ok(None);
        }
        let base = q + qoff::ENTRIES;
        let slot = |i: u32| base + ((self.rx_read + i) % count) as usize * ELEMENT_SIZE;
        let mut first = alloc::vec![0u8; ELEMENT_SIZE];
        mem.read(slot(0), &mut first);
        let n = rpc::element_count(&first);
        if n == 0 || n > MAX_ELEMENTS || n > avail {
            return Err(GpuError::InvalidArgument(
                "cozi GSP: numar de elemente invalid in mesajul primit",
            ));
        }
        let mut msg = first;
        for i in 1..n {
            let mut page = [0u8; ELEMENT_SIZE];
            mem.read(slot(i), &mut page);
            msg.extend_from_slice(&page);
        }
        self.rx_read = (self.rx_read + n) % count;
        // Our read pointer of the status queue lives in the command queue.
        mem.write_u32(self.layout.cmd_queue_offset + qoff::RX_HDR, self.rx_read);
        mem.fence();
        Ok(Some(msg))
    }

    /// How many RPCs the driver has put in the command queue so far (the
    /// next sequence number). C-108: I8 checks that the two init RPCs were
    /// queued **before** Booter (I7) when the variant asks for it.
    pub fn rpcs_sent(&self) -> u32 {
        self.tx_seq
    }

    /// Encodes and sends one RPC (driver → GSP-RM) with the next sequence number.
    pub fn send_rpc(
        &mut self,
        mem: &mut dyn SharedMem,
        function: u32,
        payload: &[u8],
    ) -> GpuResult<u32> {
        let seq = self.tx_seq;
        let msg = rpc::encode(function, seq, seq, payload)?;
        self.send_raw(mem, &msg)?;
        self.tx_seq = self.tx_seq.wrapping_add(1);
        Ok(seq)
    }

    /// Receives and checks one message (checksum, header, sequence number).
    pub fn recv_rpc(&mut self, mem: &mut dyn SharedMem) -> GpuResult<Option<rpc::Message>> {
        let Some(raw) = self.recv_raw(mem)? else {
            return Ok(None);
        };
        let m = rpc::decode(&raw)?;
        if m.queue_seq != self.rx_seq {
            return Err(GpuError::InvalidArgument(
                "RPC: numar de secventa gresit in coada de stare",
            ));
        }
        self.rx_seq = self.rx_seq.wrapping_add(1);
        Ok(Some(m))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_layout_matches_the_570_144_driver() {
        let l = QueueLayout::standard();
        // 128 queue pages + 1 page for 129 PTEs.
        assert_eq!(l.page_table_entries, 129);
        assert_eq!(l.page_table_size, 0x1000);
        assert_eq!(l.cmd_queue_offset, 0x1000);
        assert_eq!(l.stat_queue_offset, 0x41000);
        assert_eq!(l.total_size, 0x81000);
        assert_eq!(QueueLayout::msg_count(QUEUE_SIZE), 63);
        assert!(QueueLayout::new(0x40001, QUEUE_SIZE).is_err());
        assert!(QueueLayout::new(0x8000, QUEUE_SIZE).is_err());
    }

    #[test]
    fn init_writes_page_table_and_command_header_only() {
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0400_0000 + i * 0x1000)
            .collect();
        let q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        assert_eq!(mem.read_u32(0), 0x0400_0000);
        assert_eq!(mem.read_u32(128 * 8), 0x0400_0000 + 128 * 0x1000);
        let c = l.cmd_queue_offset;
        assert_eq!(mem.read_u32(c + qoff::SIZE), 0x40000);
        assert_eq!(mem.read_u32(c + qoff::MSG_COUNT), 63);
        assert_eq!(mem.read_u32(c + qoff::FLAGS), 1);
        assert_eq!(mem.read_u32(c + qoff::ENTRY_OFF), 0x1000);
        // Status queue untouched: GSP writes it.
        assert!(mem[l.stat_queue_offset..].iter().all(|&b| b == 0));
        assert_eq!(q.tx_free(&mem).unwrap(), 62);
        // Bad page lists are refused.
        let mut m2 = alloc::vec![0u8; l.total_size];
        assert!(CpuQueues::init(&mut m2, l, &pages[1..]).is_err());
        let mut odd = pages.clone();
        odd[5] += 8;
        assert!(CpuQueues::init(&mut m2, l, &odd).is_err());
    }

    #[test]
    fn rx_refused_until_gsp_wrote_a_570_header() {
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..129u64).map(|i| 0x0400_0000 + i * 0x1000).collect();
        let mut q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        assert!(matches!(q.recv_raw(&mut mem), Err(GpuError::Blocked(_))));
        assert!(matches!(q.link_rx(&mem), Err(GpuError::Blocked(_))));
    }
}
