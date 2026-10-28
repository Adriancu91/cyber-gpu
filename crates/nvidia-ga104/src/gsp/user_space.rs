//! I9c — the driver's working client and its VA space, on the driver's own
//! page tables. Built and tested **only in SIMULARE** (against
//! `gpu_sim::fake_rm` behind the real queue code); nothing here has run on
//! the board, and nothing here writes a register.
//!
//! After the golden image (I8e), tinygrad (`ops_nv.py` `PCIIface` +
//! `NVDevice.__init__`, `ip.py` `rpc_rm_alloc` / `rpc_set_page_directory`,
//! MIT, commit 9cf7829a6; read as the specification and re-written) opens a
//! second client for the real work:
//! 1. client (`NV01_ROOT`, handle `0xc1000000`), device, subdevice;
//! 2. a VA space that is *externally owned* (the driver makes the page
//!    tables) with page faulting on, `vaBase = 0x1000`;
//! 3. the `SET_PAGE_DIRECTORY` RPC: GSP-RM is told the address of the root
//!    table (PD3) — the same root the golden image's reserved range hangs
//!    from (one set of page tables for everything, as tinygrad).
//!
//! RM 570.144 (`dma.c` `deviceCtrlCmdDmaSetPageDirectory`, `gpu_vaspace.c`
//! `gvaspaceExternalRootDirCommit`, MIT) is what says why: the call is
//! accepted only on an externally-owned (or shared-management) VA space, the
//! entries must cover the VA space's limit, and an externally-owned VA space
//! can get its root only once. The later channels of this client (channel
//! group, context share, GPFIFO — the rest of I9) are not written yet.

use gpu_core::error::{GpuError, GpuResult};

use super::golden::{Rm, RmTransport};
use super::rmapi::{self, class};
use super::rpc::function;
use crate::mmu::LEVELS;

/// The client handle tinygrad uses for its work (`PCIIface.root`). DEDUS.
pub const USER_CLIENT: u32 = 0xc100_0000;

/// What the working client is made of (kept for the channels of I9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UserSpace {
    pub client: u32,
    pub device: u32,
    pub subdevice: u32,
    pub vaspace: u32,
    /// The root table (PD3) GSP-RM was told about, and its entry count.
    pub root: u64,
    pub root_entries: u32,
    /// The first handle still free after this client.
    pub next_handle: u32,
}

/// Opens the working client. `first_handle` comes from
/// [`super::golden::GoldenImage::next_handle`]; `root` is the driver's PD3
/// (`mmu::GpuMemory::root`). Stops at the first refusal from GSP-RM (the
/// error carries RM's status); never reports a VA space as usable unless
/// GSP-RM accepted `SET_PAGE_DIRECTORY`.
pub fn open_user_space<T: RmTransport + ?Sized>(
    t: &mut T,
    first_handle: u32,
    root: u64,
) -> GpuResult<UserSpace> {
    let entries = LEVELS[0].entries() as u32;
    // Checked before the first message, so a bad root sends nothing.
    rmapi::check_root(root)?;
    let mut rm = Rm {
        t,
        client: USER_CLIENT,
        next: first_handle,
    };
    rm.alloc(0, class::ROOT, &rmapi::root_params())?;
    let device = rm.alloc(
        USER_CLIENT,
        class::DEVICE,
        &rmapi::device_params(USER_CLIENT),
    )?;
    let subdevice = rm.alloc(device, class::SUBDEVICE, &rmapi::subdevice_params())?;
    let vaspace = rm.alloc(device, class::VASPACE, &rmapi::user_vaspace_params())?;
    let msg = rmapi::set_page_directory_rpc(USER_CLIENT, device, vaspace, root, entries)?;
    // The RPC result is GSP-RM's status; a non-zero one is an error
    // (`RmTransport::call`). Its reply payload carries nothing we use.
    rm.t.call(function::SET_PAGE_DIRECTORY, &msg)?;
    let next_handle = rm.next;
    if next_handle <= vaspace {
        return Err(GpuError::InvalidArgument("RM: handle-urile s-au terminat"));
    }
    Ok(UserSpace {
        client: USER_CLIENT,
        device,
        subdevice,
        vaspace,
        root,
        root_entries: entries,
        next_handle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::golden::{GoldenMemory, QueueTransport, RM_REPLY_US, init_golden_image};
    use crate::gsp::msgq::{CpuQueues, QueueLayout, SharedMem};
    use crate::gsp::sim_peer::SimGsp;
    use crate::mmu::{GpuMemory, VramWrite};
    use alloc::vec::Vec;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_rm::{FakeRm, class as c, status};
    use gpu_sim::vram::{SimVram, translate};

    struct Sim(SimVram);
    impl VramWrite for Sim {
        fn write_u64(&mut self, pa: u64, v: u64) -> GpuResult<()> {
            self.0.write_u64(pa, v).map_err(GpuError::InvalidArgument)
        }
    }

    /// Direct transport to the model (no queues).
    struct Direct(FakeRm);
    impl RmTransport for Direct {
        fn call(&mut self, f: u32, p: &[u8]) -> GpuResult<Vec<u8>> {
            match self.0.handle(f, p) {
                Some((0, r)) => Ok(r),
                Some((code, _)) => Err(GpuError::HardwareFault { code }),
                None => Err(GpuError::NotImplemented("functie necunoscuta modelului")),
            }
        }
    }

    /// SIMULARE end to end: golden image and working client through the real
    /// queues, both on the same page tables (`mmu::GpuMemory` in a simulated
    /// VRAM); the model records the root it was given, and the independent
    /// walker (`gpu_sim::vram::translate`) resolves a golden buffer from it.
    #[test]
    fn working_client_gets_the_drivers_page_tables() {
        let l = QueueLayout::standard();
        let mut shm = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut shm, l, &pages).unwrap();
        let mut gsp = SimGsp::start(&mut shm, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&shm).unwrap();
        let mut rm = FakeRm::new();
        let mut clock = SimClock::default();
        let vram = 8u64 << 30;
        let mut mem =
            GpuMemory::new(64 << 20, vram - (256 << 20), Sim(SimVram::new(vram))).unwrap();
        let (g, u) = {
            let mut t = QueueTransport {
                q: &mut q,
                mem: &mut shm,
                clock: &mut clock,
                timeout_us: RM_REPLY_US,
                poll: |m: &mut dyn SharedMem| {
                    gsp.service_with(m, |f, p| rm.handle(f, p));
                },
                events: Vec::new(),
            };
            let g = init_golden_image(&mut t, &mut mem).unwrap();
            let u = open_user_space(&mut t, g.next_handle(), mem.root()).unwrap();
            (g, u)
        };
        assert_eq!(g.next_handle(), 0xcf00_0007);
        // The root allocation takes a handle too (0xcf00_0007), as in the golden image.
        assert_eq!(
            (u.client, u.device, u.subdevice, u.vaspace, u.next_handle),
            (
                USER_CLIENT,
                0xcf00_0008,
                0xcf00_0009,
                0xcf00_000a,
                0xcf00_000b
            )
        );
        assert_eq!((u.root, u.root_entries), (mem.root(), 4));
        // The model saw the golden image's 7 allocations, then these 4.
        assert_eq!(
            rm.log.allocs[7..],
            [c::ROOT, c::DEVICE, c::SUBDEVICE, c::VASPACE].map(|x| (x, status::OK))
        );
        assert_eq!(rm.log.page_dirs, [(u.vaspace, status::OK)]);
        assert_eq!(
            rm.page_directory(USER_CLIENT, u.vaspace),
            Some((mem.root(), 4))
        );
        // From the root the model was given, a golden GR buffer resolves.
        let (_, m) = g.gr_bufs[0];
        let tr = translate(
            &mem.writer().0,
            rm.page_directory(USER_CLIENT, u.vaspace).unwrap().0,
            m.va,
        )
        .unwrap();
        assert_eq!(tr.pa, m.pa);
        // A buffer mapped after the RPC is visible through the same root.
        let late = mem.alloc_mapped(0x2000, 0x1000).unwrap();
        assert_eq!(
            translate(&mem.writer().0, u.root, late.va + 0x1008)
                .unwrap()
                .pa,
            late.pa + 0x1008
        );
    }

    fn opened() -> (Direct, UserSpace) {
        let mut d = Direct(FakeRm::new());
        let u = open_user_space(&mut d, 0x100, 0x7_0000_0000).unwrap();
        (d, u)
    }

    #[test]
    fn bad_roots_send_nothing() {
        for root in [0, 0x7_0000_0800, crate::mmu::PA_LIMIT] {
            let mut d = Direct(FakeRm::new());
            assert!(open_user_space(&mut d, 0x100, root).is_err());
            assert!(d.0.log.allocs.is_empty() && d.0.log.page_dirs.is_empty());
        }
    }

    /// The model's reading of RM's rules for SET_PAGE_DIRECTORY: each
    /// refusal comes back as RM's status code, never as success.
    #[test]
    fn rm_refusals_are_errors() {
        let (mut d, u) = opened();
        // A second root for an externally-owned VA space: INVALID_STATE.
        let again =
            rmapi::set_page_directory_rpc(USER_CLIENT, u.device, u.vaspace, u.root, 4).unwrap();
        assert_eq!(
            d.call(function::SET_PAGE_DIRECTORY, &again),
            Err(GpuError::HardwareFault {
                code: status::INVALID_STATE
            })
        );
        // Too few entries for the VA space's limit: INVALID_ARGUMENT.
        let (mut d, u) = {
            let mut d = Direct(FakeRm::new());
            let mut rm = Rm {
                t: &mut d,
                client: USER_CLIENT,
                next: 0x100,
            };
            rm.alloc(0, class::ROOT, &rmapi::root_params()).unwrap();
            let dev = rm
                .alloc(
                    USER_CLIENT,
                    class::DEVICE,
                    &rmapi::device_params(USER_CLIENT),
                )
                .unwrap();
            let va = rm
                .alloc(dev, class::VASPACE, &rmapi::user_vaspace_params())
                .unwrap();
            let plain = rm
                .alloc(dev, class::VASPACE, &rmapi::vaspace_params())
                .unwrap();
            (d, (dev, va, plain))
        };
        let (dev, va, plain) = u;
        for (msg, code) in [
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, va, 0x7_0000_0000, 2).unwrap(),
                status::INVALID_ARGUMENT,
            ),
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, va, 0x7_0000_0000, 5).unwrap(),
                status::INVALID_ARGUMENT,
            ),
            // A VA space RM manages itself: NOT_SUPPORTED.
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, plain, 0x7_0000_0000, 4).unwrap(),
                status::NOT_SUPPORTED,
            ),
            // The device handle where the VA space should be, another client.
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, dev, 0x7_0000_0000, 4).unwrap(),
                status::INVALID_OBJECT_HANDLE,
            ),
            (
                rmapi::set_page_directory_rpc(0xc1e0_0004, dev, va, 0x7_0000_0000, 4).unwrap(),
                status::INVALID_CLIENT,
            ),
        ] {
            assert_eq!(
                d.call(function::SET_PAGE_DIRECTORY, &msg),
                Err(GpuError::HardwareFault { code })
            );
        }
        // A truncated message: INVALID_PARAM_STRUCT.
        let ok = rmapi::set_page_directory_rpc(USER_CLIENT, dev, va, 0x7_0000_0000, 4).unwrap();
        assert_eq!(
            d.call(function::SET_PAGE_DIRECTORY, &ok[..40]),
            Err(GpuError::HardwareFault {
                code: status::INVALID_PARAM_STRUCT
            })
        );
        // After all the refusals the VA space has no root; then it gets one.
        assert_eq!(d.0.page_directory(USER_CLIENT, va), None);
        d.call(function::SET_PAGE_DIRECTORY, &ok).unwrap();
        assert_eq!(
            d.0.page_directory(USER_CLIENT, va),
            Some((0x7_0000_0000, 4))
        );
    }

    /// Planted mistakes in the message (the bytes our encoder writes) are
    /// caught by the model, which parses with its own hand-written offsets.
    #[test]
    fn planted_mistakes_are_caught() {
        let (_, u) = opened();
        let good =
            rmapi::set_page_directory_rpc(USER_CLIENT, u.device, u.vaspace, 0x7_0000_0000, 4)
                .unwrap();
        let fresh = || {
            let mut d = Direct(FakeRm::new());
            let u2 = {
                let mut rm = Rm {
                    t: &mut d,
                    client: USER_CLIENT,
                    next: 0x100,
                };
                rm.alloc(0, class::ROOT, &rmapi::root_params()).unwrap();
                let dev = rm
                    .alloc(
                        USER_CLIENT,
                        class::DEVICE,
                        &rmapi::device_params(USER_CLIENT),
                    )
                    .unwrap();
                rm.alloc(dev, class::SUBDEVICE, &rmapi::subdevice_params())
                    .unwrap();
                rm.alloc(dev, class::VASPACE, &rmapi::user_vaspace_params())
                    .unwrap()
            };
            assert_eq!(u2, u.vaspace);
            d
        };
        assert!(fresh().call(function::SET_PAGE_DIRECTORY, &good).is_ok());
        // (offset, value): params start at 16 in the RPC.
        for (off, v) in [
            (16usize, 0x0000_0800u32), // root not 4 KiB aligned (low half)
            (16 + 12, 3),              // aperture 3 (no such aperture)
            (16 + 16, u.device),       // hVASpace = the device
            (4, u.vaspace),            // hDevice = the VA space
            (16 + 24, 5),              // subDeviceId of a 5th GPU
        ] {
            let mut bad = good.clone();
            bad[off..off + 4].copy_from_slice(&v.to_le_bytes());
            assert!(
                fresh().call(function::SET_PAGE_DIRECTORY, &bad).is_err(),
                "greseala la {off} neprinsa"
            );
        }
    }
}
