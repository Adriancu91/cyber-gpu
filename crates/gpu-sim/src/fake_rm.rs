//! # SIMULARE — a stand-in for GSP-RM's resource manager (I8e, "golden image")
//!
//! After `GSP_INIT_DONE` the driver talks to GSP-RM with two RPCs:
//! `GSP_RM_ALLOC` (create an object: client, device, subdevice, VA space,
//! channel, compute/copy object) and `GSP_RM_CONTROL` (a command on an object).
//! This model answers both, from the raw bytes, with offsets written here
//! **independently** of `nvidia-ga104::gsp::rmapi` (which uses the
//! bindgen-generated layouts), so a mistake on one side shows up as a failed
//! test instead of two agreeing copies of the same mistake.
//!
//! What it enforces is *our reading* of the r570 rules (open-gpu-kernel-modules
//! 570.144, MIT, read: `nvos.h`, `cl0080.h`, `alloc_channel.h`,
//! `ctrl2080fifo.h`, `ctrl2080gpu.h`, `ctrl2080internal.h`, `ctrl90f1.h`,
//! `nvstatuscodes.h`; tinygrad `ip.py` `init_golden_image` for the order):
//! parents must exist and be of the right class, handles are unique, parameter
//! sizes are the 570.144 ones, the channel's VA space must exist, and — the
//! point of the golden image — a compute object can be created on a channel
//! only after its GR context buffers were promoted. Real GSP-RM checks far
//! more (and some of these differently). Sizes it reports for the GR context
//! buffers are **SIM** values, not the board's.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// `rpc_global_enums.h` (570.144).
pub const FN_GSP_RM_CONTROL: u32 = 76;
pub const FN_GSP_RM_ALLOC: u32 = 103;

/// `nvstatuscodes.h` (570.144).
pub mod status {
    pub const OK: u32 = 0x00;
    pub const INSERT_DUPLICATE_NAME: u32 = 0x19;
    pub const INVALID_ARGUMENT: u32 = 0x1f;
    pub const INVALID_CLASS: u32 = 0x22;
    pub const INVALID_CLIENT: u32 = 0x23;
    pub const INVALID_COMMAND: u32 = 0x24;
    pub const INVALID_OBJECT_HANDLE: u32 = 0x33;
    pub const INVALID_OBJECT_PARENT: u32 = 0x36;
    pub const INVALID_PARAM_STRUCT: u32 = 0x3a;
    pub const INVALID_STATE: u32 = 0x40;
}

/// Class numbers (`cl0000.h`, `cl0080.h`, `cl2080.h`, `cl90f1.h`,
/// `clc56f.h`, `clc7c0.h`, `clc7b5.h`).
pub mod class {
    pub const ROOT: u32 = 0x0;
    pub const DEVICE: u32 = 0x80;
    pub const SUBDEVICE: u32 = 0x2080;
    pub const VASPACE: u32 = 0x90f1;
    pub const CHANNEL_GPFIFO: u32 = 0xc56f;
    pub const COMPUTE: u32 = 0xc7c0;
    pub const DMA_COPY: u32 = 0xc7b5;
}

/// Control commands (the `finn`-evaluated values in the ctrl headers).
pub mod cmd {
    pub const GET_DEVICE_INFO_TABLE: u32 = 0x2080_1112;
    pub const KGR_GET_CONTEXT_BUFFERS_INFO: u32 = 0x2080_0a32;
    pub const PROMOTE_CTX: u32 = 0x2080_012b;
    pub const COPY_SERVER_RESERVED_PDES: u32 = 0x90f1_0106;
}

/// Parameter sizes in 570.144 (x86-64 C layout).
pub mod size {
    pub const ROOT: usize = 120;
    pub const DEVICE: usize = 56;
    pub const SUBDEVICE: usize = 4;
    pub const VASPACE: usize = 48;
    pub const CHANNEL: usize = 368;
    pub const DEVICE_INFO: usize = 3212;
    pub const KGR_INFO: usize = 1664;
    pub const PROMOTE: usize = 560;
    pub const RESERVED_PDES: usize = 184;
}

/// GR context buffers (`NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_*`):
/// index → (size, alignment). SIM values in the order of magnitude of a GA10x
/// (MiB for the main image, KiB for the small ones); the board reports its own.
pub const SIM_CTX_INFO: [(u32, u32); 26] = {
    let mut t = [(0u32, 0u32); 26];
    t[0] = (0x0004_2000, 0x1000); // GRAPHICS (main image)
    t[16] = (0x0002_0000, 0x1000); // GRAPHICS_PATCH
    t[17] = (0x0000_8000, 0x100); // BUNDLE_CB
    t[18] = (0x0008_0000, 0x1000); // PAGEPOOL_GLOBAL
    t[19] = (0x0013_0000, 0x1000); // ATTRIBUTE_CB
    t[20] = (0x0000_4000, 0x1000); // RTV_CB_GLOBAL
    t[21] = (0x0000_0000, 0x1000); // GFXP_POOL (not on compute)
    t[22] = (0x0000_0000, 0x1000); // GFXP_CTRL_BLK
    t[23] = (0x0000_1000, 0x1000); // FECS_EVENT
    t[24] = (0x0000_3000, 0x1000); // PRIV_ACCESS_MAP
    t
};

/// Device-info entries the model reports: (RM engine type, runlist). SIM.
pub const SIM_ENGINES: [(u32, u32); 3] = [(1, 0), (9, 1), (10, 2)]; // GR0, COPY0, COPY1

/// Buffers that must have been promoted on a channel before a compute
/// object: MAIN (0), PATCH (2), PRIV_ACCESS_MAP (10). Our reading of why the
/// golden image exists; the real list is GSP-RM's.
pub const REQUIRED_PROMOTED: [u16; 3] = [0, 2, 10];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Obj {
    class: u32,
    parent: u32,
}

/// Everything the model has seen, for the tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RmLog {
    /// (class, status) per GSP_RM_ALLOC, in order.
    pub allocs: Vec<(u32, u32)>,
    /// (command, status) per GSP_RM_CONTROL, in order.
    pub controls: Vec<(u32, u32)>,
}

/// SIMULARE: the resource manager behind the queues.
#[derive(Clone, Debug, Default)]
pub struct FakeRm {
    client: Option<u32>,
    objects: BTreeMap<u32, Obj>,
    /// channel handle → promoted buffer ids
    promoted: BTreeMap<u32, Vec<u16>>,
    reserved_pdes_copied: bool,
    pub log: RmLog,
}

fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn rd64(b: &[u8], o: usize) -> u64 {
    rd32(b, o) as u64 | (rd32(b, o + 4) as u64) << 32
}
fn wr32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

impl FakeRm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handles one RPC. Returns (RPC result, reply payload); `None` for
    /// functions this model does not know (the caller decides what to do).
    pub fn handle(&mut self, function: u32, payload: &[u8]) -> Option<(u32, Vec<u8>)> {
        match function {
            FN_GSP_RM_ALLOC => Some(self.alloc(payload)),
            FN_GSP_RM_CONTROL => Some(self.control(payload)),
            _ => None,
        }
    }

    fn class_of(&self, h: u32) -> Option<u32> {
        self.objects.get(&h).map(|o| o.class)
    }

    // rpc_gsp_rm_alloc_v03_00: hClient 0, hParent 4, hObject 8, hClass 12,
    // status 16, paramsSize 20, flags 24, reserved[4] 28, params 32.
    fn alloc(&mut self, p: &[u8]) -> (u32, Vec<u8>) {
        if p.len() < 32 {
            return (status::INVALID_PARAM_STRUCT, Vec::new());
        }
        let (client, parent, object, class, psize) = (
            rd32(p, 0),
            rd32(p, 4),
            rd32(p, 8),
            rd32(p, 12),
            rd32(p, 20) as usize,
        );
        let mut reply = p.to_vec();
        let st = if p.len() != 32 + psize {
            status::INVALID_PARAM_STRUCT
        } else {
            self.check_alloc(client, parent, object, class, &p[32..])
        };
        if st == status::OK {
            if class == class::ROOT {
                self.client = Some(client);
            } else {
                self.objects.insert(object, Obj { class, parent });
            }
        }
        wr32(&mut reply, 16, st);
        self.log.allocs.push((class, st));
        (0, reply)
    }

    fn check_alloc(&self, client: u32, parent: u32, object: u32, class: u32, params: &[u8]) -> u32 {
        if class == class::ROOT {
            return match (self.client, parent, params.len()) {
                (Some(_), _, _) => status::INSERT_DUPLICATE_NAME,
                (None, 0, size::ROOT) => status::OK,
                (None, 0, _) => status::INVALID_PARAM_STRUCT,
                _ => status::INVALID_OBJECT_PARENT,
            };
        }
        if self.client != Some(client) {
            return status::INVALID_CLIENT;
        }
        if object == client || self.objects.contains_key(&object) {
            return status::INSERT_DUPLICATE_NAME;
        }
        let parent_class = if parent == client {
            Some(class::ROOT)
        } else {
            self.class_of(parent)
        };
        let (want_parent, want_size) = match class {
            class::DEVICE => (class::ROOT, size::DEVICE),
            class::SUBDEVICE => (class::DEVICE, size::SUBDEVICE),
            class::VASPACE => (class::DEVICE, size::VASPACE),
            class::CHANNEL_GPFIFO => (class::DEVICE, size::CHANNEL),
            class::COMPUTE | class::DMA_COPY => (class::CHANNEL_GPFIFO, 0),
            _ => return status::INVALID_CLASS,
        };
        if parent_class != Some(want_parent) {
            return status::INVALID_OBJECT_PARENT;
        }
        if params.len() != want_size {
            return status::INVALID_PARAM_STRUCT;
        }
        match class {
            // NV0080_ALLOC_PARAMETERS.hClientShare (offset 4) must name our client.
            class::DEVICE if rd32(params, 4) != client => status::INVALID_ARGUMENT,
            class::CHANNEL_GPFIFO => self.check_channel(params),
            class::COMPUTE => {
                let got = self.promoted.get(&parent).cloned().unwrap_or_default();
                if REQUIRED_PROMOTED.iter().all(|b| got.contains(b)) {
                    status::OK
                } else {
                    status::INVALID_STATE // no golden context for this channel
                }
            }
            _ => status::OK,
        }
    }

    // NV_CHANNEL_ALLOC_PARAMS: gpFifoOffset 8, gpFifoEntries 16, flags 20,
    // hVASpace 28, userdOffset[0] 64, engineType 128, cid 132,
    // instanceMem 144, userdMem 168, ramfcMem 192, mthdbufMem 216
    // (NV_MEMORY_DESC_PARAMS: base 0, size 8, addressSpace 16, cacheAttrib 20).
    fn check_channel(&self, c: &[u8]) -> u32 {
        let entries = rd32(c, 16);
        if self.class_of(rd32(c, 28)) != Some(class::VASPACE) {
            return status::INVALID_OBJECT_HANDLE;
        }
        if entries < 2
            || !entries.is_power_of_two()
            || !rd64(c, 8).is_multiple_of(0x1000)
            || rd64(c, 8) == 0
        {
            return status::INVALID_ARGUMENT;
        }
        if rd32(c, 128) != 1 {
            return status::INVALID_ARGUMENT; // only GR0 in this model
        }
        // (offset, minimum size) of instance, USERD, RAMFC, method buffer.
        for (o, min) in [(144, 0x1000u64), (168, 0x20), (192, 0x200), (216, 0x5000)] {
            let (base, sz, aperture) = (rd64(c, o), rd64(c, o + 8), rd32(c, o + 16));
            if aperture != 2 || sz < min || base == 0 || !base.is_multiple_of(0x20) {
                return status::INVALID_ARGUMENT; // 2 = video memory
            }
        }
        status::OK
    }

    // rpc_gsp_rm_control_v03_00: hClient 0, hObject 4, cmd 8, status 12,
    // paramsSize 16, flags 20, params 24.
    fn control(&mut self, p: &[u8]) -> (u32, Vec<u8>) {
        if p.len() < 24 {
            return (status::INVALID_PARAM_STRUCT, Vec::new());
        }
        let (client, object, command, psize) =
            (rd32(p, 0), rd32(p, 4), rd32(p, 8), rd32(p, 16) as usize);
        let mut reply = p.to_vec();
        let st = if p.len() != 24 + psize {
            status::INVALID_PARAM_STRUCT
        } else if self.client != Some(client) {
            status::INVALID_CLIENT
        } else {
            let class = self.class_of(object);
            let params = &mut reply[24..];
            match (command, class) {
                (cmd::GET_DEVICE_INFO_TABLE, Some(class::SUBDEVICE)) => Self::device_info(params),
                (cmd::KGR_GET_CONTEXT_BUFFERS_INFO, Some(class::SUBDEVICE)) => {
                    Self::kgr_info(params)
                }
                (cmd::COPY_SERVER_RESERVED_PDES, Some(class::VASPACE)) => {
                    let st = Self::check_pdes(params);
                    if st == status::OK {
                        self.reserved_pdes_copied = true;
                    }
                    st
                }
                (cmd::PROMOTE_CTX, Some(class::SUBDEVICE)) => self.promote(client, params),
                (
                    cmd::GET_DEVICE_INFO_TABLE
                    | cmd::KGR_GET_CONTEXT_BUFFERS_INFO
                    | cmd::COPY_SERVER_RESERVED_PDES
                    | cmd::PROMOTE_CTX,
                    Some(_),
                ) => status::INVALID_OBJECT_HANDLE,
                (_, None) => status::INVALID_OBJECT_HANDLE,
                _ => status::INVALID_COMMAND,
            }
        };
        wr32(&mut reply, 12, st);
        self.log.controls.push((command, st));
        (0, reply)
    }

    // baseIndex 0, numEntries 4, bMore 8, entries 12 (100 B each:
    // engineData[16] 0, pbdmaIds[2] 64, pbdmaFaultIds[2] 72, numPbdmas 80, name 84).
    fn device_info(b: &mut [u8]) -> u32 {
        if b.len() != size::DEVICE_INFO {
            return status::INVALID_PARAM_STRUCT;
        }
        if rd32(b, 0) != 0 {
            wr32(b, 4, 0);
            b[8] = 0;
            return status::OK;
        }
        b[4..].fill(0);
        wr32(b, 4, SIM_ENGINES.len() as u32);
        for (i, (eng, runlist)) in SIM_ENGINES.iter().enumerate() {
            let e = 12 + i * 100;
            wr32(b, e + 2 * 4, *eng); // ENGINE_INFO_TYPE_RM_ENGINE_TYPE
            wr32(b, e + 3 * 4, *runlist); // ENGINE_INFO_TYPE_RUNLIST
            wr32(b, e + 80, 1);
        }
        status::OK
    }

    // engineContextBuffersInfo[8] of engine[26] {size, alignment}; [0] = GR0.
    fn kgr_info(b: &mut [u8]) -> u32 {
        if b.len() != size::KGR_INFO {
            return status::INVALID_PARAM_STRUCT;
        }
        b.fill(0);
        for (i, (sz, al)) in SIM_CTX_INFO.iter().enumerate() {
            wr32(b, i * 8, *sz);
            wr32(b, i * 8 + 4, *al);
        }
        status::OK
    }

    // hSubDevice 0, subDeviceId 4, pageSize 8, virtAddrLo 16, virtAddrHi 24,
    // numLevelsToCopy 32, levels[6] 40 (24 B: physAddress 0, size 8,
    // aperture 16, pageShift 20).
    fn check_pdes(b: &[u8]) -> u32 {
        if b.len() != size::RESERVED_PDES {
            return status::INVALID_PARAM_STRUCT;
        }
        let (page, lo, hi, n) = (rd64(b, 8), rd64(b, 16), rd64(b, 24), rd32(b, 32) as usize);
        if n == 0 || n > 6 || hi <= lo || hi - lo + 1 != page || !lo.is_multiple_of(page) {
            return status::INVALID_ARGUMENT;
        }
        for i in 0..n {
            let l = 40 + i * 24;
            if rd64(b, l) == 0
                || !rd64(b, l).is_multiple_of(0x1000)
                || rd64(b, l + 8) == 0
                || b[l + 20] == 0
            {
                return status::INVALID_ARGUMENT;
            }
        }
        status::OK
    }

    // engineType 0, hClient 4, ChID 8, hChanClient 12, hObject 16,
    // hVirtMemory 20, virtAddress 24, size 32, entryCount 40, promoteEntry[16]
    // 48 (32 B: gpuPhysAddr 0, gpuVirtAddr 8, size 16, physAttr 24,
    // bufferId 28 u16, bInitialize 30, bNonmapped 31).
    fn promote(&mut self, client: u32, b: &[u8]) -> u32 {
        if b.len() != size::PROMOTE {
            return status::INVALID_PARAM_STRUCT;
        }
        let (engine, chan_client, chan, n) =
            (rd32(b, 0), rd32(b, 12), rd32(b, 16), rd32(b, 40) as usize);
        if engine != 1 || chan_client != client || n == 0 || n > 16 {
            return status::INVALID_ARGUMENT;
        }
        if self.class_of(chan) != Some(class::CHANNEL_GPFIFO) {
            return status::INVALID_OBJECT_HANDLE;
        }
        if !self.reserved_pdes_copied {
            return status::INVALID_STATE; // the channel's VA space is not usable yet
        }
        let mut ids = Vec::new();
        for i in 0..n {
            let e = 48 + i * 32;
            let (pa, va, sz) = (rd64(b, e), rd64(b, e + 8), rd64(b, e + 16));
            let id = u16::from_le_bytes([b[e + 28], b[e + 29]]);
            let (init, nonmapped) = (b[e + 30] != 0, b[e + 31] != 0);
            if id > 12 || ids.contains(&id) {
                return status::INVALID_ARGUMENT;
            }
            // Physical half: address + size + initialize; virtual half: VA.
            if init && (pa == 0 || !pa.is_multiple_of(0x1000) || sz == 0) {
                return status::INVALID_ARGUMENT;
            }
            if !nonmapped && va == 0 {
                return status::INVALID_ARGUMENT;
            }
            if nonmapped && !init {
                return status::INVALID_ARGUMENT;
            }
            ids.push(id);
        }
        self.promoted.entry(chan).or_default().extend(ids);
        status::OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc_rpc(client: u32, parent: u32, object: u32, class: u32, params: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0u8; 32];
        for (o, v) in [
            (0, client),
            (4, parent),
            (8, object),
            (12, class),
            (20, params.len() as u32),
        ] {
            wr32(&mut p, o, v);
        }
        p.extend_from_slice(params);
        p
    }

    #[test]
    fn parents_classes_and_sizes_are_enforced() {
        let mut rm = FakeRm::new();
        let c = 0xc1e0_0004;
        let st = |r: (u32, Vec<u8>)| rd32(&r.1, 16);
        // A device before the client exists is refused.
        let mut dev = alloc::vec![0u8; size::DEVICE];
        wr32(&mut dev, 4, c);
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, c, 1, class::DEVICE, &dev))),
            status::INVALID_CLIENT
        );
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, 0, 0, class::ROOT, &[0; size::ROOT]))),
            status::OK
        );
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, 0, 0, class::ROOT, &[0; size::ROOT]))),
            status::INSERT_DUPLICATE_NAME
        );
        // Wrong size, wrong parent, unknown class.
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, c, 1, class::DEVICE, &[0; 8]))),
            status::INVALID_PARAM_STRUCT
        );
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, c, 1, class::SUBDEVICE, &[0; 4]))),
            status::INVALID_OBJECT_PARENT
        );
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, c, 1, 0x1234, &[]))),
            status::INVALID_CLASS
        );
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, c, 1, class::DEVICE, &dev))),
            status::OK
        );
        assert_eq!(
            st(rm.alloc(&alloc_rpc(c, c, 1, class::DEVICE, &dev))),
            status::INSERT_DUPLICATE_NAME
        );
        // A payload whose paramsSize lies about the length.
        let mut bad = alloc_rpc(c, 1, 2, class::SUBDEVICE, &[0; 4]);
        bad.push(0);
        assert_eq!(st(rm.alloc(&bad)), status::INVALID_PARAM_STRUCT);
        // Unknown functions are not answered at all.
        assert!(rm.handle(72, &[]).is_none());
    }
}
