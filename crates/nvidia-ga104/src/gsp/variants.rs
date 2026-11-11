//! C-108 — the settings GSP-RM gets at its first boot, as **numbered
//! variants** Adrian tries one per cold boot (`gpu lant --varianta N` in the
//! OS) until one passes I8. Built and tested **only in SIMULARE**.
//!
//! Why (session 88, after the `os-283` journal): the journal of os-283 is
//! byte-for-byte the journal of os-280 (only timestamps differ), so
//! `bDmemStack` was not the cause of `GSP_INIT_DONE = 0x59`. The comparison
//! then moved from *what* we send to *when*: all four public drivers that boot
//! GSP-RM 570 put `GSP_SET_SYSTEM_INFO` and `SET_REGISTRY` in the command
//! queue **before** Booter starts GSP-RM, and NVIDIA's own sources say why —
//! GSP-RM runs them "in early boot before OBJGPU is created"
//! (`kernel_gsp.c`, `kgspQueueAsyncInitRpcs`) and "may skip these RPCs if not
//! sent in time" (`kernel_gsp_gh100.c`). We sent them in I8, after I7 had
//! started GSP-RM (≈0.3 s later on the board). Variant 1 is exactly that
//! change; the others add, one at a time or as one driver's whole set, the
//! few field values the four drivers fill differently (C-107 table).
//!
//! Sources (read as specification, nothing copied): NVIDIA
//! open-gpu-kernel-modules 570.144 (MIT: `kernel_gsp.c`, `kernel_gsp_gh100.c`,
//! `kernel_gsp_tu102.c`, `rpc.c`), tinygrad `runtime/support/nv/ip.py` (MIT:
//! "Prefill cmd queue with info for gsp to start"), Linux nouveau
//! `nvkm/subdev/gsp/rm/r535/gsp.c` + `r570/gsp.c` + `tu102.c` and nova-core
//! `gsp/boot.rs` + `gsp/fw/commands.rs` + `gsp/commands.rs` (GPL: read only).
//! Details: `docs/os-283_ANALIZA_ORDINE_RPC.md`.
//!
//! Every value here comes from one of those sources; no register that is not
//! already written at os-283 is touched by any variant. On the board each
//! variant is a first try, with Adrian present and a "DA".

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use super::msgq::{CpuQueues, SharedMem};
use super::rpc::function;
use super::{libos, registry, sysinfo, wpr_meta};
use crate::fb_layout::FbLayout;

/// What a variant decides. Everything not listed is identical in all
/// variants (and to `os-283`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    /// `GSP_SET_SYSTEM_INFO` + `SET_REGISTRY` (+ the doorbell) go into the
    /// command queue **before** Booter starts GSP-RM (I7). `false` = as at
    /// os-283: in I8, after GSP-RM already runs.
    pub init_rpcs_before_booter: bool,
    /// `GspSystemInfo` fields the drivers fill differently.
    pub sysinfo: sysinfo::Options,
    /// Registry: the two keys, plus `RMDevidCheckIgnore = 1`.
    pub devid_check_ignore: bool,
    /// `GspFwWprMeta.flags` (0 or [`wpr_meta::FLAG_CLOCK_BOOST`]).
    pub wpr_meta_flags: u8,
    /// `GSP_ARGUMENTS_CACHED.bDmemStack`.
    pub dmem_stack: bool,
}

/// One numbered variant for `gpu variante` / `gpu lant --varianta N`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    /// 1-based; 0 is the baseline ([`OS_283`]).
    pub nr: u8,
    /// Short name (ASCII, fits a journal text row).
    pub name: &'static str,
    /// One sentence, in Romanian, for the screen: what it changes.
    pub changes: &'static str,
    /// Where the values come from.
    pub source: &'static str,
    pub settings: Settings,
}

/// Exactly what os-283 sent (and when). Not a variant: the behaviour the OS
/// keeps when no `--varianta` is given.
pub const OS_283: Settings = Settings {
    init_rpcs_before_booter: false,
    sysinfo: sysinfo::Options::OS_283,
    devid_check_ignore: false,
    wpr_meta_flags: 0,
    dmem_stack: true,
};

/// Variant 1: only the order changes.
const RPC_BEFORE: Settings = Settings {
    init_rpcs_before_booter: true,
    ..OS_283
};

const NOVA: sysinfo::Options = sysinfo::Options {
    is_passthru: false,
    is_primary: Some(false),
    host_page_size: 0,
};
const TINYGRAD: sysinfo::Options = sysinfo::Options {
    is_passthru: true,
    is_primary: Some(false),
    host_page_size: 0,
};
const NOUVEAU: sysinfo::Options = sysinfo::Options {
    is_passthru: false,
    is_primary: None,
    host_page_size: 0,
};

const SRC_ORDER: &str = "RM 570.144 kgspQueueAsyncInitRpcs (\"before OBJGPU is created\"), nouveau r535_gsp_oneinit, tinygrad ip.py (\"Prefill cmd queue\"), nova-core gsp/boot.rs";

/// The list, most probable first. Numbers are stable: a journal says
/// "varianta 3" and means this list's entry 3, now and later (new variants
/// are appended, never renumbered).
pub const VARIANTS: [Variant; 10] = [
    Variant {
        nr: 1,
        name: "rpc-inainte",
        changes: "SetSystemInfo + SetRegistry (si soneria) intra in coada INAINTE ca Booter sa porneasca GSP-RM (la I7, nu la I8); valorile raman cele de la os-283",
        source: SRC_ORDER,
        settings: RPC_BEFORE,
    },
    Variant {
        nr: 2,
        name: "rpc-inainte+nova",
        changes: "varianta 1 + setul nova-core: hostPageSize 0, bIsPrimary 0, registrul cu RMDevidCheckIgnore=1 (3 chei)",
        source: "nova-core gsp/fw/commands.rs (SetSystemInfo), gsp/commands.rs (SetRegistry)",
        settings: Settings {
            sysinfo: NOVA,
            devid_check_ignore: true,
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 3,
        name: "rpc-inainte+tinygrad",
        changes: "varianta 1 + setul tinygrad: bIsPassthru 1, bIsPrimary 0, hostPageSize 0, registrul cu 2 chei",
        source: "tinygrad runtime/support/nv/ip.py (rpc_set_gsp_system_info, rpc_set_registry_table)",
        settings: Settings {
            sysinfo: TINYGRAD,
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 4,
        name: "rpc-inainte+nouveau",
        changes: "varianta 1 + setul nouveau r570: hostPageSize 0, bIsPrimary dupa VGA, registrul cu RMDevidCheckIgnore=1 (3 chei)",
        source: "nouveau nvkm/subdev/gsp/rm/r570/gsp.c (r570_gsp_set_system_info), rm/r535/gsp.c (registrul)",
        settings: Settings {
            sysinfo: NOUVEAU,
            devid_check_ignore: true,
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 5,
        name: "rpc-inainte+clock-boost",
        changes: "varianta 1 + GspFwWprMeta.flags = CLOCK_BOOST (bitul 0), cum pune driverul NVIDIA pe GA10x",
        source: "RM 570.144 kernel_gsp_tu102.c (bBootGspRmWithBoostClocks), gsp_fw_wpr_meta.h GSP_FW_FLAGS_CLOCK_BOOST",
        settings: Settings {
            wpr_meta_flags: wpr_meta::FLAG_CLOCK_BOOST,
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 6,
        name: "rpc-inainte+passthru",
        changes: "varianta 1 + doar bIsPassthru = 1 (singura valoare in plus a lui tinygrad)",
        source: "tinygrad ip.py (bIsPassthru=1)",
        settings: Settings {
            sysinfo: sysinfo::Options {
                is_passthru: true,
                ..sysinfo::Options::OS_283
            },
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 7,
        name: "rpc-inainte+devid",
        changes: "varianta 1 + doar cheia de registru RMDevidCheckIgnore = 1 (nouveau si nova)",
        source: "nouveau rm/r535/gsp.c, nova-core gsp/commands.rs (RMDevidCheckIgnore)",
        settings: Settings {
            devid_check_ignore: true,
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 8,
        name: "rpc-inainte+stiva-libos",
        changes: "varianta 1 + bDmemStack = 0 (stiva LibOS, ce trimiteam pana la os-280; RM o alege cu RmGspStackPlacement=1)",
        source: "RM 570.144 kernel_gsp.c kgspPopulateGspRmInitArgs (RmGspStackPlacement)",
        settings: Settings {
            dmem_stack: false,
            ..RPC_BEFORE
        },
    },
    Variant {
        nr: 9,
        name: "dupa-booter+tinygrad",
        changes: "ordinea de la os-283 (RPC-urile la I8) + setul tinygrad (bIsPassthru 1, bIsPrimary 0, hostPageSize 0) — daca ordinea nu era cauza",
        source: "tinygrad ip.py",
        settings: Settings {
            sysinfo: TINYGRAD,
            ..OS_283
        },
    },
    Variant {
        nr: 10,
        name: "dupa-booter+nova",
        changes: "ordinea de la os-283 + setul nova-core (hostPageSize 0, bIsPrimary 0, RMDevidCheckIgnore=1) — daca ordinea nu era cauza",
        source: "nova-core gsp/fw/commands.rs, gsp/commands.rs",
        settings: Settings {
            sysinfo: NOVA,
            devid_check_ignore: true,
            ..OS_283
        },
    },
];

/// All variants, in order.
pub fn list() -> &'static [Variant] {
    &VARIANTS
}

/// Variant `nr` (1-based); `None` for 0 or an unknown number — the OS then
/// refuses the command without writing anything.
pub fn get(nr: u8) -> Option<&'static Variant> {
    VARIANTS.iter().find(|v| v.nr == nr)
}

/// The settings for `--varianta N`, or [`OS_283`] when no variant is given.
pub fn settings(nr: Option<u8>) -> GpuResult<Settings> {
    match nr {
        None => Ok(OS_283),
        Some(n) => get(n).map(|v| v.settings).ok_or(GpuError::InvalidArgument(
            "varianta: numar necunoscut (vezi gpu variante)",
        )),
    }
}

impl Settings {
    /// The `GSP_SET_SYSTEM_INFO` payload of this variant.
    pub fn encode_sysinfo(&self, i: &sysinfo::SystemInfoInput) -> GpuResult<[u8; sysinfo::SIZE]> {
        sysinfo::encode_with(i, &self.sysinfo)
    }

    /// The registry entries of this variant (2 or 3).
    pub fn registry_entries(&self) -> Vec<registry::Entry> {
        registry::boot_entries_with(self.devid_check_ignore)
    }

    /// The `SET_REGISTRY` payload of this variant.
    pub fn encode_registry(&self) -> GpuResult<Vec<u8>> {
        registry::encode(&self.registry_entries())
    }

    /// The `GspFwWprMeta` page of this variant.
    pub fn encode_wpr_meta(
        &self,
        fb: &FbLayout,
        s: &wpr_meta::SysmemInputs,
    ) -> GpuResult<[u8; wpr_meta::SIZE]> {
        wpr_meta::encode_with_flags(fb, s, self.wpr_meta_flags)
    }

    /// The `GSP_ARGUMENTS_CACHED` (RMARGS) page of this variant.
    pub fn encode_gsp_args(
        &self,
        q: &super::msgq::QueueLayout,
        shared_mem_pa: u64,
    ) -> GpuResult<[u8; libos::GSP_ARGS_SIZE]> {
        libos::encode_gsp_args_with(q, shared_mem_pa, self.dmem_stack)
    }

    /// Puts the two init RPCs in the command queue (RAM only; the doorbell is
    /// the caller's, through `falcon::ring_doorbell`). Returns their sequence
    /// numbers. Called at I7 before Booter when `init_rpcs_before_booter`,
    /// else at I8 (os-283 order).
    pub fn queue_init_rpcs(
        &self,
        q: &mut CpuQueues,
        mem: &mut dyn SharedMem,
        i: &sysinfo::SystemInfoInput,
    ) -> GpuResult<[u32; 2]> {
        let sys = self.encode_sysinfo(i)?;
        let reg = self.encode_registry()?;
        let a = q.send_rpc(mem, function::GSP_SET_SYSTEM_INFO, &sys)?;
        let b = q.send_rpc(mem, function::SET_REGISTRY, &reg)?;
        Ok([a, b])
    }

    /// I8's check: when the variant wants the init RPCs before Booter, they
    /// must already be in the queue (I7 did it); refuses otherwise instead of
    /// silently repeating os-283. When it does not, I8 sends them itself.
    pub fn init_rpcs_ready_for_i8(&self, q: &CpuQueues) -> GpuResult<bool> {
        match (self.init_rpcs_before_booter, q.rpcs_sent()) {
            (true, n) if n >= 2 => Ok(true),
            (true, _) => Err(GpuError::Blocked(
                "I8: varianta cere SetSystemInfo + SetRegistry in coada INAINTE de Booter (I7), dar nu sunt",
            )),
            (false, 0) => Ok(false),
            (false, _) => Err(GpuError::Blocked(
                "I8: RPC-urile de pornire sunt deja in coada, dar varianta le vrea la I8",
            )),
        }
    }

    /// Writes the values this variant sends into the journal (C-108: "valorile
    /// trimise, în jurnal, la fiecare variantă"), as `valoare` / `text` rows.
    pub fn journal(&self, nr: Option<u8>, j: &mut Journal) {
        let (n, name) = match nr.and_then(get) {
            Some(v) => (u32::from(v.nr), v.name),
            None => (0, "os-283"),
        };
        j.push(Event::Value {
            what: "varianta I8 (C-108): numarul",
            value: n,
        });
        j.push(Event::text("varianta I8: numele", name.as_bytes()));
        let rows: [(&'static str, u32); 7] = [
            (
                "varianta: RPC-urile de pornire inainte de Booter (1 = la I7, 0 = la I8)",
                u32::from(self.init_rpcs_before_booter),
            ),
            (
                "varianta: GspSystemInfo.bIsPassthru",
                u32::from(self.sysinfo.is_passthru),
            ),
            (
                "varianta: GspSystemInfo.bIsPrimary (2 = dupa VGA)",
                self.sysinfo.is_primary.map_or(2, u32::from),
            ),
            (
                "varianta: GspSystemInfo.hostPageSize",
                self.sysinfo.host_page_size as u32,
            ),
            (
                "varianta: chei de registru",
                self.registry_entries().len() as u32,
            ),
            (
                "varianta: GspFwWprMeta.flags",
                u32::from(self.wpr_meta_flags),
            ),
            (
                "varianta: GSP_ARGUMENTS_CACHED.bDmemStack",
                u32::from(self.dmem_stack),
            ),
        ];
        for (what, value) in rows {
            j.push(Event::Value { what, value });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::sysinfo::tests::workstation;

    /// T121: the list is well-formed and every variant encodes exactly the
    /// difference its sentence states.
    #[test]
    fn t121_variants_are_numbered_named_and_sourced() {
        assert_eq!(list().len(), 10);
        for (i, v) in list().iter().enumerate() {
            assert_eq!(usize::from(v.nr), i + 1);
            assert!(v.name.is_ascii() && v.name.len() <= gpu_core::diag::TEXT_LEN);
            assert!(!v.changes.is_empty() && !v.source.is_empty());
            assert_eq!(get(v.nr), Some(v));
        }
        assert_eq!(get(0), None);
        assert_eq!(get(11), None);
        assert_eq!(settings(None).unwrap(), OS_283);
        assert_eq!(settings(Some(1)).unwrap(), RPC_BEFORE);
        assert!(settings(Some(99)).is_err());
        // Variant 1 differs from os-283 only in the order.
        assert_eq!(
            Settings {
                init_rpcs_before_booter: false,
                ..RPC_BEFORE
            },
            OS_283
        );
        // 1..=8 all put the RPCs before Booter; 9 and 10 keep the old order.
        for v in list() {
            assert_eq!(
                v.settings.init_rpcs_before_booter,
                v.nr <= 8,
                "varianta {}",
                v.nr
            );
        }
        // The names of 2..=8 say "rpc-inainte"; 9, 10 say "dupa-booter".
        for v in list() {
            let prefix = if v.nr <= 8 {
                "rpc-inainte"
            } else {
                "dupa-booter"
            };
            assert!(v.name.starts_with(prefix), "varianta {}", v.nr);
        }
        // No two variants are the same.
        for a in list() {
            for b in list() {
                assert!(a.nr == b.nr || a.settings != b.settings);
            }
        }
    }

    #[test]
    fn t121b_each_variant_changes_only_what_it_says() {
        let ws = workstation();
        let base = OS_283.encode_sysinfo(&ws).unwrap();
        assert_eq!(base, sysinfo::encode(&ws).unwrap());
        for v in list() {
            let s = v.settings;
            let b = s.encode_sysinfo(&ws).unwrap();
            // The only bytes that may differ from os-283.
            for (k, (x, y)) in base.iter().zip(b.iter()).enumerate() {
                let allowed = k == sysinfo::off::B_IS_PASSTHRU
                    || k == sysinfo::off::B_IS_PRIMARY
                    || (sysinfo::off::HOST_PAGE_SIZE..sysinfo::off::HOST_PAGE_SIZE + 8)
                        .contains(&k);
                assert!(x == y || allowed, "varianta {} octetul {k}", v.nr);
            }
            assert_eq!(
                b[sysinfo::off::B_IS_PASSTHRU],
                u8::from(s.sysinfo.is_passthru)
            );
            assert_eq!(
                b[sysinfo::off::B_IS_PRIMARY],
                u8::from(s.sysinfo.is_primary.unwrap_or(ws.is_primary))
            );
            let n = s.registry_entries().len();
            assert_eq!(n, if s.devid_check_ignore { 3 } else { 2 });
            let decoded = registry::decode(&s.encode_registry().unwrap()).unwrap();
            assert_eq!(decoded.len(), n);
            assert_eq!(decoded[0].name, b"RMForcePcieConfigSave");
            assert_eq!(decoded[1].name, b"RMSecBusResetEnable");
            if n == 3 {
                assert_eq!(decoded[2].name, b"RMDevidCheckIgnore");
            }
            let q = super::super::msgq::QueueLayout::standard();
            let args = s.encode_gsp_args(&q, 0x0500_0000).unwrap();
            assert_eq!(args[libos::B_DMEM_STACK_OFF], u8::from(s.dmem_stack));
        }
        // The sets named after a driver match the C-107 table.
        assert_eq!(get(3).unwrap().settings.sysinfo, TINYGRAD);
        assert!(
            TINYGRAD.is_passthru
                && TINYGRAD.is_primary == Some(false)
                && TINYGRAD.host_page_size == 0
        );
        assert!(!NOVA.is_passthru && NOVA.is_primary == Some(false) && NOVA.host_page_size == 0);
        assert!(
            !NOUVEAU.is_passthru && NOUVEAU.is_primary.is_none() && NOUVEAU.host_page_size == 0
        );
        assert_eq!(get(5).unwrap().settings.wpr_meta_flags, 1);
        assert!(!get(8).unwrap().settings.dmem_stack);
    }

    #[test]
    fn t121c_journal_rows_name_the_variant_and_its_values() {
        let mut j = Journal::new(64);
        get(3).unwrap().settings.journal(Some(3), &mut j);
        let rows: Vec<alloc::string::String> = j.events().map(|e| alloc::format!("{e}")).collect();
        assert_eq!(rows[0], "valoare varianta I8 (C-108): numarul = 0x00000003");
        assert_eq!(
            rows[1],
            "text varianta I8: numele = \"rpc-inainte+tinygrad\""
        );
        assert!(
            rows.iter()
                .any(|r| r.ends_with("inainte de Booter (1 = la I7, 0 = la I8) = 0x00000001"))
        );
        assert!(rows.iter().any(|r| r.ends_with("bIsPassthru = 0x00000001")));
        assert!(
            rows.iter()
                .any(|r| r.ends_with("hostPageSize = 0x00000000"))
        );
        let mut j = Journal::new(64);
        OS_283.journal(None, &mut j);
        let rows: Vec<alloc::string::String> = j.events().map(|e| alloc::format!("{e}")).collect();
        assert_eq!(rows[1], "text varianta I8: numele = \"os-283\"");
        assert!(
            rows.iter()
                .any(|r| r.ends_with("bIsPrimary (2 = dupa VGA) = 0x00000002"))
        );
    }

    #[test]
    fn t121d_i8_check_refuses_the_wrong_order() {
        let l = super::super::msgq::QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        // Nothing queued yet: variant 1 refuses, os-283 goes on and sends.
        assert!(RPC_BEFORE.init_rpcs_ready_for_i8(&q).is_err());
        assert!(!OS_283.init_rpcs_ready_for_i8(&q).unwrap());
        let seqs = RPC_BEFORE
            .queue_init_rpcs(&mut q, &mut mem, &workstation())
            .unwrap();
        assert_eq!(seqs, [0, 1]);
        assert_eq!(q.rpcs_sent(), 2);
        assert!(RPC_BEFORE.init_rpcs_ready_for_i8(&q).unwrap());
        assert!(OS_283.init_rpcs_ready_for_i8(&q).is_err());
    }
}
