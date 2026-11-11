//! I8c (part) — the payload of `SET_REGISTRY` (`PACKED_REGISTRY_TABLE`),
//! built and tested **only in SIMULARE**.
//!
//! ```text
//! +0  size        u32   whole table: header + entries + names + data
//! +4  numEntries  u32
//! +8  entries[n]  16 B each: nameOffset u32, type u8 (+3 pad), data u32, length u32
//! ... names, NUL-terminated, then binary/string data, in entry order
//! ```
//! Offsets are from the start of the table. A DWORD entry keeps its value in
//! `data` (`length` = 4) and adds only its name to the blob.
//!
//! Sources (MIT, open-gpu-kernel-modules@570.144): `g_os_nvoc.h` (the two
//! structs, type codes 0–3), `arch/nvalloc/unix/src/registry.c`
//! (`regCountEntriesAndSize`, `regCopyEntriesToPackedBuffer`) — read,
//! re-written. The two keys tinygrad sends on GA102 (study, proposal C) are
//! `RMForcePcieConfigSave=1` and `RMSecBusResetEnable=1`; NVIDIA's own Linux
//! driver writes the same two in `osinit.c`. Layout of the 16-byte entry is C
//! layout arithmetic (u32, u8, 3 pad, u32, u32), not a bindgen check: the
//! header that defines it pulls in most of RM.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use super::{get_u32, put_u32};

pub const HEADER_SIZE: usize = 8;
pub const ENTRY_SIZE: usize = 16;
pub const TYPE_DWORD: u8 = 1;
pub const TYPE_BINARY: u8 = 2;
pub const TYPE_STRING: u8 = 3;
/// Our limit; the boot table has two entries.
pub const MAX_ENTRIES: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Dword(u32),
    Binary(Vec<u8>),
    /// Stored as given (include the trailing NUL if GSP-RM expects one).
    String(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: &'static str,
    pub value: Value,
}

/// The two entries tinygrad sends on GA102 with 570.144 (and NVIDIA's own
/// driver writes in `osinit.c`). Values proven on real boards by others,
/// not by us: "dovedite empiric".
pub fn boot_entries() -> [Entry; 2] {
    [
        Entry {
            name: "RMForcePcieConfigSave",
            value: Value::Dword(1),
        },
        Entry {
            name: "RMSecBusResetEnable",
            value: Value::Dword(1),
        },
    ]
}

/// `RMDevidCheckIgnore = 1`: nouveau r570 and nova-core add it to the two
/// above ("allows GSP-RM to boot even if the PCI dev ID is not found" —
/// nova's own comment); RM and tinygrad do not. A C-108 variant.
pub fn devid_check_ignore() -> Entry {
    Entry {
        name: "RMDevidCheckIgnore",
        value: Value::Dword(1),
    }
}

/// The boot table of a variant: the two entries, plus `RMDevidCheckIgnore`
/// when asked (nouveau/nova order: the two first, then the third).
pub fn boot_entries_with(devid_ignore: bool) -> Vec<Entry> {
    let mut v = boot_entries().to_vec();
    if devid_ignore {
        v.push(devid_check_ignore());
    }
    v
}

/// Builds the table. Names must be ASCII without NUL.
pub fn encode(entries: &[Entry]) -> GpuResult<Vec<u8>> {
    if entries.is_empty() || entries.len() > MAX_ENTRIES {
        return Err(GpuError::InvalidArgument("registru: 1..64 intrari"));
    }
    let mut blob_at = HEADER_SIZE + ENTRY_SIZE * entries.len();
    let mut size = blob_at;
    for e in entries {
        if e.name.is_empty() || !e.name.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(GpuError::InvalidArgument("registru: nume invalid"));
        }
        size += e.name.len() + 1;
        size += match &e.value {
            Value::Dword(_) => 0,
            Value::Binary(d) | Value::String(d) => d.len(),
        };
    }
    let size32 =
        u32::try_from(size).map_err(|_| GpuError::InvalidArgument("registru prea mare"))?;
    let mut t = alloc::vec![0u8; size];
    put_u32(&mut t, 0, size32);
    put_u32(&mut t, 4, entries.len() as u32);
    for (i, e) in entries.iter().enumerate() {
        let at = HEADER_SIZE + i * ENTRY_SIZE;
        put_u32(&mut t, at, blob_at as u32);
        t[blob_at..blob_at + e.name.len()].copy_from_slice(e.name.as_bytes());
        blob_at += e.name.len() + 1; // NUL already there
        let (ty, data, len) = match &e.value {
            Value::Dword(v) => (TYPE_DWORD, *v, 4u32),
            Value::Binary(d) | Value::String(d) => {
                let ty = if matches!(e.value, Value::Binary(_)) {
                    TYPE_BINARY
                } else {
                    TYPE_STRING
                };
                t[blob_at..blob_at + d.len()].copy_from_slice(d);
                let off = blob_at as u32;
                blob_at += d.len();
                (ty, off, d.len() as u32)
            }
        };
        t[at + 4] = ty;
        put_u32(&mut t, at + 8, data);
        put_u32(&mut t, at + 12, len);
    }
    debug_assert_eq!(blob_at, size);
    Ok(t)
}

/// One decoded entry (for the simulated GSP and for logs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub name: Vec<u8>,
    pub value: Value,
}

/// Reads a table back, checking every offset against the table size.
pub fn decode(t: &[u8]) -> GpuResult<Vec<Decoded>> {
    let bad = || GpuError::InvalidArgument("registru: tabel corupt");
    if t.len() < HEADER_SIZE {
        return Err(bad());
    }
    let size = get_u32(t, 0) as usize;
    let n = get_u32(t, 4) as usize;
    if size != t.len() || n == 0 || n > MAX_ENTRIES || HEADER_SIZE + n * ENTRY_SIZE > size {
        return Err(bad());
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let at = HEADER_SIZE + i * ENTRY_SIZE;
        let name_off = get_u32(t, at) as usize;
        let ty = t[at + 4];
        let data = get_u32(t, at + 8);
        let len = get_u32(t, at + 12) as usize;
        let name_end = t
            .get(name_off..)
            .and_then(|r| r.iter().position(|&b| b == 0))
            .ok_or_else(bad)?;
        let name = t[name_off..name_off + name_end].to_vec();
        let value = match ty {
            TYPE_DWORD if len == 4 => Value::Dword(data),
            TYPE_BINARY | TYPE_STRING => {
                let d = t
                    .get(data as usize..data as usize + len)
                    .ok_or_else(bad)?
                    .to_vec();
                if ty == TYPE_BINARY {
                    Value::Binary(d)
                } else {
                    Value::String(d)
                }
            }
            _ => return Err(bad()),
        };
        out.push(Decoded { name, value });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::rpc;

    #[test]
    fn boot_table_layout() {
        let t = encode(&boot_entries()).unwrap();
        // 8 + 2*16 + "RMForcePcieConfigSave\0" (22) + "RMSecBusResetEnable\0" (20)
        assert_eq!(t.len(), 82);
        assert_eq!(get_u32(&t, 0), 82);
        assert_eq!(get_u32(&t, 4), 2);
        assert_eq!(get_u32(&t, 8), 40); // first name right after the entries
        assert_eq!(t[12], TYPE_DWORD);
        assert_eq!(get_u32(&t, 16), 1);
        assert_eq!(get_u32(&t, 20), 4);
        assert_eq!(get_u32(&t, 24), 62);
        assert_eq!(&t[40..62], b"RMForcePcieConfigSave\0");
        assert_eq!(&t[62..82], b"RMSecBusResetEnable\0");
        let d = decode(&t).unwrap();
        assert_eq!(d[1].name, b"RMSecBusResetEnable");
        assert_eq!(d[1].value, Value::Dword(1));
    }

    #[test]
    fn binary_and_string_roundtrip_through_an_rpc() {
        let entries = [
            Entry {
                name: "A",
                value: Value::Dword(7),
            },
            Entry {
                name: "Blob",
                value: Value::Binary(alloc::vec![1, 2, 3]),
            },
            Entry {
                name: "S",
                value: Value::String(b"x\0".to_vec()),
            },
        ];
        let t = encode(&entries).unwrap();
        let msg = rpc::encode(rpc::function::SET_REGISTRY, 3, 3, &t).unwrap();
        let back = rpc::decode(&msg).unwrap();
        assert_eq!(back.function, 73);
        let d = decode(&back.payload).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(d[1].value, Value::Binary(alloc::vec![1, 2, 3]));
        assert_eq!(d[2].value, Value::String(b"x\0".to_vec()));
    }

    #[test]
    fn refuses_bad_input_and_corrupt_tables() {
        assert!(encode(&[]).is_err());
        assert!(
            encode(&[Entry {
                name: "a b",
                value: Value::Dword(0)
            }])
            .is_err()
        );
        let mut t = encode(&boot_entries()).unwrap();
        assert!(decode(&t[..81]).is_err()); // size field disagrees
        put_u32(&mut t, 8, 500); // name offset outside the table
        assert!(decode(&t).is_err());
        let mut t = encode(&boot_entries()).unwrap();
        t[12] = 9; // unknown type
        assert!(decode(&t).is_err());
    }
}
