//! First-fit range allocator with coalescing (invariant M3). Used for VRAM
//! offsets, GPU virtual address ranges and the CPU backend's arena. It
//! manages numbers, never touches memory.

use alloc::collections::BTreeMap;

use gpu_core::error::{GpuError, GpuResult};

use crate::region::align_up;

pub struct RangeAllocator {
    base: u64,
    size: u64,
    /// start -> len, non-overlapping, non-adjacent (always coalesced).
    free: BTreeMap<u64, u64>,
    /// start -> len.
    used: BTreeMap<u64, u64>,
}

impl RangeAllocator {
    pub fn new(base: u64, size: u64) -> GpuResult<Self> {
        if size == 0 || base.checked_add(size).is_none() {
            return Err(GpuError::InvalidArgument("regiune invalida"));
        }
        let mut free = BTreeMap::new();
        free.insert(base, size);
        Ok(RangeAllocator {
            base,
            size,
            free,
            used: BTreeMap::new(),
        })
    }

    pub fn total(&self) -> u64 {
        self.size
    }
    pub fn free_bytes(&self) -> u64 {
        self.free.values().sum()
    }
    pub fn used_bytes(&self) -> u64 {
        self.used.values().sum()
    }

    /// Allocate `size` bytes aligned to `align` (power of two). Returns the start.
    pub fn alloc(&mut self, size: u64, align: u64) -> GpuResult<u64> {
        if size == 0 {
            return Err(GpuError::InvalidArgument("marime zero"));
        }
        if !align.is_power_of_two() {
            return Err(GpuError::InvalidArgument(
                "aliniere care nu e putere a lui 2",
            ));
        }
        let mut hit = None;
        for (&start, &len) in &self.free {
            let Some(a) = align_up(start, align) else {
                continue;
            };
            let pad = a - start;
            if pad.checked_add(size).is_some_and(|need| need <= len) {
                hit = Some((start, len, a));
                break;
            }
        }
        let Some((start, len, a)) = hit else {
            return Err(GpuError::OutOfMemory { requested: size });
        };
        self.free.remove(&start);
        if a > start {
            self.free.insert(start, a - start);
        }
        let end = a + size;
        let free_end = start + len;
        if free_end > end {
            self.free.insert(end, free_end - end);
        }
        self.used.insert(a, size);
        Ok(a)
    }

    /// Release the range that starts at `start`. Double free and unknown
    /// starts are errors, not silent no-ops.
    pub fn release(&mut self, start: u64) -> GpuResult<u64> {
        let len = self.used.remove(&start).ok_or(GpuError::InvalidHandle)?;
        let mut s = start;
        let mut l = len;
        if let Some((&ps, &pl)) = self.free.range(..start).next_back()
            && ps + pl == start
        {
            self.free.remove(&ps);
            s = ps;
            l += pl;
        }
        if let Some(&nl) = self.free.get(&(start + len)) {
            self.free.remove(&(start + len));
            l += nl;
        }
        self.free.insert(s, l);
        Ok(len)
    }

    pub fn size_of(&self, start: u64) -> Option<u64> {
        self.used.get(&start).copied()
    }

    /// Verify M3. Cheap enough to call from tests after every operation.
    pub fn check(&self) -> Result<(), &'static str> {
        let end = self.base + self.size;
        let mut all: alloc::vec::Vec<(u64, u64, bool)> = self
            .free
            .iter()
            .map(|(&s, &l)| (s, l, true))
            .chain(self.used.iter().map(|(&s, &l)| (s, l, false)))
            .collect();
        all.sort_unstable();
        let mut cursor = self.base;
        let mut prev_free = false;
        for (s, l, is_free) in all {
            if l == 0 {
                return Err("interval gol");
            }
            if s < cursor {
                return Err("suprapunere");
            }
            if s > cursor {
                return Err("gaura neevidentiata");
            }
            if is_free && prev_free {
                return Err("intervale libere necoalescate");
            }
            prev_free = is_free;
            cursor = s + l;
        }
        if cursor != end {
            return Err("total gresit");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn basic() {
        let mut a = RangeAllocator::new(0x1000, 0x10000).unwrap();
        let x = a.alloc(0x100, 0x1000).unwrap();
        assert_eq!(x, 0x1000);
        let y = a.alloc(0x100, 0x100).unwrap();
        assert_eq!(y, 0x1100);
        a.check().unwrap();
        assert_eq!(a.release(x), Ok(0x100));
        assert_eq!(a.release(x), Err(GpuError::InvalidHandle));
        a.release(y).unwrap();
        a.check().unwrap();
        assert_eq!(a.free_bytes(), 0x10000);
        assert!(matches!(
            a.alloc(0x20000, 1),
            Err(GpuError::OutOfMemory { .. })
        ));
        assert!(a.alloc(0, 1).is_err());
        assert!(a.alloc(1, 3).is_err());
    }

    #[test]
    fn randomized_invariants() {
        // Deterministic LCG, no external crates.
        let mut seed = 0x2482_10de_u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let mut a = RangeAllocator::new(0, 1 << 24).unwrap();
        let mut live: Vec<u64> = Vec::new();
        for _ in 0..5000 {
            if live.is_empty() || rnd() % 3 != 0 {
                let size = 1 + rnd() % 65536;
                let align = 1u64 << (rnd() % 13);
                if let Ok(s) = a.alloc(size, align) {
                    assert_eq!(s % align, 0);
                    live.push(s);
                }
            } else {
                let i = (rnd() as usize) % live.len();
                a.release(live.swap_remove(i)).unwrap();
            }
            a.check().unwrap();
            assert_eq!(a.free_bytes() + a.used_bytes(), a.total());
        }
        for s in live {
            a.release(s).unwrap();
        }
        a.check().unwrap();
        assert_eq!(a.free_bytes(), 1 << 24);
    }
}
