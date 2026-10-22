//! Generation-tagged buffer table (invariant M6) on top of a range allocator.

use alloc::vec::Vec;

use gpu_core::backend::{BufferDesc, BufferId};
use gpu_core::error::{GpuError, GpuResult};

use crate::range::RangeAllocator;
use crate::region::check_access;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Allocation {
    /// Start inside the managed space (VRAM offset, arena offset, ...).
    pub start: u64,
    pub size: u64,
    pub desc: BufferDesc,
}

struct Slot {
    generation: u32,
    alloc: Option<Allocation>,
}

pub struct BufferTable {
    space: RangeAllocator,
    slots: Vec<Slot>,
    free_slots: Vec<u32>,
}

impl BufferTable {
    pub fn new(space: RangeAllocator) -> Self {
        BufferTable {
            space,
            slots: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    pub fn space(&self) -> &RangeAllocator {
        &self.space
    }

    pub fn allocate(&mut self, desc: BufferDesc) -> GpuResult<(BufferId, Allocation)> {
        let align = desc.align.max(1);
        let start = self.space.alloc(desc.size, align)?;
        let a = Allocation {
            start,
            size: desc.size,
            desc,
        };
        let index = match self.free_slots.pop() {
            Some(i) => {
                self.slots[i as usize].alloc = Some(a);
                i
            }
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    alloc: Some(a),
                });
                (self.slots.len() - 1) as u32
            }
        };
        Ok((
            BufferId {
                index,
                generation: self.slots[index as usize].generation,
            },
            a,
        ))
    }

    pub fn get(&self, id: BufferId) -> GpuResult<Allocation> {
        let s = self
            .slots
            .get(id.index as usize)
            .ok_or(GpuError::InvalidHandle)?;
        match s.alloc {
            Some(a) if s.generation == id.generation => Ok(a),
            _ => Err(GpuError::InvalidHandle),
        }
    }

    /// Resolve `[offset, offset+len)` inside buffer `id` to a range in the space.
    pub fn resolve(&self, id: BufferId, offset: u64, len: u64) -> GpuResult<u64> {
        let a = self.get(id)?;
        check_access(offset, len, a.size)?;
        Ok(a.start + offset)
    }

    pub fn release(&mut self, id: BufferId) -> GpuResult<Allocation> {
        let a = self.get(id)?;
        self.space.release(a.start)?;
        let s = &mut self.slots[id.index as usize];
        s.alloc = None;
        s.generation = s.generation.wrapping_add(1);
        self.free_slots.push(id.index);
        Ok(a)
    }

    pub fn live(&self) -> usize {
        self.slots.iter().filter(|s| s.alloc.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_handles_rejected() {
        let mut t = BufferTable::new(RangeAllocator::new(0, 4096).unwrap());
        let (a, _) = t.allocate(BufferDesc::device(1024)).unwrap();
        assert_eq!(t.resolve(a, 1000, 24), Ok(1000));
        assert!(matches!(
            t.resolve(a, 1000, 25),
            Err(GpuError::OutOfBounds { .. })
        ));
        t.release(a).unwrap();
        let (b, _) = t.allocate(BufferDesc::device(512)).unwrap();
        assert_eq!(b.index, a.index, "slot reused");
        assert_eq!(
            t.get(a),
            Err(GpuError::InvalidHandle),
            "old generation rejected"
        );
        assert_eq!(t.release(a), Err(GpuError::InvalidHandle));
        assert!(t.get(b).is_ok());
        assert_eq!(t.live(), 1);
    }
}
