//! Diagnostic journal. Every register access a backend performs on real
//! hardware is recorded with its value (rule D-013: before/after values in the
//! log). The journal is bounded so it never grows the kernel heap without
//! limit; the oldest entries are dropped and counted.

use alloc::collections::VecDeque;
use core::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    ConfigRead {
        offset: u16,
        value: u32,
    },
    ConfigWrite {
        offset: u16,
        before: u32,
        written: u32,
        after: u32,
    },
    MmioRead {
        bar: u8,
        offset: u64,
        value: u32,
    },
    MmioWrite {
        bar: u8,
        offset: u64,
        before: u32,
        written: u32,
        after: u32,
    },
    Step {
        name: &'static str,
        ok: bool,
    },
    Note(&'static str),
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Event::ConfigRead { offset, value } => {
                write!(f, "cfg  r {offset:#05x} = {value:#010x}")
            }
            Event::ConfigWrite {
                offset,
                before,
                written,
                after,
            } => {
                write!(
                    f,
                    "cfg  w {offset:#05x}: {before:#010x} <- {written:#010x} => {after:#010x}"
                )
            }
            Event::MmioRead { bar, offset, value } => {
                write!(f, "bar{bar} r {offset:#08x} = {value:#010x}")
            }
            Event::MmioWrite {
                bar,
                offset,
                before,
                written,
                after,
            } => {
                write!(
                    f,
                    "bar{bar} w {offset:#08x}: {before:#010x} <- {written:#010x} => {after:#010x}"
                )
            }
            Event::Step { name, ok } => write!(f, "pas {name}: {}", if ok { "ok" } else { "ESEC" }),
            Event::Note(n) => write!(f, "nota: {n}"),
        }
    }
}

pub struct Journal {
    events: VecDeque<Event>,
    cap: usize,
    dropped: u64,
    sink: Option<fn(&Event)>,
}

impl Journal {
    pub fn new(cap: usize) -> Self {
        Journal {
            events: VecDeque::new(),
            cap: cap.max(1),
            dropped: 0,
            sink: None,
        }
    }
    /// Same journal, and every event is also handed to `sink` the moment it
    /// is pushed (C-51: the OS writes each register access with its own
    /// timestamp as it happens, so a hang in the middle of a step still
    /// leaves the accesses before it on the stick).
    pub fn with_sink(cap: usize, sink: fn(&Event)) -> Self {
        let mut j = Self::new(cap);
        j.sink = Some(sink);
        j
    }
    pub fn push(&mut self, e: Event) {
        if let Some(s) = self.sink {
            s(&e);
        }
        if self.events.len() == self.cap {
            self.events.pop_front();
            self.dropped += 1;
        }
        self.events.push_back(e);
    }
    pub fn events(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }
    pub fn len(&self) -> usize {
        self.events.len()
    }
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
    /// Number of writes recorded (config + MMIO). Level-0 probes must keep this at 0.
    pub fn writes(&self) -> usize {
        self.events
            .iter()
            .filter(|e| matches!(e, Event::ConfigWrite { .. } | Event::MmioWrite { .. }))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn sink_sees_every_event_even_dropped_ones() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static SEEN: AtomicUsize = AtomicUsize::new(0);
        fn count(_: &Event) {
            SEEN.fetch_add(1, Ordering::Relaxed);
        }
        let mut j = Journal::with_sink(1, count);
        j.push(Event::Note("a"));
        j.push(Event::Note("b"));
        j.push(Event::Note("c"));
        assert_eq!(SEEN.load(Ordering::Relaxed), 3);
        assert_eq!((j.len(), j.dropped()), (1, 2));
    }

    #[test]
    fn bounded_and_counted() {
        let mut j = Journal::new(2);
        j.push(Event::Note("a"));
        j.push(Event::MmioRead {
            bar: 0,
            offset: 0,
            value: 0x174000a1,
        });
        j.push(Event::ConfigWrite {
            offset: 4,
            before: 0,
            written: 2,
            after: 2,
        });
        assert_eq!(j.len(), 2);
        assert_eq!(j.dropped(), 1);
        assert_eq!(j.writes(), 1);
        assert_eq!(
            j.events().next().unwrap().to_string(),
            "bar0 r 0x000000 = 0x174000a1"
        );
    }
}
