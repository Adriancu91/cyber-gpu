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
    /// C-96: one GSP-RM sequencer command, journaled before it runs
    /// (`index` within its request; `args` are the command's words in
    /// order, `n` of them meaningful).
    SeqCmd {
        index: u32,
        op: &'static str,
        n: u8,
        args: [u32; 5],
    },
    /// C-99: a long wait that ended well — `what` was awaited, the answer
    /// came after `waited_us` (so the board's log says how long GSP-RM took).
    Waited {
        what: &'static str,
        waited_us: u64,
    },
    /// C-102: a value read for diagnosis (a register or a field of a GSP
    /// message), named — so the board's log says where an error came from.
    Value {
        what: &'static str,
        value: u32,
    },
    /// 11 Nov 2026 (after os-279): a short text read from a GSP message or
    /// a name looked up for a number (an `NV_STATUS` name, a message name,
    /// the `source` of a NOCAT record). `bytes[..len]` is the text; anything
    /// not printable ASCII is shown as `.`.
    Text {
        what: &'static str,
        len: u8,
        bytes: [u8; TEXT_LEN],
    },
}

/// Longest text an [`Event::Text`] carries (longer texts are cut).
pub const TEXT_LEN: usize = 32;

impl Event {
    /// An [`Event::Text`] from any bytes: cut at [`TEXT_LEN`] and at the
    /// first NUL (C strings), the rest filled with zeros.
    pub fn text(what: &'static str, s: &[u8]) -> Event {
        let end = s
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(s.len())
            .min(TEXT_LEN);
        let mut bytes = [0u8; TEXT_LEN];
        bytes[..end].copy_from_slice(&s[..end]);
        Event::Text {
            what,
            len: end as u8,
            bytes,
        }
    }
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
            Event::SeqCmd { index, op, n, args } => {
                write!(f, "secv #{index} {op}")?;
                for a in args.iter().take(usize::from(n)) {
                    write!(f, " {a:#x}")?;
                }
                Ok(())
            }
            Event::Waited { what, waited_us } => {
                write!(f, "asteptat {what}: raspuns dupa {waited_us} us")
            }
            Event::Value { what, value } => write!(f, "valoare {what} = {value:#010x}"),
            Event::Text {
                what,
                len,
                ref bytes,
            } => {
                write!(f, "text {what} = \"")?;
                for &b in bytes.iter().take(usize::from(len).min(TEXT_LEN)) {
                    let c = if (0x20..0x7f).contains(&b) && b != b'"' {
                        b as char
                    } else {
                        '.'
                    };
                    write!(f, "{c}")?;
                }
                write!(f, "\"")
            }
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
    /// 12 Nov 2026 (os-295): a text longer than [`TEXT_LEN`], in pieces of
    /// [`TEXT_LEN`] bytes — the first as `what`, the following ones as `more`
    /// (e.g. "motivul esecului" / "motivul esecului (urmare)"), so the whole
    /// text reaches the stick even when the screen is gone. An empty text
    /// gives one empty event. Pieces are cut on bytes, so a multi-byte
    /// character on a boundary shows as `.` on both sides.
    pub fn push_text(&mut self, what: &'static str, more: &'static str, s: &str) {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            self.push(Event::text(what, b""));
            return;
        }
        for (i, piece) in bytes.chunks(TEXT_LEN).enumerate() {
            self.push(Event::text(if i == 0 { what } else { more }, piece));
        }
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
    fn text_events_are_cut_and_printable() {
        let e = Event::text("sursa NOCAT", b"GSP\0garbage after the NUL");
        assert_eq!(e.to_string(), "text sursa NOCAT = \"GSP\"");
        let long: Vec<u8> = (0..60u8).map(|i| b'a' + i % 26).collect();
        let e = Event::text("lung", &long);
        assert_eq!(
            e.to_string(),
            "text lung = \"abcdefghijklmnopqrstuvwxyzabcdef\""
        );
        let e = Event::text("ciudat", &[b'o', b'k', 0x01, 0xff, b'"', b' ', b'x']);
        assert_eq!(e.to_string(), "text ciudat = \"ok... x\"");
        assert_eq!(Event::text("gol", b"").to_string(), "text gol = \"\"");
    }

    /// os-295: the reason a step failed must reach the stick whole, in
    /// pieces of TEXT_LEN, the first labelled differently from the rest.
    #[test]
    fn long_text_is_journaled_in_pieces() {
        let mut j = Journal::new(16);
        j.push_text("motiv", "motiv (urmare)", "eroare hardware 0x57");
        let rows: Vec<String> = j.events().map(|e| e.to_string()).collect();
        assert_eq!(rows, ["text motiv = \"eroare hardware 0x57\""]);

        let mut j = Journal::new(16);
        let long = "argument invalid: RM_CONTROL: raspunsul e pentru alta comanda";
        assert_eq!(long.len(), 61);
        j.push_text("motiv", "motiv (urmare)", long);
        let rows: Vec<String> = j.events().map(|e| e.to_string()).collect();
        assert_eq!(
            rows,
            [
                "text motiv = \"argument invalid: RM_CONTROL: ra\"",
                "text motiv (urmare) = \"spunsul e pentru alta comanda\"",
            ]
        );
        // Put back together, the pieces are the whole text.
        let joined: String = j
            .events()
            .map(|e| match e {
                Event::Text { len, bytes, .. } => core::str::from_utf8(&bytes[..usize::from(*len)])
                    .unwrap()
                    .to_string(),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(joined, long);

        let mut j = Journal::new(16);
        j.push_text("motiv", "motiv (urmare)", "");
        assert_eq!(j.events().next().unwrap().to_string(), "text motiv = \"\"");
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
