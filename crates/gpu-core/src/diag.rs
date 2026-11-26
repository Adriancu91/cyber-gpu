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

/// C-145: whether the live destination must have an event on the stick
/// before the library goes on ([`Flush::Now`]: the event and everything held
/// before it) or may hold it in memory for a while ([`Flush::Later`]: a
/// routine event inside a [`Routine`] stretch).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flush {
    Now,
    Later,
}

/// C-145: a live destination that may hold routine events: `line` gets every
/// event in push order with its [`Flush`]; `flush` is called when the held
/// events must reach the stick without a new event (the end of a routine
/// stretch, [`Journal::flush_live`]).
#[derive(Clone, Copy, Debug)]
pub struct LiveSink {
    pub line: fn(&Event, Flush),
    pub flush: fn(),
}

/// C-145: which events are routine during a long run of the same accesses
/// (`kronos --gpu`: the VRAM window moves, the TLB invalidate, the
/// work-channel doorbell and the notes that come with them). Only BAR0
/// **writes** at `writes` and notes equal to one of `notes` can be routine;
/// steps, values, texts, reads, configuration accesses, sequencer commands
/// and waits are never routine. At most `max_pending` routine events are
/// held: the next one goes out with [`Flush::Now`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Routine {
    pub writes: &'static [u64],
    pub notes: &'static [&'static str],
    pub max_pending: u32,
}

impl Routine {
    pub fn covers(&self, e: &Event) -> bool {
        match e {
            Event::MmioWrite { bar: 0, offset, .. } => self.writes.contains(offset),
            Event::Note(n) => self.notes.contains(n),
            _ => false,
        }
    }
}

/// C-145: what the live destination was handed during routine stretches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveStats {
    /// Events handed with [`Flush::Later`].
    pub held: u64,
    /// Times the held events had to go out ([`Flush::Now`] after held
    /// events, or a `flush` call).
    pub flushes: u64,
    /// Held right now (not yet on the stick).
    pub pending: u32,
}

pub struct Journal {
    events: VecDeque<Event>,
    cap: usize,
    dropped: u64,
    /// os-473 (27 Nov 2026): every event ever pushed, kept or dropped.
    pushed: u64,
    /// os-473: every write ever pushed, kept or dropped. Never decreases.
    writes_total: u64,
    sink: Option<fn(&Event)>,
    live: Option<LiveSink>,
    routine: Option<Routine>,
    stats: LiveStats,
}

impl Journal {
    pub fn new(cap: usize) -> Self {
        Journal {
            events: VecDeque::new(),
            cap: cap.max(1),
            dropped: 0,
            pushed: 0,
            writes_total: 0,
            sink: None,
            live: None,
            routine: None,
            stats: LiveStats::default(),
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
    /// C-145: like [`Self::with_sink`], with a destination that may hold
    /// routine events during a [`Routine`] stretch ([`Self::set_routine`]).
    /// Outside such a stretch every event is handed with [`Flush::Now`]:
    /// the same as `with_sink`.
    pub fn with_live_sink(cap: usize, live: LiveSink) -> Self {
        let mut j = Self::new(cap);
        j.live = Some(live);
        j
    }

    /// C-145: starts (`Some`) or ends (`None`) a routine stretch. Ending it,
    /// or changing it, first puts the held events on the stick. The memory
    /// journal is the same either way: every event, in order.
    pub fn set_routine(&mut self, r: Option<Routine>) {
        self.flush_live();
        self.routine = r.filter(|r| r.max_pending > 0);
    }

    pub fn routine(&self) -> Option<Routine> {
        self.routine
    }

    /// C-145: the held events go to the stick now (e.g. once per token).
    pub fn flush_live(&mut self) {
        if self.stats.pending > 0 {
            if let Some(l) = self.live {
                (l.flush)();
            }
            self.stats.pending = 0;
            self.stats.flushes += 1;
        }
    }

    pub fn live_stats(&self) -> LiveStats {
        self.stats
    }

    pub fn push(&mut self, e: Event) {
        if let Some(s) = self.sink {
            s(&e);
        }
        if let Some(l) = self.live {
            let hold = self
                .routine
                .is_some_and(|r| r.covers(&e) && self.stats.pending < r.max_pending);
            if hold {
                (l.line)(&e, Flush::Later);
                self.stats.pending += 1;
                self.stats.held += 1;
            } else {
                (l.line)(&e, Flush::Now);
                if self.stats.pending > 0 {
                    self.stats.pending = 0;
                    self.stats.flushes += 1;
                }
            }
        }
        self.pushed += 1;
        if is_write(&e) {
            self.writes_total += 1;
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
    /// Writes among the events still kept (the ring). When the ring is
    /// full, this number goes DOWN as old writes are dropped — so it must
    /// not be used to ask "did anything write since ...": use
    /// [`Self::writes_total`] for that (os-473).
    pub fn writes(&self) -> usize {
        self.events.iter().filter(|e| is_write(e)).count()
    }

    /// os-473 (27 Nov 2026): every write pushed since the journal was made,
    /// dropped ones included. It never decreases, so the difference of two
    /// readings is the number of writes pushed between them, however full
    /// the ring is. (On the workstation the OS ring of 4 096 events filled
    /// up and `writes()` fell while the read tour ran: a false "the tour
    /// journaled a write".)
    pub fn writes_total(&self) -> u64 {
        self.writes_total
    }

    /// os-473: every event pushed since the journal was made, dropped ones
    /// included. A mark for [`Self::events_since`].
    pub fn pushed(&self) -> u64 {
        self.pushed
    }

    /// os-473: the events pushed after the mark `pushed_before` (a value of
    /// [`Self::pushed`]) that are still kept, oldest first. If some of them
    /// were already dropped, only the kept ones are given.
    pub fn events_since(&self, pushed_before: u64) -> impl Iterator<Item = &Event> {
        let new = self.pushed.saturating_sub(pushed_before);
        let kept_new = usize::try_from(new)
            .unwrap_or(usize::MAX)
            .min(self.events.len());
        self.events.iter().skip(self.events.len() - kept_new)
    }
}

fn is_write(e: &Event) -> bool {
    matches!(e, Event::ConfigWrite { .. } | Event::MmioWrite { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn writes_total_never_falls_when_the_ring_drops_writes() {
        // os-473: a full ring of writes, then only reads. `writes()` falls
        // (the old behaviour the tour tripped on), `writes_total()` does not.
        let mut j = Journal::new(4);
        for i in 0..4 {
            j.push(Event::MmioWrite {
                bar: 0,
                offset: i,
                before: 0,
                written: 1,
                after: 1,
            });
        }
        let (w0, t0, p0) = (j.writes(), j.writes_total(), j.pushed());
        j.push(Event::Note("citire"));
        j.push(Event::Note("citire"));
        assert_eq!((w0, j.writes()), (4, 2), "the ring lost two writes");
        assert_eq!(j.writes_total(), t0, "no write was pushed");
        assert_eq!(j.pushed() - p0, 2);
        assert_eq!(j.events_since(p0).count(), 2);
        // More new events than the ring holds: only the kept ones.
        for _ in 0..10 {
            j.push(Event::Note("x"));
        }
        assert_eq!(j.events_since(p0).count(), 4);
        assert_eq!(j.events_since(j.pushed()).count(), 0);
    }

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

    /// C-145: a routine stretch holds only the routine events, never more
    /// than `max_pending`; every step, value and other write goes out at
    /// once and takes the held ones with it; the order is the push order and
    /// the memory journal is unchanged.
    #[test]
    fn routine_stretch_holds_only_routine_events() {
        use std::sync::Mutex;
        use std::vec::Vec;
        static LOG: Mutex<Vec<(String, Flush)>> = Mutex::new(Vec::new());
        static FLUSHES: Mutex<u32> = Mutex::new(0);
        fn line(e: &Event, f: Flush) {
            LOG.lock().unwrap().push((e.to_string(), f));
        }
        fn flush() {
            *FLUSHES.lock().unwrap() += 1;
        }
        const W: [u64; 2] = [0x1700, 0xbb0090];
        const N: [&str; 1] = ["permis"];
        let w = |off: u64| Event::MmioWrite {
            bar: 0,
            offset: off,
            before: 0,
            written: 1,
            after: 1,
        };
        let mut j = Journal::with_live_sink(64, LiveSink { line, flush });
        j.push(w(0x1700)); // no stretch yet: Now
        j.set_routine(Some(Routine {
            writes: &W,
            notes: &N,
            max_pending: 3,
        }));
        j.push(Event::Note("permis")); // held
        j.push(w(0x1700)); // held
        j.push(w(0x1704)); // not in the list: Now
        j.push(Event::MmioRead {
            bar: 0,
            offset: 0x1700,
            value: 1,
        }); // reads never routine
        j.push(w(0xbb0090)); // held
        j.push(w(0x1700)); // held
        j.push(w(0x1700)); // held (3rd)
        j.push(w(0x1700)); // over the cap: Now
        j.push(Event::Note("altceva")); // other note: Now
        j.push(w(0x1700)); // held
        j.push(Event::Step {
            name: "pas",
            ok: true,
        }); // steps never held
        j.push(w(0x1700)); // held
        j.set_routine(None); // the end flushes
        j.push(w(0x1700)); // after the stretch: Now
        let log = LOG.lock().unwrap().clone();
        let flags: Vec<Flush> = log.iter().map(|x| x.1).collect();
        use Flush::{Later as L, Now as N_};
        assert_eq!(flags, [N_, L, L, N_, N_, L, L, L, N_, N_, L, N_, L, N_]);
        // Same events, same order as the memory journal.
        let mem: Vec<String> = j.events().map(|e| e.to_string()).collect();
        let sent: Vec<String> = log.iter().map(|x| x.0.clone()).collect();
        assert_eq!(mem, sent);
        assert_eq!(
            *FLUSHES.lock().unwrap(),
            1,
            "only the end of the stretch calls flush"
        );
        let st = j.live_stats();
        assert_eq!(st.held, 7);
        assert_eq!(st.pending, 0);
        assert_eq!(st.flushes, 4);
        // Without a live sink the stretch changes nothing.
        let mut k = Journal::new(4);
        k.set_routine(Some(Routine {
            writes: &W,
            notes: &N,
            max_pending: 3,
        }));
        k.push(w(0x1700));
        assert_eq!((k.len(), k.live_stats().held), (1, 0));
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
