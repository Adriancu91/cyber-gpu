//! T234 (C-157 p.2, os-423 comparison): how many global loads of a NAK listing wait for their value before the next
//! load is issued. Same rule as `tools/nak/citiri_seriale.py`: an `ld.global` writes scoreboard `wr:N`; if an
//! instruction up to the next `ld.global` waits on N (`wt=xxxxxx`, rightmost digit = scoreboard 0), the load is
//! "serial". Text only (the listings recorded by `tools/nak/build*.sh`), MEASURED in the cloud, nothing on a board.
//!
//! Why: `mm_tc` ran on the board at 1.03 TFLOP/s instead of 8-17 (os-423). The listing shows every one of its
//! 96 loads waited before the next. A future "loads ahead" program must show loads in flight in this same test.

use std::path::PathBuf;

struct Loads {
    total: usize,
    serial: usize,
    max_in_flight: usize,
}

fn listing(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/nak")
        .join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn scoreboard<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let at = line.find(key)? + key.len();
    line.get(at..)
}

fn analyse(text: &str) -> Loads {
    let lines: Vec<&str> = text.lines().collect();
    let loads: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].contains("ld.global"))
        .collect();
    let (mut serial, mut run, mut max_in_flight) = (0, 0, 0);
    for (a, &i) in loads.iter().enumerate() {
        let end = loads.get(a + 1).copied().unwrap_or(lines.len() - 1);
        let sb = scoreboard(lines[i], "wr:")
            .and_then(|s| s.chars().next())
            .and_then(|c| c.to_digit(10));
        let waited = sb.is_some_and(|sb| {
            lines[i + 1..=end].iter().any(|x| {
                scoreboard(x, "wt=").is_some_and(|m| {
                    let m: Vec<char> = m.chars().take(6).collect();
                    m.len() == 6 && m[5 - sb as usize] == '1'
                })
            })
        });
        run += 1;
        max_in_flight = max_in_flight.max(run);
        if waited {
            serial += 1;
            run = 0;
        }
    }
    Loads {
        total: loads.len(),
        serial,
        max_in_flight,
    }
}

#[test]
fn t234a_mm_tc_waits_for_every_load() {
    let l = analyse(&listing("out2/mm_tc.sm86.txt"));
    assert_eq!((l.total, l.serial, l.max_in_flight), (96, 96, 1));
}

#[test]
fn t234b_all_matmul_programs_have_the_same_shape() {
    for (name, n) in [
        ("out2/mm_tc128.sm86.txt", 128),
        ("out2/mm_s8.sm86.txt", 128),
        ("out2/mm_tc_gelu.sm86.txt", 65),
        ("out/mm_train_tile.sm86.txt", 32),
        ("out/mm_tv_128x128.sm86.txt", 80),
        ("out/mm_tv_64x64k8.sm86.txt", 24),
    ] {
        let l = analyse(&listing(name));
        assert_eq!((l.total, l.serial, l.max_in_flight), (n, n, 1), "{name}");
    }
}

#[test]
fn t234c_the_rule_sees_loads_in_flight_where_there_are_some() {
    // Attention gen 3 (passed on the board): clamped index, loads into a register array, zero chosen afterwards.
    let l = analyse(&listing("out/attn_fwd_tile3.sm86.txt"));
    assert_eq!((l.total, l.serial), (80, 8));
    assert!(l.max_in_flight >= 16, "{}", l.max_in_flight);
    let q = analyse(&listing("out/q8_fused32.sm86.txt"));
    assert_eq!((q.total, q.max_in_flight), (7, 4));
}

#[test]
fn t234d_planted_listing() {
    // Two loads on different scoreboards, the wait only after both: 2 in flight, 1 serial (the second, waited at end).
    let text = "a = ld.global.b32 [r0] // wr:0\nb = ld.global.b32 [r2] // wr:1\nc = fadd a b // wt=000011\n";
    let l = analyse(text);
    assert_eq!((l.total, l.serial, l.max_in_flight), (2, 1, 2));
    // The same, but the first value is used before the second load: 1 in flight.
    let text =
        "a = ld.global.b32 [r0] // wr:0\nx = f2f a // wt=000001\nb = ld.global.b32 [r2] // wr:1\n";
    assert_eq!(analyse(text).max_in_flight, 1);
}
