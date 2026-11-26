//! C-157 (os-467, `docs/C-157_OS467_ATTN0_TC_BLOCAJ.md`): the two board-free checks of `nvidia_ga104::program_lint`
//! over every GPU program we have (`tools/nak/*.comp`, the NAK listings in `tools/nak/out*/`). Not SIMULARE: a check of
//! our sources and of NAK's printed listing; nothing here runs a program.

use nvidia_ga104::program_lint::{barriers_outside_uniform, coopmat_shared_strides};
use std::path::PathBuf;

fn nak_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/nak")
}

fn files(dir: &PathBuf, ext: &str) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(ext))
        .map(|p| {
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            (n, std::fs::read_to_string(&p).unwrap())
        })
        .collect();
    v.sort();
    v
}

/// Z1: the stride rule over every program source; the list of programs that fail it.
#[test]
fn z1_coopmat_strides_over_every_program() {
    let srcs = files(&nak_dir(), ".comp");
    assert!(srcs.len() > 60, "{} surse", srcs.len());
    let mut with_coopmat = 0;
    let mut bad = Vec::new();
    for (n, s) in &srcs {
        if s.contains("coopMatStore") || s.contains("coopMatLoad") {
            with_coopmat += 1;
        }
        let f = coopmat_shared_strides(s);
        if !f.is_empty() {
            println!("{n}: {f:?}");
            bad.push(n.clone());
        }
    }
    println!(
        "{} surse, {} cu matrice cooperative, refuzate: {:?}",
        srcs.len(),
        with_coopmat,
        bad
    );
    assert!(with_coopmat >= 15);
    assert!(
        bad.is_empty(),
        "programe cu pasul matricelor cooperative nealiniat: {bad:?}"
    );
}

/// Z2: every `bar.sync` of every listing of the second NAK tool (Mesa 26.3.1, `out2/`) sits in a uniform block. The
/// first tool (Mesa 25.0.7, `out/`) marks fewer blocks uniform: 11 of its listings keep barriers in blocks it did not mark,
/// all of them programs that ran on the board (os-3xx–4xx) — there the check proves nothing either way, so it is only
/// counted, not judged.
#[test]
fn z2_barriers_only_in_uniform_blocks() {
    let mut checked = 0;
    for (n, l) in files(&nak_dir().join("out2"), ".sm86.txt") {
        if !l.contains("bar.sync") {
            continue;
        }
        checked += 1;
        let bad = barriers_outside_uniform(&l);
        assert!(
            bad.is_empty(),
            "out2/{n}: bar.sync in afara unui bloc uniform la liniile {bad:?}"
        );
    }
    let mut old_unproven = Vec::new();
    let mut old_checked = 0;
    for (n, l) in files(&nak_dir().join("out"), ".sm86.txt") {
        if l.contains("bar.sync") {
            old_checked += 1;
            if !barriers_outside_uniform(&l).is_empty() {
                old_unproven.push(n);
            }
        }
    }
    println!(
        "out2: {checked} listari cu bariere, toate in blocuri uniforme; out: {old_checked}, nedovedite {old_unproven:?}"
    );
    assert!(checked >= 13, "{checked}");
    assert_eq!(old_unproven.len(), 11, "{old_unproven:?}");
}

/// Z2, planted: the os-467 program's listing with one `block.u` header made non-uniform before a barrier is caught.
#[test]
fn z2_planted_non_uniform_barrier_is_caught() {
    let l = std::fs::read_to_string(nak_dir().join("out2/attn_back_tile0_tc.sm86.txt")).unwrap();
    assert!(barriers_outside_uniform(&l).is_empty());
    let planted = l.replacen("block.u 4 L6 [3] -> {", "block 4 L6 [3] -> {", 1);
    assert_ne!(planted, l);
    assert_eq!(barriers_outside_uniform(&planted).len(), 1);
}

/// Z1, planted: the os-467 stride (rows of 65 floats) put back into today's source is refused at the store.
#[test]
fn z1_planted_stride_65_is_caught() {
    let s = std::fs::read_to_string(nak_dir().join("attn_back_tile0_tc.comp")).unwrap();
    assert!(coopmat_shared_strides(&s).is_empty());
    let planted = s.replace("ROW = 68u", "ROW = 65u");
    assert_ne!(planted, s);
    let f = coopmat_shared_strides(&planted);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(
        f.iter()
            .all(|f| f.call == "coopMatStore" && f.array == "cs" && f.stride_bytes == Some(260))
    );
}
