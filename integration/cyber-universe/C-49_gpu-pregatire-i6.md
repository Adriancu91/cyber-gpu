# Propunere pentru sesiunea OS: comanda `gpu pregatire-i6` (doar citire) — C-49

*1 nov 2026, sesiunea Cyber-GPU 30. Doar o propunere: OS-ul decide, cu „da”-ul lui Adrian. Nimic de aici nu e aplicat în `cyber-universe`.
Planul ferestrei: `docs/FEREASTRA_C-49.md`.*

## Ce face
Citește de pe placă, **fără nicio scriere**, tot ce îi trebuie pasului I6 (FWSEC-FRTS, prima scriere a driverului nostru, într-o fereastră
ulterioară): siguranța „ucode version” a FWSEC și semnătura aleasă din VBIOS, mărimea VRAM, dacă WPR2 e deja ridicat, regiunea FRTS.
Se poate rula în fereastra următoare, înaintea probei tinygrad (după tinygrad WPR2 ar fi ridicat până la oprirea completă).

## Ce cere de la OS
- **Fereastra BAR0 de 16 MiB** (azi `ReadOnlyBar0Window::len()` = 4 MiB): siguranțele sunt la `0x824xxx`. Aceeași fereastră doar-citire,
  pagină cu pagină, fără `WRITABLE`; doar lungimea `nvidia_ga104::fuse::FUSE_WINDOW` (16 MiB) în loc de `regs::INIT_WINDOW`.
  BAR0 al plăcii are 16 MiB (K0, 23 oct), deci nicio citire nu iese din BAR.
- Fixarea `cyber-gpu` pe `d6de57e` sau mai nou. Doar adăugări în interfața publică (`fuse`, `i6_check`).

## Schiță (pentru `kernel/src/drivers/gpu.rs`, lângă `InitStep::Fwsec`)
```rust
InitStep::I6Check => {
    // 1. FWSEC location on the 4 MiB window (as `gpu fwsec`).
    let loc = locate_fwsec(&mut win4, &mut journal, &mut clock);
    // 2. Fuse, VRAM size, WPR2 on a 16 MiB read-only window (same type, len = FUSE_WINDOW).
    let fw = &nvidia_ga104::firmware::GSP_570_144;
    let size = |r| fw.iter().find(|p| p.role == r).map(|p| p.size).unwrap_or(0);
    use nvidia_ga104::firmware::GspRole::{Bootloader, GspRm};
    let r = loc.and_then(|l| nvidia_ga104::i6_check::i6_check(
        &mut win16, &mut journal, &l.desc, size(Bootloader), size(GspRm)));
    match r {
        Ok(c) => {
            lines.push(format!("{p}: siguranta FWSEC {:#x}: versiunea {} -> semnatura {} din VBIOS",
                c.signature.fuse_offset, c.signature.fuse_version, c.signature.index));
            lines.push(format!("{p}: VRAM {} MiB; FRTS {:#x}..{:#x}", c.scratch42, c.frts.0, c.frts.1));
            lines.push(match c.wpr2 {
                None => format!("{p}: WPR2 coborat (pornire rece): bun"),
                Some((s, e)) => format!("{p}: WPR2 DEJA RIDICAT {s:#x}..{e:#x}: trebuie oprire completa"),
            });
            lines.push(format!("{p}: criteriul I6-citire: {}", if c.passes() { "TRECE" } else { "NU TRECE" }));
        }
        Err(e) => lines.push(format!("{p}: I6-citire: {e}")),
    }
    // + "0 scrieri" from the window's `refused` counter, as for the other steps.
}
```

## Ce trebuie să apară pe placa reală
4 citiri în jurnal (`0x8241e0`, `0x1183a4`, `0x1fa828`, plus tabelele VBIOS ca la `gpu fwsec`), VRAM `8192 MiB`, semnătura `0`, `1` sau `2`,
`WPR2 coborat`, `criteriul I6-citire: TRECE`, „0 scrieri”. Un `0xbadf…` la siguranță = eroare de acces (oprire, poză), nu o valoare.
