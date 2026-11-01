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

---

# C-51 (1 nov): partea a doua — jurnalul pe loc și `gpu i6 --scrie` (pentru fereastra cu prima scriere)

*Pe placa reală rulează doar Cyber Universe (decizia din 1 nov 12:00); proba tinygrad e anulată. Fixarea: `cyber-gpu` ≥ `22436f2`.*

## A. Jurnalul fiecărui acces, cu ora, pe loc (pentru toate comenzile `gpu`)
`gpu_core::diag::Journal::with_sink(cap, sink)` cheamă `sink(&event)` la fiecare citire/scriere, în clipa în care se face. OS-ul pune ora
și scrie rândul în log și în `raport/*.txt` de pe stick (de preferat golit pe stick înainte de pașii care scriu în placă):
```rust
fn gpu_sink(e: &gpu_core::diag::Event) { crate::info!("gpu: {}", e); }   // the log line already carries the timestamp
let mut journal = gpu_core::diag::Journal::with_sink(4096, gpu_sink);
```
Formatul rândului e cel din `Event` (`bar0 w 0x110624: 0x00000000 <- 0x00000080 => 0x00000080`); după fereastră îl citim cu
`nvidia_ga104::i6_log::{parse_line, check_i6}`.

## B. `gpu i6 --scrie` (doar în stick-ul de test, cu `hw-write`; cere `DA` la tastatură)
Ordinea (planul `docs/FEREASTRA_C-49.md`, pașii 1–5):
1. `i6_check` (ca la `gpu pregatire-i6`); dacă `!passes()` → stop.
2. `fwsec::read_ucode_and_signature(win4, …, &loc, check.signature.index)` → `gsp::hs_load::fwsec_frts_image(&loc.desc, &ucode, &sig,
   check.frts.0)` → imaginea copiată de OS într-o **zonă de RAM contiguă sub 4 GiB** (~60 KiB, aliniată la 256 B), adresa fizică → `DmaImage`.
3. `gpu_pci::bus_master::set_bus_master(cfg, &BusMasterPermit::new("C-51 I6"), &mut journal, true)` → păstrează valoarea veche.
4. `gsp::falcon::fwsec_frts_dma(&mut GuardedBar0::new(win16_rw, &mut journal, Some(permit))?, clock, &HsUcode::from_fwsec_v3(&loc.desc)?,
   &image, check.frts)` — **singura** fereastră BAR0 cu drept de scriere, tot de 16 MiB; `GuardedBar0` refuză orice în afara GSP/SEC2.
5. `bus_master::restore_command(cfg, &permit, &mut journal, old)` — **și la eroare** în pasul 4.
6. Rândurile: `pas I6 FWSEC-FRTS: ok`, WPR2 `start..end`, numărul de scrieri (`journal.writes()`).

Cod nou în OS: zona de RAM pentru DMA (cu adresa fizică), fereastra BAR0 cu scriere doar pentru această comandă, confirmarea `DA`.
Propunerea în hub: `idei/2026-11-01_cyber-gpu-i6-in-os.md`.
