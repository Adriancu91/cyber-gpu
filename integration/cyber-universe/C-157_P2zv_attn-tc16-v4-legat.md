# C-157 P2zv — `attn_back_tile_tc16_v4` / `_v4pf` legate în pasul mixt rapid: varianta rapidă aleasă singură pe formele aliniate (sesiunea Cyber-GPU 253, 26 nov 2026)

**PREDARE.** Cod de legătură, **nerulat pe placă**, testat **doar în SIMULARE** (T263). Nota întreagă:
`cyber-gpu/docs/C-157_ATTN_BACK_TC16_V4_LEGAT.md` (planul și predicțiile W1–W7 urcate înainte de cod, `a922bab`).

## Ce e

Pasul mixt rapid de azi (`--atentie-tc16`, P6-231) pune programul al doilea al atenției înapoi (dQ, dK, dV) pe
`attn_back_tile_tc16` (`AttnTcProgram::Tc16`). Proba P2zq (P6-233, `--atentie-tc16-v4`, os-465) va spune care dintre
`attn_back_tile_tc16_v4` și `_v4pf` e mai rapidă. Ca să nu mai așteptați o sesiune Cyber-GPU după P2zq, legătura e gata acum:
`nvidia_ga104::train_step::pack_mixed_fast_attn4(…, attn, attn_fast, attn0, …)` cu
`attn_fast = Some(AttnTcFast::V4(adresa))` sau `Some(AttnTcFast::V4pf(adresa))` — **doar câștigătoarea**. Pe fiecare lansare
`AttnTc` alege varianta rapidă dacă forma e aliniată (`attn_back_tile_tc16_v4::fits_cbuf`), altfel `attn` (tc16) ca până acum.
QMD-ul rapid diferă de cel tc16 doar prin adresa programului și numărul de registre (72 la v4, 104 la v4pf); memoria partajată
(16 KiB, o barieră), grila, blocul și cbuf-ul sunt aceleași. Alegerea iese ca `TcPick::AttnTc16V4` / `TcPick::AttnTc16V4pf`.

## Ce e nou la noi (doar adăugări)

- `train_step::{AttnTcFast, TcPick::AttnTc16V4, TcPick::AttnTc16V4pf, pack_mixed_fast_attn4}`;
  `pack_mixed_fast_attn3(…)` = `pack_mixed_fast_attn4(…, None, …)`, aceiași octeți (verificat pe mini și #513, W1).
- `batch_ram::pack_mixed_fast_via_ram_attn4` (pentru `--lot-prin-ram`), la fel.
- `TcPick` are două variante noi: strică doar un `match` complet; la voi `fast_picks_text` folosește o listă explicită —
  adăugați `TcPick::AttnTc16V4` / `AttnTc16V4pf` acolo ca să apară în rândul de alegeri. `StepProfile::from_mixed` le numește singur.

## Ce ar face OS-ul (propunere; decide sesiunea OS și centrul de comandă)

**Doar după ce P2zq trece pe placă** (aceleași cuvinte ca tc16 și cel puțin ×1,15 la `atentie-513-b4` pentru câștigătoare):
încărcați câștigătoarea lângă `attn_back_tile_tc16` și, în locul de la `gpu_i14.rs` unde azi chemați
`pack_mixed_fast_attn(…, at, …)` / `pack_mixed_fast_via_ram_attn(…)`, chemați `pack_mixed_fast_attn4(…, at, Some(AttnTcFast::…), Attn0Programs { tc: 0, v4: 0 }, …)`
(sau cu adresele dS, dacă P2zs / P2zt au trecut între timp). Rândul propus: `kt pas-mixt attn: v4 N / tc16 M`.
Pe #513 aștept 12 rapide / 0 tc16.
**Aștept (scris înainte):** pierderile și gradientul **identice bit cu bit** cu pasul pe tc16 (aceeași aritmetică); timpul
programului cât a arătat P2zq (W7, DEDUS). Dacă P2zq nu trece, nu schimbați nimic.

**Nou pentru placă:** nimic în plus față de P2zq (aceleași programe: v4 SHA-256 `b19c8451…5a86`, v4pf `3ba898a9…f853`);
niciun registru GPU nou, nicio comandă nouă. Cod: `crates/nvidia-ga104/src/train_step.rs`, `crates/nvidia-ga104/src/batch_ram.rs`;
test `crates/nvidia-ga104/tests/sim_c157_attn_tc16_v4_mixed.rs` (T263, SIMULARE). Nicio schimbare incompatibilă pentru OS.
