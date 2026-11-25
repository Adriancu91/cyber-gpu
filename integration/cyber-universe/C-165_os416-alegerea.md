# C-165 după os-416: alegerea cursei = `mm_tv_128x128` pe toate 15 formele (sesiunea Cyber-GPU 219, 25 nov 2026)

**Pentru sesiunea OS. Nimic nou pentru placă în afară de ce a trecut deja în cursă.**

## Ce am verificat (pe procesor, T225, `crates/nvidia-ga104/tests/c165_os416.rs`)
- Raportul vostru os-416 (rândurile cursei, copiate în `crates/nvidia-ga104/tests/rapoarte/os416-cursa.txt`) citit cu `parse_race`:
  60 de rânduri, toate „=”, 15 forme / 147 de lansări = exact formele pasului #513.
- `MmChoice::from_race` → **toate 15 formele pe `mm_tv_128x128`** (câștigul cel mai mic 11,2 %, pe 768×768×1024).
- Economia din raport: noi 141 388 µs, voi 141 387 µs.
- `Plan::with_mm_choice(alegerea)` mută exact cele 147 de înmulțiri pe `mm_tv_128x128`; aceeași arenă, același număr de lansări.

## Cum o legați (două căi, același rezultat)
1. Fără raport la pornire (nou, doar adăugări): `MmChoice::all(&mm_train_variants::forms(&plan.step(..)?), RACE_WINNER_513)`
   (`RACE_WINNER_513 = 3` = `mm_tv_128x128`), apoi `Plan::new_tiles3(..).with_mm_choice(alegerea)` + `.with_colsum_split()` cum aveți
   lista #513. Programe în plus de încărcat: `plan.mm_choice_programs()` = `[mm_tv_128x128]` (SHA-256 `5d3affd4…9e23ebc`, 90 de registre).
2. Din raport: `parse_race(text).choice()`.

## Ce prezicem pe placă (DEDUS, de verificat)
- Biții: aceiași ca oracolul (în os-416 toate variantele au dat biții martorului).
- Pasul #513 întreg: **~341 ms ≈ 3 000 simb/s** (interval 325–345 ms). Voi ați socotit ~333 ms din timpii izolați; noi aplicăm raportul
  128×128 / martor (0,690) pe înmulțirile **din pas** (430,5 ms, profilul os-412), fiindcă cursa a mers la 1 785 MHz, iar pașii la 1 950.
- Dacă pasul iese peste 345 ms: profilul pe programe (`--profil`) ne spune dacă `mm_tv_128x128` e mai lentă în pas decât în cursă.

## După
`mm_tc` (os-352, os-384) — tot nerulate; apoi pasul mixt pe #513 cu `LOSS_SCALE_513`.
