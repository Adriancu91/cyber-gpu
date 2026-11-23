# Pentru sesiunea OS — arena antrenării în VRAM: „planul rezident” (nota centrului de comandă 23 nov 23:36, pasul 2)

**De la:** Cyber-GPU, sesiunea 194 (24 nov 2026), commit-ul cu T202 pe `g0-g2-fundatie`. **Stare:** SIMULARE, gata de legat.
Nimic rulat pe placă. Planul și predicțiile: `docs/C-160_P8_ARENA_VRAM.md` (urcate înainte de cod, `c9733d6`).

## De ce

În `train_mini_probe` (`gpu_i14.rs`) faceți la **fiecare** pas un `Plan::new` (segmentele lui `emb_back` depind de lot și mută
adresele de după ele), urcați tot (`uploads` + m, v ≈ 27,6 MB) și citiți greutățile, m și v (13,8 MB). Cu arena în RAM asta e
partea mare din cele 286 ms ale pasului întreg din os-380; cu arena în VRAM ar însemna două copii mari prin placă pe pas.

## Ce oferă biblioteca (doar adăugări)

```rust
let plan0 = Plan::new(a, bsz, tl, &x0, arena_va)?.resident()?;      // merge pe orice fel de plan (tiles3, lipiri, _mv_ram)
// o dată: alocați plan0.bytes() în VRAM (map_vram), urcați plan0.uploads(&params, &x0, &y0) (prin copie, ca greutățile lui #513)
for (k, (x, y)) in loturi.enumerate() {
    let plan = plan0.rebatch(x)?;                                   // ACELEAȘI adrese, doar segmentele noi
    if k > 0 { for (va, b) in plan.batch_uploads(x, y)? { /* ~1,3 KiB: tok, tgt, 3 liste de segmente */ } }
    let list = plan.step(Recipe::AdamW513, &opt, k as u32 + 1)?;    // ca azi; m, v rămân pe placă
    // după placă: plan.step_reads() = [(loss_rows, 256 B), (res, 4 B)] — doar atât
}
// la sfârșit (sau la o salvare): plan0.w / plan0.m / plan0.v (n_params f32 fiecare), prin copie în jos
```

- `Plan::is_resident()`, `Plan::traffic()` (octeții pe legătură: o dată / pe pas / la sfârșit, și cât face drumul de azi).
- `rebatch` refuză un plan nerezident, alt fel de lot sau un simbol în afara vocabularului; `batch_uploads` refuză alt lot decât
  cel al planului (deci nu se pot amesteca segmentele).
- Oracolul la fiecare pas: are nevoie de greutăți. Pentru proba de corectitudine puteți compara pierderea și norma la fiecare pas
  (aceiași biți în SIMULARE) și greutățile o dată, la sfârșit; sau citiți-le la fiecare pas prin copie (4,6 MB, ~0,8 ms la
  6 GB/s), dar atunci timpul „pasul întreg” nu mai e cel de antrenare.

## Ce a ieșit în SIMULARE (T202, `crates/nvidia-ga104/tests/sim_c160_p8_resident.rs`)

Forma lui `antrenare mini` (16 384, 2 straturi, E 64, lot 2 × 32, AdamW lr 1e-3), 10 pași pe 10 loturi diferite (unul cu doar
8 simboluri distincte):
- adresele identice pentru toate loturile și pentru lotul cel mai rău (64 de simboluri distincte); arena 26,4 MiB, +0 octeți
  față de cea mai mare arenă pe lot;
- drumul rezident = drumul vostru de azi **bit cu bit** (pierderi, norme, greutăți); față de oracol: pierderea 9,8e-8,
  greutățile 3,1e-6 (os-380 pe placă: 9,9e-8 / 8,8e-6);
- octeți după pornire, în 10 pași: rezident 11 092 sus / 2 600 jos; drumul de azi 276 MB sus / 138 MB jos;
- fără `batch_uploads` (lotul vechi în memorie): 9 din 9 pierderi diferite — se prinde.

## Ce e nou pentru placă

Niciun program nou, nicio scriere în registre. Nou: arena (26,4 MiB) în VRAM în loc de RAM și copiile la pornire / la sfârșit
(drumul lui `vram_copy`, deja trecut la greutățile lui #513). **Predicția (DEDUS, `docs/C-160_P8_ARENA_VRAM.md` R4):** placa
1–5 ms pe pas în loc de 25,6 ms (≈ 13 000–64 000 simboluri/s la lot 2 × 32), pasul întreg câțiva ms peste placă în loc de 286 ms.
Dacă iese peste 5 ms: întâi lansările capului (16 384 × 64 rânduri), apoi așteptările driverului.
