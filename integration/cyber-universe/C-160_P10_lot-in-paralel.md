# Pentru sesiunea OS — lotul următor pregătit în paralel cu pasul (ordinea 20:11, pasul 5)

**De la:** Cyber-GPU, sesiunea 196 (24 nov 2026), commit `d52f264` pe `g0-g2-fundatie`. **Stare:** SIMULARE, gata de legat,
oprit implicit. Nimic rulat pe placă. Planul și predicțiile: `docs/C-160_P10_LOT_IN_PARALEL.md` (urcate înainte de cod, `044bb0c`).
Se sprijină pe arena rezidentă (`C-160_P8_arena-rezidenta.md`); are sens după ce arena în VRAM merge pe placă.

## Ce oferă biblioteca (doar adăugări)

```rust
let base = Plan::new(a, bsz, tl, &x0, arena_va)?.resident()?   // (+ lipiri / with_recompute, ca înainte)
    .with_batch_slots()?;                                       // NOU: al doilea loc de lot, la capătul arenei
// o dată: base.slot(0)?.rebatch(&x0)?.uploads(..) ca azi, plus base.slot(1)?.slot_uploads()
//         (pasul mixt: și base.slot(1)?.mixed_uploads(S))
let plan_k = base.slot(k % 2)?.rebatch(&x_k)?;                  // pasul k folosește locul k % 2
plan_k.batch_uploads(&x_k, &y_k)?;  plan_k.step(..)?;  plan_k.step_reads();   // ca la arena rezidentă
```

- `BatchVas` + `Plan::batch_area()` = locul lotului (intrări, ținte, listele de segmente ale simbolurilor, pierderile pe rând,
  blocul `res` de 64 B). `Plan::batch_slots()` = 1 sau 2. `slot(k)` refuză `k ≥ 2` și planurile cu un singur loc.
- Locul 0 = **exact** adresele planului rezident; locul 1 stă după vechiul capăt: arena crește cu **3 840 B** la lot 2 × 32.
  Restul (greutăți, gradient, m, v, activări) e comun. `with_recompute` păstrează cele două locuri.
- Lansările celor două locuri sunt aceleași; diferă doar adresele lotului din argumente.

## Ordinea pe placă (propunere)
Scrie lotul 1 în locul 0 și lotul 2 în locul 1; trimite pasul 1 și pasul 2 (unul după altul, în aceeași coadă). Apoi, la
fiecare pas N terminat (semaforul lui): citește pierderile + norma din locul lui N, scrie lotul N+2 în **același** loc, trimite
pasul N+2. Placa are mereu un pas în coadă; procesorul atinge doar locul pe care placa nu-l folosește.

## Ce a ieșit în SIMULARE (T204)
Forma lui `antrenare mini`, 10 pași pe 10 loturi diferite, AdamW F32. Drumul „în paralel” (lotul următor scris **la mijlocul**
pasului curent, pierderile citite **după** ce pasul următor a pornit): **aceiași biți** cu drumul pas cu pas (pierderi, norme,
greutăți), oracolul 9,9e-8 / 1,5e-5; la fel cu ambele lipiri + recalcularea. Cu un singur loc, aceeași ordine dă **10/10
pierderi** și ~1,15 milioane de greutăți greșite — deci fără al doilea loc ordinea „în paralel” nu e voie.

## Ce e nou pentru placă
**Nimic** în registre, niciun program, nicio lansare nouă: doar 3 840 B în plus în arenă și altă ordine a scrierilor/citirilor
procesorului. Câștigul estimat (DEDUS): timpul în care placa stă azi între pași (~50–300 µs), adică ~2–15 % din pas când arena
e în VRAM. AdamW cu m, v în RAM (`_mv_ram`) rămâne doar în simulare (condiția lui Adrian, 23 nov 20:36).
