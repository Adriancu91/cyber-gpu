# C-157 p.2y — `colsum` împărțit pe rânduri în pasul #513 (`Plan::with_colsum_split`) — gata de legat în SIMULARE

Pentru sesiunea OS. De la Cyber-GPU, sesiunea 210 (24 nov 2026). Planul și predicțiile: `cyber-gpu/docs/C-157_P2y_COLSUM_IMPARTIT.md`
(scrise înainte de cod, `885a44b`); comparația os-407: `cyber-gpu/docs/C-157_P2_COMPARATII.md`.

## De ce
os-407: pasul #513 778 ms, `colsum` **24,3 ms** (73 de lansări × ~333 µs: un fir pe coloană adună toate cele 1 024 de rânduri).
După `--mm-placi` (P6-188) pasul ar fi ~175–295 ms (DEDUS), deci `colsum` ar ajunge la 8–14 % din pas.

## Cum se leagă
- Plan: `Plan::new_tiles3(...)?.with_mm_tile().with_colsum_split().resident()?` (sau `.with_colsum_split()` pe orice plan;
  se păstrează prin `resident()`, `with_recompute()`, `with_batch_slots()`). Se combină cu toate celelalte opțiuni.
- **Arena:** `plan.bytes()` cuprinde acum și **zona de lucru** de după `plan.end` (`plan.colsum_scratch()` = (p0, p1, cuvinte);
  la #513 2 × 512: **768 KiB** în plus). Dacă arena e alocată din `plan.bytes()` (cum face `gpu_i14.rs`), nu e nimic de schimbat.
  Zona nu se urcă și nu se citește: placa o scrie și o citește singură în timpul pasului.
- **Lista:** 385 → **483** de lansări la #513 (`launches_per_step` le numără): fiecare `colsum` devine `colsum_part` (+ `colsum` peste
  sumele parțiale; + încă un `colsum` la LayerNorm), cu același tag și strat; restul listei identic, în aceeași ordine. Cu „cel mult
  255 pe trimitere” rămân **2 trimiteri** (483 ≤ 510). Aplicată și în `step_until_opt` și `step_mixed`.
- **Programele de încărcat:** pe lângă `programs()`, `fused_programs()`, `mm_tile_programs()` și **`colsum_split_programs()`**
  (un program: `colsum_part`, SHA-256 `740f5d439b357d37c06685384cdfd4fb7ba5cbda84c3023f49a77565d59725ac`, 20 de registre,
  bloc 256 × 1 × 1, fără memorie partajată). `Prog::index()` merge acum pe `PROGRAMS_ALL_SPLIT` (25); indicii vechi neschimbați.
  `Prog` are o variantă nouă (`ColsumPart`): un `match` complet pe `Prog` în OS cere un braț în plus.
- **Placa simulată a OS-ului:** înregistrează `gpu_sim::cmdproc::colsum_part_on_cpu` (fără `Needs`).
- **Profilul:** `colsum_part` apare ca program separat; `colsum` are 171 de lansări în loc de 73 (cele noi sunt mici: 32 de rânduri).

## Ce se vede (SIMULARE, T216)
- **Nu aceiași biți** (altă ordine a sumelor), dar: pasul mic (10 pași, sgd + adamw_513, pe `new`, `_3` + plăci, `_3` + lipiri + plăci)
  = PyTorch / oracolul, **pierderile identice** cu lista fără opțiune, gradientul de 2–3 ori **mai aproape** de oracolul f64
  (1,2e-7 față de 2,4–3,6e-7). Un strat la lățimile #513: aceleași cifre.
- Deci criteriul pasului #513 (`ORACLE_513_SYNTH`, pașii 1–9 la 1e-3) rămâne același; amprenta biților se schimbă.

## Ce e nou pentru placă
Programul **`colsum_part`** (neprobat pe placă; aceleași operații ca `colsum`, grila ⌈c/256⌉ × 32) și zona de lucru de 768 KiB în
arenă. Niciun registru nou.

## Ce așteptăm (DEDUS, NETESTAT)
`colsum` **24,3 → ~2,5–5 ms** pe pas, pasul **−19–22 ms** (azi 778 → ~757 ms; după `--mm-placi` câștigul e 7–12 % din pas).

## Propunere
Opțiune separată (de ex. `--colsum-impartit`) după `--mm-placi`, ca să se vadă fiecare câștig singur; la o diferență față de oracol,
întoarcere la lista fără opțiune.

## Altceva mic (fără placă): ceasul procesorului
În raportul os-407, „ceasurile: placa / pasul întreg = 1 022 ‰”: TSC-ul e socotit cu frecvența măsurată la pornire în 10 ms
(3 106 925 kHz), dar mediana 5 × 50 ms dă 3 002 035 kHz (+34,9 ‰). Toți timpii pe ceasul procesorului ies cu ~3,4 % prea mici
(pasul întreg 760,8 → ~787 ms; „așteptarea” mini 3 519 → ~3 642 µs ≥ placa 3 620). Propunere: timpii socotiți cu frecvența „acum”
sau calibrarea de la pornire pe 50 ms.
