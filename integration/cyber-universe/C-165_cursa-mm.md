# C-165 — cursa de noapte: variantele lui `mm_train_tile` (nota pentru sesiunea OS)

**Cyber-GPU, sesiunea 211, 24 nov 2026, în cloud, FĂRĂ placă.** Totul de aici e verificat **doar în SIMULARE** (T217).
Nota completă: `docs/C-165_CURSA_MM.md`.

## Ce e gata (doar adăugări în interfața publică)
- `nvidia_ga104::mm_train_variants::VARIANTS` — 5 programe, același bufer constant ca `mm_train` (`mm_train::Args::cbuf()`),
  **aceiași biți** ca `mm_train` / oracolul în SIMULARE. Indicele 0 = `mm_train_tile` (martorul); variante noi doar la coadă.
  Pentru fiecare: `name`, `code`, `sha256`, `registers`, `block()` (256 × 1 × 1 la toate), `shared()` (pentru
  `ComputeQmd::to_bytes_with_shared`), `grid(&args)`, `Variant::flops(&args)` (pentru TFLOP/s).
- `nvidia_ga104::mm_train_variants::forms(&launches)` — formele distincte ale lansărilor `mm_train` / `mm_train_tile` dintr-un pas,
  cu numărul de lansări pe formă. Pe pasul #513 (`Plan::new_tiles3`, rezident, 2 × 512): **147 de lansări** (lista exactă:
  rândurile `kt t217 forma #513` din `cargo test -p nvidia-ga104 --test sim_mm_train_variants -- --nocapture`).

## Rândul de test propus (îl face sesiunea OS, nu noi): `test-placa --cursa <listă>`
Condiția din decizie: prima cursă abia după ce `mm_train_tile` a trecut o dată singur pe placă (os-410, `--mm-placi`).
Pentru fiecare formă din `forms(...)` și fiecare variantă din listă:
1. SHA-256 al codului = `sha256` din tabel (altfel: varianta sărită, scris în raport).
2. Aceleași date de intrare (aceeași sămânță) pentru toate variantele unei forme; C umplut întâi cu o valoare-santinelă.
3. O lansare de încălzire, apoi ~5–11 lansări cronometrate → **timpul median**.
4. **Verdictul biților:** C-ul fiecărei variante comparat **bit cu bit** cu C-ul martorului (`mm_train_tile`, indicele 0) pe aceeași
   placă. Orice diferență = varianta respinsă pentru acea formă (nu se folosește niciodată în pas).
5. Tabelul din raport: `variantă · formă (M×N×K, opțiuni) · µs median · TFLOP/s · biți (= / ≠)`; la final, pe formă, cea mai
   rapidă variantă cu biți „=”. Plus puterea / starea P (A4) și temperatura.

**Nimic nou pentru placă în afară de programe:** aceeași cale de lansare (QMD + pushbuffer) ca la `mm_train_tile`; niciun
registru, nicio comandă nouă. Grila **trebuie** luată din varianta lansată (`v.grid(&args)`): o grilă a altei variante poate lăsa o
parte din C nescrisă (T217 arată că se prinde prin comparație, dar n-are ce căuta acolo).

## Ce urmează la noi
`Plan::with_mm_choice(...)`: pentru fiecare formă, varianta câștigătoare din raportul cursei (aceiași biți → aceeași amprentă a
pasului), apoi variante pentru `mm_train_gelu` și pentru atenție.

## După os-410 (adăugat la sfârșitul sesiunii 211)
Condiția „`mm_train_tile` a trecut o dată singur pe placă” e îndeplinită (os-410). **Rugăminte:** în aceeași pornire cu cursa, A4 —
starea P, ceasul SM și puterea citite în timpul cursei / pasului. Comparația os-410 (`docs/C-157_P2_COMPARATII.md`) arată ambele
programe la ~1/7 din plafonul lor, la fel pe toate formele: bănuiala nr. 1 e ceasul, nu programul.

## Pasul 2 — `Plan::with_mm_choice` (sesiunea 213, 24 nov 20:15–20:40; SIMULARE) — PREDARE
Totul e pregătit ca, **după raportul cursei**, câștigătorii să intre în pasul #513 fără cod nou la noi. Doar adăugări:
- `mm_train_variants::RaceRow { m, n, k, flags, variant, median_ns, same_bits }` — un rând al tabelului cursei.
- `mm_train_variants::MmChoice::from_race(&rows)` — pe fiecare formă alege cea mai rapidă variantă cu biți „=” față de martor,
  **doar dacă e cu cel puțin 3 % mai rapidă** decât martorul (`MIN_GAIN_PERMILLE = 30`; altfel rămâne martorul, ca zgomotul să nu
  schimbe alegerea). O variantă cu alți biți nu e aleasă niciodată; fără rândul martorului pe acea formă, nicio alegere.
  Se poate face și de mână: `MmChoice::new()` + `set(m, n, k, flags, indice)`.
- `Plan::with_mm_choice(alegerea)` = `with_mm_tile()` + alegerea: lansările formelor alese merg pe varianta lor, cu grila ei;
  restul pe `mm_train_tile`. Aceeași arenă, aceleași 385 de lansări, aceleași cbuf-uri; alegerea goală = `with_mm_tile` exact.
  Trece prin `resident()`, `with_recompute()`, `with_colsum_split()`, `step_until_opt`.
- Programele de încărcat: în plus față de azi, `plan.mm_choice_programs()` (doar variantele alese, o dată fiecare):
  `programs() + fused_programs() + mm_tile_programs() + mm_choice_programs() + colsum_split_programs()`.
- `Prog` are 4 valori noi (`MmTv128x64`, `MmTv64x128`, `MmTv128x128`, `MmTv64x64K8`) cu indicii 25–28
  (`PROGRAMS_ALL_CHOICE`, 29 de programe); **indicii vechi 0–24 nu se schimbă**; `check_programs()` verifică acum 29 de SHA-256.
  `Prog::race_index()` / `Prog::of_race_index(i)` leagă `Prog` de `VARIANTS[i]`. Dacă OS-ul are un `match` pe `Prog` fără `_`,
  are nevoie de un braț în plus (în `antrenare_sim.rs` există deja `_ =>`).

**Teste T219 (SIMULARE, pe procesor):** pasul #513 cu o alegere dintr-un raport inventat = lista `with_mm_tile` cu doar formele
alese schimbate (385 lansări, 147 pe variante, blocuri 64 960 → 34 016); pasul de aur mic cu **toate cele 4 variante în același pas**
= `with_mm_tile` **bit cu bit** în 10 pași (sgd și adamw_513, pe `_3` și `_3` + lipiri), verdictul față de PyTorch TRECE; regulile
alegerii (biți diferiți, sub 3 %, fără martor, egalitate → indicele mic, tabel plin).

**Propunere pentru OS (după cursă, decide sesiunea OS):** opțiunea `--mm-alegere` după `--mm-placi`, cu tabelul cursei (din raport
sau din fișierul cursei de pe stick) → `MmChoice::from_race` → `with_mm_choice`; în raport, rândul alegerii (formă → variantă).
Nou pentru placă: doar programele variantelor deja rulate în cursă; niciun registru, nicio comandă nouă.

## Pasul 3 — cititorul raportului: `parse_race` (sesiunea 215, 24 nov 22:15–23:20; pe procesor) — PREDARE
Ca alegerea să iasă **direct din textul raportului cursei** (fără rânduri copiate de mână). Doar adăugări în `mm_train_variants`:
- `parse_race(text) -> ParsedRace` citește rândurile voastre (P6-190, `race_row` din `gpu_i14.rs`, formatul de azi, neschimbat):
  `…: cursa: <varianta> | MxNxK f<opțiuni> (x<N> in pas) | <µs> us median | … TFLOP/s | biti <=|DIFERA|…>`.
  Se păstrează doar **ultima** cursă din text (antetul `cursa mm (C-165, …)` o ia de la capăt). Rândurile „- us”, numele necunoscute și
  rândurile stricate nu devin `RaceRow` (numărate în `untimed` / `unknown`); nimic nu se blochează.
- `ParsedRace { rows, counts, untimed, unknown }`; `ParsedRace::choice()` = `MmChoice::from_race(&rows)`;
  `ParsedRace::saved_ns(&alegere)` = cât ar câștiga pasul (Σ (martor − ales) × lansări; aceeași sumă DEDUSĂ ca rândul vostru „pasul ar scadea cu”).
- Timpul e cel din raport, rotunjit la 0,1 µs; alegerea iese la fel ca din timpii întregi (T221).

**Cum se poate lega (decide sesiunea OS):** după `--cursa-mm`, aceeași pornire poate face `parse_race(&rândurile_raportului)` →
`choice()` → `Plan::with_mm_choice` pentru un pas de probă `--mm-alegere`; sau, mai simplu, OS-ul păstrează alegerea direct din
variabilele cursei (`best` pe formă) și ignoră cititorul — rezultatul e același. Pentru noi cititorul e unealta de comparație după cursă.

**Teste T221 (pe procesor, nimic simulat pe placă):** raportul #513 făcut cu șabloanele voastre copiate (15 forme × 5 variante, cu un
rând SARIT, unul NETERMINAT, unul DIFERA, liniile „cea mai rapida”, tabloul A4, rezumatul) → 73 de rânduri, aceleași forme și lansări
ca pasul, aceeași alegere ca din timpii adevărați, `with_mm_choice` mută exact formele alese; două curse → doar ultima; rânduri stricate
și străine lăsate deoparte. Nou pentru placă: nimic.
