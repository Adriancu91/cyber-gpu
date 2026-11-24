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
