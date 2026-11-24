# C-157 punctul 2 după os-392 — profilul pasului de antrenare pe programe (plan + predicții, scrise înainte de cod)

Sesiunea 203, 24 nov 2026. Tot ce e aici e **SIMULARE / DEDUS**; nimic nu rulează pe RTX 3070 Ti din cloud.

## De ce
os-392: placa face pasul lui `antrenare mini` în **3 602 µs** (69 de lansări). Costul fix al lansărilor e ~4 µs fiecare (os-331, confirmat
de lipire: −26 µs pentru 6 lansări) → ~0,3 ms. Restul de **~3,3 ms e calcul**, și nu știm pe ce program. La generare avem de la os-309
„timpul pe fel de lansare” (`kronos_perf::KindProfile`); la antrenare nu. OS-ul citește deja **toate înregistrările** la 3 pași din 10
(`RecordPlan::DIET`, „citite toate la 3 pași”), deci cifrele există pe placă — lipsește doar împărțirea lor.

## Ce se adaugă (doar adăugări, nimic schimbat în `train_step`)
`nvidia_ga104::train_profile`:
- `StepProfile::from_launches(&[Launch])` — numele fiecărei lansări (eticheta `inainte:…` / `inapoi:…` / `opt:…` și programul), în ordinea
  listei; `with_copies(sus, jos)` pentru cele 5 + 2 copieri ale lotului prin RAM (T209), puse în față și în coadă.
- `add_step(&[u64])` (timpii în ns, aceeași lungime) și `add_records(records, first_payload)` (prin `transfer::analyze_sequence`, aceleași
  verificări: lipsă / ordine / ceas înapoi = refuz, nimic ghicit).
- `by_tag()`, `by_prog()`, `by_stage()` — sume pe pas (straturile adunate la aceeași etichetă), mediana peste pași, partea din total în
  promile, sortat descrescător; `lines(top)` — rândurile de text pentru OS (română fără diacritice, ca restul jurnalului).

## Predicții
- **Q1 (structură):** suma pe etichete = suma pe programe = suma pe etape = timpul plăcii al pasului, la ns, pe fiecare pas.
- **Q2 (refuzuri):** lungimi diferite, o înregistrare lipsă, ceasul înapoi → eroare, nu profil parțial.
- **Q3 (lipire):** pe lista lipită etichetele `inapoi:rezidual-ln*` și `inapoi:gelu` dispar, apar `…+rezidual` / `…+gelu`; restul identic.
- **Q4 (pe placă, DEDUS din formă — de verificat când OS-ul tipărește profilul):**
  1. `inapoi:cap-dX` e **primul**, 0,3–1,0 ms: `mm_train` cu 16 blocuri (din 48 de multiprocesoare) și K = 16 384 → 1 024 de ture
     seriale (două plăci 16×16 din memorie + barieră), fără nimic care să ascundă așteptarea memoriei.
  2. apoi `eroarea` (`xent`, 64 de blocuri × 16 384) 50–200 µs și `opt:pas` (1,15 M greutăți, ~32 MB trafic) 60–120 µs;
  3. `inainte:logit` și `inapoi:cap-dW` (K = 64, 4 096 de blocuri) sub 60 µs fiecare;
  4. cele 24 de lansări `colsum` cu un singur bloc + cele 8 `attn_*` cu 8 blocuri: câte ~5–15 µs, împreună ~0,3–0,5 ms.
  Dacă Q4.1 iese adevărat, tăietura următoare e K împărțit pe mai multe blocuri pentru `cap-dX` (sau `mm_tc` în pasul mixt).
  Dacă nu, profilul spune unde e timpul și asta e tăietura.

## Test
T210 (`cargo test --release -p nvidia-ga104 --test sim_c157_p2_profil`), SIMULARE: lista reală a lui `antrenare mini` (nelipit 75,
lipit 69, cu copierile lotului 76), timpi sintetici și înregistrări codate ca pe placă (`gpu_sim` nu are ceasul plăcii: numerele lui
testează numărarea, nu spun nimic despre RTX 3070 Ti).
