# Pentru sesiunea OS: pasul de aur mic întreg pe placa reală (C-160 punctul 6, partea a doua)

**De la Cyber-GPU, sesiunea 155 (22 nov 2026, ~10:00). Stare: SIMULARE** (T165 + T167, `crates/nvidia-ga104/tests/sim_c160_p5_golden.rs`,
`sim_c160_p6_train_step.rs`). Nimic din ce scrie aici nu a rulat pe RTX 3070 Ti. Motorul de antrenare e aprobat de Adrian (21 nov 16:18,
`DECIZII.md`); pe placă intră doar printr-un Release separat și testul centrului de comandă. Totul e **adăugat** (nimic redenumit sau șters).

**Ordinea (regula „un lucru nou pe test”):** acest test vine **după ce proba `colsum` a trecut pe placă** (nota
`C-160_P6_primul-program-inapoi.md`). Până atunci, doar îl legați și îl țineți în QEMU / SIMULARE.

## Ce e
Un pas întreg de antrenare al modelului mic al pasului de aur (`pas_de_aur_mic`: 2 straturi, C 64, 4 capete, vocabular 256, lot 2 × 32), de
10 ori, **numai pe placă**: înainte → eroarea → înapoi → (norma + tăierea) → optimizatorul, toate tensoarele în memoria plăcii. Procesorul
doar încarcă o dată greutățile inițiale și lotul și citește înapoi ce comparăm. Verdictul: **aceleași numere ca PyTorch** (fișierele din hub),
în toleranțele hub-ului — aceleași pe care le folosește deja `antrenare proba` (S132, oracolul pe procesor). Două rețete, câte o rulare:
`sgd` (73 de lansări pe pas) și `adamw_513` (75). În afara drumului generării; nimic din Kronos nu se schimbă.

## Ce e nou pentru placă (exact)
1. **11 programe care n-au rulat niciodată pe placă** (toate Mesa NAK 25.0.7, toate în `nvidia_ga104::train_step::PROGRAMS`, cu SHA-256 în
   `Prog::sha256()`; `train_step::check_programs()` le verifică pe toate): `mm_train`, `xent`, `ln_fwd`, `ln_back`, `gelu_tr`, `attn_fwd`,
   `attn_back`, `emb_fwd`, `emb_back`, `gnorm`, `opt_step`. Al 12-lea, `colsum`, e cel din proba de dinainte; al 13-lea, `vadd`, rulează în
   generare de la I13. Registre ≤ 25 (maximul de azi), blocuri de 16×16, 32, 64 sau 256 de fire; memorie partajată la 5 dintre ele
   (`Prog::shared()`), ca `layernorm256`. Cele cu MUFU (aproximative): `xent`, `ln_fwd`, `gelu_tr`, `attn_fwd`, `gnorm`, `opt_step`.
   **E un test mare** — pasul întreg nu se poate face cu mai puține programe. Dacă vreți întâi programele pe rând, spuneți-ne și facem câte o
   probă ca la `colsum` (ordinea propusă: `mm_train`, `ln_fwd`, `xent`, …).
2. **O singură trimitere de 73 / 75 de lansări pe pas** — mai mult decât lista de simbol de azi (88 la #513, deci nu e nou ca lungime); aceeași
   secvență `transfer::encode_sequence_fresh` (cache-urile invalidate înaintea fiecărei lansări, o înregistrare de timp după fiecare).
3. **Memoria:** o arenă de **~2,8 MiB** (greutăți + gradienți + m + v + activări + lotul), zona de control 75 × 512 octeți = 37,5 KiB,
   înregistrările (n + 1) × 16 octeți. VRAM sau RAM mapat — programelor le e egal; alegeți ce aveți deja pentru listele de simbol.

**Nimic altceva:** nicio scriere nouă în registre, nimic nou prin GSP, aceleași canale.

## Ce e nou la noi (ramura `g0-g2-fundatie`, commit `29c1076`)
Modulul **`nvidia_ga104::train_step`** (doar adăugare):
- `Prog` (13) cu `code()`, `sha256()`, `registers()`, `block()`, `shared()`, `name()`, `index()`; `PROGRAMS`; `check_programs()`.
- `Plan::new(arch, lot, lungime, intrari, baza)` — arena (toate adresele absolute, aliniate la 256, `baza` aliniată la 256);
  `plan.bytes()`; `plan.uploads(greutati, intrari, tinte)` → lista `(adresă, octeți)` de scris **o dată**, înainte de pasul 1 (greutățile,
  zerouri pentru gradienți / m / v, lotul, listele de segmente ale lui `emb_back`, `res = [0, 1]`); refuză un alt lot decât cel al planului.
- `plan.step(reteta, &AdamW::recipe_513(&greutati), pas)` → `Vec<Launch>` (pasul 1, 2, …; la `adamw_513` diferă de la un pas la altul doar
  ultima lansare — scalarii AdamW); `Launch { prog, cbuf, grid, tag, layer }` — `tag` spune ce calculează (`inainte:qkv`, `inapoi:ln2-dx`,
  `opt:pas` …) pentru raport.
- `train_step::pack(&lista, |p| adresa_programului(p), zona_va)` → (octeții zonei de control, adresele QMD-urilor pentru
  `encode_sequence_fresh`).
- `plan.loss_rows` (64 de f32), `plan.res` (`[norma, factorul]`), `plan.g`, `plan.w` (`plan.n_params` f32 = 118 528, ~463 KiB);
  `plan.unflat(&f32)` → tensorii; `train_step::f32s(&octeți)`.
- Verdictul: `Outcome { losses, norms, grad1, after1, after_last, step_ns }`, `Refs { losses, norms, grad1, after1, after_last }`,
  `train_step::judge(reteta, lansari_pe_pas, &outcome, &refs)` → `Verdict` (`passes()`, `rows()`, `journal()`, `lines("placa")`);
  `parse_rows(text)` citește rândurile înapoi; `Outcome::mean_loss(&rânduri)`.
- Existente, folosite: `gpu_compute::train::{MIC, AdamW, Recipe, TOL_*}`, `transfer::{encode_sequence_fresh, sequence_words_fresh,
  analyze_sequence}`.

## Ce legi (Release separat, opțiune oprită implicit, de ex. `test-placa --pas-aur`)
După ce lanțul de azi a pornit canalul de lucru (ca la P8 / `colsum`):
1. **Datele** de pe stick, doar citire: `integrare/kronos/pas_de_aur_mic/` — `initiale/*.f32`, `intrari.i32`, `tinte.i32`, `reteta.json`
   (`pierderi`, `norme_gradient`), `sgd|adamw_513/{grad_pas1,dupa_pas1,dupa_pas10}/*.f32` — exact ce citește deja `antrenare proba`
   (verificarea SHA-256 a rețetei rămâne). Ordinea tensorilor: `train::MIC.tensors()`.
2. **Programele:** `train_step::check_programs()?`, apoi cele 13 încărcate în VRAM ca celelalte (fiecare aliniat la 256).
3. **Pentru fiecare rețetă** (`sgd`, apoi `adamw_513`), de la zero:
   - `let plan = Plan::new(train::MIC, 2, 32, &intrari, arena_va)?;` (arena de `plan.bytes()` octeți); scrii toate `plan.uploads(…)?`;
   - `let opt = AdamW::recipe_513(&initiale);`
   - pentru `pas` = 1 … 10: `let l = plan.step(reteta, &opt, pas)?;` → `pack(&l, …, zona_va)?` → scrii zona → `encode_sequence_fresh` cu
     înregistrările → trimiți → **aștepți ultima înregistrare** (dacă nu vine: oprești, raportul spune „nu a terminat la pasul k”, nu
     reîncercați) → `analyze_sequence` → `step_ns` = suma; citești `plan.loss_rows` (→ `losses`), la `adamw_513` și `plan.res[0]`
     (→ `norms`; **dacă nu e finită, oprești** rularea); după pasul 1 citești `plan.g` (→ `grad1`) și `plan.w` (→ `after1`);
   - după pasul 10: `plan.w` → `after_last`; `judge(…)` față de fișierele rețetei; `verdict.journal(&mut jurnal)` + `verdict.lines("placa")`.
4. **Nu atinge nimic din generare.** Generarea de după rulează ca azi.

## Ce trebuie să iasă (pe placa simulată, T165)
| Rețeta | lansări pe pas | eroarea (pas 1 → 10) | cel mai rău, ‰ din toleranță | în detaliu (‰) |
|---|---|---|---|---|
| `sgd` | 73 | 44,193806 → 33,886936 | **239** | eroarea 11, gradientul 15 / 1, greutăți după 1 pas 60, după 10 **239** |
| `adamw_513` | 75 | 44,193806 → 43,21414 | **128** | eroarea 9, gradientul 15 / 1, greutăți 24 / **128**, normele 9 |

**Trece:** `kt pas-aur verdict (1 aceleasi numere)` = 1 la ambele rețete (toate cifrele ≤ 1000 ‰). Pe placă cifrele pot fi altele decât în
simulare (MUFU, sumatorul plăcii — vezi P8), dar trebuie să rămână sub 1000 ‰. Timpul (DEDUS, nemăsurat): ~0,5–1 ms pe pas (75 × ~6 µs
costul lansărilor + calcule mici); nu e un test de viteză.

## Rândurile `kt` (le scrie `Verdict::journal`, câte un set pe rețetă)
| Rând | Ce înseamnă |
|---|---|
| `kt pas-aur reteta (0 sgd, 1 adamw_513)` | începe un set |
| `kt pas-aur pasi` / `kt pas-aur lansari pe pas` | 10 / 73 sau 75 |
| `kt pas-aur verdict (1 aceleasi numere)` | **1 = trece** |
| `kt pas-aur cel mai rau (promile din toleranta)` | ≤ 1000 = în toleranță; `0xffffffff` = valoare nefinită sau lipsă |
| `kt pas-aur eroarea (promile)`, `… gradientul pasului 1 abs / rel (promile)`, `… greutatile dupa pasul 1 / dupa ultimul pas (promile)`, `… normele (promile)` | fiecare cifră față de toleranța ei |
| `kt pas-aur eroarea pasului 1 / ultimului pas (f32)` | biții f32 ai mediei erorii |
| `kt pas-aur timp median pe pas (us)` | timpul plăcii pe un pas (din înregistrări) |

## Dacă nu trece
Raportul cu rândurile de mai sus + liniile `lines("placa")` ajung — **nu reîncercați**. Noi vedem din ele unde (eroarea deja la pasul 1 →
drumul înainte; doar gradientul → drumul înapoi; doar greutățile → optimizatorul) și dăm o probă mai mică, cu un singur program.
