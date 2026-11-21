# Pentru sesiunea OS: P8 — primul calcul cu circuitele speciale („tensor cores”) pe placa reală (C-160 punctul 1)

**De la Cyber-GPU, sesiunea 139 (21 nov 2026, ~17:11–17:20). Stare: SIMULARE** (T150, `crates/nvidia-ga104/tests/sim_c160_p8.rs`).
Nimic din ce scrie aici nu a rulat pe RTX 3070 Ti. **Aprobat pe placă de Adrian** (21 nov 16:18, „Da la tot: cheie, motor, pasul P8”;
`DECIZII.md`). Totul e **adăugat** (nimic redenumit sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.

## Ce e
Un singur calcul mic, **în afara drumului generării**: placa face `D = A·B + C` pentru matrice de 16×16 (A, B numere pe 16 biți, C, D pe 32
de biți) cu circuitele speciale, de 6 ori, cu 6 seturi de numere fixe. OS-ul citește D și îl judecă pe procesor. Nimic din Kronos nu se
schimbă; niciun simbol nu depinde de acest test.

## Programul ales și de ce
**`mma16_f16`** (`tools/nak/mma16_f16.comp`, compilat de Mesa NAK **26.3.1**, a doua unealtă; documentul `docs/C-147_PUNCTUL5_TENSOR.md`).
- SHA-256: **`f8c0db68b654372d94ed96c767c7775c32909298a1b77076ddda81a66afa1ef9`** (1 248 octeți, 78 de instrucțiuni) — `nvidia_ga104::mma16::{MMA16_F16, MMA16_F16_SHA256}`.
- De ce acesta și nu `mma16_s8` (8 biți): (1) e cel **modelat și testat** în simulare (T142, T150), `mma16_s8` e doar compilat; (2) numerele
  pe 16 biți cu acumulare pe 32 de biți sunt exact ce ar folosi motorul de antrenare (C-160) la înmulțirile mari; (3) are o referință
  exactă pe procesor și o toleranță scrisă dinainte, deci rezultatul e „da/nu” fără interpretare; 8 biți ar cere întâi o decizie de exactitate.

## Ce e nou pentru placă (exact)
1. **Primul program din unealta Mesa 26.3.1** (toate programele de azi sunt din 25.0.7).
2. **Instrucțiunile circuitelor speciale:** 2 × `hmma.m16n8k16.f32` și 4 × `movm.16.m8n8.trans` (rearanjarea lui B) — niciodată rulate de noi.
3. **Bufferul constant de 0x800 octeți** (câmpul `cbuf0_size` din QMD = 0x800, azi 0x100), cu adresele la **0x700** (nu la 0x28).
4. 32 de registre pe fir (azi maxim 25), un bloc de 32 de fire (un singur warp), fără memorie partajată, fără barieră.
5. Programul **citește din RAM-ul procesorului** A, B (pe 16 biți) și C (citiri de 64 de biți) și **scrie 1 KiB în RAM** (D) — același fel de
   acces ca programul de copiere de azi.

**Nimic altceva:** nicio scriere nouă în registre, nimic nou prin GSP, aceleași canale, aceeași secvență cu înregistrări de timp.

## Ce e nou la noi (ramura `g0-g2-fundatie`, commit `82178f7`)
- Modulul **`nvidia_ga104::mma16_probe`**:
  - `SETS` (6 seturi: `identitate`, `uniform-1..3`, `scari-1..2`), `ProbeSet::{inputs, page, inputs_fnv}`;
  - pagina de 4 KiB a unui set: A la `A_OFF` = 0x000, B la `B_OFF` = 0x200, C la `C_OFF` = 0x400, **D la `D_OFF` = 0x800 umplut cu NaN**
    (`D_FILL` = `0x7fc00000`, ca un element nescris să nu poată trece), restul zero;
  - `launch_bytes(prog_va, cbuf_va, page_va)` → (cbuf 0x800 octeți, QMD) — verifică alinierea (pagina la 4 KiB, cbuf la 256);
  - `judge_set(set, d_bytes, ns)` → `SetResult` (verdictul, abaterea maximă în ‰ din toleranță, modelele potrivite); `Probe::{add, passes,
    rows, journal, lines}`; `parse_mma16(text)` citește rândurile înapoi din jurnalul de pe stick.
- În oracol, `gpu_compute::mma::{Rounding, AddModel, ADD_MODELS, mma16_blocked, matching_models, worst_permille}` (10 modele candidate
  ale felului în care adună circuitele: câte 1, 2, 4, 8 sau 16 termeni odată × rotunjire sau tăiere).
- Existente (sesiunea 126): `nvidia_ga104::mma16::{MMA16_F16, MMA16_F16_SHA256, cbuf, qmd, CBUF_SIZE, PUSH_BASE}`, `gpu_compute::mma::{judge, tolerance, mma16_exact}`.

## Ce legi (Release separat, opțiune oprită implicit, de ex. `test-placa --p8`)
După ce lanțul de azi a pornit canalul de lucru (ca la orice test cu programe), o singură dată:
1. **Programul:** încarcă `MMA16_F16` în VRAM ca pe celelalte programe (aceeași cale); verifică întâi SHA-256 = `MMA16_F16_SHA256` (refuză altfel).
2. **O zonă VRAM de 4 KiB** pentru cbuf + QMD: cbuf la +0, QMD la +0x800 (aliniat la 256).
3. **6 pagini de RAM** (sau una refolosită), aliniate la 4 KiB, mapate pentru placă ca RAM-ul gazdă al copiei (`sys`).
4. Pentru fiecare set `i` din `SETS` (0..6), **câte o trimitere**:
   - scrii `SETS[i].page()` în pagină (D devine NaN);
   - `let (cb, q) = mma16_probe::launch_bytes(prog_va, zona_va, pagina_va)?`; scrii `cb` la zona+0 și `q` la zona+0x800;
   - `transfer::encode_sequence(&[zona_va + 0x800], rec.va, first)` → `run_records` (2 înregistrări) → `analyze_sequence` = timpul `ns`;
   - citești cei **1 024 de octeți de la pagină + `D_OFF`** și `probe.add(mma16_probe::judge_set(i, &d, Some(ns))?)`.
   - dacă placa nu termină (înregistrarea nu vine): oprești proba, rândul `kt mma16 seturi` cu câte s-au făcut, raportul spune „nu a terminat
     la setul i”; nu reîncerci.
5. La final: `probe.journal(&mut journal)` + `probe.lines("placa")` în raport. **Nu atinge nimic din generare.**

## Oracolul, toleranța, ce citim înapoi
- **Oracolul pe procesor:** `mma16_exact` — fiecare produs exact, cei 17 termeni ai unui element adunați în f64 și rotunjiți o dată.
- **Toleranța (scrisă în C-147 punctul 5, înainte de orice rulare, neschimbată):** fiecare element la cel mult
  **`17 · 2⁻²³ · (|c| + Σ|a·b|)`** de oracol. Nu cerem identic bit cu bit: NVIDIA nu documentează ordinea și rotunjirea adunărilor.
- **Ce citim:** doar D (1 KiB pe set) + cele 2 înregistrări de timp.
- **Bonus de măsurătoare:** D se compară bit cu bit și cu cele 10 modele de adunare; T150b arată că cele 6 seturi **deosebesc orice două
  modele**, deci masca spune exact cum adună GA104 (sau 0 = alt fel, tot bun dacă e în toleranță). E prima cifră despre asta pe Ampere de
  consum; o folosim la motorul de antrenare (cât de exact iese o sumă lungă).

## Rândurile `kt` (le scrie `Probe::journal`)
| Rând | Ce înseamnă |
|---|---|
| `kt mma16 seturi` | câte seturi s-au judecat (6) |
| `kt mma16 seturi in toleranta` | **6 = trece** |
| `kt mma16 elemente exacte` | din 1 536, egale bit cu bit cu oracolul |
| `kt mma16 abaterea maxima (promile din toleranta)` | ≤ 1000 = în toleranță |
| `kt mma16 modele potrivite (masca)` | bitul i = `ADD_MODELS[i]` (bitul 8 = `bloc8-taiere`, ipoteza din simulare) |
| `kt mma16 timp median (ns)` | timpul unei lansări pe ceasul plăcii |
| `kt mma16 set: index / intrari (fnv) / verdict / exacte / promile / modele (masca) / timp (ns)` | câte unul pe set; verdict 1 = în toleranță, 0 = nu, 2 = invalid (D nescris / NaN) |

Amprentele intrărilor (`kt mma16 set: intrari (fnv)`), ca să știm că s-au trimis exact aceste numere:
`identitate 0x4c78b133`, `uniform-1 0x64e5d34e`, `uniform-2 0xf9b32da5`, `uniform-3 0x1a21544b`, `scari-1 0xa8bf6c7d`, `scari-2 0x6f0725c7`.

## Ce trebuie să iasă
- **Trece:** `kt mma16 seturi in toleranta` = 6, setul `identitate` cu **256/256 exacte** (D = B), abaterea maximă ≤ 1000 ‰.
- În SIMULARE (modelul e o ipoteză, nu o dovadă): 6/6, 650/1 536 exacte, 107 ‰, masca `0x100` (`bloc8-taiere`).
- Timpul (DEDUS, nemăsurat): o lansare de un warp, ~3–10 µs cu tot cu pornirea; nu contează pentru viteză — rostul e „calculează corect”.

## Dacă nu trece — ce spune fiecare semn
| Ce vezi | Ce înseamnă probabil |
|---|---|
| verdict 2 la toate seturile (D rămas NaN) | programul nu a scris: cbuf-ul nu e la 0x700 / mărimea 0x800 nu a ajuns în QMD / programul nu a pornit |
| `identitate` nu trece, restul nici | așezarea lui A sau B (transpusă, deplasată) — nu aritmetica |
| `identitate` trece, celelalte nu | aritmetica circuitelor diferă de ce credem — **oprim P8**, raportăm cifrele, nu ajustăm toleranța |
| placa nu termină | canalul rămâne ca după orice lansare blocată; raportul cu setul la care s-a oprit |

La orice eșec: nimic din generare nu e atins, testul obișnuit rămâne la îndemână. Un singur lucru nou în acest Release.
