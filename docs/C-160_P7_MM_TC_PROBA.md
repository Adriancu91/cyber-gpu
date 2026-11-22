# C-160 punctul 7, partea a patra — proba `mm_tc` pentru placă (seturi fixe + timpul la forma qkv a lui #513)

Sesiunea 160, 22 nov 2026, în cloud, **fără placă**. Planul, marginea și predicțiile de mai jos sunt scrise și urcate **înainte** de cod;
rezultatul în SIMULARE vine la sfârșit. Nimic de aici nu a rulat pe RTX 3070 Ti.

## De ce
`mm_tc` (înmulțirea antrenării pe circuitele speciale, `docs/C-160_P7_MM_TC.md`) e singurul drum spre „mai repede decât PyTorch”
(`docs/C-160_P7_VITEZA_ESTIMARE.md` §2). În SIMULARE e corect (T168), dar pe placă n-a rulat. Ca la `colsum` (os-349), îl trimitem întâi
**singur**, într-o probă mică, separată de pasul de aur: un singur lucru nou pe test.

## Ce e nou pentru placă (exact)
- binarul `mm_tc` (Mesa NAK 26.3.1, SHA-256 `f60cd944…1ee3`, 64 de registre, bloc 128) — al doilea program al uneltei 26.3.1 după
  `mma16_f16` (os-335);
- **memoria partajată 24 KiB pe bloc + o barieră** (cea mai mare de până acum: 2–4 KiB la `ln_fwd`/`xent`) — câmpurile QMD din
  `ComputeQmd::to_bytes_with_shared(&mm_tc::SHARED)`, ca la programele cu memorie partajată care au trecut;
- grilele cu y > 1 (⌈n/64⌉ × ⌈m/64⌉), argumentele la cbuf 0 + 0x700 (ca la P8);
- **setul de timp** cere ~83 MB într-un singur buffer de **VRAM** (cel mai mare de până acum; modelul lui #513 ocupă ~270 MiB în bucăți).
Nimic nou prin GSP, nicio scriere nouă în registre.

## Seturile (fixe, aceiași octeți pe OS și în teste)
| # | nume | m × n × k | opțiuni | ce încearcă |
|---|---|---|---|---|
| 0 | `16x16x16` | 16 × 16 × 16 | — | o singură placă 16×16, K < 32 (felia completată cu zero) |
| 1 | `70x130x33-tb-bias` | 70 × 130 × 33 | TB, BIAS | margini pe toate trei dimensiunile, grila 3 × 2, forma `linear` (greutatea `[out, in]`) |
| 2 | `64x64x96-ta` | 64 × 64 × 96 | TA | A întors (forma dW), 3 felii de K |
| 3 | `100x72x200-ta-tb` | 100 × 72 × 200 | TA, TB | amândouă întoarse, K nedivizibil cu 32 |
| 4 | `32x48x64-limite-f16` | 32 × 48 × 64 | TB, BIAS | rânduri cu valori mici (sub 2⁻¹⁴: f16 subnormal, unele → 0), mari (până la 65 504), amestecate |
| 5 | `qkv-513` | 6 144 × 2 304 × 768 | TB, BIAS | **timpul**: qkv înainte pe o bucată #513 (12 × 512 rânduri); judecat pe 3 benzi de 16 rânduri |

- Intrările: o funcție de dispersie (splitmix64) pe (sămânță, indice) → f32; OS-ul le poate scrie bucată cu bucată fără să țină
  matricea întreagă (`ProbeSet::{a_at, b_at, bias_at}`). Seturile 0–4: valori f32 oarecare (deci și rotunjirea `F2F.F16.F32` e probată);
  setul 5: activări în [−2, 2), greutăți în [−0,05, 0,05).
- Un buffer pe set: A, B, deplasarea, C, apoi **gărzi** (256 de cuvinte); C și gărzile umplute cu NaN (`0x7fc00000`) înainte — un element
  nescris nu poate trece, o scriere după C se vede.
- Setul 5 se lansează de **5 ori** (aceleași intrări, C rescris identic), fiecare cu timpul ei; rândul de timp = mediana; se citesc
  înapoi doar benzile de rânduri 0–15, 3 072–3 087, 6 128–6 143 (toate cele 2 304 de coloane) + garda. Dacă OS-ul nu poate da 83 MB de
  VRAM, setul 5 se sare (rândul `qkv omis`), iar seturile 0–4 rămân proba de corectitudine.

## Marginea (deja scrisă, înainte de orice rulare — T168, `gpu_compute::mm_tc::margin`)
Față de referința pe intrările rotunjite la f16 (produsele exacte în f64): **(K + 3)·2⁻²³·(Σ|â·b̂| + |deplasare|) + 2⁻¹²⁶** pe element.
Neschimbată pentru placă. Verdictul unui set: toate elementele citite în margine, gărzile neatinse.

## Predicțiile (scrise acum)
1. **Verdictul: 6 / 6 seturi în margine** — încredere ~85 %. Riscul principal nu e aritmetica (P8 a trecut cu aceeași instrucțiune), ci
   lucrurile noi: memoria partajată de 24 KiB (dacă QMD-ul o declară greșit → blocare sau gunoi) și grila cu y > 1.
2. **Bit cu bit modelul nostru** (`gpu_compute::mm_tc::model`, sumatorul potrivit pe amprenta os-335): **toate elementele citite** —
   încredere doar ~50 %: sumatorul e o potrivire pe 12 numere, nu dovedit bit cu bit, iar ordinea în care NAK înlănțuie cele două
   `hmma.m16n8k16` pe placa 16×16 n-a fost verificată pe placă. Dacă nu: verdictul tot trece (marginea acoperă și rotunjirea la cel mai
   apropiat), iar amprentele ieșirii ne spun pe ce set să căutăm.
3. **Abaterea maximă: ≤ 100 ‰** din margine (modelul dă sub ~50 ‰ în T168; dublu pentru un sumator puțin diferit).
4. **Timpul la `qkv-513`** (21,7 GFLOP; DEDUS în `C-160_P7_MM_TC.md` §6): **1,3–2,7 ms ≈ 8–17 TFLOP/s**. Sub 5 TFLOP/s (> 4,3 ms) =
   frâna e altundeva decât credem (ocuparea cu 24 KiB + 64 de registre, ceasul, citirile pe element) — de lămurit înainte de plăci mai
   mari. Peste 17 = listarea ne-a speriat degeaba.
5. Seturile 0–4 sunt prea mici pentru timp: ~5–10 µs fiecare, cât costă o lansare (P8: ~6,1 µs).

## Ce NU face
Nu schimbă `mm_tc`, nu intră în pasul de aur (acela rămâne F32). Pe placă **după** pasul de aur mic F32 (C-160 p.6) — un singur lucru nou
pe test.

## Surse
Aceleași ca `docs/C-160_P7_MM_TC.md`. Forma probei (buffer cu gărzi NaN, rânduri `kt …`, amprente FNV) = `colsum_probe` (os-349).

## Rezultatul în SIMULARE (scris după cod, sesiunea 160)
Cod: `crates/nvidia-ga104/src/mm_tc_probe.rs`, testul T171 `crates/nvidia-ga104/tests/sim_c160_p7_mm_tc_probe.rs` (6 teste + 5 în modul).
- Seturile 0–4 prin procesorul de comenzi simulat (QMD-ul cu 24 KiB + bariera din `launch_bytes`): **5 / 5 în margine**, 22 188 / 22 188
  bit cu bit modelul (normal: „placa” simulată *este* modelul), cel mai rău **69 ‰** (setul cu limitele f16; celelalte 10–40 ‰) — în
  predicția 3 (≤ 100 ‰).
- `qkv-513` judecat pe cele 3 benzi din ieșirea modelului: în margine, 5 ‰; modelul pe benzi ~0,6 s pe procesor. Că benzile judecă exact ce
  a scris lansarea întreagă: T171d (un produs 160 × 130 × 40 rulat întreg, citit pe benzi = modelul pe benzi, octet cu octet).
- Greșeli plantate prinse (18+): fără lansare, TB pierdut, K − 1, M − 16, două rânduri din A schimbate. Un QMD fără memoria partajată e
  refuzat de modelul procesorului de comenzi (T171c), nu ghicit.
- Amprentele de comparat pe placă: tabelul din `integration/cyber-universe/C-160_P7d_mm_tc-proba.md`.
- `tools/check.sh`: **807 PASS**. Predicțiile 1, 2, 4 și 5 rămân **NETESTATE** până la placă.
