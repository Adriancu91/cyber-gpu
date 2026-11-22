# C-160 punctul 4 — programele 9 și 10 ale antrenării: norma + tăierea gradientului (`gnorm`) și pasul optimizatorului (`opt_step`)

Sesiunea 152, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

Cu ele, toate cele 10 programe din lista studiului (`docs/C-160_MOTOR_STUDIU.md`) există în SIMULARE.

## Programul 9 — `gnorm`: norma totală și factorul de tăiere
Ce face (ca `clip_grad_norm_` din PyTorch): N = √(Σ g²) pe **toți** tensorii (driverul ține gradienții unul după altul, într-un
singur buffer), apoi c = min(1; max_norm / (N + 1e-6)). Pasul AdamW înmulțește apoi fiecare gradient cu c.

**Fără `atomic`:** blocul b adună `x[b·chunk .. (b+1)·chunk)` (firul t ia din 256 în 256, de la b·chunk + t, de la 0), apoi suma
blocului din `ln_fwd` (fluture în warp, cele 8 sume în ordine). Aceiași biți la fiecare rulare. Trei moduri:
- `Partial` (1): pătrate → `part[b]`, câte unul pe bloc;
- `Finish` (2), un bloc: suma sumelor parțiale → `res[0] = N`, `res[1] = c`;
- `Whole` (3), un bloc: ambele, pentru un buffer de cel mult un „chunk”.
Driverul: o lansare dacă totul încape într-un chunk, altfel două (`Partial`, apoi `Finish`).

`tools/nak/gnorm.comp` → Mesa NAK 25.0.7: **2 160 de octeți**, 135 de instrucțiuni, 19 registre, 32 de octeți memorie partajată,
o barieră; SHA-256 `253c8e00f63e65c78283b4296cd9963e04146facd8231090d933d5d37322a594`. Bufferul constant 0: x / part / res la
0x28 / 0x30 / 0x38, n 0x40, chunk 0x44, modul 0x48, max_norm 0x4c; baza grupurilor 0x0 = 0. Aritmetica: `ffma.ftz` (pătratele),
`fadd.ftz` (sumele), la final un `mufu.sqrt` și un `mufu.rcp`: **NAK împarte prin înmulțire cu inversul** (c = RCP(N + 1e-6) ·
max_norm, două rotunjiri). NaN rămâne NaN (ca `torch.clamp`): un gradient NaN dă c = NaN — driverul trebuie să oprească pasul.

## Programul 10 — `opt_step`: pasul SGD / AdamW pe toți parametrii
Un fir pe element, blocuri de 256 (w, g, m, v: câte un buffer, tensorii unul după altul). Întâi g ← g · c (c scris de `gnorm`, sau
1,0 scris de driver la rețeta fără tăiere: g · 1 e exact).
- **SGD** (`sgd`): w ← w + (−lr)·g;
- **AdamW** (`adamw_513`, ordinea PyTorch 2.14): w ← w·(1 − lr·wd); m ← m + (1−β1)·(g − m); v ← v·β2 + ((1−β2)·g)·g;
  w ← w + (−lr/(1−β1ᵗ))·(m / (√v / √(1−β2ᵗ) + eps)).

Scalarii pasului îi rotunjește driverul la f32 exact ca oracolul (funcție nouă `train::AdamW::scalars`, folosită acum și de
`AdamW::step` — același rezultat, T151 neschimbat). În program totul e `precise`, deci NAK **nu contopește** nicio înmulțire cu o
adunare (verificat: niciun `ffma` în cod): SGD și m, v sunt operațiile f32 ale lui PyTorch, **bit cu bit** (cu FTZ). √ și cele două
împărțiri sunt unități aproximative: `mufu.sqrt` + două `mufu.rcp` (împărțire = înmulțire cu inversul, ca la `gnorm`).

`tools/nak/opt_step.comp` → Mesa NAK 25.0.7: **976 de octeți**, 61 de instrucțiuni, 19 registre; SHA-256
`8e15a8b6410bef4b91bee949b332805ee261aa6a9a06d2a7b668f42c240b3b12`. Bufferul constant 0: g / coef / w / m / v la
0x28 / 0x30 / 0x38 / 0x40 / 0x48, n 0x50, modul 0x54, cei 8 scalari de la 0x58 (−lr, 1 − lr·wd, 1−β1, β2, 1−β2, √(1−β2ᵗ),
−lr/(1−β1ᵗ), eps). Mesa reconstruit în cloud: `emb_back` a ieșit identic (aceeași sumă SHA-256).

Adăugări (nimic redenumit, OS-ul compilează neschimbat): `nvidia_ga104::{gnorm, opt_step}` (`GNORM`, `OPT_STEP`, SHA-256,
`REGISTERS`, `BLOCK`, `CBUF_BYTES`, `gnorm::SHARED`, `Mode`, `Args::{cbuf, grid}`), `gpu_compute::layer::{GNORM_SQUARE,
GNORM_FINISH, gnorm_block_gpu, gnorm_finish_gpu, grad_norm_gpu, OptScalars, opt_step_gpu}`, `gpu_compute::train::{AdamW::scalars,
AdamScalars}`, `gpu_sim::cmdproc::{GNORM_NEEDS, gnorm_on_cpu, opt_step_on_cpu}`. Test: `crates/nvidia-ga104/tests/sim_c160_opt.rs`.

## Marginile, scrise înainte de măsurare
u = 2⁻²⁴; pentru fiecare unitate aproximativă a plăcii 2⁻²² loc în plus (E), ca la celelalte programe.
- **N:** toate pătratele au același semn, deci lanțul de k fma al unui fir + cele 12 adunări ale sumei blocului dau ≤ 1,01·(k + 12)·u
  relativ, la fel a doua etapă; e_S = suma lor (+ n·FTZ absolut). N: ½·e_S + u + E relativ (+ √(n·FTZ)).
- **c:** rel(N) + 3u + E (adunarea, RCP-ul, produsul). Dacă c adevărat ≥ 1, placa trebuie să dea exact 1.
- **AdamW, w:** pe placă √ (u + E), RCP(bc2) (u + E) și produsul (u), + eps (u), RCP(den) (u + E), · m (u), · −pas (u) → 3E + 7u;
  oracolul are ≤ 5u; adunarea finală se rotunjește o dată de fiecare parte:
  |w_placă − w_oracol| ≤ 1,01·(3E + 12u)·|actualizare| + 2,01·u·|w| + 4·FTZ.
- **AdamW, m și v; SGD:** bit cu bit (date fără numere subnormale).

## Ce a ieșit (SIMULARE, lansări prin procesorul de comenzi față de oracol)
**T163 — `gnorm`:** 10 forme (n = 1 … 300 000, chunk 256 … 65 536, scări 1e-18 … 1e18, până la 1 172 de blocuri): **≤ 7,4 % din
margine** (11,4 % fără loc pentru MUFU). Pe gradientul **PyTorch** al pasului 1 din `pas_de_aur_mic/adamw_513`: N = 10,759223 =
norma lui PyTorch din `reteta.json` (10,759222984…), abaterea relativă **0** (sub o unitate f32), la trei chunk-uri; c = 0,092943504.
În plus: c = 1 exact sub prag, c < 1 la N = max (1/(1 + 1e-6)), max_norm ≠ 1, totul 0 → (0; 1), NaN → (NaN; NaN), ∞ → (∞; 0);
aceiași biți la două rulări; o lansare / două lansări; refuzuri (terminare cu 2 blocuri — s-ar bate pe `res`, bloc greșit, QMD fără
memorie partajată, baza grupurilor ≠ 0); **8 greșeli plantate prinse** (fără pătrate, ultimul bloc uitat, doar primul bloc, fără
rădăcină, ultimul element uitat, N/2, c cu 2·max, c fără limita 1).

**T164 — `opt_step`:** SGD bit cu bit egal cu `train::sgd_step` (n = 1 … 20 000; m, v neatinse). AdamW pe 6 cazuri (pasul 1 … 10,
gradienți 1e-12 … 1e6, cu și fără tăiere): m și v **bit cu bit**, w **≤ 71,2 % din margine**, 99,9 % din elemente cu exact biții
oracolului (restul diferă prin RCP·înmulțire față de împărțirea lui PyTorch). Cu √ și ambele RCP mutate cu ±2 unități (cum ar putea
greși placa): ≤ 65,2 % din margine. **8 greșeli plantate prinse** (fără scăderea greutății, fără corecția lui β1 / β2, β-uri
inversate, semnul gradientului, gradientul deplasat, tăiere aplicată când nu trebuie, eps × 1000).

**Pasul de aur mic cu optimizatorul pe placa simulată** (gradienții din oracolul pe procesor, T151; apoi pe placa simulată
`gnorm` în două lansări + `opt_step`), 10 pași, față de **PyTorch**:
| Rețeta | după pasul 1 | după pasul 10 | toleranța | în plus |
|---|---|---|---|---|
| `sgd` | 1,19e-7 | 2,38e-7 | 1e-6 | bit cu bit egal cu rularea oracolului |
| `adamw_513` | 1,94e-6 | 1,09e-5 | 1e-4 | normele celor 10 pași față de PyTorch ≤ 1,8e-7 relativ; față de oracol 7,5e-9 / 1,4e-6 |

*Corectat pe drum (cinstit):* am pus la început două praguri în plus, ghicite, nu scrise din socoteală: „după 10 pași ≤ 1e-5 față de
PyTorch” — greșit, chiar oracolul e la 1,05e-5 (T151); apoi „≤ 1e-6 față de oracol după 10 pași” — măsurat 1,41e-6: gradienții se
recalculează din greutăți care diferă în ultimii biți, iar m/(√v + eps) al lui AdamW mărește diferența unde g ≈ 0. Criteriul rămas
pentru 10 pași = toleranța din hub (1e-4); după pasul 1 (doar unitățile aproximative) ≤ 1e-7. Marginile T163/T164 n-au fost atinse.

**Ce NU e dovedit:** nimic pe placă. Cât de mult greșesc `mufu.sqrt` / `mufu.rcp` ale lui GA104 (am lăsat 2⁻²²; P8 și LayerNorm-ul
nou au trecut pe placă cu RSQ / EX2 / RCP în margini asemănătoare). Viteza: `gnorm` citește gradienții o dată; `opt_step` e o trecere
prin memorie (5 citiri + 3 scrieri pe element, 32 de octeți) — la #513 o trecere prin toată starea fixă a optimizatorului pe pas (nemăsurat).

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
`gnorm`: memoria partajată + o barieră + `shfl.bfly` (ca `ln_fwd`), un `mufu.sqrt` (nou ca unitate pe placă). `opt_step`: citire și
scriere în același loc (w, m, v), `mufu.sqrt` + `mufu.rcp`. Driverul trebuie să citească `res` după `gnorm` și să **oprească** pasul
dacă N nu e finit (PyTorch ar strica greutățile cu NaN).

## Ce urmează
Punctul 5: pasul de aur mic **cap-coadă** pe placa simulată — toate cele 10 programe legate (înainte, eroarea, înapoi, norma,
AdamW), fără niciun calcul pe procesor în afara oracolului de comparat; apoi `pas_de_aur_513` cât permite simularea.
