# Ordinea 20:11, pasul 2 — circuitele speciale în înmulțirile antrenării, pe forma lui `antrenare mini` (SIMULARE)

Sesiunea 191, 23 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE** sau **DEDUS**: nimic nu a rulat pe RTX 3070 Ti.

## De ce acum
Decizia lui Adrian din 23 nov 20:11 (DECIZII.md): (1) pasul întreg de antrenare pe placa reală (`test-placa --antrenare-mini`, îl face
OS-ul, C-161 nota 18:16), apoi **(2) circuitele speciale în înmulțirile antrenării**. Pe GA104, prin Mesa NAK, circuitele speciale dau
doar **f16 × f16 → f32** (bf16 nu e oferit: `docs/C-147_PUNCTUL5_TENSOR.md`), deci „bf16” înseamnă la noi f16 + **scalarea erorii**.
Pasul mixt există deja (`Plan::step_mixed`, T169), dar a fost probat doar pe modelul de 256 de simboluri, unde scalarea **nu s-a dovedit
necesară** (T169: S = 1 și S = 1 024 au dat același gradient — softmax-ul era ascuțit).

`antrenare mini` (OS, P6-157 / P6-162) are **vocabularul lui Kronos, 16 384 de simboluri**, pornește din greutăți mici (σ 0,02), deci
softmax-ul e aproape uniform: d(logits) pentru simbolurile care nu sunt ținta ≈ (1/16 384) / 64 ≈ **9,5·10⁻⁷** — sub cel mai mic număr
normal f16 (6,1·10⁻⁵), în zona „subnormală”, unde f16 mai are doar ~4 biți. Aici scalarea ar trebui să conteze. Asta e pasul care va urca
pe placă după `--antrenare-mini` F32 și după proba `mm_tc` (os-352), deci trebuie verificat **pe forma lui**, înainte.

## Ce se face (doar test nou + notă; codul bibliotecii neschimbat)
T199, `crates/nvidia-ga104/tests/sim_c160_p7_mixed_mini.rs`:
- modelul `antrenare mini`: vocabular 16 384, context 32, E 64, 4 capete, 2 straturi; lot 2 × 32; greutățile de start ca în OS (uniforme
  ±0,0346, bias 0, LayerNorm 1; generatorul nostru, alt șir de numere); simbolurile lotului pseudo-aleatoare fixe (nu avem datele lui
  Kronos în repo); **AdamW ca `antrenare mini`**: lr 1·10⁻³, β 0,9 / 0,999, ε 1·10⁻⁸, decay 0,1, tăiere la 1,0; 5 pași pe același lot;
- referința = **oracolul pe procesor** (`gpu_compute::train::loss_and_grad` + tăiere + `AdamW::step`), aceleași greutăți de start;
- trei rulări pe placa simulată, câte o trimitere pe pas: lista F32 (`Plan::step`, martor), **pasul mixt S = 1 024** și pasul mixt S = 1;
- așezarea simplă (`Plan::new`, cea care a trecut pe placă în os-351) și, pentru S = 1 024, atenția pe plăci `_3` (`Plan::new_tiles3`).

Măsurile (față de oracol): eroarea fiecărui pas (relativ); gradientul pasului 1 pe fiecare tensor, ‖Δ‖/‖g‖ (după împărțirea la S);
**„rândurile rare” ale lui `emb_token`** = simbolurile care nu apar nici ca intrare, nici ca țintă în lot — gradientul lor vine doar din
d(logits) mici, deci acolo se vede f16; **mișcarea greutăților** după 5 pași: ‖(w − w₀) − (w_ref − w₀)‖ / ‖w_ref − w₀‖ pe tensor
(măsura cea mai strictă: compară pașii, nu greutățile întregi), fără `in_proj_bias` (partea k are gradient 0 exact — zgomot, T169 §2).

## Predicțiile, scrise ÎNAINTE de prima rulare (se raportează oricum ar ieși)
1. **Martorul F32** față de oracol: eroarea ≤ 1·10⁻⁵ relativ, gradientul ≤ 1·10⁻³ pe tensor (toleranțele de până acum).
2. **Mixt, S = 1 024**: eroarea fiecărui pas ≤ 1·10⁻³; gradientul pasului 1 ≤ 5·10⁻³ pe fiecare tensor (fără `in_proj_bias`);
   rândurile rare ale lui `emb_token` **≤ 1·10⁻³** (intrări f16 normale, ~2⁻¹¹ relativ, ~2–5·10⁻⁴ medie pătratică);
   mișcarea greutăților ≤ 5·10⁻² pe tensor (fără `in_proj_bias`).
3. **Mixt, S = 1**: rândurile rare ale lui `emb_token` **între 5·10⁻³ și 5·10⁻²** (pasul subnormal 6·10⁻⁸ față de valori ~16 pași →
   rotunjirea ≈ 1/(16·√12) ≈ 1,8 %), adică **de cel puțin 10 ori mai rău decât S = 1 024**. Pe tot `emb_token` (dominat de rândurile
   țintelor, d(logits) ≈ −1/64, normal în f16) diferența rămâne mică: raport S=1 / S=1 024 **sub 3** — măsura pe tensor întreg ascunde
   efectul, ca în T169.
4. **AdamW ascunde mare parte din efect la mișcarea greutăților** (la primul pas m̂/√v̂ = semnul lui g, indiferent de mărime): pe rândurile
   rare, S = 1 dă o mișcare mai rea decât S = 1 024, dar **cu cel mult 10 ori** (nu cât la gradient).
5. **Atenția pe plăci `_3` + mixt S = 1 024**: aceleași cifre ca așezarea simplă mixtă, în 2× (programele `_3` dau aceiași biți ca cele
   vechi, T195).

**Greșeli plantate care trebuie prinse:** (a) S pus în eroare, dar norma comparată cu 1,0 în loc de S (tăiere de ~1 000 de ori prea tare)
→ prinsă de mișcarea greutăților (peste 5·10⁻²); (b) rândurile rare calculate greșit (o țintă trecută drept rară) → testul de construcție al măsurii o
refuză.

Ce NU dovedește: nimic despre placă; modelul sumatorului din circuitele speciale e cel potrivit pe os-335 (P8), nu dovedit bit cu bit;
viteza nu se măsoară aici (estimarea pentru `mm_tc` rămâne ~8–17 TFLOP/s, `docs/C-160_P7_MM_TC.md` §6, DEDUS).

## Rezultatele (SIMULARE, T199, 6 teste, ~43 s) — toate cele 5 predicții au ieșit

Startul: eroarea 9,741 (ln 16 384 = 9,704 → softmax aproape uniform, cum prezisesem); 16 319 din 16 384 de rânduri ale lui `emb_token` sunt
„rare”. Lista mixtă are **27 de lansări pe `mm_tc` pe pas** (toate înmulțirile modelului) + o singură `mm_train` 1×1×1 (c/S).

| Măsura (față de oracolul pe procesor) | F32 (martor) | Mixt S = 1 024 | Mixt S = 1 | Mixt S = 1 024, atenția `_3` |
|---|---|---|---|---|
| Eroarea, cel mai rău pas din 5 (relativ) | 0 | 7,6·10⁻⁶ | 6,7·10⁻⁶ | 7,7·10⁻⁶ |
| Gradientul pasului 1, cel mai rău tensor | 2,9·10⁻⁶ | 8,7·10⁻⁴ (`ln1.bias`, stratul 0) | 8,9·10⁻⁴ | 8,6·10⁻⁴ |
| `emb_token` întreg | 1,4·10⁻⁶ | 4,7·10⁻⁴ | 5,1·10⁻⁴ | 4,7·10⁻⁴ |
| **Rândurile rare ale lui `emb_token`** | 2,4·10⁻⁷ | **4,8·10⁻⁴** | **7,8·10⁻³** | 4,8·10⁻⁴ |
| Mișcarea greutăților, cel mai rău tensor | 1,5·10⁻⁴ | 1,7·10⁻² (`out_proj`, stratul 0) | 1,9·10⁻² | 1,7·10⁻² |
| Mișcarea rândurilor rare | 7,1·10⁻⁶ | 2,8·10⁻³ | 1,6·10⁻² | 2,8·10⁻³ |

Față de predicții:
1. Martorul F32: eroarea identică, gradientul 2,9·10⁻⁶ (≤ 1·10⁻³) — **DA**.
2. S = 1 024: eroarea 7,6·10⁻⁶ (≤ 1·10⁻³), gradientul 8,7·10⁻⁴ (≤ 5·10⁻³), rândurile rare 4,8·10⁻⁴ (≤ 1·10⁻³; prezis ~2–5·10⁻⁴),
   mișcarea 1,7·10⁻² (≤ 5·10⁻²) — **DA**.
3. S = 1: rândurile rare **7,8·10⁻³** (în intervalul prezis 5·10⁻³ – 5·10⁻²; prezisesem ~1,8 % — a ieșit ~0,8 %, deci ceva mai bine),
   **de 16,0 ori mai rău** decât S = 1 024 (prezis ≥ 10); pe tot `emb_token` raportul e **1,07** (prezis < 3: măsura pe tensor întreg
   ascunde efectul) — **DA**.
4. AdamW micșorează efectul: mișcarea rândurilor rare S = 1 / S = 1 024 = **5,7** (prezis ≤ 10, mai puțin decât cele 16 de la gradient) — **DA**.
5. Atenția `_3`: aceleași cifre ca așezarea simplă (diferențe în a treia cifră) — **DA**.

Greșelile plantate: norma comparată cu 1,0 în loc de S → mișcarea greutăților **0,55** (prinsă, de 11 ori peste 5·10⁻²); eroarea
rămâne bună (2,6·10⁻³ — doar eroarea nu ajunge ca probă, la fel ca în T169); o țintă trecută drept „rară” → refuzată.

**Ce înseamnă:** pe vocabularul lui Kronos **scalarea erorii e necesară** (fără ea, gradientul celor 16 319 simboluri rare pierde ~16× din
precizie, fiindcă d(logits) ≈ 10⁻⁶ cade în zona subnormală a f16), iar cu S = 1 024 pasul mixt rămâne lângă oracolul F32 la toate măsurile.
Asta corectează concluzia din T169 („pe acest set nu s-a dovedit necesară”), care era adevărată doar pentru modelul de 256 de simboluri.
Mișcarea greutăților de ~1,7 % față de oracol după 5 pași e mai mare decât la F32 (1,5·10⁻⁴) — normal pentru f16: AdamW (m/√v) mărește
diferențele mici ale gradienților mici; de urmărit pe o antrenare lungă prin curba erorii, nu prin greutăți bit cu bit.

Ce urmează pe placă (în ordine, fiecare test separat, prin OS): `--antrenare-mini` F32 (C-161, al OS-ului) → proba `mm_tc` (os-352,
GATA DE TEST) → **`--antrenare-mini --mixt`** (nota pentru OS `integration/cyber-universe/C-160_P7f_antrenare-mini-mixt.md`).

`tools/check.sh`: vezi `docs/IMPLEMENTATION_STATUS.md`, sesiunea 191.

## Surse
- Scalarea erorii: Micikevicius et al., „Mixed Precision Training”, ICLR 2018 (idee). Formatul f16 și zona subnormală: IEEE 754-2008.
  Invarianța AdamW la o scară constantă a gradientului: Kingma & Ba, „Adam”, 2015. Codul e al nostru.
