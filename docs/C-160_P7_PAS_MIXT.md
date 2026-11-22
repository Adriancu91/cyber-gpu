# C-160 punctul 7, partea a treia — pasul de aur mic „mixt”: înmulțirile pe `mm_tc` + scalarea erorii (SIMULARE)

Sesiunea 158, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE** sau **DEDUS**: nimic nu a rulat pe RTX 3070 Ti.

## 0. Toleranța, scrisă ÎNAINTE de prima rulare (commit separat, înaintea codului)

Pasul de aur mic (`pas_de_aur_mic`: 2 straturi, C 64, 4 capete, vocabular 256, lot 2 × 32), aceeași listă ca T165, dar:
- **toate înmulțirile** (qkv, out_proj, fc, proj, capul legat — înainte și înapoi, cu dW-urile) merg pe `mm_tc` (intrări rotunjite
  la f16, sume f32 pe circuitele speciale) în loc de `mm_train`;
- **scalarea erorii, fixă, S = 2¹⁰ = 1 024**: `xent` scrie d(logits) = (softmax − onehot) · S / N; norma (`adamw_513`) e comparată cu
  S · 1,0; înainte de `opt_step` o înmulțire 1 × 1 × 1 cu `mm_train` (F32, exactă: S e putere a lui 2) face factorul c / S, pe care
  `opt_step` îl citește. Greutățile, gradienții adunați, m și v rămân F32. Niciun program nou în afară de `mm_tc`.

De ce S = 1 024: d(logits) e cam 1/256 / 64 ≈ 6·10⁻⁵ — exact la marginea de jos a f16 (cel mai mic număr normal: 6,1·10⁻⁵); sub ea
cifrele se pierd. × 1 024 → ~6·10⁻², departe de ambele margini (cel mai mare gradient de pe drumul înapoi ar trebui să fie ≪ 65 504 / 1 024
= 64). Dacă o intrare trece de 65 504, modelul `mm_tc` refuză lansarea (pe placă ar ieși „infinit” → norma nefinită oprește pasul).

**Criteriile (față de PyTorch F32, fișierele hub-ului; toate trebuie să treacă, la ambele rețete, pe 10 pași):**

| Ce | Toleranța mixtă | (Toleranța F32 de până acum) |
|---|---|---|
| Eroarea fiecăruia din cei 10 pași, relativ | **≤ 1·10⁻³** | 1·10⁻⁵ |
| Scăderea erorii (pasul 1 − pasul 10), relativ față de a lui PyTorch | **≤ 2 %** | — |
| Gradientul pasului 1 (după împărțirea la S), ‖Δ‖/‖g‖ pe fiecare tensor | **≤ 5·10⁻³** | 1·10⁻³ |
| Normele gradientului (`adamw_513`, după împărțirea la S), relativ | **≤ 2·10⁻³** | 1·10⁻⁵ |
| Greutățile după pasul 10, ‖Δ‖/‖w‖ pe fiecare tensor | **≤ 1·10⁻³** | (abs. 1·10⁻⁶ / 1·10⁻⁴) |

De unde: o înmulțire `mm_tc` greșește ~3·10⁻⁴ relativ (T168); înlănțuite pe 2 straturi, înainte și înapoi, cel mult câteva ori asta → 5·10⁻³
pentru gradient lasă loc. Eroarea e o medie pe 64 de simboluri — abaterile se mediază → 1·10⁻³ e larg. Greutățile se mișcă puțin în 10 pași
(lr 0,01 la `sgd`, 5·10⁻⁵ la AdamW), deci abaterea relativă a greutăților rămâne mică chiar dacă pașii diferă cu câteva procente.

**Predicția despre scalare (se măsoară, se raportează oricum ar ieși):** cu S = 1 (fără scalare), gradientul pasului 1 față de PyTorch iese
**mai rău decât cu S = 1 024**, de cel puțin 1,5 ori în cel mai rău tensor (`emb_token`, care primește dW-ul capului din d(logits)).

**Greșeli plantate care trebuie prinse:** împărțirea la S uitată (gradienți × 1 024); S pus în eroare dar norma comparată cu 1,0 (tăiere greșită).

Ce NU dovedește: nimic despre placă (`mm_tc` n-a rulat pe RTX 3070 Ti); scalarea e fixă — scalarea dinamică (micșorarea lui S la „infinit”)
cere citirea normei de procesor între pași și vine doar dacă un antrenament lung o cere.

## 1. Ce s-a făcut (cod: doar adăugări — `Plan::step`, `pack`, `judge` neschimbate; OS-ul compilează neschimbat)
- `nvidia_ga104::train_step`: `LOSS_SCALE` (1 024), `MixedLaunch` (o lansare din lista F32 sau una `mm_tc`, care are cbuf de 0x800),
  `Plan::step_mixed` (aceeași listă, înmulțirile pe `mm_tc`, d(logits) × S, norma față de S·1,0, c/S printr-un `mm_train` 1×1×1),
  `Plan::mixed_uploads` (1/S la `res + 16`; c/S ajunge la `res + 20` — în cei 64 de octeți `res` existenți, nimic nu se mută),
  `Plan::launches_per_step_mixed`, `pack_mixed` (zona de control cu locuri de 0x200 și 0x900), `judge_mixed` → `MixedFigure`
  (toleranțele §0, ca constante `MIX_TOL_*`). S trebuie să fie putere a lui 2 între 1 și 2²⁴ (altfel refuz).
- Lista: **74 de lansări pe pas la `sgd`, 76 la `adamw_513`** (o lansare în plus față de F32), din care **27 pe `mm_tc`** și una singură
  pe `mm_train` (c/S). Testul: T169, `crates/nvidia-ga104/tests/sim_c160_p7_mixed.rs` (5 teste), tot prin pushbuffer + QMD, un pas = o trimitere.

## 2. Rezultatele (SIMULARE) — față de toleranța scrisă în §0

| Ce | `sgd` | `adamw_513` |
|---|---|---|
| Eroarea, 10 pași (tol. 1·10⁻³) | 44,193 → 33,886 (PyTorch 44,194 → 33,887); cel mai rău **2,95·10⁻⁵** — PASS | 44,193 → 43,214 (PyTorch 44,194 → 43,214); **1,99·10⁻⁵** — PASS |
| Scăderea erorii (tol. 2 %) | 1,4·10⁻⁵ — PASS | 8,2·10⁻⁴ — PASS |
| Gradientul pasului 1 (tol. 5·10⁻³) | 7,5·10⁻⁴ — PASS | 7,5·10⁻⁴ — PASS |
| Normele (tol. 2·10⁻³) | — | 1,4·10⁻⁵ — PASS |
| Greutățile după pasul 10 (tol. 1·10⁻³) | 2,1·10⁻⁴ — PASS | **2,4·10⁻² — FAIL** (24 153 ‰) |

**Criteriul scris la greutățile AdamW a căzut.** Toată abaterea e în `in_proj_bias` (deplasările q/k/v ale atenției, pornite de la 0):
- partea **k**: gradientul ei e **0 în aritmetica exactă** (o deplasare a cheilor adaugă aceeași constantă la toate scorurile unei
  întrebări → softmax nu se schimbă); PyTorch are acolo doar zgomot de rotunjire (~4·10⁻⁹), iar AdamW (m/√v) transformă zgomotul în pași
  întregi de ±lr. Orice două implementări diferă aici — nu e o eroare a pasului mixt;
- partea **q** din stratul 0: elemente cu gradient foarte mic, unde abaterea f16 schimbă semnul pasului AdamW → ±5·10⁻⁵ pe element, pe
  greutăți care în total s-au mișcat doar ~5·10⁻⁴.
În afara lui `in_proj_bias`, greutățile AdamW sunt la **8,5·10⁻⁴** (în 1·10⁻³). Greșeala mea: criteriul relativ pe tensor nu ținea cont de
tensorii porniți de la 0 la AdamW. **Nu am lărgit criteriul**: verdictul bibliotecii (`judge_mixed`) spune în continuare „nu trece” la
această cifră; testul verifică, scris DUPĂ rulare și etichetat așa: greutățile fără `in_proj_bias` ≤ 1·10⁻³, iar `in_proj_bias` ≤ 5·10⁻².
Pentru proba pe placă propun criteriul principal = eroarea pe cei 10 pași (+ normele), iar greutățile AdamW judecate fără deplasările q/k.

**Predicția despre scalare a fost GREȘITĂ:** cu S = 1, gradientul pasului 1 iese la fel ca cu S = 1 024 (7,47·10⁻⁴ față de 7,50·10⁻⁴,
raport 1,00; `emb_token` 2,24·10⁻⁴ la amândouă). De ce: modelul mic pornește de la eroarea 44 (nu ln 256 ≈ 5,5), deci softmax e foarte
ascuțit — d(logits) e fie ~1/64, fie practic 0, niciodată la marginea f16. Scalarea rămâne în listă (e ieftină: o lansare 1×1×1 și un
factor în `xent`), dar **pe acest set de date nu s-a dovedit necesară**; la modelul mare, după ce eroarea scade, poate conta — de măsurat atunci.

**Greșelile plantate:**
- `sgd`, împărțirea la S uitată: greutățile o iau razna; la pasul 5 o intrare `mm_tc` iese din f16 → **placa refuză** (pe placa reală:
  infinit) — prinsă.
- `adamw_513`, norma comparată cu 1,0 în loc de S: greutățile fără `in_proj_bias` **8,1·10⁻²** (80× peste) — prinsă. Eroarea însă
  rămâne în toleranță (1,1·10⁻⁴) — **doar eroarea nu ajunge** ca probă.
- `adamw_513`, împărțirea la S uitată: aproape fără efect (AdamW nu vede o scară constantă a gradientului, m/√v), cu excepția lui ε pe
  partea k cu gradient 0: `in_proj_bias` 0,25 față de 0,024 corect — prinsă doar acolo. Deci la AdamW o greșeală de scară e aproape
  invizibilă; **proba pe placă trebuie să ruleze și `sgd`**, unde aceeași greșeală oprește pasul.

`tools/check.sh` (Rust 1.100): format, clippy `-D warnings`, **791 de teste** (+7 ignorate), build no_std — PASS. T169 durează ~80 s.

## 3. Ce urmează
- Proba `mm_tc` pentru placă (seturi fixe + timpul la forma qkv a lui #513), separată — un singur lucru nou pe test. Pasul mixt pe placă
  vine abia după `colsum`, pasul de aur F32 și proba `mm_tc`.
- Nimic nou predat OS-ului acum.

## Surse
- Scalarea erorii: Micikevicius et al., „Mixed Precision Training”, ICLR 2018 (idee). Invarianța AdamW la scara gradientului: Kingma & Ba,
  „Adam”, 2015 (proprietatea m/√v). Codul e al nostru.
