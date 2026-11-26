# C-157 — `attn_back_tile_tc`: dQ, dK, dV pe circuitele speciale (f16 × f16 → f32)

Sesiunea 243, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**.
Nimic de aici nu a rulat pe RTX 3070 Ti; tot ce iese pe procesor e **SIMULARE**, timpii sunt **DEDUS**.

## 1. De ce

Studiul din sesiunea 242 (`docs/C-157_ATENTIE_TC.md`) a arătat că atenția pasului mixt #513 ia 20,6 ms din 106 (MĂSURAT os-431 / os-445)
și că pe circuitele speciale ar coborî la ~5,3–9,7 ms (DEDUS). Cel mai mare program e `attn_back_tile_3` (9,2 ms, 3 produse:
dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO); studiul l-a pus primul. Precizia e bună doar dacă gradienții sunt deja înmulțiți cu scara 2¹⁴ —
**și pasul mixt îi are deja așa** (`train_step`: `scale = loss_scale / n` pe d(logits), S = 2¹⁴), deci programul nou nu are nevoie
de nicio scară în plus: aceleași intrări ca azi.

## 2. Ce va fi (plan)

- **Același contract ca `attn_back_tile_3`:** aceleași argumente și același buffer constant (`attn_back::Args`, `Mode::Columns`),
  doar modul 1, aceeași grilă [⌈T/64⌉, H, B], blocuri de 256 de fire, aceleași locuri de scriere în dqkv. Pentru OS ar fi
  **un program de schimbat în listă, nimic altceva** (în spatele unei opțiuni, ca `--mm-tc`).
- **Lucrul:** blocul bx face rândurile t0 … t0+63 ale capului (b, h), 64 de coloane e. Cele 8 subgrupuri (8 × 32 de fire) țin câte
  2 acumulatori 16 × 16 (rândurile 16·(w/2), coloanele 32·(w%2) + 0 / 16), ca `mm_tc`.
  - dQ: pentru jt = 0 … bx: placa dS [t][j] (zero peste diagonală) și placa K [j][e] copiate în memoria partajată ca **f16**, apoi
    4 pași de 16 cu `coopMatMulAdd` (`HMMA.16816.F32`).
  - dK / dV (aceeași buclă, două treceri): pentru it = bx … ultima: placa dS (sau P) [i][t] și Q (sau dO) [i][e]; A = transpusa plăcii
    (`coopMatLoad` cu ordinea pe coloane), deci fără copie transpusă.
  - Pe placa diagonală, pașii de 16 care sunt numai zerouri pentru un subgrup se sar (dQ: k > rândurile lui; dK / dV: k < rândurile lui).
  - La final acumulatorii trec printr-o placă f32 64 × 64 din memoria partajată și se scriu cu aceleași gărzi ca azi (t < T, e < d).
- **Memoria partajată:** 2 plăci f16 de 64 × 64 (16 KiB) + placa f32 de ieșire (16 KiB) = **32 768 octeți** (azi 33 024), o barieră.
- **Modelul pe procesor** (`gpu_compute::attn_tc`): ordinea programului, fiecare pas de 16 = `mma::mma16_hw` cu sumatorul potrivit pe
  placă în os-335 (`GA104_OS335`), ca la `mm_tc`. **Judecata** = `mm_tc::reference` + `mm_tc::margin` pe produsele mascate ale fiecărui cap
  (marginea scrisă înainte de os-335, aceeași ca la înmulțiri).
- Compilat cu a doua unealtă NAK (Mesa 26.3.1, `tools/nak/build2.sh`), dacă unealta se construiește în sesiune; altfel compilarea
  rămâne pasul următor (modelul și judecata nu depind de ea).

## 3. Predicțiile (scrise ÎNAINTE de cod)

- **B1 — aceiași biți ca înmulțirea întreagă.** Modelul cu ordinea programului (plăcile de peste diagonală nesărite deloc, pașii de
  zerouri de pe diagonală săriți) dă **exact aceiași biți** ca `mm_tc::model` pe produsele întregi ale capului (dS mascat · K,
  dSᵀ · Q, Pᵀ · dO, K = T), pe ≥ 6 forme (T multiplu și nemultiplu de 64, d = 64 și d < 64, B > 1, H > 1): zerourile adunate la început
  sau la sfârșit nu schimbă suma în modelul sumatorului.
- **B2 — în margine.** Pe aceleași forme și pe un cap #513 (T = 512, d = 64, mărimile pasului mixt: P softmax, dO ~ 1·10⁻⁵ × 2¹⁴),
  fiecare element din dQ, dK, dV e **în marginea `mm_tc`** (verdict `InMargine`, cel mai rău sub 300 ‰ din margine).
- **B3 — cât diferă de F32 de azi.** Față de `attn_back_col_gpu` (programul F32 de azi), eroarea relativă (norma) pe capul #513 e
  **între 2·10⁻⁴ și 1·10⁻³** la dQ, dK, dV (studiul: 3,9–5,3·10⁻⁴ cu valori asemănătoare).
- **B4 — greșelile plantate se prind.** dS în loc de dSᵀ la dK (ordinea pe rânduri), P în loc de dS la dK, masca uitată pe diagonală
  (NaN peste diagonală în dS / P trebuie să nu schimbe niciun bit), un pas de 16 lipsă: fiecare iese din margine sau schimbă biții.
- **B5 — programul (doar dacă se compilează în sesiune).** 0 memorie locală, **≤ 64 de registre** (4 blocuri de 256 pe SM, față de
  3 azi), `hmma.m16n8k16.f32` prezent (2 pe fiecare `coopMatMulAdd` 16 × 16), fără `ffma` în buclele produselor.
- **B6 — timpul (DEDUS, din studiu, neschimbat):** 9,2 ms azi → **~2,4–4,5 ms**, câștig ~4,7–6,8 ms pe pas (4–6 % din placa mixtă).
  Se vede doar pe placă.

## 4. Rezultatul (sesiunea 243, ~06:10; T252 pe procesor = SIMULARE; programul = COMPILAT, NU RULAT)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| B1 | modelul cu ordinea programului = `mm_tc::model` pe produsele întregi, bit cu bit, ≥ 6 forme | **6 forme, 228 936 de elemente, toate aceiași biți** (și cu pașii de zerouri săriți, și fără) | CONFIRMATĂ |
| B2 | totul în marginea `mm_tc`, cel mai rău < 300 ‰ | `InMargine` pe 7 forme (cele 6 + capul #513); cel mai rău **84 ‰** (T = 17), la #513 **21 ‰** | CONFIRMATĂ |
| B3 | față de F32 de azi, capul #513: 2·10⁻⁴ … 1·10⁻³ | dQ **2,99·10⁻⁴**, dK **2,92·10⁻⁴**, dV **2,80·10⁻⁴** (toate formele: 2,7–3,5·10⁻⁴) | CONFIRMATĂ |
| B4 | greșelile plantate se prind; NaN peste diagonală nu schimbă nimic | NaN în dS / P peste diagonală → **aceiași biți**; masca uitată → modelul refuză (pe placă ar ieși NaN); dK fără transpusă, P în loc de dS la dK, ultimul pas de 16 lipsă → toate **`NuTrece`**; dQ și dV neatinse de greșelile din dK | CONFIRMATĂ |
| B5 | 0 memorie locală, ≤ 64 registre → 4 blocuri pe SM (azi 3), `hmma` 2 pe `coopMatMulAdd`, fără `ffma` | **48 registre**, 0 memorie locală, 0 vărsări, **4 × `hmma.m16n8k16.f32`** (2 `coopMatMulAdd` × 2), **0 `ffma`**, 786 instrucțiuni, 12 576 octeți, `Shared size: 32768`. **Dar** memoria partajată (32 KiB × 4 = 128 KiB > 100 KiB pe SM) ține tot **3 blocuri** pe SM, ca azi; registrele ar permite 5 | PE JUMĂTATE |
| B6 | 9,2 ms → ~2,4–4,5 ms | — (doar pe placă) | NOT TESTED |

Unealta: Mesa 26.3.1 construită din nou în cloud; `mm_tc_v4` a ieșit **identic** (`488ca341…`), deci aceeași unealtă ca la os-443.
SHA-256 `attn_back_tile_tc.sm86.sass` = `527d8dd1dc5000c26541561e8c60f59f4b5dab0d034a4d5ee593ae415aed5c65`.

Cod (doar adăugări): `gpu_compute::attn_tc` (`model`, `model_with`, `Variant`, `head_products`, `head_values`, `judge`),
`nvidia_ga104::attn_back_tile_tc` (octeții, suma, registrele, memoria partajată, grila, `shape_ok`), `tools/nak/attn_back_tile_tc.comp`,
testul `crates/gpu-compute/tests/c157_attn_back_tc.rs` (T252a–c) și 2 teste în modul (suma, ce spune dump-ul).

### Ce înseamnă (pe scurt)

1. **Programul există și se compilează** pentru placă: aceleași argumente ca `attn_back_tile_3`, deci pentru OS ar fi doar alt program
   în listă. Pe procesor (SIMULARE) dă dQ, dK, dV cu o abatere de ~3·10⁻⁴ față de F32 — cât costă f16 însăși, ca la înmulțiri.
2. **Se lucrează doar cu circuitele speciale** în bucle (0 `ffma`), cu 48 de registre în loc de 80.
3. **Ce am greșit în predicție:** memoria partajată, nu registrele, limitează câte blocuri stau pe SM (3, ca azi). Remediul e mic:
   placa f32 de ieșire poate folosi locul celor două plăci f16 (16 KiB în total) → până la 5 blocuri pe SM.

### Ce NU e dovedit / ce urmează

- Nimic nu a rulat pe placă; timpul (B6) se vede doar acolo. Modelul presupune sumatorul potrivit în os-335.
- Încă **nu e legat** în pasul de antrenare (`train_step`) și nu are înlocuitor în `gpu-sim`: fără asta OS-ul nu-l poate încerca.
  Pașii următori, în ordine: (1) varianta cu 16 KiB memorie partajată; (2) înlocuitorul din `gpu-sim` + opțiunea în pasul mixt;
  (3) PREDARE către OS cu rândul de test.

**Ce e nou pentru placă:** un program nou compilat (`attn_back_tile_tc`), încă netrimis OS-ului. **Nicio schimbare incompatibilă.**

## 5. Sesiunea 244 — înlocuitorul din `gpu-sim` + opțiunea în pasul mixt (plan și predicții scrise ÎNAINTE de teste)

26 nov 2026, ~06:40, în cloud, fără placă. Tot ce urmează e **SIMULARE** (pe procesor) sau **DEDUS** (aritmetică).

### 5.1 O greșeală găsită în nota de mai sus (sesiunea 243)

Nota spunea „același buffer constant ca `attn_back_tile_3`”. **Nu e adevărat.** Programul e compilat cu a doua unealtă
(Mesa 26.3.1), care pune argumentele la **0x700** în cbuf 0 (ca la `mm_tc`), nu la 0x28. Dump-ul o arată:
`ldc c[0x0][+0x700]` … `c[0x0][+0x738]` (qkv, p, dout, ds, dqkv, T, C, H, d, mod). Dacă OS-ul i-ar fi dat buffer-ul
lui `attn_back_tile_3`, programul ar fi citit modul 0 la 0x738 și ar fi ieșit fără să scrie nimic (dQ, dK, dV vechi /
zero → antrenare greșită, fără nicio eroare). Nimic nu fusese predat, deci nu s-a stricat nimic.

Reparat (doar în `attn_back_tile_tc`, care nu era folosit de nimeni în afara acestui repo): `PUSH_BASE = 0x700`,
`CBUF_BYTES = 0x800` (era greșit 0x68), `cbuf(&Args)` și `cbuf_from_f32(&[u8])` (mută octeții 0x28..0x68 la 0x700..0x740).
Un test nou citește toate locurile `c[0x0]` din dump și verifică că sunt doar 0x0/0x4/0x8 sau 0x700..0x740.

### 5.2 Ce s-a adăugat (doar adăugări în interfața folosită de OS)

- `gpu_sim::cmdproc::attn_back_tile_tc_on_cpu` + `ATTN_BACK_TILE_TC_NEEDS` (32 768 B, 1 barieră): citește la 0x700,
  calculează cu `gpu_compute::attn_tc::model` (f16, sumatorul potrivit în os-335 — IPOTEZĂ), refuză ce nu modelează.
- `train_step::MixedLaunch::AttnTc(Launch)` (variantă nouă; OS-ul folosește `MixedLaunch` doar cu `matches!`, deci nu se
  strică nimic), `train_step::attn_tc_list` (mută lansările `attn_back_tile_3` ale listei mixte pe `attn_back_tile_tc`),
  `train_step::pack_mixed_fast_attn` (ca `pack_mixed_fast`, plus adresa lui `attn_back_tile_tc`), `TcPick::AttnTc`,
  `batch_ram::pack_mixed_fast_via_ram_attn`. `pack_mixed_fast` / `pack_mixed_fast_via_ram` rămân aceiași octeți.

### 5.3 Predicțiile (T253, scrise înainte să rulez testele)

- **C1 — înlocuitorul = modelul.** Pe un cap mic și pe forme cu grilă parțială, ce scrie `attn_back_tile_tc_on_cpu` =
  `attn_tc::model` bit cu bit (doar rândurile și capetele din grilă); modul 0 nu scrie nimic; d > 64 și o intrare
  peste f16 (1·10⁶ în dS) sunt refuzate.
- **C2 — drumul în pasul mixt (mini, `new_tiles3`, 2 straturi).** `attn_tc_list` mută exact 2 lansări pe pas (una pe
  strat); zona de control crește cu 2 × (0x900 − 0x200) = 3 584 B; același număr de QMD-uri. Cu `attn_tc_va = 0` și
  fără `AttnTc`, `pack_mixed_fast_attn` = `pack_mixed_fast` octet cu octet.
- **C3 — antrenarea mini în SIMULARE (5 pași, S = `LOSS_SCALE`).** Pierderile rămân finite și scad; biții **diferă**
  de pasul mixt cu atenția F32 (f16 în dQ, dK, dV), dar puțin: eroarea relativă a pierderilor ≤ 1·10⁻⁴ pe pas, gradientul
  pasului 1 (normă relativă) între 1·10⁻⁵ și 2·10⁻³, greutățile după pasul 5 ≤ 1·10⁻⁴.
- **C4 — lista #513 (lot 2, nelipită).** 12 lansări `AttnTc` + 147 `mm_tc_v4`; zona crește cu 12 × 1 792 = 21 504 B;
  QMD-ul fiecărei `AttnTc`: adresa programului = `attn_tc_va`, 48 de registre, cbuf 0x800, aceeași grilă [8, 12, 2].
- **C5 — greșelile se prind.** `attn_tc_va = 0` cu `AttnTc` → refuz; `pack_mixed_gelu` / `pack_mixed_fast` cu `AttnTc` →
  refuz; treapta 16 KiB (sub 32 KiB) → refuz; un cbuf pus greșit la 0x28 (cum spunea nota veche) → placa simulată nu
  scrie dQ, dK, dV → gradientul pasului 1 diferă vizibil (> 1 %).
- **C6 — câte blocuri pe SM la treapta 64 KiB (DEDUS, `occupancy`).** Pasul mixt rapid merge la treapta 64 KiB
  (os-431): `attn_back_tile_3` (33 024 B) → **1 bloc** pe SM (ambele citiri ale rezervei de 1 KiB);
  `attn_back_tile_tc` (32 768 B) → 2 blocuri fără rezervă, 1 cu rezervă. Deci nota veche („3 blocuri, ca azi”) era
  la 100 KiB, treaptă pe care nu o folosim; varianta cu 16 KiB ar da 3–4 blocuri la 64 KiB.

### 5.4 Rezultatul C1–C6 (T253, SIMULARE; C6 DEDUS)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| C1 | înlocuitorul = `attn_tc::model` bit cu bit; modul 0 nimic; d > 64 și f16 depășit refuzate | **135 864 de elemente** pe 5 forme (și grilă parțială) aceiași biți, nimic scris în afara grilei; modul 0 nimic; ambele refuzate | CONFIRMATĂ |
| C2 | 2 lansări mutate, zona + 3 584 B, fără `AttnTc` aceiași octeți | 2 `AttnTc`, zona 87 296 → 90 880 B (+3 584), aceiași octeți | CONFIRMATĂ |
| C3 | pierderi ≤ 1·10⁻⁴, gradient 1·10⁻⁵ … 2·10⁻³, greutăți ≤ 1·10⁻⁴, biți diferiți | pierderea 9,7191 → 8,8299 (scade); pierderi **1,2·10⁻⁶**, gradient **1,9·10⁻⁴**, greutăți **3,8·10⁻⁴**; biți diferiți | **PE JUMĂTATE** — greutățile au ieșit de ~4× peste predicție (AdamW face din diferențe mici pe gradienți aproape de zero pași întregi de lr); rămân sub toleranța pasului mixt față de PyTorch (1·10⁻³) |
| C4 | 12 `AttnTc` + 147 `mm_tc_v4`, zona + 21 504 B, QMD: adresa, 48 registre, grila [8, 12, 2] | exact așa | CONFIRMATĂ |
| C5 | refuzurile; cbuf la 0x28 → gradient greșit > 1 % | toate refuzurile; cbuf la 0x28 → programul simulat nu scrie nimic, iar **pasul e refuzat** (lansarea următoare citește memorie nescrisă) — prins mai devreme decât prezisesem | CONFIRMATĂ (altfel decât prezis) |
| C6 | la 64 KiB: `attn_back_tile_3` 1 / 1, `attn_back_tile_tc` 2 / 1, 16 KiB → 3–4 | (1, 1), (2, 1), **(4, 3)** | CONFIRMATĂ |

## 6. `attn_back_tile_tc16` — aceeași aritmetică în 16 KiB (sesiunea 244)

**Ce.** `tools/nak/attn_back_tile_tc16.comp`: plăcile f16 sa / sb stau câte două într-un cuvânt de 32 de biți
(`packFloat2x16`, rotunjire la cel mai apropiat par — dump-ul: `f2fp.pack_ab` fără sufix = RNE, verificat în sursa NAK),
iar placa f32 de ieșire se scrie **peste ele** după o barieră (toate subgrupurile au terminat produsele). Încărcarea
plăcilor în circuite (`coopMatLoad` din tabloul de cuvinte): pasul și deplasarea sunt în cuvinte (sursa NAK,
`nak_nir_lower_cmat.c`: „stride is in elements of the pointed to type”), adică indicele f16 / 2. Nimic altceva nu se schimbă.

**Compilat** (Mesa NAK 26.3.1, aceeași unealtă — `attn_back_tile_tc` și `mm_tc_v4` ies identic): **48 registre, 16 384 B
partajată**, 0 memorie locală, 0 vărsări, 4 `hmma`, aceleași `ldsm` / `movm` ca programul de 32 KiB, 0 `ffma`, 785 instrucțiuni.
SHA-256 `attn_back_tile_tc16.sm86.sass` = `8cbbab7ed17988f5c6030fbb3ffcee7163f24a47a670a9ae4ae91b7be8ee5286`.
(Compilarea s-a făcut înainte să scriu predicțiile de mai jos; ele privesc doar testele.)

**Predicții (scrise înainte de teste):**
- **D1** — în SIMULARE, pasul mixt mini cu `attn_back_tile_tc16` = cu `attn_back_tile_tc`, bit cu bit (același model —
  verifică doar drumul: alegerea, QMD-ul, refuzurile).
- **D2** — lista #513: 12 `AttnTc16`, QMD cu 16 384 B partajată și 48 registre; același cbuf ca la 32 KiB.
- **D3** — cu lotul prin RAM: `pack_mixed_fast_via_ram_attn(…, None, …)` = `pack_mixed_fast_via_ram` octet cu octet.
- **D4 (doar pe placă)** — `attn_back_tile_tc16` dă **aceleași cuvinte** ca `attn_back_tile_tc` pe orice intrare.
  Timpul: 1 → 3–4 blocuri pe SM la treapta 64 KiB (DEDUS); câștigul față de 9,2 ms (`attn_back_tile_3`) rămâne de măsurat.

**Rezultat D1–D3 (T253d, SIMULARE):** D1 mini cu `tc16` = cu `tc` bit cu bit (pierderi, gradient, greutăți) — CONFIRMATĂ (verifică
drumul, nu placa); D2 #513: 12 `attn_back_tile_tc16`, QMD 16 384 B / 48 registre, același cbuf ca la 32 KiB, aceeași zonă — CONFIRMATĂ;
D3 prin RAM: `None` = packer-ul vechi octet cu octet, `tc16` ales, lista cu `AttnTc` refuzată de packer-ul vechi — CONFIRMATĂ.
D4 (aceleași cuvinte pe placă, timpul) — NOT TESTED. Verificările întregi: 1 238 PASS (8 ignorate).

**PREDARE** către OS: `integration/cyber-universe/C-157_P2zm_atentia-tc.md` (proba „vechi lângă nou” întâi, apoi opțiunea în pasul mixt).
