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

## 4. Rezultatul

(se completează după cod)
