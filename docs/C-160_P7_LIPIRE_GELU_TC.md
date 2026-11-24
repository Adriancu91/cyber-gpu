# C-157 p.2 — GELU' lipit și în pasul MIXT (`mm_tc_gelu`) — plan + predicții, scrise ÎNAINTE de cod

**Sesiunea 199, 24 nov 2026, cloud, fără placă.** Continuarea lui `docs/C-160_P7_LIPIRE_GELU.md` (sesiunile 192–193): acolo lipirea
s-a făcut pe `mm_train` (pasul F32), iar `step_mixed` refuză `with_fused_gelu`. Aici: aceeași lipire pe `mm_tc` (circuitele
speciale). Tot ce e aici e DEDUS până la secțiunea „Rezultatul”.

## 1. Ce se lipește

În pasul mixt, pe strat, azi:
1. `mm_tc` „inapoi:proj-dX”: `dwide = dx · W_proj` (intrări rotunjite la f16, sume f32), scris în memorie;
2. `gelu_tr` mod 1 „inapoi:gelu”: citește `u` și `dwide`, scrie `dwide · (Φ(u) + u·φ(u))`.

Lipit: un program nou **`mm_tc_gelu`** = `mm_tc` neschimbat (aceleași felii f16, aceleași `coopMatMulAdd`, aceeași placă 64×64 în
memoria partajată) care, la scrierea finală, în loc de `v (+ bias)` citește `u[rând, coloană]` și scrie `v · fma(u, φ, Φ)` cu exact
operațiile lui `gelu_tr` mod 1 (metoda din sesiunea 193: `fma(u, pdf, cdf)` scris explicit). `u` stă în locul lui `bias` (0x718),
fanionul BIAS e refuzat. `mm_tc` și `gelu_tr` rămân neschimbate (alți octeți ar cere o nouă probă pe placă).

În bibliotecă (doar adăugări): modulul `nvidia_ga104::mm_tc_gelu` (cod, SHA-256, registre, `Args`), varianta nouă
`MixedLaunch::TcGelu`, `pack_mixed_gelu(…, tc_va, tc_gelu_va, …)`; `step_mixed` pe un plan `with_fused_gelu` dă `TcGelu` în loc
de refuz; `pack_mixed` (cel de azi) refuză `TcGelu` cu mesaj clar. În simulator: `gpu_sim::cmdproc::mm_tc_gelu_on_cpu` = modelul
`mm_tc` + `gelu_tr_gpu_back` pe fiecare element.

## 2. Predicțiile

- **H1 (biți):** pasul mixt (S = 1 024) cu `mm_tc_gelu` = pasul mixt fără, **bit cu bit** (pierderi, gradientul pasului 1, greutățile
  de la sfârșit): forma `antrenare mini` 5 pași AdamW, cu și fără `ln_back_res`.
- **H2 (listarea — riscul principal):** `mm_tc` vine din a doua unealtă (Mesa 26.3.1), `gelu_tr` din prima (Mesa 25.0.7). Partea
  GELU din `mm_tc_gelu` trebuie să aibă **aceleași 23 de operații** ca `gelu_tr` mod 1 (un `MUFU.EX2`, un `MUFU.RCP`, aceleași
  constante, `ffma`(u, φ, Φ) apoi `fmul`(acc, ·)), iar partea de înmulțire aceleași instrucțiuni ca `mm_tc` (aceleași `HMMA`, în
  aceeași ordine). Dacă NAK 26.3.1 contopește altfel: încerc `precise` doar unde trebuie (ca la `add_exact`); dacă nici așa, renunț
  și `step_mixed` rămâne cu refuzul.
- **H3 (lansări):** L lansări mai puține pe pasul mixt (`antrenare mini`: 2); `mm_tc` + `mm_tc_gelu` = câte `mm_tc` erau înainte.
- **H4 (greșeli plantate):** `u` al altui strat → alți biți; epilogul „fără `ffma`” (Φ + u·φ rotunjit de două ori) → alți biți;
  `pack_mixed` vechi pe o listă cu `TcGelu` → refuzat, nimic scris.
- **H5 (resurse):** registre ≤ 72 (`mm_tc`: 64), memoria partajată ca `mm_tc` (24 KiB), aceeași grilă. Câștigul (DEDUS, ca G5): la
  #513 ~1,8 GB citiri/scrieri și 12 lansări mai puțin pe bucată, ~4–5 ms — **se vede doar pe placă**.

## 3. Ce e nou pentru placă (dacă H1–H4 ies)

Un singur program, `mm_tc_gelu`, după ce `--antrenare-mini --mixt` (os-384) a trecut pe placă cu `mm_tc` și după `mm_train_gelu`.
Nimic în registre, nimic prin GSP.
