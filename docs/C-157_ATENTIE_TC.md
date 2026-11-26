# C-157 — atenția pe circuitele speciale (f16): cât ar lua din cele ~20,6 ms ale pasului mixt #513

Sesiunea 242, 26 nov 2026, în cloud, **fără placă**. Timpii plăcii sunt MĂSURAȚI de OS (os-431, os-443); tot ce e calculat aici
e **DEDUS** (aritmetică pe procesor). Proba de precizie rulează pe procesor (**SIMULARE**, nu placa). Nimic nu a rulat pe
RTX 3070 Ti în sesiunea asta și **nu se scrie niciun program nou pentru placă** — e un studiu, ca să știm dacă merită.

## 1. De ce acum

Sesiunea 241 (`docs/C-157_VALURI_V4.md`) a arătat că pe pasul mixt (placa 106 ms, os-445) înmulțirile sunt ~73 ms și restul ~33 ms,
din care **atenția F32 = 20,6 ms (19 % din placă)** — cea mai mare bucată în afara înmulțirilor. Înmulțirile au trecut deja pe
circuitele speciale (`mm_tc_v4`, ~8,2 TFLOP/s); atenția nu.

## 2. Ce avem (citit, nu presupus)

- Forma #513: lot B = 2, T = 512, 12 capete de d = 64, 12 straturi, atenție cauzală (doar sub diagonală), plăci de 64 × 64
  → pe fiecare cap 8 × 9 / 2 = **36 de perechi de plăci** din 64.
- Programele (gen 3, `--placi3`) și ce înmulțiri fac:
  - `attn_fwd_tile3`: S = Q·Kᵀ, softmax, O = P·V (P scris în memorie, F32) — **2 produse**;
  - `attn_back_tile0_3`: dP = dO·Vᵀ, apoi dS = P·(dP − dot)·scale — **1 produs**;
  - `attn_back_tile_3`: dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO — **3 produse**.
- os-431 (MĂSURAT, 12 lansări din fiecare): `attn_back_tile_3` **9 192 µs**, `attn_fwd_tile3` **6 207 µs**, `attn_back_tile0_3`
  **5 182 µs** → 20 581 µs (1 715 µs pe strat).
- `mm_tc_v4` (os-443, MĂSURAT): qkv-513 la 64 KiB = 2 623 680 ns, ~8,28 TFLOP/s. Înmulțirile F32 ale pasului (`mm_tv_128x128`,
  os-431): 223 234 µs pentru aceleași ~599 GFLOP ale pasului mixt → ~2,7 TFLOP/s (DEDUS).
- Rotunjirea f16 a procesorului (`gpu_compute::mm_tc::round_f16`) = cea a programului (`F2F.F16.F32`), cu subnormale.

## 3. Predicțiile (scrise ÎNAINTE de cod și de calcule)

- **A1 — cât lucrează azi.** Cele 6 produse ale atenției au ~2,7 GFLOP pe strat (~33 GFLOP pe pas); atenția F32 merge cu
  **~1,6 TFLOP/s** în medie — sub înmulțirile F32 ale pasului (~2,7) și de ~5 ori sub `mm_tc_v4`. Cel mai încet program pe FLOP e
  `attn_back_tile0_3` (un singur produs + trecerea a doua): **~1 TFLOP/s**.
- **A2 — memoria nu e frâna de azi.** P și dS scrise / citite în F32 (doar plăcile de sub diagonală) ≈ **100–130 MiB pe strat** →
  la 450–600 GB/s ≈ 0,2–0,3 ms pe strat, **sub 20 %** din cele 1,7 ms măsurate. Cu P / dS în f16: jumătate.
- **A3 — cât ar lua circuitele speciale.** Cu produsele la eficiența `mm_tc_v4` (un plafon de jos) și de până la 2 ori mai încet
  (produse cu K mic: 64 la S și dP, plăci scrise des), plus memoria P / dS, atenția pasului ar ajunge la **~5–10 ms** (de la 20,6) →
  câștig **~10–15 ms = ~10–14 % din placa pasului mixt**. Doar ducerea atenției F32 la eficiența înmulțirilor F32 (fără f16) ar lua
  cel mult ~8 ms → circuitele speciale câștigă de cel puțin ~1,3 ori mai mult.
- **A4 — precizia (SIMULARE, pe procesor, un cap #513 cu T = 512, d = 64, valori inventate dar de mărimea reală).** Intrările celor 6
  produse rotunjite la f16, sumele în F32, dO înmulțit cu scara **2¹⁴** (ca `mm_tc`): eroarea relativă (norma) față de calculul în
  f64 la O, dQ, dK, dV **≤ 1·10⁻³**. Fără scară, cu dO de mărimea ~1·10⁻⁵ (cât are pasul), dS cade în subnormale f16 și eroarea la
  dQ / dK urcă **≥ 1·10⁻²** → scara e obligatorie, ca la înmulțiri.
- **A5 — ordinea.** Primul program de trecut pe f16 ar fi `attn_back_tile_3` (cel mai mare, 9,2 ms; 3 produse cu K până la 512, cel
  mai aproape de forma `mm_tc_v4`); apoi `attn_fwd_tile3`; `attn_back_tile0_3` ultimul (un produs cu K = 64).

## 4. Ce se face în sesiune

Cod doar de adăugare, în `nvidia_ga104::attn_tc_study` (aritmetică, fără nimic pentru placă): numărul de perechi de plăci și FLOP-ii
celor 6 produse, timpii os-431, traficul P / dS, estimarea pe circuitele speciale (calibrată pe os-443) și proba de precizie f16 pe
procesor. Test T251 pe procesor (rândurile os-431 citite din `tests/rapoarte/os431-pas.txt`).
