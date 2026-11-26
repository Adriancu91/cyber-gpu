# C-157 — `attn_back_tile0_tc` legat în pasul mixt rapid (ambele programe „înapoi” ale atenției pe circuitele speciale)

Sesiunea 250, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**.
Nimic de aici nu a rulat pe RTX 3070 Ti; tot ce iese pe procesor e **SIMULARE**, timpii sunt **DEDUS**.

## 1. De ce

`attn_back_tile0_tc` (dP = dO·Vᵀ pe circuitele speciale, restul F32 ca azi) e gata și compilat din sesiunea 245
(`docs/C-157_ATTN_BACK0_TC.md`, T254: aceiași biți ca modelul, 56 de registre), dar **nu e legat**: pasul mixt rapid
(`pack_mixed_fast_attn`) știe să mute doar programul al doilea (`attn_back_tile_3` → `attn_back_tile_tc` / `tc16`).
`attn_back_tile0_3` e cel mai încet program pe FLOP (os-431: 5 182 µs pe pas, MĂSURAT). Aici fac legătura, la fel cum
sesiunea 244 a legat `attn_back_tile_tc` (T253), **doar prin adăugări**.

## 2. Ce va fi (plan)

- **`gpu-sim`:** `attn_back_tile0_tc_on_cpu` + `ATTN_BACK_TILE0_TC_NEEDS` (33 024 B, o barieră): citește argumentele la
  **0x700** (qkv / p / dout / ds la 0x700…0x718, T 0x728, C 0x72c, H 0x730, d 0x734, mod 0x738, scara 0x73c); mod ≠ 0 → nu
  scrie nimic; pe grilă [⌈T/64⌉, ≤ H, ≤ B] scrie rândurile dS = `gpu_compute::attn_tc::back0_model` (aceeași ipoteză a
  sumatorului din os-335); d > 64, grila peste H / T, d ≠ C/H sau o intrare care nu e f16 finit → refuz, nimic scris.
- **`train_step` (doar adăugări):** `MixedLaunch::Attn0Tc(Launch)` (o lansare `attn_back_tile0_3`, mod 0, mutată),
  `attn0_tc_list(list)` (mută toate lansările `attn_back_tile0_3`; refuză una pe care `attn_back_tile0_tc::cbuf_from_f32`
  ar refuza-o), `TcPick::Attn0Tc`, și `pack_mixed_fast_attn2(…, attn, attn0_va, …)` = `pack_mixed_fast_attn` + adresa
  programului nou (0 = neîncărcat). `pack_mixed_fast_attn` devine `pack_mixed_fast_attn2` cu `attn0_va = 0` (aceiași octeți).
  `pack_mixed` / `pack_mixed_gelu` / `pack_mixed_fast` refuză o listă cu `Attn0Tc`. QMD-ul: 56 de registre, 33 024 B
  partajați, o barieră, blocuri de 256, aceeași grilă; zona: cbuf 0x800 + QMD 0x100 (ca `AttnTc`).
- **Proba:** testul nou `crates/nvidia-ga104/tests/sim_c157_attn0_tc_mixed.rs` (T260), după modelul T253.

## 3. Predicțiile (scrise ÎNAINTE de cod)

- **F1 — înlocuitorul = modelul.** Pe ≥ 5 forme (T multiplu / nemultiplu de 64, d < 64, B > 1, H > 1, o grilă parțială)
  scrie exact biții lui `back0_model` în rândurile grilei și **nimic** în afara ei; mod 1 → nimic scris; d > 64 refuzat;
  o intrare V peste f16 → refuzat, nimic scris.
- **F2 — drumul pe lista mini (`new_tiles3`).** `attn0_tc_list` mută **exact o lansare pe strat** (2), aceleași etichete;
  pe o listă simplă (fără `_3`) e identitatea; zona crește cu 2 × (0x900 − 0x200) = 3 584 B; `pack_mixed_fast_attn2` cu
  `attn0_va = 0` și fără `Attn0Tc` = `pack_mixed_fast_attn` **octet cu octet**. Ambele mutări împreună: 2 `AttnTc` +
  2 `Attn0Tc`.
- **F3 — antrenarea mini în SIMULARE (5 pași, S = `LOSS_SCALE`), ambele programe pe circuite față de F32 (ambele `_3`).**
  Pierderile scad și rămân finite, biții **diferă**; abaterea pierderilor ≤ 1·10⁻⁴; gradientul pasului 1
  **2·10⁻⁴ … 8·10⁻⁴** (în T253, doar cu programul al doilea: 1,9·10⁻⁴; la capul #513 dQ / dK au crescut ×1,4 cu ambele);
  greutățile ≤ `MIX_TOL_W_REL_NORM` = 1·10⁻³. Față de rularea doar cu `AttnTc`: biți diferiți, gradientul ≤ 6·10⁻⁴.
- **F4 — lista #513 (lot 2, nelipită).** 12 `Attn0Tc` + 12 `AttnTc` + 147 `mm_tc_v4`; zona crește cu 24 × 1 792 =
  43 008 B față de `pack_mixed_fast`; QMD-ul `Attn0Tc`: adresa programului, **56** de registre, grila [8, 12, 2], cbuf cu
  T = 512 la 0x728 și mod 0 la 0x738.
- **F5 — greșelile se prind.** `Attn0Tc` cu `attn0_va = 0` → refuz; adresă nealiniată → refuz; `pack_mixed_fast_attn` /
  `pack_mixed_fast` / `pack_mixed_gelu` cu `Attn0Tc` → refuz; treapta 32 KiB (sub 33 024 + rezerva) → refuz; o lansare mod 1
  pusă ca `attn_back_tile0_3` → `attn0_tc_list` refuză; cbuf-ul lăsat la 0x28 (fără mutarea la 0x700) → programul simulat
  citește mod 0 la 0x738 = 0, dar T = 0 → **refuz** în înlocuitor (pasul nu trece).
- **F6 — câte blocuri pe SM la 64 KiB (DEDUS, `occupancy`).** `attn_back_tile0_tc` = **1** (memoria partajată, ca
  `attn_back_tile0_3`); registrele ar permite 4.
- **F7 — timpul (DEDUS, doar pe placă):** cel din sesiunea 245 (E7): 5,2 ms → ~1,8–3,2 ms pe pas #513. NOT TESTED.
