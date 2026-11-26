# C-157 — `attn_back_tile0_tc`: dP = dO·Vᵀ pe circuitele speciale (f16 × f16 → f32), restul (dot, dS) în F32 ca azi

Sesiunea 245, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**.
Nimic de aici nu a rulat pe RTX 3070 Ti; tot ce iese pe procesor e **SIMULARE**, timpii sunt **DEDUS**.

## 1. De ce

Studiul din sesiunea 242 (`docs/C-157_ATENTIE_TC.md`) a pus `attn_back_tile0_3` (modul 0 al atenției înapoi: dP = dO·Vᵀ,
dot = Σ P·dP, dS = P·(dP − dot)·scale) pe locul doi: **5 182 µs** pe pas în os-431 (MĂSURAT), cel mai încet program pe FLOP
(1,05 TFLOP/s). `attn_back_tile_3` (dQ, dK, dV) a trecut deja pe circuitele speciale (sesiunile 243–244, PREDARE P2zm).
Aici trece doar **produsul** dP pe circuitele speciale; lanțul `dot` și trecerea a doua (dS) rămân F32, în aceeași ordine.

## 2. Ce va fi (plan)

- **Același contract ca `attn_back_tile0_3`:** aceleași argumente (`qkv, p, dout, ds, dqkv, T, C, H, d, mod, scale`), doar modul 0,
  grila [⌈T/64⌉, H, B], blocuri de 256 de fire, aceleași locuri scrise în `ds`. Compilat cu a doua unealtă (Mesa NAK 26.3.1), deci
  argumentele stau la **0x700** în cbuf 0 (lecția sesiunii 244) — se verifică din dump, nu se presupune.
- **Trecerea 1, pentru jt = 0 … bx:** placa dO [r][e] și placa V [j][e] (64 × 64) copiate ca f16 în memoria partajată (zero în afara
  lui T și d); cele 8 subgrupuri țin câte 2 acumulatori 16 × 16 ai plăcii dP [r][j] (rândurile 16·(w/2), coloanele 32·(w%2) + 0 / 16),
  4 pași de 16 pe e cu `coopMatMulAdd`; B = Vᵀ se încarcă din placa V cu ordinea pe coloane (fără copie transpusă).
  Acumulatorii trec în placa f32 dP din memoria partajată (aceeași așezare „sucită” `sk` ca azi), se scrie dP în `ds` pentru j ≤ i,
  se încarcă placa P și lanțul `dot` (firele 0 … 63, j crescător, `fma`) — **exact codul de azi**.
- **Trecerea 2** (dS = P·(dP − dot)·scale peste dP din `ds`, 0 deasupra diagonalei): **neschimbată**, copiată din `attn_back_tile0_3.comp`.
- **Memoria partajată:** plăcile f16 dO / V (2 × 8 KiB) și placa f32 dP (16 KiB) folosesc **același loc** (după o barieră), plus placa
  P 64 × 65 f32 → **33 024 B** (cât azi).
- **Modelul pe procesor** (`gpu_compute::attn_tc::back0_model` / `back0_dp`): dP al fiecărui cap = `mm_tc::model` al produsului
  dO_h · V_hᵀ (K = d, 4 pași de 16 din zero, sumatorul potrivit în os-335 — IPOTEZĂ); apoi lanțul `dot` și dS cu aceleași funcții
  ca `layer::attn_back_row_gpu` (F32 de azi). **Judecata:** dP în marginea `mm_tc` (`mm_tc::judge` pe produsul fiecărui cap).

## 3. Predicțiile (scrise ÎNAINTE de cod)

- **E1 — aceiași biți acolo unde trebuie.** (a) dP din model = `mm_tc::model` pe produsul întreg dO_h·V_hᵀ, bit cu bit, pe ≥ 6 forme
  (T multiplu / nemultiplu de 64, d = 64 / d < 64 / d nemultiplu de 16, B > 1, H > 1). (b) Dacă în locul dP-ului f16 se pune dP-ul
  F32 de azi, coada (dot, dS) dă **exact biții** lui `layer::attn_back_gpu` — adică partea F32 nu s-a schimbat.
- **E2 — în margine.** dP e `InMargine` pe toate formele + un cap #513 (T = 512, d = 64, dO ~ 1·10⁻⁵ × 2¹⁴); cel mai rău < 300 ‰.
- **E3 — cât diferă de F32 de azi (capul #513).** dP: eroare relativă (normă) **2·10⁻⁴ … 1·10⁻³**. dS: **mai mare decât la dP**, din
  scăderea dP − dot (se pierd cifre când dP ≈ dot): **3·10⁻⁴ … 3·10⁻³**.
- **E4 — ambele programe înapoi pe circuitele speciale (capul #513).** dS din `back0_model`, apoi dQ, dK, dV din `attn_tc::model`,
  față de `attn_back_gpu` F32: dQ și dK **3·10⁻⁴ … 3·10⁻³** (peste B3 = 2,9–3,0·10⁻⁴, din dS-ul nou); dV **exact aceiași biți** ca
  în B3 (dV nu depinde de dS).
- **E5 — greșelile plantate se prind.** (a) NaN deasupra diagonalei în P → **niciun bit schimbat**; (b) V fără transpusă
  (dO·V în loc de dO·Vᵀ, pe d = 64 = T) → dP `NuTrece`; (c) masca uitată în lanțul `dot` (j > i adunat) → dS diferit (> 1 %);
  (d) dO fără scara 2¹⁴ (1·10⁻⁵): dP cade în subnormale f16, eroarea dS **≥ 1·10⁻²**.
- **E6 — programul (doar dacă unealta se construiește în sesiune).** 0 memorie locală, **≤ 64 registre**, `hmma.m16n8k16.f32` prezent,
  `ffma` doar în lanțul `dot` (nu în produs), memoria partajată ≤ 33 024 B, argumentele la 0x700..0x740.
- **E7 — timpul (DEDUS, doar pe placă):** 5,2 ms azi → **~1,8–3,2 ms** (traficul P / dS ~67 MiB pe strat rămâne: ~1,4–1,9 ms pe pas
  la 450–600 GB/s, plus produsul pe circuite ~0,3–1 ms) → câștig **~2–3,4 ms** (2–3 % din placa mixtă). Mai puțin decât studiul
  (3,5–4,2 ms), fiindcă studiul punea și trecerea a doua la viteza `mm_tc_v4`.

## 4. Rezultatul

(se completează după teste)
