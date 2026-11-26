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

## 4. Rezultatul (sesiunea 245, ~08:00; T254 pe procesor = SIMULARE; programul = COMPILAT, NU RULAT)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| E1 | (a) dP în ordinea programului = `mm_tc::model` pe produsul întreg, bit cu bit, ≥ 6 forme; (b) coada F32 cu dP-ul de azi = dS-ul de azi | (a) **6 forme, 136 579 de elemente** (plăcile calculate) aceiași biți; plăcile de deasupra diagonalei nu se calculează (0); (b) **aceiași biți** pe toate cele 6 forme (184 963 de elemente dS) | CONFIRMATĂ |
| E2 | dP `InMargine`, cel mai rău < 300 ‰ | `InMargine` pe 7 forme; cel mai rău **74 ‰** (T = 17), la #513 **32 ‰** | CONFIRMATĂ |
| E3 | capul #513: dP 2·10⁻⁴ … 1·10⁻³; dS **mai mare** decât dP, 3·10⁻⁴ … 3·10⁻³ | dP **2,98·10⁻⁴**, dS **3,01·10⁻⁴** (celelalte forme: dP 2,9–3,1·10⁻⁴, dS 2,8–3,4·10⁻⁴) | **PE JUMĂTATE** — intervalele țin, dar dS ≈ dP: scăderea dP − dot nu strică precizia (dot e mic față de dP pe aceste valori) |
| E4 | ambele programe înapoi pe circuite: dQ, dK 3·10⁻⁴ … 3·10⁻³; dV = B3 bit cu bit | dQ **4,25·10⁻⁴**, dK **4,19·10⁻⁴** (în B3, doar cu `attn_back_tile_tc`: 2,99 / 2,92·10⁻⁴); dV **aceiași biți** ca B3 (2,80·10⁻⁴) pe toate formele | CONFIRMATĂ |
| E5 | NaN peste diagonală în P → niciun bit; V fără transpusă → `NuTrece`; masca uitată în `dot` → dS diferit; dO fără scară → dS ≥ 1·10⁻² | NaN → aceiași biți; V fără transpusă → `NuTrece` (primul element); masca uitată → dS NaN; **fără scară: 2,95·10⁻³** (cu scară 3,0·10⁻⁴) | **PE JUMĂTATE** — (d) greșită: dO ~ 1·10⁻⁵ încă păstrează ~7 biți ca subnormal f16, deci eroarea crește doar de ~10 ori, nu până la 1 %. Scara rămâne necesară (de 10 ori mai bine), dar nu e „obligatorie” ca la dQ / dK (unde fără scară ieșea 17 %) |
| E6 | 0 memorie locală, ≤ 64 registre, `hmma` prezent, `ffma` doar în `dot`, partajată ≤ 33 024 B, argumentele la 0x700 | **56 registre** (azi 68), 0 memorie locală, 0 vărsări, **16 × `hmma.m16n8k16.f32`** (bucla de 4 pași desfăcută), **8 `ffma.ftz`, toate în registrul lui `dot`**, `Shared size: 33024`, cbuf citit doar la 0x0–0x8 și 0x700–0x73c | CONFIRMATĂ |
| E7 | 5,2 ms → ~1,8–3,2 ms | — (doar pe placă) | NOT TESTED |

Unealta: Mesa 26.3.1 construită din nou în cloud; `mm_tc_v4` (`488ca341…`) și `attn_back_tile_tc16` (`8cbbab7e…`) ies
**identic**, deci aceeași unealtă ca în sesiunile 243–244.
SHA-256 `attn_back_tile0_tc.sm86.sass` = `95fc392e6de3cf3bc2f5f93a33487e2ebefdedf38951f35ab1a9168540c1ef8d`.

Câte blocuri pe SM (DEDUS): registrele (56 × 256) ar permite 4; memoria partajată (33 024 B, ca azi) ține **1 bloc** la treapta
64 KiB a pasului mixt rapid, la fel ca `attn_back_tile0_3`. Câștigul de timp vine deci doar din produs (circuitele speciale în loc
de 1 024 `ffma` pe fir și placă), nu din mai multe blocuri. O variantă cu placa P în f16 n-ar mai păstra biții cozii F32 — nu am
făcut-o.

Cod (doar adăugări): `gpu_compute::attn_tc::{Back0Variant, back0_dp_with, back0_tail, back0_model, back0_model_with, HeadDp,
head_dp, head_block, judge_dp}`, `gpu_compute::mm_tc::permille` (era privată, acum publică — adăugare), `nvidia_ga104::attn_back_tile0_tc`
(octeții, suma, registrele, memoria, `cbuf` la 0x700, grila, `shape_ok`), `tools/nak/attn_back_tile0_tc.comp`, testul
`crates/gpu-compute/tests/c157_attn_back0_tc.rs` (T254a–c) și 3 teste în modul (suma, ce spune dump-ul, unde citește cbuf-ul).

### Ce înseamnă (pe scurt)

1. **Al doilea program al atenției înapoi are acum varianta pe circuitele speciale** (compilat, nu rulat): calculează dP pe circuitele
   speciale, restul exact ca azi. Pe procesor (SIMULARE) dS diferă de F32 cu ~3·10⁻⁴, cât costă f16 însăși.
2. **Ambele programe înapoi pe circuite** dau dQ / dK cu ~4,2·10⁻⁴ față de F32 (doar unul: ~3·10⁻⁴) — tot sub toleranța mixtă 1·10⁻³.
3. Pentru OS ar fi tot „alt program în listă”, cu buffer-ul constant mutat la 0x700 (ca la `attn_back_tile_tc`).

### Ce NU e dovedit / ce urmează

- Nimic nu a rulat pe placă; timpul (E7) se vede doar acolo. Modelul presupune sumatorul potrivit în os-335.
- Încă **nu e legat** în pasul mixt și nu are înlocuitor în `gpu-sim`. Pașii următori, în ordine: (1) înlocuitorul din `gpu-sim`
  + opțiunea în pasul mixt (lansările `attn_back_tile0_3` → `attn_back_tile0_tc`, ca `attn_tc_list`); (2) proba antrenării mini
  în SIMULARE cu ambele; (3) PREDARE către OS.

Verificările întregi (`tools/check.sh`): **1 245 PASS** (8 ignorate).

**Ce e nou pentru placă:** un program nou compilat (`attn_back_tile0_tc`), încă netrimis OS-ului. **Nicio schimbare incompatibilă.**
