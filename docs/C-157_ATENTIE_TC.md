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

## 5. Rezultatul (sesiunea 242, ~04:50; T251, 5/5 pe procesor; timpii = DEDUS din cei MĂSURAȚI de OS, precizia = SIMULARE pe procesor)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| A1 | ~2,7 GFLOP pe strat; ~1,6 TFLOP/s azi, sub înmulțirile F32 (~2,7); `attn_back_tile0_3` ~1 TFLOP/s | **2 718 MFLOP** pe strat (32,6 GFLOP pe pas); azi **1,58 TFLOP/s**; înmulțirile F32 ale pasului **2,68**; pe programe: `attn_fwd_tile3` 1,75, `attn_back_tile0_3` **1,05**, `attn_back_tile_3` 1,77 | CONFIRMATĂ |
| A2 | P / S / dS în F32: 100–130 MiB pe strat, sub 20 % din timpul de azi | **135 MiB** pe strat (10 treceri × 13,5 MiB: 5 înainte, 2 + 3 înapoi); 2,8–3,7 ms pe pas = **13,5–18,1 %** din 20,6 ms | PE JUMĂTATE (volumul cu 4 % peste interval; concluzia „memoria nu e frâna” ține) |
| A3 | pe circuitele speciale ~5–10 ms; câștig ~10–15 ms = ~10–14 % din placă; F32 la viteza înmulțirilor F32 cel mult ~8 ms câștig | **5,3–9,7 ms**; câștig **10,8–15,2 ms = 10,2–14,4 %** din placa mixtă (106 ms); F32 la 2,68 TFLOP/s: 12,2 ms, câștig **8,4 ms** (de 1,3–1,8 ori mai puțin decât f16) | CONFIRMATĂ (F32: 8,4 față de „cel mult ~8” — la limită) |
| A4 | f16 + scara 2¹⁴: eroare ≤ 1·10⁻³ la O, dQ, dK, dV; fără scară, dQ / dK ≥ 1·10⁻² | cu scara: O **3,9·10⁻⁴**, dQ **5,3·10⁻⁴**, dK **5,1·10⁻⁴**, dV **3,9·10⁻⁴**; fără scară: dQ **0,17**, dK **0,17** (dS cade în subnormale f16), dV 1,7·10⁻³; F32 de azi: ~3–6·10⁻⁷ | CONFIRMATĂ |
| A5 | ordinea: `attn_back_tile_3`, apoi `attn_fwd_tile3`, `attn_back_tile0_3` ultimul | câștig estimat: `attn_back_tile_3` **4,7–6,8 ms**, `attn_back_tile0_3` **3,5–4,2 ms**, `attn_fwd_tile3` 2,7–4,2 ms | PE JUMĂTATE (primul — da; dar `attn_back_tile0_3` e azi cel mai încet pe FLOP, deci câștigă cel puțin cât înainte) |

Cod (doar adăugări): `nvidia_ga104::attn_tc_study` (`AttnForm`, `M513_ATTN`, `PROGRAMS`, `tc_pair_ps`, `tc_program_ns`,
`tc_total_ns`, `f32_memory_ns`, `f32_at_rate_ns`) și testul `crates/nvidia-ga104/tests/c157_atentie_tc.rs` (T251a–e).

### Ce înseamnă (pe scurt)

1. **Atenția pe circuitele speciale ar lua ~11–15 ms din cei 106 ai plăcii** (DEDUS) — cea mai mare tăietură rămasă în afara
   înmulțirilor. Doar rescrierea în F32 ar lua cel mult ~8 ms; f16 câștigă de 1,3–1,8 ori mai mult.
2. **Precizia e bună doar cu scara pe gradient** (2¹⁴, ca la înmulțiri): eroare ~5·10⁻⁴, adică cât rotunjirea f16 însăși. Fără scară
   gradienții dQ / dK se strică (17 %). Deci un program f16 de atenție înapoi primește dO deja înmulțit cu scara, ca `mm_tc`.
3. **Ordinea propusă:** `attn_back_tile_3` (3 produse cu K până la 512, forma cea mai apropiată de `mm_tc_v4`), apoi
   `attn_back_tile0_3` și `attn_fwd_tile3` (amândouă ~3–4 ms). O idee de verificat mai târziu: dP calculat direct în programul
   înapoi ar scoate o trecere prin VRAM și o lansare.

### Ce NU e dovedit

- Nimic nu a rulat pe placă. Costul pe pereche de plăci e cel al lui `mm_tc_v4` pe qkv-513 (K = 768); produsele atenției au K mic
  (64 la S și dP) și o formă triunghiulară (blocurile de pe ultimele rânduri au de 8 ori mai mult de lucru) — de aici intervalul ×1–×2,
  dar nici ×2 nu e garantat.
- Traficul P / S / dS e numărat din sursa programelor (doar plăcile de sub diagonală, fără zerourile de deasupra); citirile Q / K / V /
  dO (mai ales din L2) nu sunt în model.
- Proba de precizie e pe valori inventate (Q, K, V ~ N(0, 1), dO ~ 1·10⁻⁵), un singur cap; pe activările reale ale lui Kronos poate
  ieși altfel (mai ales dacă scorurile sunt mari și softmax-ul foarte ascuțit).

**Ce e nou pentru placă:** nimic (studiu). **Nicio schimbare incompatibilă** (doar adăugări). Următorul pas posibil, fără placă:
`attn_back_tile_tc` (dQ, dK, dV pe `mma` f16, plan + predicții întâi, SIMULARE cu oracolul de aici) — sau, dacă vin teste noi de pe
placă, comparațiile lor au întâietate.
