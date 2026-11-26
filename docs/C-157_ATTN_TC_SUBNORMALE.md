# C-157 — setul `ascutit` iese din marginea `mm_tc`: valorile subnormale f16 și marginea „pe exponentul nominal”

Sesiunea 246, 26 nov 2026, în cloud, **fără placă**. Răspuns la nota OS din P6-227 (sesiunea OS 07:10): în proba „vechi lângă
nou” a atenției pe circuitele speciale, setul `ascutit` (amp 6) iese din marginea `mm_tc` **chiar cu modelul nostru**
(`attn_tc::model`): primul element în afară nr. 394 pe cap, 3,0900992·10⁻⁷ față de 3,0901583·10⁻⁷, marginea 3,6·10⁻¹².
Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**. Tot ce iese pe procesor e **SIMULARE**.

## 1. Cauza (DEDUS din cod, de verificat la S1–S2)

Sumatorul potrivit pe placă în os-335 (`mma::GA104_OS335`, modelul `mma16_hw`) aliniază termenii unui pas după **exponentul
nominal** al produsului: `exp(a) + exp(b)`, iar o valoare f16 **subnormală** (|x| < 2⁻¹⁴ ≈ 6,1·10⁻⁵) are exponentul nominal −14,
oricât de mică ar fi. Un produs cu un factor subnormal ridică deci „cuanta” pasului (2^(E−24)) mult peste mărimea lui reală.
Marginea `mm_tc::margin` (scrisă înainte de os-335) e proporțională cu Σ|â·b̂| — mărimile reale. Pentru intrări normale cele
două coincid ca ordin (2^(ea+eb) ≤ |â·b̂| < 4·2^(ea+eb)); pentru subnormale nu: marginea devine prea strâmtă.

În `ascutit` softmax-ul e ascuțit (qkv în [−6, 6)): multe P și dS sunt ~10⁻⁶–10⁻⁹, adică subnormale f16; proba nu are scara 2¹⁴
a pasului mixt (dO e în [−1, 1)). Nu e o greșeală a programului, ci o margine care nu știe de subnormale.

## 2. Ce adaug (doar adăugări; vechea margine rămâne și verdictul ei rămâne raportat)

- `mm_tc::margin_nominal(p, r, c)`: aceeași formulă, cu |â·b̂| înlocuit de **u = max(|â·b̂|, 2^(eâ+eb̂))** pentru fiecare produs
  nenul (eâ, eb̂ = exponenții f16, subnormalele −14). Derivarea e cea din `docs/C-160_P7_MM_TC.md` §3 cu u în locul lui |â·b̂|
  (cuanta unui pas e ≤ 2⁻²⁴·max(u, |acc|)). Fără nicio valoare subnormală: pentru produse
  normale 2^(ea+eb) ≤ |â·b̂|, deci u = |â·b̂| — marginea e **aceeași, bit cu bit**.
- `mm_tc::judge_nominal`, `attn_tc::judge_nominal`, `attn_tc::judge_dp_nominal`: judecata cu marginea nouă.
- **Cinste:** marginea nouă e scrisă **după** ce am văzut ratarea, deci nu e o predicție, ci o corectură a marginii; propun OS-ului
  să afișeze ambele verdicte (vechi și nou), nu să-l ascundă pe cel vechi.

## 3. Predicțiile (scrise ÎNAINTE de cod)

| # | Ce | Prezis |
|---|---|---|
| S1 | Reproducerea: P din `attn_fwd_tile_gpu`, dS din `attn_back_gpu`, `attn_tc::model`, `attn_tc::judge` pe `ascutit` | `NuTrece` la indexul 394, 3,0900992·10⁻⁷ / 3,0901583·10⁻⁷ / 3,6·10⁻¹² (aceleași cifre ca OS-ul) |
| S2 | Elementul 394: în rândul lui, cel puțin un factor f16 subnormal nenul | DA |
| S3 | `judge_nominal` pe cele 5 seturi mici ale probei | toate `InMargine`; `ascutit` sub 1000 ‰ |
| S4 | Pe cele 4 seturi care trec azi și pe cele 6 forme din T252 (scara 2¹⁴) | ‰ cel mai rău identic între margini sau foarte apropiat (puține subnormale) |
| S5 | Greșelile plantate (dK netranspus, P în loc de dS, fără mască, ultimul pas uitat) pe `ascutit` | toate prinse și de marginea nouă |
| S6 | Varianta „scara în probă”: dO × 2¹⁴ pe `ascutit`, marginea veche | **NU ajunge** (P însuși e subnormal în dV = Pᵀ·dO) — incert |
| S7 | `margin_nominal` = `margin` bit cu bit pe un produs fără subnormale | DA |

## 4. Rezultatul

Testul T256 (`crates/nvidia-ga104/tests/sim_c157_attn_tc_subnormal.rs`, 6 teste), toate în **SIMULARE**:

| # | Ce a ieșit | Predicția |
|---|---|---|
| S1 | `NuTrece` la 394, 3,0900992·10⁻⁷ față de 3,0901583·10⁻⁷, marginea 3,647·10⁻¹² — **aceleași cifre ca OS-ul** | CONFIRMATĂ |
| S2 | elementul 394 citește o valoare f16 subnormală nenulă (`mm_tc::reads_subnormal`) | CONFIRMATĂ |
| S3 | marginea nouă: **5 / 5 seturi în margine**; `ascutit` 46 ‰ (cu vechea: în afară) | CONFIRMATĂ |
| S4 | pe celelalte seturi ‰ identic (46, 45, 40 …), același număr de elemente exacte | CONFIRMATĂ |
| S5 | dK netranspus, P în loc de dS, ultimul pas uitat: prinse. „Fără mască”: cu zerourile de deasupra diagonalei (ce ține proba) dă aceleași numere — nu se vede; cu 0,5 deasupra diagonalei e prinsă | PE JUMĂTATE (lista mea pusese „fără mască” fără să țină cont de zerouri) |
| S6 | cu dO × 2¹⁴ în probă, **și marginea veche trece** pe `ascutit`: 796 ‰ (cea nouă 46 ‰) | **GREȘITĂ** (preziceam că nu ajunge) |
| S7 | fără subnormale, `margin_nominal` = `margin` bit cu bit (768 de elemente); cu o subnormală, mai largă | CONFIRMATĂ |
| — | dP al lui `attn_back_tile0_tc` (sesiunea 245) pe cele 5 seturi: ambele margini identice, în margine (21–40 ‰) | — |

**Ce propun OS-ului (P6-227), în ordinea preferinței:**
1. În proba `--atentie-tc`, pe lângă verdictul de azi (`attn_tc::judge`), încă un rând cu `attn_tc::judge_nominal`
   (marginea care știe de subnormale). Așteptat pe placă, dacă placa = modelul: vechea **4 / 5** (`ascutit` în afară, aceleași cifre),
   noua **5 / 5**. Dacă placa iese și din marginea nouă, atunci placa nu se poartă ca modelul cu subnormalele — asta ar fi o informație nouă.
2. Nu schimbați setul și nu puneți scara în probă: cu scara 2¹⁴ vechea margine trece la 796 ‰ (aproape de limită), deci proba ar
   pierde exact cazul care arată purtarea plăcii la subnormale.
3. Pasul 2 (`--atentie-tc16` în pasul mixt rapid) poate fi legat independent de asta: pasul mixt are deja scara 2¹⁴.

**Nou pentru placă:** nimic (doar judecata pe procesor). **Compatibilitate:** doar adăugări (`mm_tc::{margin_nominal, judge_nominal,
reads_subnormal}`, `attn_tc::{judge_nominal, judge_dp_nominal}`); `judge` / `judge_dp` dau exact aceleași verdicte ca înainte.
