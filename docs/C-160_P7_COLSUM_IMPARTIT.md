# C-160 punctul 7 — `colsum` împărțit pe rânduri (după os-349)

Sesiunea 159, 22 nov 2026, în cloud, **fără placă**. Planul, marginea și predicțiile au fost scrise și urcate **înainte** de cod (commit `5d582a5`);
rezultatul e la sfârșit.

## De ce
os-349: `colsum` e pe placă exact modelul nostru (3 778 / 3 778 bit cu bit), dar un fir adună singur toate rândurile coloanei lui:
~0,6–0,7 µs pe rând (MĂSURAT, buffer în RAM-ul procesorului). La bucata #513 (6 144 de rânduri, 73 de `colsum` pe pas) asta ar fi
~165–275 ms, mai mult decât tot pasul PyTorch (DEDUS, `tools/buget-antrenare/colsum_timp.py`, `docs/C-157_P2_COMPARATII.md`).

## Ce facem (planul, înainte de cod)
- Program nou **`colsum_part`** (`tools/nak/colsum_part.comp`): aceleași argumente ca `colsum` + `chunk` (rânduri pe bucată); grila
  **⌈c/256⌉ × G × 1**, G = ⌈rows / chunk⌉. Blocul (x, g) adună, în ordine, rândurile g·chunk … min((g+1)·chunk, rows) − 1 ale coloanelor
  lui și scrie suma parțială la `p0[g·c + k]` (și `p1[g·c + k]` la LayerNorm). Aceleași operații ca `colsum` (doar add / mul / fma, fără
  MUFU, fără memorie partajată, fără `atomic`).
- A doua trecere = **`colsum`-ul de azi, neschimbat** (deja dovedit pe placă), în modul deplasare, peste cele G rânduri de sume parțiale:
  `p0 → s0` și, la LayerNorm, `p1 → s1`. Deci 2 lansări la o deplasare, 3 la LayerNorm.
- Același rezultat la fiecare rulare (ordinea e fixă), dar **alt rezultat decât `colsum`** (altă ordine a sumelor) → model nou pe procesor
  (`gpu_compute::layer::colsum_split_gpu`), care urmează codul NAK.

## Marginea (scrisă acum, înainte de orice măsurătoare)
Pentru o coloană cu R rânduri, G bucăți, u = 2⁻²⁴: fiecare sumă trece prin cel mult (chunk − 1) + (G − 1) ≤ R − 1 rotunjiri de adunare
(pentru că (chunk − 1)(G − 1) ≥ 0), deci **termenii cu u ai marginii lui `colsum` (T155) rămân valabili neschimbați**. Se schimbă doar
numărul de aduceri la zero (FTZ): R + G adunări în loc de R. Marginea:
- dβ (și deplasarea): R·u·Σ|dy| + u·|dβ| + **(R + G + 1)**·2⁻¹²⁶;
- dγ: Σ|dy|·(rs·u·|μ| + 5u·|x̂|) + R·u·Σ|dy·x̂| + u·|dγ| + **(R + G + 1)**·2⁻¹²⁶.
În cod: `gpu_compute::train::colsum_split_margin` = `colsum_margin` + G·2⁻¹²⁶.

## Ce trebuie să iasă în SIMULARE (predicții scrise acum)
1. Pe aceleași forme ca T155 (1…2 048 rânduri, C = 1…768) și pe chunk ∈ {1, 7, 64, R, > R}: **în marginea de mai sus, ≤ 50 %**.
2. chunk ≥ R (o singură bucată) → **bit cu bit `colsum`** (a doua trecere adună un singur număr la 0.0, exact).
3. Pe setul `ln-8x768` cu chunk = 3 și pe 6 144 × 768 cu chunk = 64: **abaterea față de oracolul f64 nu mai mare decât a lui `colsum`**
   (sume mai scurte → de obicei mai mică; „nu mai mare” e predicția, nu certitudinea).
4. Greșeli plantate prinse: o bucată sărită, ultima bucată trunchiată greșit (rows nedivizibil cu chunk), p0 și p1 inversate.
5. Viteza (DEDUS, de verificat doar pe placă): la 6 144 × 768, chunk = 64 → 288 de blocuri; trecerea 1 ~64 rânduri pe fir, trecerea 2
   ~96 de rânduri → ~0,1 ms pe `colsum` în loc de ~2–4 ms; pasul #513: ~7–10 ms pentru toate cele 73 (≈ 5–6 % din 156 ms).

## Ce NU face
Nimic pe placă. Nu schimbă `colsum` (nici binarul, nici `colsum_probe`). Proba pe placă, dacă o vrem, vine separat, după pasul de aur.

## Rezultatul (SIMULARE, T170, `crates/nvidia-ga104/tests/sim_colsum_part.rs`)
- **Binarul:** `tools/nak/out/colsum_part.sm86.sass`, Mesa NAK **25.0.7** (aceeași unealtă ca `colsum`, reconstruită în cloud; `colsum` a
  ieșit identic, `7a0fa2bd…`). SHA-256 **`740f5d439b357d37c06685384cdfd4fb7ba5cbda84c3023f49a77565d59725ac`**, 121 de instrucțiuni,
  **20 de registre**, fără MUFU / barieră / `atom`. Aritmetica e cea a lui `colsum` (`fadd.ftz`, `fmul.ftz`, `ffma.ftz` în aceeași ordine);
  bucata vine din `sr[0x26]` (blocul y) + baza din cbuf 0x4 (lăsată 0), `chunk` la 0x64.
- **Predicția 1:** în marginea scrisă, **cel mai rău 48 %** (≤ 50 % — trece, dar strâns: setul 3 × 257, unde și `colsum` era la 447 ‰,
  dominat de rotunjirea oracolului). Placa simulată = modelul `colsum_split_gpu` bit cu bit pe toate cele 70 de rulări.
- **Predicția 2:** o singură bucată = `colsum` bit cu bit — da.
- **Predicția 3:** abaterea față de f64 nu mai mare: 8 × 768 (numere sintetice, nu setul probei), chunk 3 → **egală** (3,7e-9); 6 144 × 768, chunk 64 → **4,8e-7 față de
  4,9e-6** (de ~10 ori mai mică, sumele sunt mai scurte).
- **Predicția 4:** greșelile plantate (o bucată sărită, ultima bucată trunchiată, p0/p1 inversate) — prinse.
- **Viteza:** nemăsurată (doar pe placă). Bucata propusă: `colsum_part::default_chunk` = ⌈√R⌉ rotunjit la 8 (6 144 → 80, G = 77).
- `tools/check.sh`: **796 de teste** (+7 ignorate), format, clippy, build no_std — PASS.

## Ce urmează
Nepredat pentru placă. Ordinea pe placă rămâne: pasul de aur mic F32 (predat, cu `colsum` vechi), apoi proba `mm_tc`; `colsum_part`
intră ca probă separată (un lucru nou: binarul nou + grila 2D, a doua trecere fiind `colsum` deja dovedit) când ajungem la viteza pasului.
