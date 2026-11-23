# Ordinea 20:11, pasul 4 — recalcularea activărilor (stratul întreg refăcut la întoarcere)

Sesiunea 195, 24 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE**; memoria e **DEDUS** din adresele planului (aritmetică, fără placă).

## De ce
Pasul de antrenare (`nvidia_ga104::train_step::Plan`) păstrează de la „înainte” până la „înapoi”, pentru **fiecare** strat:
μ și 1/σ ale celor două LayerNorm, `h1`, `qkv` (3·C), P, `att`, `x_mid`, `h2`, `u` (4·C), `act` (4·C) — adică ~16·C numere pe
simbol + P. Planul cu P recalculat (`new_p_recalc`, `C-160_P7_P_RECALCULAT.md`) a scos doar P. Studiul (`C-160_MOTOR_STUDIU.md` §2)
nota pasul următor: **păstrăm doar intrarea stratului** (C numere pe simbol) și refacem restul la întoarcere — costă calcul în plus,
dar memoria activărilor scade de ~16 ori. Decizia 23 nov 20:11 îl pune al patrulea în ordinea motorului de antrenare.

## Ideea (scrisă înainte)
Ca la P: programele „înainte” (`ln_fwd`, `mm_train`, atenția, `vadd`, `gelu_tr`) n-au `atomic`, iar ordinea fiecărei sume e fixă
→ **aceeași lansare pe aceleași intrări dă aceiași biți**. Greutățile nu se schimbă în timpul pasului (gradienții merg în `g`), iar
intrarea stratului l (`x_in` = ieșirea stratului l − 1) rămâne păstrată. Deci la întoarcerea stratului l putem relansa exact
„înainte”-ul lui și obținem aceleași `h1`, `qkv`, P, `att`, `x_mid`, `h2`, `u`, `act`, bit cu bit.

Concret, **doar adăugări** (interfața veche neatinsă; `Plan::new` și toate celelalte dau aceleași adrese și aceleași liste):
- `Plan::with_recompute(self) -> GpuResult<Plan>`: refă arena aceluiași plan (aceleași programe, plăci, lipiri, m/v, rezident)
  cu **un singur set** de bufere de strat, comun tuturor straturilor (μ, 1/σ ×2, `h1`, `qkv`, P, `att`, `x_mid`, `h2`, `u`, `act`);
  pe strat rămâne doar ieșirea lui (firul rezidual `x_out`). Primul strat are aceleași adrese ca înainte; restul arenei se mută.
- În `forward_backward`, la întoarcerea stratului l < L − 1 (ultimul are setul încă întreg), **înainte** de orice lansare „înapoi”
  a stratului: 8 lansări cu eticheta `recalcul:*` — ln1, qkv, atenția (același program ca la „înainte”), out, rezidual1, ln2, fc,
  gelu. `proj` și `rezidual2` nu trebuie (scriu `x_out`, care e păstrat). `out` scrie în `dtmp`, liber în acel moment.
- Relansarea separată a lui P (`inapoi:atentia-P`) dispare la straturile refăcute (atenția refăcută scrie deja P).
- `Plan::recomputes()`, `Plan::recompute_launches()` (= 8·(L − 1)); `launches_per_step` le numără; `p_relaunches` = 0 pe un plan
  cu recalculare. `Plan::resident` păstrează recalcularea.
- Niciun program GPU nou: **nimic nou pentru placă** (doar mai multe lansări ale programelor deja probate).

## Predicții (scrise acum)
1. **SIMULARE, pasul de aur mic** (2 straturi, `sgd` și `adamw_513`, 10 pași): `Plan::new().with_recompute()` = `Plan::new()`
   **bit cu bit** (erorile, normele, gradienții și greutățile după pasul 1, greutățile după pasul 10); verdictul față de PyTorch TRECE.
   Lansări: 73 → **81** (`sgd`), 75 → **83** (`adamw_513`): +8·(L − 1) = +8.
2. Combinat cu ce e deja: `new_tiles3` + ambele lipiri + recalculare = același plan fără recalculare, bit cu bit, +8 lansări;
   `new_p_only_tiles3` + recalculare = bit cu bit, lansări −(L − 1) + 8·(L − 1) = **+7**.
3. Model cu **4 straturi** (greutăți inventate, `adamw_513`, 3 pași, plăci `_3` + lipiri): bit cu bit, **+24** lansări.
4. **Greșeli plantate:** (a) fără lansările `recalcul:*` → straturile 0 … L − 2 folosesc activările ultimului strat → biți
   diferiți, iar pasul mic `sgd` iese din toleranța față de PyTorch; (b) `recalcul:ln1` al stratului 0 citind intrarea stratului 1
   → biți diferiți (prins).
5. `with_recompute().resident()` rămâne cu recalculare (aceeași memorie ca `with_recompute()` pe un plan rezident, aceleași lansări).
6. **Memoria arenei (DEDUS, `Plan::bytes`, marja ±5 %)**, față de P recalculat (`C-160_P7_P_RECALCULAT.md`):
   - #513, lot 12 × 512: 6,15 → **~3,25 GiB**;
   - #651 (24 de straturi), lot 4 × 512: 5,42 → **~3,40 GiB**; lot 6 × 512: 6,76 → **~3,73 GiB**;
   - #651, lot **12 × 512** (lotul adevărat al lui #513, cu P recalculat nu încăpea): **~4,73 GiB** — sub 7 GiB;
   - cel mai mare lot × 512 al lui #651 sub 7 GiB: **~25** (azi 6).
   Ce rămâne mare după recalculare: greutăți + gradient + m + v (16 B pe parametru: 1,46 GiB la #513, 2,73 GiB la #651) și
   logit-urile cu gradientul lor (128 KiB pe simbol) — nu activările.
7. **Costul (DEDUS, nemăsurat):** pe strat refăcut, „înainte”-ul fără `proj` (8·C² din 12·C² înmulțiri pe simbol + atenția);
   pasul întreg ≈ 3 × „înainte” → **~+15–25 % timp pe pas** la #513 (față de planul fără recalculare; față de P recalculat ceva
   mai puțin, fiindcă relansarea lui P e inclusă). Se măsoară doar pe placă.

## Ce NU face
Nu schimbă niciun program GPU și niciun plan existent. Nu alege încă **ce** straturi se refac (toate sau din k în k — un pas
următor, dacă memoria rămasă o cere). Nimic pe placa reală; pe placă doar prin OS, după comanda centrului de comandă.
