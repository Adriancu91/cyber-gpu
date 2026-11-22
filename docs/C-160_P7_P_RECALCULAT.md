# C-160 punctul 7 — atenția înapoi fără P păstrat (P recalculat), pentru memoria lui #513 / #651

Sesiunea 167, 22 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Memoria e **DEDUS** din adresele planului (aritmetică, fără placă); corectitudinea se verifică doar în **SIMULARE**.

## De ce
Pasul de antrenare (`nvidia_ga104::train_step::Plan`) păstrează pentru fiecare strat matricea P a atenției (B·H·T² numere F32) de la
„înainte” până la „înapoi”. La lotul adevărat al lui #513 (12 × 512) asta înseamnă 12 × ~144 MiB = ~1,7 GiB doar pentru P, iar
`docs/C-160_MOTOR_STUDIU.md` arăta deja că „F32 cu P păstrat **nu încape**; cu P recalculat încape”. La #651 (24 de straturi)
diferența e lotul 4 față de 6.

## Ideea (scrisă înainte)
Programele atenției n-au `atomic`, iar ordinea fiecărei sume e fixă → **aceeași lansare pe aceleași intrări dă aceiași biți, de fiecare
dată** (și pe placă: aceleași instrucțiuni, inclusiv MUFU, pe aceleași numere). Deci P-ul stratului l se poate **reface exact** la
întoarcere relansând `attn_fwd` pe `qkv`-ul stratului (care e păstrat oricum): scrie același P (și același `att`, bit cu bit, peste
el însuși — inofensiv). Nu trebuie niciun program nou și nicio statistică nouă (max / sumă pe rând).

Concret, doar adăugări:
- `Plan::new_p_recalc(...)` (aceleași argumente ca `Plan::new`): **un singur** buffer P, comun tuturor straturilor. `Plan::new`
  rămâne neschimbat (aceleași adrese, aceleași 73 / 75 de lansări).
- În `forward_backward`, la întoarcerea stratului l, imediat înainte de cele două lansări `attn_back`: dacă un strat de după l
  folosește același buffer P (deci l-a suprascris), o lansare `attn_fwd` cu eticheta `inapoi:atentia-P` (aceleași argumente ca la
  „înainte”). Ultimul strat nu are nevoie (P-ul lui e încă acolo) → **L − 1 lansări în plus pe pas**. Regula depinde doar de
  adrese, deci `Plan::new` dă 0 lansări în plus.
- `launches_per_step` / `launches_per_step_mixed` le numără; `Plan::p_relaunches()` spune câte sunt.

## Predicții (scrise acum)
1. **SIMULARE, pasul de aur mic** (2 straturi, `sgd` și `adamw_513`, 10 pași, aceleași date ca T165): cu P recalculat **toate**
   numerele (erorile pe pași, normele, gradienții și greutățile după pasul 1, greutățile după pasul 10) **identice bit cu bit** cu P
   păstrat; deci și verdictul față de PyTorch neschimbat.
2. Lansări: **76 / 74** pe pas (față de 75 / 73), adică +1 = L − 1; memoria arenei mai mică cu exact un buffer P al pasului mic
   (2·4·32²·4 = 32 KiB + 256 de octeți de gol = 33 024 B).
3. Un model mic cu **4 straturi** (greutăți inventate, `adamw_513`, 3 pași): tot bit cu bit, +3 lansări.
4. **Greșeală plantată:** P comun **fără** relansări → straturile 0 … L − 2 folosesc P-ul ultimului strat → gradienții diferă
   (prinși), iar pasul `sgd` iese din toleranța față de PyTorch.
5. **Memoria arenei (DEDUS, din adresele planului — `Plan::bytes`):**
   - #513, lot 12 × 512 (o bucată): P păstrat **~7,7 GiB** (peste limita de ~7 GiB a workstation-ului), P recalculat **~6,2 GiB** (încape);
   - #651, lot 4 × 512: ~6,5 GiB → ~5,4 GiB; lot 6 × 512: ~8,4 GiB (nu încape) → **~6,8 GiB** (încape).
   (Pe lângă ce socotea `C-160_MOTOR_STUDIU.md`, planul ține și logit-urile + gradientul lor, 2 × n·V, și dS-ul unui strat.)
   Marja predicției: ±3 %.
6. **Costul (DEDUS, nemăsurat):** o atenție „înainte” în plus pe strat, fără ultimul: cu `attn_fwd_tile` (~0,53 ms pe strat la B 4,
   proba predată) ~1,6 ms pe strat la B 12 → **~17 ms pe bucată #513** (11 straturi), ~+6–11 % din pasul de atenție pe plăci și
   ~11 % din bucata PyTorch (156 ms). Un program care face doar P (fără P·V) ar înjumătăți asta — pasul următor, cu plan separat.

## Ce NU face
Nu schimbă niciun program GPU și nici `Plan::new`. Nu leagă programele „pe plăci” de pas (rămâne pasul următor). Nimic nou pentru
placă: relansarea folosește programul `attn_fwd` care e deja în lista pasului de aur mic (os-351).

## Rezultatul (după cod, sesiunea 167)
| Predicția | Ieșit | |
|---|---|---|
| 1. pasul mic, 10 pași, P recalculat = P păstrat bit cu bit | `sgd` și `adamw_513`: **DA**, verdictul față de PyTorch TRECE | PASS (SIMULARE) |
| 2. 76 / 74 de lansări, −33 024 B | 76 / 74, −33 024 B | PASS |
| 3. 4 straturi, 3 pași AdamW, +3 lansări, bit cu bit | 140 de lansări (+3), **DA** | PASS (SIMULARE) |
| 4. fără relansări → prins | 4 straturi: biți diferiți; pasul mic `sgd`: grad_abs 6,3e-2 în `straturi.0.atentie.in_proj_weight` (toleranța 1e-4 ×600), verdictul NU TRECE | PASS (SIMULARE) |
| 5. memoria arenei (±3 %) | #513 12 × 512: **7,70 → 6,15 GiB** (est. 7,7 / 6,2); #651 4 × 512: 6,49 → 5,42 (6,5 / 5,4); #651 6 × 512: **8,38 → 6,76** (8,4 / 6,8) | PASS (DEDUS) |
| 6. costul (~17 ms pe bucată #513 cu `attn_fwd_tile`) | nemăsurat | NOT TESTED |

Ce înseamnă: **lotul adevărat al lui #513 (12 × 512) încape acum sub 7 GiB** (cu P păstrat nu încăpea), iar #651 urcă de la lotul
4 la 6. Codul: `Plan::new_p_recalc`, `Plan::p_rebuilt`, `Plan::p_relaunches` (doar adăugări; `Plan::new` dă aceleași adrese și
aceleași 73 / 75 de lansări). Testele: T177 în `crates/nvidia-ga104/tests/sim_c160_p5_golden.rs` și `sim_c160_p6_train_step.rs`.

**Ce nu e dovedit:** în SIMULARE „aceiași biți la a doua lansare” e adevărat prin construcție (modelul e o funcție). Pe placa reală
se sprijină pe faptul că programul n-are `atomic` și are ordinea sumelor fixă — de verificat o dată pe placă: proba atenției pe
plăci (predată) poate lansa `attn_fwd_tile` de două ori pe același set și compara amprentele P (propunere pentru un test viitor,
nimic nou de legat acum).

**Pasul următor:** un program care face doar P (fără P·V) pentru relansare (cost ~½), apoi programele „pe plăci” în pas.
