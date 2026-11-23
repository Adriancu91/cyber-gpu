# C-160 punctul 7 — `attn_p` legat în pas: refacerea lui P fără P·V

Sesiunea 175, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE**; costul e **DEDUS** (nemăsurat).

## De ce
`Plan::new_p_recalc` (sesiunea 167) ține un singur buffer P pentru toate straturile și, la întoarcere, relansează `attn_fwd`
întreg pe stratul care și-a pierdut P-ul (L − 1 relansări pe pas). `attn_fwd` face și P·V, care atunci nu trebuie. Programul
`attn_p` (sesiunea 168, T178) face doar P, cu aceleași operații în aceeași ordine — costul DEDUS ≈ 0,51 din `attn_fwd`.
OS-ul a anunțat (în `stare/cyber-universe.md`, 23 nov 04:38) că `T::AttnP` nu-i cere nimic înainte: `gpu_i10.rs` are brațul
`_ => 0`, deci compilează și după ce `train_step::Prog` primește varianta nouă.

## Ce se schimbă (doar adăugări)
- `train_step::Prog::AttnP` (cod, SHA-256, registre, bloc, nume `attn_p` = ale modulului `attn_p`).
- **`PROGRAMS` rămâne cele 13** (ce încarcă OS-ul azi la `--pas-aur` nu se schimbă); nou `PROGRAMS_P_ONLY` = cele 13 + `AttnP`
  la sfârșit, deci `Prog::index()` al celor vechi rămâne același; `check_programs` verifică și SHA-256 al lui `attn_p`.
- `Plan::new_p_only(…)` = aceeași arenă ca `new_p_recalc` (un singur P), dar relansarea de la întoarcere e **`attn_p`** cu
  exact argumentele (cbuf, grilă) ale lansării `attn_fwd` a stratului; `Plan::programs()` = lista de încărcat pentru planul dat.
  `Plan::new` și `Plan::new_p_recalc` trimit exact ce trimiteau.
- Pentru OS, când va lega: un rând `T::AttnP => cp.register_program(code, c::attn_p_on_cpu)` în `gpu_i10.rs` și încărcarea
  după `plan.programs()` în loc de `ts::PROGRAMS` în `gpu_i14.rs` — doar dacă vrea planul nou. Nimic nu e obligatoriu.

## Predicții (scrise acum)
1. **Lista:** `new_p_only` are aceeași arenă (aceiași octeți) ca `new_p_recalc` și același număr de lansări (73 + 1 `sgd`,
   75 + 1 `adamw_513`, la mixt + 1); relansarea stratului 0 e `Prog::AttnP` cu cbuf-ul și grila lansării `inainte:atentia`
   a stratului 0; `att` nu mai e rescris (nicio altă lansare nu se schimbă).
2. **Pasul de aur mic, 10 pași, ambele rețete (SIMULARE):** greutățile, gradientul pasului 1 și erorile **bit cu bit** egale cu
   `Plan::new` (P păstrat) și cu `new_p_recalc`; verdictul față de PyTorch trece.
3. **4 straturi, 3 pași AdamW:** bit cu bit, 3 relansări `attn_p`.
4. **Greșeli plantate prinse:** relansarea `attn_p` scoasă (straturile 0…L−2 folosesc P-ul ultimului strat); relansarea pe
   `qkv`-ul altui strat.
5. **Costul (DEDUS, nemăsurat):** cele L − 1 relansări costă ≈ 0,51 din `attn_fwd` fiecare → la #513 (12 straturi) se câștigă pe
   pas ≈ 11 × 0,49 ≈ 5,4 lansări `attn_fwd`. Pe placă nu e nimic nou de cerut acum: `attn_p` n-a rulat pe placă; o probă pe
   placă vine doar odată cu pasul de aur pe placă (o comandă nouă).

## Rezultatul
(se completează după cod)
