# C-157 p.2x — `mm_train` pe plăci de 64 × 64 (`mm_train_tile`), aceiași biți

**Sesiunea 208, 24 nov 2026, în cloud, FĂRĂ placă.** Planul și predicțiile sunt scrise **înainte** de compilare și de teste.

## De ce
os-403 (profilul pasului #513 pe placă) a arătat `mm_train` = **64 %** din pas, la **~0,83 TFLOP/s** (plafonul F32 al plăcii
~21,7). După atenția `_3` (nota P2v), `mm_train` ajunge ~90 % din pas (DEDUS, `docs/C-157_P2_COMPARATII.md`). `mm_train` de azi
e treapta 3 din seria matmul (plăci 16 × 16, un rezultat pe fir): fiecare număr citit din memoria partajată dă o singură
înmulțire. Calea cu circuitele speciale (`--mixt`, f16) rămâne separată; aici e drumul **F32**, cel care dă exact numerele
oracolului.

## Ce e programul
`tools/nak/mm_train_tile.comp` — aceeași treabă și **aceleași argumente** (bufferul constant 0, 0x28–0x57) ca `mm_train`:
`C = op(A)·op(B) (+ bias)`, opțiunile TA / TB / BIAS. Diferă doar împărțirea:

| | `mm_train` | `mm_train_tile` |
|---|---|---|
| bloc | 16 × 16 fire → 16 × 16 din C | 256 de fire → **64 × 64** din C |
| pe fir | 1 rezultat | **4 × 4** (rândurile ty + 16i, coloanele tx + 16j) |
| felia lui K | 16 | 16 |
| memorie partajată | 2 KiB | **8 KiB** (16 × 64 din A, 16 × 64 din B) |
| citiri partajate pe înmulțire | 2 | **0,5** |
| grila | ⌈N/16⌉ × ⌈M/16⌉ | ⌈N/64⌉ × ⌈M/64⌉ |

Încărcarea plăcilor depinde de TA / TB doar ca firele vecine să citească adrese vecine; valorile din memoria partajată sunt
aceleași. Ideea: siboehm SGEMM_CUDA kernel 5 (MIT, doar ideea) și `matmul_2d.comp` (al nostru).

## Predicții (scrise înainte)
1. **Biții:** fiecare element din C rămâne lanțul de `ffma.ftz` p = 0..K−1 pornind de la +0,0, apoi un `fadd.ftz` cu
   bias → **bit cu bit egal cu `mm_train`** și cu `reference::matmul_t_fma_ftz`, pe toate cele 8 combinații de opțiuni și pe
   margini (M, N, K care nu se împart la 64 / 16). Dacă NAK contractă altfel (de ex. bias-ul într-un FMA) → greșit, se vede în test.
2. **Registre:** 16 sume + 8 valori + adrese ≈ **40–64**; fără memorie locală (buclele fixe desfăcute). Peste 64 → mai puține
   blocuri pe SM, de notat.
3. **Viteza pe placă (DEDUS, NETESTAT):** siboehm pe A6000: treapta 3 → treapta 5 = ×5,4; noi avem 4 × 4 (nu 8 × 8) și NAK,
   deci **×3–6 față de `mm_train`: ~2,5–5 TFLOP/s**. În pasul #513 (2 × 512, atenția `_3`): `mm_train` ~600 ms → **~120–250 ms**,
   pasul ~770–800 → **~300–450 ms** (≈ 2 300–3 400 simb/s). Ce nu știm: conflictele de bancă reale, ocuparea cu NAK.
4. **Corectitudinea în SIMULARE:** modelul pe procesor (`gpu_sim::cmdproc::mm_train_tile_on_cpu`) urmează plăcile programului;
   predicție: = `mm_train_on_cpu` bit cu bit pe toate formele pasului mic și pe formele #513 tăiate.

## Ce urmează după (nu în sesiunea asta)
- Proba pe placă (ca `mm_tc_probe`): seturile mici + `qkv-513`, comparat bit cu bit cu `mm_train` pe aceeași placă, timpul median.
- Legarea în pas: `Plan::...` cu `mm_train_tile` în locul lui `mm_train` (aceiași biți → aceeași amprentă a pasului).
- `mm_train_gelu` pe aceleași plăci.

## Rezultat (sesiunea 208, după compilare și teste)
| Ce | Predicție | Ieșit | Stare |
|---|---|---|---|
| Compilarea | — | Mesa NAK 25.0.7 reconstruit în cloud; `mm_train` reconstruit = `69d54ef2…` (neschimbat); `mm_train_tile` de două ori = aceeași sumă | PASS |
| SHA-256 | — | `f8acce2165870280d06271c6b0a60ca96ebc418b23fc7abb8ff496e08bbf094d`, 1 436 instrucțiuni, 22 976 octeți | — |
| Registre | 40–64 | **34**, memorie locală 0 (predicția era prea largă: NAK refolosește registrele adreselor) | mai bine decât prezis |
| Codul | lanț de FMA + un fadd pe element | 256 `ffma.ftz` (bucla de 16 desfăcută), 16 `fadd.ftz` (bias), 128 citiri partajate | conform |
| Biții (T214, SIMULARE) | = `mm_train` = oracolul | bit cu bit pe 8 combinații × 6 forme cu plăci parțiale + 3 produse × 3 straturi; −0,0; refuzuri (opțiune 8, 2 KiB, fără barieră, blocuri 16×16); grilă prea mică → restul lui C neatins | PASS (6/6) |
| Viteza pe placă | ×3–6 (~2,5–5 TFLOP/s) | — | NOT TESTED |

**Atenție, cinstit:** modelul de pe procesor urmează sursa GLSL, nu execută codul binar; „aceiași biți pe placă” rămâne
predicție până la proba pe placă.

**Următorul pas:** `Plan::with_mm_tile` (aceeași listă de lansări, `mm_train` → `mm_train_tile`, aceeași amprentă în SIMULARE),
apoi proba mică pentru placă; `mm_train_gelu` pe aceleași plăci.
