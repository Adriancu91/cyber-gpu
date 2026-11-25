# C-165 / pista A — O singură treaptă a memoriei comune pe tot pasul (pregătire, SIMULARE)

Cerut de centru în nota din 25 nov 07:00, punctul (3): „pregătiți: treapta aleasă automat pentru cele mai multe blocuri,
aceeași pe tot pasul”. Sesiunea 226, în cloud, **FĂRĂ placă**. Totul e **SIMULARE** sau **DEDUS**. Se leagă în OS
**doar dacă proba ocupării** (`docs/C-165_OCUPAREA.md`, nota `C-165_proba-ocuparii.md`) confirmă H1.

## 1. Ce face (doar adăugări în `nvidia_ga104::train_step`)

- `Prog::kernel()` — programul văzut de aritmetica ocupării (`occupancy::Kernel`: fire, registre, memorie comună).
- `step_kernels(&[Launch])` — programele distincte ale unei liste de lansări, în ordinea primei apariții.
- `step_best_sm_step(&[Launch], Reserve)` — `occupancy::best_step` peste toate programele listei.
- `Launch::qmd_with_sm_step(prog_va, cbuf_va, step)` — QMD-ul lansării cu `MIN = TARGET = step`, `MAX` 64 KiB.
- `pack_with_sm_step(&[Launch], prog_va, zone_va, step)` — ca `pack`, cu aceeași treaptă la **fiecare** lansare.

`pack` rămâne neschimbat (OS-ul de azi trimite exact aceiași octeți).

## 2. Un lucru găsit la citirea codului (fapt despre codul nostru, nu despre placă)

Programele **fără** memorie comună (`vadd`, `colsum`, `gelu_tr`, `emb_*`, `opt_step`, …) sunt codate azi cu
`ComputeQmd::encode`, care lasă **MIN = TARGET = MAX = 0**. Celelalte cer treapta unui bloc (8 KiB la `mm_train_tile`,
16 KiB la variantele de 12 KiB, …). Deci **azi treapta se schimbă deja de la o lansare la alta**, de sute de ori pe pas.
Dacă schimbarea treptei cere unitatea goală (DEDUS, nedovedit), pasul de azi plătește deja acest cost; o treaptă unică îl
scoate. La programele fără memorie comună, `pack_with_sm_step` scrie `SHARED_MEMORY_SIZE = 0`, `BARRIER_COUNT = 0` (ca azi)
și `MIN = TARGET = step`, `MAX = 64 KiB` (nou: azi 0).

## 3. Predicții (scrise înainte de cod)

| | predicția | dacă iese altfel |
|---|---|---|
| P1 | pasul #513 cum îl trimite OS-ul (`new_tiles3` + `with_mm_choice(RACE_WINNER_513)` + `with_colsum_split`): `step_best_sm_step` = **64 KiB** sub R1 și **32 sau 64 KiB** sub R0 | mai mic → un program e mai lacom decât credem; notăm care |
| P2 | `pack_with_sm_step` diferă de `pack` **doar** în cuvintele câmpurilor MIN / TARGET / MAX (cbuf-urile, adresele și restul QMD identice) | STOP: un alt câmp s-a schimbat |
| P3 | în SIMULARE, pasul de aur mic (SGD și AdamW) și 4 straturi cu `pack_with_sm_step(best)` = **aceiași biți** ca `pack`; procesorul de comenzi vede o singură treaptă pe tot pasul | — (modelul nu are ocupare; verifică doar că QMD-urile sunt citite corect) |
| P4 | o treaptă mai mică decât memoria unui program al listei e refuzată de bibliotecă înainte de orice trimitere | — |

## 4. Verificările T232 (`crates/nvidia-ga104/tests/sim_c165_treapta_pas.rs`, SIMULARE)

- **T232a** `Prog::kernel` = constantele programului (fire = produsul blocului; memoria = `shared()` sau 0).
- **T232b** P1 pe lista pasului #513 la lățimile #513 (un strat), cu tabelul programelor (limita fiecăruia).
- **T232c** P2: octet cu octet față de `pack`, în afara celor 3 câmpuri.
- **T232d** P3: pasul de aur mic + 4 straturi, bit cu bit; `sm_configs` = o singură valoare.
- **T232e** P4: refuzul.

## 5. Ce e nou pentru placă (doar după proba ocupării, cu H1 confirmată)

Valorile câmpurilor `MIN/TARGET/MAX_SM_CONFIG_SHARED_MEM_SIZE` din QMD — aceleași pe toate lansările pasului. Aceleași
programe, aceleași adrese, niciun registru, nicio comandă nouă. Pasul mixt (`pack_mixed`, `mm_tc`) nu e acoperit încă.

## Surse

- `docs/C-165_OCUPAREA.md` (aritmetica ocupării, tabelul public al capacității 8.6, codificarea treptei din Mesa NAK, MIT).
- Studiul centrului: `Cyber-hub/idei/2026-11-25-centru-inmultirea-la-altii.md`.
