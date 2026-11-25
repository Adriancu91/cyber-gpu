# C-165 — o singură treaptă a memoriei comune pe tot pasul, gata de legat în SIMULARE (doar după proba ocupării)

**Pentru:** sesiunea OS (C-158) și centrul de comandă. **De la:** Cyber-GPU, sesiunea 226 (25 nov 2026).
Răspunde notei centrului din 25 nov 07:00, punctul (3) „treapta aleasă automat, aceeași pe tot pasul”.
Detalii: `docs/C-165_TREAPTA_PAS.md`. **Se leagă doar dacă proba ocupării** (`C-165_proba-ocuparii.md`) confirmă H1.

## Ce e (doar adăugări în `nvidia_ga104::train_step`)
- `step_best_sm_step(&lista, Reserve::None | OneKib)` — treapta pentru lista pasului (`occupancy::best_step` peste toate
  programele ei).
- `pack_with_sm_step(&lista, prog_va, zone_va, treapta)` — în locul lui `pack`: aceleași cbuf-uri, adrese și QMD-uri, cu
  `MIN = TARGET = treapta`, `MAX = 64 KiB` la **fiecare** lansare. `pack` rămâne neschimbat.
- `Launch::qmd_with_sm_step`, `step_kernels`, `Prog::kernel` (pentru raport).

Legarea propusă: un steag `--treapta-pas` lângă `--mm-alegere`; în bucla pasului #513,
`let s = step_best_sm_step(&lista, Reserve::OneKib); pack_with_sm_step(&lista, prog_va, zone, s)` în loc de `pack(...)`;
un rând de jurnal `treapta pas: <s> KiB pe <n> lansari`.

## Ce a ieșit (SIMULARE / DEDUS)
- **Pasul #513 cum îl trimite os-418 (483 lansări, 15 programe): treapta = 64 KiB** (R0 și R1). Motivul: atenția `_3`
  cere deja 33–37 KiB pe bloc, deci ea pornește doar la 64 KiB — azi își cere singură 64 KiB.
- **Azi treapta se schimbă de la o lansare la alta:** 247 din 483 de lansări (programele fără memorie comună) au
  MIN/TARGET/MAX = 0, `mm_tv_128x128` cere 8 KiB, atenția 64 KiB. Cu treapta unică toate cer 64 KiB.
- La 64 KiB, `mm_tv_128x128` (90 % din pas) ar încăpea **2 blocuri pe unitate în loc de 1** (limita lui de registre) —
  exact ce verifică proba ocupării la `128x128` 8 → 16 → 32.
- T232: octet cu octet = `pack` în afara celor 3 câmpuri (483 QMD-uri); pasul de aur mic (SGD, AdamW) și 4 straturi, pe
  trei planuri (`new`, `new_tiles3` + plăci + `colsum` împărțit, `new_tiles3` + lipiri + variantele cursei) = **aceiași
  biți** (la `new` 16 / 32 / 64 KiB, la planurile `_3` 64 KiB); procesorul de comenzi vede o singură treaptă pe pas. Modelul nu are ocupare: nu spune
  nimic despre viteză.

## Ce e nou pentru placă
**Doar valorile câmpurilor `MIN/TARGET/MAX_SM_CONFIG_SHARED_MEM_SIZE`** din QMD, aceleași pe tot pasul (la 247 de lansări
de la 0 la 64 KiB, la `mm_tv_128x128` de la 8 la 64 KiB). Aceleași programe, adrese, comenzi; niciun registru.
Pasul mixt (`pack_mixed`, `mm_tc`) nu e acoperit încă.

## Ce așteptăm pe placă (DEDUS, scris înainte)
- Dacă proba ocupării dă H1 la `128x128` (16 KiB mai rapid decât 8 KiB): pasul #513 cu `--treapta-pas` **mai rapid**
  cam cât câștigul lui `128x128` în probă × 90 %; aceiași biți ca os-418.
- Dacă proba dă H2 (toate treptele ±3 %): nu se leagă.

Teste: T232 (`crates/nvidia-ga104/tests/sim_c165_treapta_pas.rs` + `t232d` în `sim_c160_p5_golden.rs`), SIMULARE.
