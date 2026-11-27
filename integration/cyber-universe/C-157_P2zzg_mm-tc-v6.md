# C-157 p.2 — `mm_tc_v6`: `mm_tc_v5` cu scrierile f16 împerecheate (gata pentru probă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 263, 27 nov 2026, în cloud, fără placă. Detalii și predicții: `docs/C-157_MM_TC_V6_PLAN.md`.

## Ce e
Același program ca `mm_tc_v5` (aceeași lansare, același cbuf, 128 de fire, plăci 64 × 64, aceeași regulă `fits`, aceiași 12 KiB,
**aceiași biți**), dar în buclă fiecare fir scrie cele 4 valori f16 ale unui vector **dintr-o singură scriere** de 64 de biți în loc
de 4 scrieri de 2 octeți; când A sau B e transpus, felia e ținută pe coloane, ca scrierile să rămână una lângă alta. Socoteala (T275):
la v5, pe înmulțirile transpuse (dX și dW, două treimi din pasul #513), scrierile loveau aceeași bancă de 8 și de 16 ori.

| | |
|---|---|
| Binarul | `tools/nak/out2/mm_tc_v6.sm86.sass`, SHA-256 `8c761fa50321ee891f510a6b70ffac88778312be6b399747411cbfa637df980f` |
| Rust | `nvidia_ga104::mm_tc_v6::{MM_TC_V6, MM_TC_V6_SHA256, REGISTERS (128), BLOCK, SHARED (12 288 B), grid, fits}` — `Args` / `cbuf` / `grid` = ale lui `mm_tc` |
| Simulare | `gpu_sim::cmdproc::{mm_tc_v6_on_cpu, MM_TC_V6_NEEDS}` (refuză produsele nealiniate, ca v5) |
| QMD | exact ca `mm_tc_v5` (registre 128, 12 KiB, treapta 64 KiB); diferă doar adresa programului |

**Înainte de lansare: `mm_tc_v6::fits(&args)`** (= `mm_tc_v4::fits`). Dacă zice nu → `mm_tc_pf`.

## Ce cerem OS-ului (când are placa; după P2zzf, nu în locul lui)
1. Încă un rând în proba `--mm-tc-64k`, **după `mm_tc_v5`**: aceleași seturi aliniate + qkv-513, cu `mm_tc_v6`.
   - **Așteptat:** aceiași biți ca `mm_tc` pe toate seturile aliniate (și pe `64x64x96-ta`, `100x72x200-ta-tb`), gărzile neatinse.
   - **Viteza prezisă (DEDUS, scrisă înainte):** qkv-513 (o înmulțire netranspusă) **1,70–1,87 ms** (v5: 1,868; ×1,00–1,10).
     Proba **nu vede** câștigul principal (e la înmulțirile transpuse); seturile mici transpuse sunt prea mici ca să se vadă viteza.
2. Legarea în pas o pregătim noi după rândul (1) (ca la v5: o alegere `TcPick` nouă, doar adăugări). Predicția pentru pas, față de
   același pas cu v5: înmulțirile ×1,03–1,20 → placa ×0,885–0,979 la lot 2.

## Ce e nou pentru placă
Doar programul `mm_tc_v6` (SHA de mai sus). Nicio scriere în registre, niciun firmware, nicio schimbare la `mm_tc` / `mm_tc_pf` /
`mm_tc_v4` / `mm_tc_v5`. Nicio schimbare incompatibilă (doar adăugări).
