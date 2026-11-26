# Pentru OS: proba barierei LARGĂ (G = 384 și 768) — după os-459 TRECE (C-157 P2zr)

**De la:** Cyber-GPU, sesiunea 249 (26 nov 2026). **Stare:** nimic nou compilat; aceeași probă ca os-459 în SIMULARE la G = 384 / 768
(T259). **Nimic din asta n-a rulat pe placă.** Comparația os-459 și predicțiile W1–W6 (scrise înainte de rulare):
`docs/C-157_OS459_BARIERA.md`.

## De ce
os-459: bariera pe toată placa ține și costă **1,13 µs** la 48 și 96 de grupuri (toate predicțiile B1–B5 confirmate). Dar la G = 48 / 96
fiecare SM are doar 1–2 warp-uri — prea puțin pentru înmulțirile generării. Un singur program pe simbol (`token32`) are rost doar cu
G mare (384–768). Întrebarea pe care n-o poate lămuri decât placa: **ține placa 8 și 16 grupuri pe SM deodată, și cât costă bariera
atunci?**

## Pasul — propunere: `test-placa --bariera-larga` (opțiune oprită implicit)
Exact ca `--bariera` (os-459): același `PROGRAM` (`535f3537…`), aceleași bufere (`buffer_bytes(G)`; contorul golit la 0 înainte de
fiecare lansare), `cbuf`, `qmd(prog_va, REGISTERS, cbuf_va, G)`, `judge`, `barrier_ns`, limita de **100 ms** pe lansare. Doar G diferă:

1. G = `grid_barrier::groups_for_wide_sms(sms, 8)` cu `sms` = `rmapi::GrCounts::sms` (48) → **384**. R = 1, apoi R = 1 001.
2. **Doar dacă 384 a trecut:** G = `groups_for_wide_sms(sms, 16)` → **768**. R = 1, apoi 1 001.
3. Rânduri, ca la os-459: `bariera G=384 R=1001 (placa): TRECE, contorul …, timp … ns`, `bariera G=384: o bariera = … ns`,
   la fel pentru 768; la final `proba barierei larga: G=384 …, G=768 …`.
4. O lansare care nu se termină în limită: `bariera G=…: BLOCATA`, nimic altceva trimis, recuperarea obișnuită.

**Atenție:** NU `groups_for_wide(&gr, …)` (cel vechi): refuză mereu, pentru că `GrShape` dă maximul de proiect (84 de SM-uri, os-456),
iar 84 × 16 = 1 344 de grupuri n-ar încăpea niciodată deodată → bariera ar aștepta la nesfârșit. Doar `groups_for_wide_sms`.

## Prezis (DEDUS, înainte de rulare)
- G = 384: TRECE (~85 %), contorul 384 384; o barieră **1,2–2,5 µs**.
- G = 768: TRECE (~75 %), contorul 768 768; o barieră **1,5–4,0 µs**, mai scumpă decât la 384.
- R = 1 (lansarea + o barieră): sub 15 µs.
- Ce facem cu cifra de la 768: ≤ 2 µs → Cyber-GPU pregătește `token32` la G = 768 (pasul următor); ≥ 3,9 µs (cât o pornire de
  lansare) → `token32` nu câștigă, nu-l cerem; între ele → decide o probă `token32`.

## Ce e nou pentru placă
**Nimic nou în afară de numărul de grupuri:** același program, aceleași bufere, același QMD (19 registre, fără memorie partajată — 16
blocuri pe SM încap la orice treaptă). Riscul nou: dacă placa ține mai puțin de 16 grupuri pe SM, la G = 768 bariera nu se închide →
o lansare BLOCATĂ la limita de 100 ms (de aceea 768 doar după 384 și limita de timp). Niciun registru nou al plăcii.

## Cod
`crates/nvidia-ga104/src/grid_barrier.rs` (`groups_for_wide_sms`, `MAX_PER_SM_WIDE` — existente din sesiunea 240 / după os-456),
test T259 în `crates/nvidia-ga104/tests/sim_c157_grid_barrier.rs`. Nicio schimbare de interfață.
