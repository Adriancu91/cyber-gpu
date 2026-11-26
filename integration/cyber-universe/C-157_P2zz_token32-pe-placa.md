# PREDARE P2zz — `token32` pe placă, G = 384 apoi G = 768, sub o opțiune oprită implicit (Cyber-GPU, sesiunea 256, 26 nov 2026)

**Nou pentru placă:** un singur program, `kronos_token::TOKEN32` (Mesa NAK 25.0.7, 92 de registre, 2 176 B partajați, SHA-256
`40d9749cc8074f639ca0eb3fafae9686b2ca4d2491e868b3b2544ec7faa1ad5f`), care face tot simbolul într-o lansare. Niciun registru nou.
Bariera pe toată placa la G = 384 / 768 a trecut deja (os-465). Predicțiile, scrise înainte: `cyber-gpu/docs/C-157_TOKEN32_PREDICTII.md`
(pe scurt: aceleași 16 simboluri; G = 384 probabil **mai lent** decât azi, ~1,26 ms; G = 768 ~1,0 ms; driverul mult mai puțin).

**Opțiunea propusă:** `--token32` (oprită implicit), după generarea obișnuită, în același lanț (aceeași încălzire a plăcii).

**Rețeta (totul există în `nvidia_ga104::kronos_token`, doar adăugări, la commit-ul din `stare/cyber-gpu.md`):**
1. `TOKEN32` urcat o dată în VRAM, SHA-256 verificat, ca celelalte programe.
2. Pe poziție: aceleași `Launch` ale listei `lipita-ln-32` + argmax de azi → `work_list32(&launches)`, scris în VRAM.
3. `sms` = `rmapi::GrCounts::sms` (48). `G = token32_groups_for_sms(sms, 8)` (= 384); a doua rulare cu `(sms, 16)` (= 768),
   **doar dacă G = 384 a trecut**. NU `token32_groups_for` (refuză mereu, os-456).
4. Bufferul constant: `token_cbuf_for(lista, contor, n, G)` (G la 0x0c); contorul pus la 0 înainte de fiecare lansare.
5. QMD: `token32_qmd_wide(program, cbuf, G, sms)` — alege singur treapta memoriei comune (32 KiB la 384, 64 KiB la 768). NU
   `token32_qmd` (treapta 8 KiB: la G = 384 grupurile n-ar încăpea deodată → bariera n-ar mai deschide).
6. O singură lansare pe poziție; aceeași limită de timp ca la proba barierei.

**Ce așteptăm în raport, pe fiecare G:** cele 16 simboluri și amprenta logit-urilor (lista de azi: `0x6c8501f0`) — trece = identice;
contorul barierei = (n − 1) · G după fiecare poziție (n = operațiile listei, ~88); timpul median pe simbol (placa) și cu driverul, lângă cele ale listei de azi din
același lanț. Dacă nu termină: la ce G și la ce poziție.

**Nicio schimbare incompatibilă.** Dacă la G = 384 iese mai lent, nu e o greșeală: e una dintre cele două ramuri prezise.
