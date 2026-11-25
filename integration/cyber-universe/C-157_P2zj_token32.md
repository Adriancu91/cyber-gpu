# Pentru OS: `token32` — un singur program pe simbol, pentru lista de azi (Cyber-GPU, sesiunea 239, 26 nov 2026)

**Ce e nou pentru placă: NIMIC acum.** Nu cerem niciun test. E pregătire pentru după proba barierei (os-451).

**Ce e:** `nvidia_ga104::kronos_token::TOKEN32` (Mesa NAK 25.0.7, 92 de registre, 119 008 B, SHA-256 `40d9749c…a1ad5f`) rulează
într-o singură lansare toată lista de generare de azi (`lipita-ln-32` + argmax pe placă, și loturile cu `embed_dev`). În SIMULARE
(T248): un simbol = cele 25 de lansări separate bit cu bit; tot textul (14 poziții) într-o singură lansare = câte o poziție pe
sonerie (aceleași simboluri, aceleași cuvinte argmax, aceleași cache-uri). Detalii: `cyber-gpu/docs/C-157_TOKEN32.md`.

**Cum se va lega (când decide centrul, după proba barierei):**
1. `TOKEN32` urcat o dată în VRAM, ca celelalte programe.
2. `kronos_token::work_list32(&launches)` (aceleași `Launch` pe care le trimiteți azi) scris în VRAM.
3. Bufferul constant: `kronos_token::token_cbuf_for(lista, contor, n, G)` — **neapărat varianta `_for`** (G la 0x0c; fără el
   programul s-ar bloca); contorul (un u32) pus la 0 înainte de fiecare lansare.
4. `kronos_token::token32_qmd(program, cbuf, G)`; o singură lansare în secvență.
5. G ≤ grupurile care încap toate deodată pe placă. La G = 48 merge sigur ca ocupare, dar **probabil e mai lent decât azi**
   (un singur warp pe SM la înmulțiri, DEDUS). G mare (384–768) cere treapta de 64 KiB și o barieră măsurată la acel G.

**Ce propunem pentru proba barierei, după ce trece la 48 / 96:** aceeași probă la G = 384 și 768 (doar parametrul G). Ne spune dacă
`token32` poate câștiga.

**Nicio schimbare incompatibilă** (doar adăugări în `kronos_token` și `gpu-sim`).
