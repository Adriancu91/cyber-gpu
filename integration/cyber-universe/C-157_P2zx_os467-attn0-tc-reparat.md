# C-157 P2zx → cyber-universe: de ce os-467 nu a terminat, și programele reparate (sesiunea 255, 26 nov 2026)

Nota întreagă: `cyber-gpu/docs/C-157_OS467_ATTN0_TC_BLOCAJ.md`. **Nou pentru placă: aceleași două programe, recompilate**
(`attn_back_tile0_tc`, `attn_back_tile0_tc_v4`); nimic altceva.

1. **Cauza (DEDUS din listarea NAK, nu e bariera):** placa dP din memoria partajată avea rânduri de 65 de `float`. NAK scrie
   acumulatorii cu scrieri de 64 de biți; cu rânduri de 260 de octeți jumătate din ele cădeau pe adrese `4 mod 8` — adresă
   partajată nealiniată, care oprește firul pe placă; blocul nu se mai termină („ultima înregistrare = 0x0”). Toate cele 6 bariere
   sunt în blocuri uniforme (verificat). `gpu-sim` nu putea prinde: rulează un model pe procesor în locul instrucțiunilor.
   v4 avea aceeași greșeală (ar fi blocat la fel).
2. **Reparația:** rânduri de **68** (272 B, multiplu de 16). Aceeași aritmetică, aceeași ordine → **aceiași biți** în SIMULARE
   (toate testele vechi ale celor două programe și ale pasului mixt trec neschimbate). Se schimbă doar constantele modulelor:
   - `attn_back_tile0_tc`: SHA-256 `f904a08bd110296b38cbee87f4cc5af597f2f0ea8a911c4789c48fd40981b8e2`, 56 registre,
     **`SHARED.bytes` = 33 792** (era 33 024), constanta nouă `DP_ROW = 68`;
   - `attn_back_tile0_tc_v4`: SHA-256 `0bbe619e5911131cc4c87c996839562df74f58eda1181c6916bc96fdd277d2ff`, 56 registre, 33 792 B.
   Numele și semnăturile nu se schimbă; dacă voi citiți constantele din modul (`a0::SHARED`, `a0::ATTN_BACK_TILE0_TC_SHA256`),
   e destul să mutați dependența pe commit-ul acestei note. Unde aveți scris de mână „33 024” pentru aceste două programe, devine
   33 792 (doar text).
3. **Verificarea nouă (cererea voastră):** `nvidia_ga104::program_lint` — (a) orice `coopMatLoad` / `coopMatStore` în memoria
   partajată cu pasul de rând ne-multiplu de 16 octeți e refuzat (din 74 de programe, prindea exact cele două); (b) orice
   `bar.sync` dintr-un bloc neuniform al listării NAK 26.3.1 e refuzat (toate 13 listările trec; la unealta veche 25.0.7 regula nu
   dovedește nimic, 11 listări au bariere în blocuri nemarcate, toate rulate bine pe placă). Rulează în testele noastre
   (`lint_c157_os467`), nu cere nimic de la voi.
4. **Pe placă (aștept):** același rând ca os-467:
   `test-placa --atentie-placi --atentie2 --atentie3 --atentie-tc --atentie-tc0 --atentie-tc0-v4`.
   `atentie-tc0` TRECE 5 / 5 seturi (abaterea ~3e-4, sub 1e-3), apoi `atentie-tc0-v4` v4 = tc pe 4 / 4 seturi aliniate. Timpul
   `attn_back_tile0_tc` la `atentie-513-b4`: estimarea E7 (~1,8–3,2 ms pe pas față de 5,2) cu cel mult ~5 % în plus (lanțul `dot`
   citește acum cu conflict de 4 pe bancă) — DEDUS. P2zs / P2zt / P2zu rămân valabile, cu programele noi.
