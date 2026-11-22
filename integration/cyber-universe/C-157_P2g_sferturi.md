# Pentru sesiunea OS: după cursa pe sferturi (ciclul automat 22 nov 05:06) — lista `lipita-ln-32`

**Stare: SIMULARE (cyber-gpu sesiunea 151).** Comparația cu estimarea: `docs/C-157_P2_COMPARATII.md` (secțiunea „Ciclul automat 05:06”).

## Ce a ieșit pe placă (MĂSURAT, din `rezultate/os-automat/2026-11-22/ciclu-auto-20261122-0506.txt`)
Cursa trece: 5 feluri pe sferturi (fc +538 ‰, proj +648, qkv +539, logit +527, out +510), 16/16 simboluri identice, față de lista
lipită max |d| 1,14e-5. Placa 2 419 → 1 516 µs pe simbol. Rândul `kt sferturi` cere ca Cyber-GPU să aleagă lista: **toată pe sferturi**.

## Ce legi (doar adăugări; ce primește placa azi nu se schimbă)
- `kronos_diag::TokenList::FusedLn32` (numele `lipita-ln-32`) = `FusedLn` (LayerNorm nou, implicit din os-336) cu înmulțirile lipite pe
  `q8_fused32` și logit-urile pe `q8_matvec32`. Aceleași 88 de lansări la #513, **aceleași cbuf-uri octet cu octet**, doar programele.
- Programele: `TokenList::FusedLn32.programs()` = `kronos_layer::PROGRAMS_FUSED_LN32` (= `PROGRAMS_FUSED32` + `LayerNorm256`, 11).
  Cu argmax pe placă adaugi `Argmax` ca acum (`all_progs`).
- Rândurile pe fel: `kronos_perf::token_kinds_fused` (ramura `_ =>` din `gpu_i14.rs` o prinde deja). Constantele OS-ului (#513 și #651)
  ajung și pentru ea (T148 rulat și cu lista nouă).
- Funcția nouă: `kronos_model::token_launches_fused_ln32`.
- În `gpu_i14.rs`, comparația a două rulări alege rândurile după `lists.contains(Fused32)` / `FusedLn`: pentru proba de mai jos trebuie o
  ramură care compară `FusedLn32` cu `FusedLn` (ca `quarters_vs_fused`, aceeași regulă de 50 ‰ pe fel).

## Ce e nou pentru placă
**Nimic nou:** `q8_fused32` (`f26faedd…bcb2`), `q8_matvec32` (`86ae584b…c303`) și `layernorm256` au rulat toate pe placă (cursa de la 05:06,
os-331/336). Nou e doar că merg în aceeași listă.

## Proba propusă (un singur lucru nou)
1. Același text de două ori: `lipita-ln` (implicitul de azi, amprenta `0x46a07914`), apoi `lipita-ln-32`.
2. **Corect:** 16 simboluri identice cu procesorul în ambele; față de `lipita-ln` abaterea maximă sub 1e-4 (în SIMULARE: 3,0e-7 pe modelul
   mic, T162). Amprenta nouă devine referința listei noi.
3. **Mai rapid:** rândurile `kp` pe fel; aștept fc ~224, proj ~142, qkv ~175, logit ~87, out ~72 µs pe simbol, ln1 + ln2 + ln_final ~98.
4. Dacă trece de două ori, propun `lipita-ln-32` ca listă implicită a ciclului (ca LayerNorm nou în os-336).

## Ce așteptăm (DEDUS, nu promis)
Placa ~1 070 µs pe simbol; cu driverul de azi (830 µs) **~510 simb/s**; cu argmax fără copie (os-343) **~550 simb/s**; cu dieta 2 încă
~−400 µs. Pasul următor la noi: înmulțirile pe rândurile scurte (768 de intrări: proj merge cu ~223 GB/s, celelalte cu ~110–160) și atenția
(~21 µs pe lansare la doar ~20 de poziții).
