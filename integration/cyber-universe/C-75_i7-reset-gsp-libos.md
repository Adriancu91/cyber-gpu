# C-75 — I8 oprit pe placă la cozile GSP: cauza și reparația (pentru sesiunea OS)

**De la:** cyber-gpu, sesiunea 51 (5 nov 2026, ~13:00 ora României). **Testat doar în SIMULARE** + regula verificată pe jurnalul real.

## Ce arată jurnalul real (`rezultate/2026-11-05-workstation-os242/os246-gpu-jurnal-002.txt`)
```
bar0 w 0x110040: 0x00000000 <- 0x09c71000 => 0x00000000
bar0 w 0x110044: 0x00000000 <- 0x00000000 => 0x00000000
```
Cutia GSP 0 **nu a păstrat** adresa LibOS (scris `0x09c71000`, citit 0). Booter a mers (cutia SEC2 0 = 0, WPR2 lărgit la ~190 MiB,
RISC-V activ), dar GSP-RM a pornit **fără** adresa argumentelor LibOS → nu și-a creat cozile → I8 „antetul cozii de stare nu are forma 570.144”.
`verifica_jurnal_lant os246-gpu-jurnal-002.txt 0x09c61000` dă acum exact asta la `i7-booter-gsp`.

Cauza (DEDUS): după FWSEC-FRTS falcon-ul GSP rămâne cum l-a lăsat FWSEC (nucleu falcon, mod HS); nouveau și RM îl **resetează în modul RISC-V**
înainte să scrie cutiile. Este exact „Completarea 08:41” din `stare/cyber-universe.md` (întrebarea 1) — confirmată de jurnal.

## Ce oferă cyber-gpu (fixare ≥ commit-ul acestei note; doar adăugări)
- `falcon::booter_load_dma_libos(bar, clock, ucode, image, wpr_meta_addr, libos_addr)` — I7 complet, în ordinea plăcii:
  1. `gsp_reset_into_riscv`: `0x1103c0` = 1, apoi 0 (+ așteptarea curățării), `0x111668` (BCR_CTRL GSP) = `0x111`, recitit cu CORE_SELECT = 1;
  2. `write_libos_mailboxes`: `0x110040`/`0x110044` = adresa LibOS, **recitite** — dacă nu s-au păstrat, oprire ÎNAINTE de Booter (`HardwareFault{code: valoarea citită}`);
  3. Booter pe SEC2 exact ca până acum;
  4. `0x110080` (`FALCON_OS`) = 570144 (`GSP_RM_APP_VERSION_570_144`); 5. așteptarea RISC-V activ.
- `i6_log::check_i7_libos(events, wpr_meta, None)` + `chain_log` o folosește singur când pasul I7 începe în fereastra GSP (`ChainJournal::i7_libos`).
- `booter_load_dma` / `check_i7` rămân neschimbate.

## Ce ar trebui schimbat în OS (P6-56)
- La I7: în locul scrierilor proprii în `0x110040/44` + `booter_load_dma` + `0x110080`, un singur apel `booter_load_dma_libos(…, I7Prep::libos)`.
  Scrieri noi față de os-246: doar 3 (`0x1103c0` ×2, `0x111668`), toate în fereastra GSP, deja permise de `GuardedBar0`.
- Nimic altceva nu se schimbă (I6, I8, golden, I9 la fel). Test în QEMU/simulare; pe placă: o fereastră cu Adrian, pornire rece.
- Așteptat pe placă: rândul `0x110040 … => 0x09c71000` (păstrat). Dacă tot 0 → oprire curată înainte de Booter (date noi, fără risc).

## Răspunsurile la întrebările din C-75
1. `0x111388` = `0xa0`: bitul 7 (`ACTIVE_STAT`) = 1 → RISC-V **activ**; bitul 4 (`HALTED`) = 0. GSP-RM rula.
2. Așteptarea de ~2 s nu e cauza: GSP-RM nu avea adresa LibOS (cutia citită 0), deci nu avea unde să-și facă cozile — ar fi așteptat la nesfârșit.
3. Formatul cozii de stare nu se poate judeca până nu există coada; rămâne cel din 570.144 (simularea îl verifică). Se vede la fereastra următoare.

## Turul de după (faptul `i6.trece`)
În cyber-gpu, `i6.trece` din `read_tour` înseamnă în continuare „I6 se poate încerca” (neschimbat). Nou: faptul `i6.wpr2.etapa` =
`coborat (pornire rece)` / `FRTS (dupa I6)` / `largit pentru GSP-RM (dupa I7)` / `alta forma (de analizat)` (`I6Check::wpr2_stage`).
Pentru faptul vostru `i6.trece` (= rezultatul I6): după I7, „lărgit pentru GSP-RM” este normal și înseamnă că I6 a trecut înainte.
