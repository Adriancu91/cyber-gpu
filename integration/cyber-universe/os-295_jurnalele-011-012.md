# Pentru sesiunea OS — os-295, jurnalele 011 și 012: starea plăcii după un I7 / Booter Unload eșuat (cyber-gpu `3404313` sau mai nou)

Sesiunea Cyber-GPU 94, 12 noiembrie 2026. Analiza: `docs/os-295_ANALIZA_011_012.md`. Pe scurt: la repetarea variantei 1 (jurnalul 011, stick la ~1 s/rând),
**I7 a picat** — Booter a terminat cu 0, dar RISC-V-ul GSP a rămas `CPUCTL = 0x10`; scrierile au fost identice cu 010 (care a trecut); cauza nu se poate stabili
din jurnal. `gpu recupereaza` (jurnalul 012) a primit de la Booter Unload codul **0x88** (nepublicat); WPR2 a rămas ridicat.

## 1. Fără nicio schimbare în OS (doar legarea)

`kernel/Cargo.toml`: cele 5 crate-uri la `rev = "3404313"` (sau mai nou; conține și `e49ca76` = varianta 1 implicită, `45dfc77` = motivul eșecului pe stick). De atunci:

- după un I7 eșuat (`booter_load_dma_libos`), înainte ca eroarea să ajungă la lanț, jurnalul primește un rând `nota: I7: … starea placii (doar citiri):` și **9 rânduri**
  `valoare … = 0x…` (`PGC6 SCRATCH_14`, `SEC2 CPUCTL`, `SEC2 MAILBOX0`, `SEC2 MAILBOX1`, `GSP RISCV_CPUCTL (bit 7 = activ)`, `GSP MAILBOX0`, `GSP MAILBOX1`,
  `GSP RISCV_BCR_CTRL`, `GSP FALCON_OS`) — doar citiri din ferestrele GSP/SEC2 și PGC6, deja permise;
- la fel după un Booter Unload cu cod (`booter_unload_dma`, `gpu recupereaza`), după rândul `MAILBOX0 (Booter a dat eroare) = …`;
- așteptarea „RISC-V activ” după Booter are limita **5 s** (`falcon::RISCV_ACTIVE_US`, ca nova) în loc de 2 s — I7 poate dura cu 3 s mai mult **doar** când pică.

Interfața publică: doar adăugări (`falcon::{journal_i7_failure, I7_FAIL_RISCV_OFF, I7_FAIL_BOOTER_CODE, UNLOAD_FAIL_STATE, I7_FAIL_EXTRA_READS}`); valoarea
`RISCV_ACTIVE_US` s-a schimbat (2 → 5 s). `verifica_jurnal_lant` și `compara_variante` citesc jurnalele noi fără schimbare (rândurile `nota`/`valoare` nu sunt pași).

## 2. Ce ține de OS (C-113), văzut din jurnalele GPU

- În 011 lentoarea a început **la rândul ~1440 (316,6 s), în mijlocul încărcării Booter-ului prin DMA**, și a rămas ~0,83–0,96 s/rând până la capăt; 012 a fost lent de la primul rând.
  I7 a picat în singura rulare lentă — corelație, nu cauză dovedită (`docs/os-295_ANALIZA_011_012.md` §1.5).
- Propunerea de acolo (jurnal în RAM, scris pe stick în bucăți pentru pașii sensibili la timp) ar scoate variabila „timp” din I6–I8. Driverul n-are nimic de schimbat pentru asta:
  sinkul C-51 e al OS-ului; `Journal` din `gpu-core` e deja în RAM.

## 3. Testul următor pe placă (după ce stick-ul scrie normal)

Pornire rece → `gpu lant` (fără `--varianta`; = varianta 1) → `DA` → după repornire `trimite`. **`gpu recupereaza` nu se mai rulează** până nu decide Adrian despre
FWSEC-SB înainte de Booter Unload (analiza §2.3) — ar da același 0x88. `compara_variante 0x9c66000 gpu-jurnal-010.txt <jurnalul nou>` pentru citire.
