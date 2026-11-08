# Fereastra de luni 9 nov 2026 — partea Cyber-GPU (C-83 / C-87)

*Scrisă duminică 8 nov, ~07:45 (ora României), sesiunea Cyber-GPU 68 (cloud, fără placă). Lista de pași a OS-ului (P6-63) o scrie sesiunea OS
în `stare/cyber-universe.md`; aici e ce așteaptă driverul de la fiecare pas și cum citim jurnalul după. **Nimic de aici nu a rulat pe RTX 3070 Ti**:
I6 și I7 au trecut pe placă (5 nov), tot ce urmează e doar SIMULARE. Regulile din `docs/FEREASTRA_C-49.md` rămân: pornire rece, oprire la primul
rezultat neașteptat, nicio reîncercare în aceeași fereastră, firmware-ul nu intră niciodată în repo sau în hub.*

## Ce trebuie să existe
- OS: Release-ul cel mai nou al sesiunii OS (acum `os-275`), stick-ul de test `cyber-universe-gpu-scrie-usb.img`. OS-ul folosește cyber-gpu `34a9793`.
- Stick-ul cu firmware (`firmware\nvidia\ga102\gsp\`, „stick-ul normal”, văzut de OS la `/usb2`): cele 4 fișiere 570.144 — `booter_load-570.144.bin`,
  `booter_unload-570.144.bin`, `bootloader-570.144.bin`, `gsp-570.144.bin` — din pachetul oficial `linux-firmware` (sumele SHA-256 așteptate sunt în
  `crates/nvidia-ga104/src/firmware.rs`; OS-ul le verifică singur, I4).
- Modelul de control `mic_q8_0.kgr` (pentru I14) e deja în imaginea stick-ului de test. **#513** (`kronos_513_q8_0.kgr` + `.sha256`, 105,7 MiB) stă pe
  stick-ul normal, în `kronos\` (pe stick-ul de test nu încape) → în OS: `/usb2/kronos/kronos_513_q8_0.kgr`.

## Lanțul (`gpu lant` + „DA”, o singură dată) — `WINDOW_CHAIN_C83`, 17 pași

| # | Pas | Pe placă | Scrieri noi față de ce a trecut deja | Trece dacă |
|---|---|---|---|---|
| 1 | `pregatire-i6` | doar citire | — | WPR2 coborât, 0 scrieri |
| 2 | `bus-master-pornit` | 1 scriere în configurație | — | bitul 2 pus |
| 3–4 | `i6-fwsec-frts`, `i6-wpr2-citit` | ca pe 5 nov (TRECE) | — | WPR2 la ultimul bloc de 128 KiB |
| 5 | ⚠️ `i7-booter-gsp` | **ordinea nouă C-75**: reset GSP în RISC-V, cutiile LibOS recitite, Booter, `FALCON_OS` | **3**: `0x1103c0` ×2, `0x111668` (fereastra GSP) | rândul `0x110040 … => <adresa LibOS>` (păstrat, nu 0) |
| 6 | `i7-citit` | doar citire | — | GSP RISC-V activ |
| 7 | ⚠️ `i8-gsp-init` | prima dată pe placă | soneria GSP + secvențiatorul (doar ferestrele GSP/SEC2) | `GSP_INIT_DONE` primit |
| 8 | `i8e-golden` | mesaje RPC | o sonerie GSP pe mesaj | fiecare răspuns RM = 0 |
| 9 | `i9-canal-lucru` | tabelele de pagini în VRAM | fereastra VRAM (4 registre) | răspunsuri RM = 0, citirea înapoi = ce am scris |
| 10 | `i10-add64` | primul calcul | soneria de lucru `0xbb0090` (din antetele MIT, încă neobservată) | 64 de adunări identice bit cu bit cu procesorul |
| 11 | `i10-cost-lansare` | 16 lansări cu marcă de timp | — | 17 înregistrări, timp crescător |
| 12 | `i11-copiere` | 8 MiB RAM → VRAM → RAM | niciun registru nou; **nou: placa citește/scrie prin DMA paginile RAM ale OS-ului** | întoarcerea identică, viteza din ceasul plăcii |
| 13 | `i12-q8-matvec` | 3072 × 768 Q8_0 | — | identic bit cu bit |
| 14 | `i13-strat` | un strat 768/12 pe 4 poziții | — (~12 MB VRAM) | abatere ≤ 1e-4 + 1e-4·\|x\| |
| 15 | `i14-simbol` | `mic_q8_0.kgr`, 3 texte, 35 de lansări pe simbol | — | simbolurile = vectorii lui Kronos, logit-uri ≤ 1e-3 |
| 16–17 | `bus-master-oprit`, `tur-dupa` | curățare (rulează și după un eșec) | — | Bus Master la loc |

Apoi, dacă tot lanțul a trecut: **oprire completă + cablul scos 30 s** (lanțul nu rulează de două ori la aceeași pornire), pornire de pe stick-ul de test
și `run kronos --gpu --model /usb2/kronos/kronos_513_q8_0.kgr --n 32 Ana are mere.` + „DA” — rulează iar tot lanțul, apoi #513 în pasul `i14-simbol`
(dacă e timp pentru o singură pornire, se poate sări direct la această comandă). `kronos --gpu` cu #513 (generare continuă, oprire la prima eroare, VRAM ~142 MiB verificat înainte); la orice eroare
Kronos continuă pe procesor. „Un singur program pe simbol” (`token.comp`) **nu** intră luni: abia după ce lansările separate trec pe placă.

**Cel mai probabil punct de oprire:** pasul 5 sau 7 (prima dată cu ordinea nouă la I7; prima dată I8 pe placă). O oprire curată acolo e un rezultat bun:
jurnalul spune exact ce a lipsit.

## După fereastră (noi, din jurnalul copiat în `rezultate/`)
```
cargo run --release -p nvidia-ga104 --example verifica_jurnal_lant -- gpu-jurnal-00N.txt <adresa GspFwWprMeta tipărită de OS> [apeluri RM | -]
```
Din sesiunea 68 unealta recunoaște singură lanțul C-83 (17 pași, tipărește „lantul: C-83”) și verifică și I11–I14: doar fereastra VRAM și soneria
de lucru, orice altă scriere e numită cu pasul ei (T107). Ieșire: 0 = TRECE, 3 = oprit curat (Bus Master la loc), 2 = de analizat.

### Potrivire cu lista OS (P6-63, `stare/cyber-universe.md` 8 nov 20:50, Release `os-275`) — sesiunea Cyber-GPU 71
Recitită duminică 8 nov, ~23:55. Pașii, ordinea și „ce înseamnă rezultatul” din lista OS sunt aceiași ca aici; am potrivit doar trei lucruri de mai sus
(cele 4 fișiere de firmware, #513 pe al doilea stick la `/usb2`, pornirea rece separată pentru `kronos --gpu`).
**Jurnalul lui `kronos --gpu`** se verifică cu aceeași unealtă: citit în codul OS (`kernel/src/drivers/gpu_i14.rs`, doar citire), cererea #513 rulează
*în interiorul* pasului `i14-simbol`, după modelul de control, cu aceeași funcție (`run_model`): doar fereastra VRAM, tabelele noastre de pagini și
soneria de lucru, niciun apel RM — exact ce regula lui I14 acceptă. Rândurile de text „kronos --gpu: …” nu sunt rânduri de pas, deci nu încurcă
verdictul. Un eșec al cererii #513 nu face I14 „ESEC” (OS-ul continuă pe procesor) — de aceea, după fereastră, citim și rândul
`kronos --gpu: … simboluri generate pe placa` / `kronos --gpu: ESEC: …`, nu doar verdictul uneltei. Verificat prin citire de cod, nu rulat.
