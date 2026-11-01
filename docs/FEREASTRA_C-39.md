# Fereastra următoare pe workstation (după C-39) — propunere

> **1 nov 2026:** pasul 1 (I4) a TRECUT pe placă. Planul nou: [FEREASTRA_C-49.md](FEREASTRA_C-49.md).

*29 oct 2026, sesiunea Cyber-GPU 18. Propunere, nu sarcină: o face Adrian, cu el de față. Nimic de aici nu a rulat pe placă.*

**Înainte:** un Release OS cu cyber-gpu fixat pe `ef97110` sau mai nou (nota `integration/cyber-universe/C-39_gpu-firmware-antete.md`).
Stick-ul OS are deja firmware-ul 570.144 (verificat 29 oct: 4/4 oficiale).

## 1. `gpu firmware` (I4) — doar citire, 0 accese la placă
Trebuie: 4 rânduri „ok: … = oficial”, sub fiecare „antet citit: …” (la cele două Booter: `HS v2: 2 semnaturi de 384 octeti, 1 aplicatii`),
`LICENCE.nvidia: prezenta`, ultimul rând `criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4)`. Poză.
Dacă tot apare „antet NECITIT”: poză cu rândul (textul erorii ne spune ce cuvânt din antet nu se potrivește) și **nu** treci la pasul 2.

## 2. ⚠️ SCRIE ÎN PLACA VIDEO — proba de referință cu tinygrad (C-20 A, aprobată 26 oct)
Neschimbată față de `docs/FEREASTRA_C-36.md` §3 (stick Ubuntu 24.04 live, laptopul pentru SSH, ~30 min, oprire sigură descrisă acolo).
De ce înaintea driverului nostru: jurnalul ei (`wreg:`) e „jurnalul de aur” cu care comparăm simularea înainte ca driverul nostru să scrie
ceva în placă.

## 3. I5+ cu driverul nostru — NU în fereastra următoare
- **I5** (aranjamentul VRAM, `GspFwWprMeta`, imaginea GSP-RM în RAM): se calculează în memoria calculatorului; **nu scrie în placă**, dar are
  nevoie de mărimea VRAM citită de pe placă (registru doar-citire). Se poate adăuga ca `gpu plan-wpr` (doar citire + calcul) — propunere
  pentru OS după fereastra 2.
- **I6** (FWSEC-FRTS) = **prima scriere a driverului nostru în placă** (încărcarea ucode-ului semnat în falcon prin DMA, pornirea lui).
  Doar după ce jurnalul probei tinygrad e în hub și comparat cu simularea; în spatele opțiunii `hw-write` (oprită implicit) și al unei
  confirmări scrise de Adrian în fereastră. Lista exactă o scriem atunci.
