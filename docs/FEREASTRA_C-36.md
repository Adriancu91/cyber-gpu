# Fereastra de pe workstation I3 → I4 (hub C-36) — pașii exacți

*Scris 28 oct 2026, sesiunea Cyber-GPU 15 (cloud, fără placă). Fereastra posibilă: **29 oct, cu Adrian de față**.
Ordinea e fixă: pasul următor începe **doar** dacă cel dinainte a trecut. Orice rezultat neașteptat = poză + oprire, fără reîncercări în aceeași fereastră.*

**Pașii 1 și 2 nu scriu nimic în placa video.** Doar pasul 3 scrie în placă și e marcat clar.

## 0. Înainte (pe Windows, ~10 min)
1. **Stick-ul OS**: arhiva cea mai nouă din hub, `integrare/cyber-universe/stick/` (C-35). Trebuie să aibă `gpu vbios --detalii` (C-31).
   Dacă are și `gpu fwsec` și `gpu firmware` (propunerea C-36 pentru OS), cu atât mai bine; dacă nu, la pasul 1c și 2 se folosește varianta B.
2. **Fișierele firmware pe același stick** (le ai de la centrul de comandă):
   - `firmware\nvidia\ga102\gsp\booter_load-570.144.bin`, `booter_unload-570.144.bin`, `bootloader-570.144.bin`, `gsp-570.144.bin`;
   - `firmware\LICENCE.nvidia`.
3. Verificare pe Windows, după copiere (PowerShell, în folderul `firmware\nvidia\ga102\gsp` de pe stick):
   `Get-FileHash *.bin -Algorithm SHA256 | Format-List` → sumele trebuie să înceapă așa:
   `booter_load` **4497E3EF…**, `booter_unload` **8E63DB5B…**, `bootloader` **82428F53…**, `gsp` **A8C3EBEE…**
   (sumele întregi: `docs/COMPATIBILITATE_FIRMWARE.md`). Poză sau copie a textului.
4. Kronos oprit și salvat pe workstation; Secure Boot oprit doar pentru test (ca la fereastra din 28 oct).

## 1. I3 — VBIOS (doar citire, 0 scrieri) — ~5 min
Pornire de pe stick → `cyber>`.

| # | Comanda | Trece dacă | Dacă nu trece |
|---|---|---|---|
| 1a | `gpu vbios` | lista de imagini merge **după** `0x24600` (nu se mai rupe la „NV”/`0x4e56`); cel puțin o imagine FWSEC; BIT cu jetonul `0x70`; rândul „complet … da”; „0 scrieri” | poză cu toate rândurile → **STOP** pentru placă (pasul 2 se poate face totuși: nu atinge placa) |
| 1b | `gpu vbios --detalii` | rândul „rezumat” spune `criteriul I3: indeplinit`; nicio „ruptura”; „0 scrieri” | poze cu **toate** rândurile (asta ne spune exact unde greșim) → STOP |
| 1c | `gpu fwsec` *(dacă există în OS)* | ultimul rând `criteriul I3b: TRECE`; aplicația `0x85` în tabela PMU; descriptorul `v3`; „0 scrieri” | poză cu cele 5 rânduri; **nu oprește** pasul 2 (FWSEC contează abia la pornirea GSP, care nu se face azi) |

Poze cu fiecare ecran. `log gpu 40` → poză la final.

## 2. I4 — fișierele de pe stick (0 accese la placă) — ~5 min
**Doar dacă 1a și 1b au trecut** (vezi mai sus excepția).

**Varianta A** — OS-ul are `gpu firmware`:
`gpu firmware` → trebuie 4 rânduri „ok: … = oficial”, sub fiecare „antet citit: …”, `LICENCE.nvidia: prezenta` și ultimul rând
`criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4); 0 scrieri, 0 accese la placa`. Poate dura până la ~30 s (citește ~64 MB de pe stick).

Ce înseamnă un rând greșit:
| Pe ecran | Ce înseamnă | Ce faci |
|---|---|---|
| `LIPSESTE de pe stick` | fișierul nu e în `firmware\nvidia\ga102\gsp\` sau are alt nume | poză; după fereastră îl pui la locul lui |
| `ALT FISIER: … octeti` | altă versiune sau copiere întreruptă | poză; recopiezi fișierul (pasul 0.2–0.3) |
| `ALT CONTINUT` | mărime bună, conținut stricat | poză; recopiezi fișierul |
| `antet NECITIT de parserul nostru` | fișierul e cel bun, **greșeala e în codul nostru** | poză; continui (pasul 3 nu folosește parserul nostru) |

**Varianta B** — OS-ul nu are încă `gpu firmware`: `ls /usb/firmware/nvidia/ga102/gsp` → poză (cele 4 nume lungi se văd în OS?);
sumele le-ai verificat deja pe Windows la pasul 0.3. Trece dacă OS-ul vede toate cele 4 nume și sumele de pe Windows au fost bune.

## 3. Ce urmează — **doar dacă 1 și 2 au trecut** — ⚠️ SCRIE ÎN PLACA VIDEO
**Proba de referință cu tinygrad** (C-20 A, aprobată de tine pe 26 oct, se rulează doar cu tine de față). Lista completă și comanda exactă:
`docs/TEST_MATRIX.md`, secțiunea **P**. Pe scurt:
- trebuie: al doilea stick cu **Ubuntu 24.04 live**, laptopul în aceeași rețea (pentru SSH), cablu de rețea la workstation, ~30 min;
- pornești workstation-ul de pe stick-ul Ubuntu (la GRUB: `e`, adaugi `modprobe.blacklist=nouveau,nvidia` la linia `linux`);
- de pe laptop, prin SSH: `sudo CONFIRM_ADRIAN=da bash /media/ubuntu/CYBER/tools/proba-tinygrad/proba.sh /media/ubuntu/CYBER`
  (stick-ul cu date trebuie să conțină și `tools/proba-tinygrad/proba.sh` din acest repo);
- **ce scrie în placă**: driverul tinygrad pornește placa (registre și memoria video) și face o adunare de 64 de numere. **Nu scrie în
  memoria permanentă a plăcii (VBIOS)**; după o oprire completă, placa pornește din nou ca din fabrică;
- se oprește singur, **fără nicio scriere**, dacă: lipsește `CONFIRM_ADRIAN=da`, placa `10de:2482` nu e găsită, nouveau/nvidia e încărcat,
  sumele firmware-ului nu se potrivesc;
- rezultat bun: rândul `PROBA: OK` și fișierul `jurnal.txt` cu mii de rânduri `wreg:` (asta e „jurnalul de aur” pentru driverul nostru).

**Oprire sigură, oricând:** `Ctrl+C` în fereastra SSH → `sudo poweroff`. Dacă nu răspunde: ții apăsat butonul de pornire 10 s, scoți cablul
de alimentare 30 s, apoi pornești în Windows. Verificare la final: în Windows, `nvidia-smi` arată placa și Kronos pornește normal.
Dacă `nvidia-smi` nu vede placa: încă o oprire completă cu cablul scos 30 s; dacă tot nu, poză și ne scrii (nu reîncerca proba).

**NU se face în fereastra asta:** pornirea plăcii cu driverul nostru (I5–I10: Bus Master, FWSEC, GSP). Codul există doar în simulare și nu e
în OS; vine într-o fereastră separată, după ce comparăm jurnalul probei cu simularea, cu o listă nouă.

## 4. După fereastră
Pozele + `raport/*.txt` de pe stickul OS + folderul `proba-tinygrad/<data>/` (dacă s-a făcut pasul 3) → hub `rezultate/2026-10-29-workstation-gpu/`.
**Fișierele firmware nu se pun în hub** (doar pozele cu rândurile de pe ecran).
