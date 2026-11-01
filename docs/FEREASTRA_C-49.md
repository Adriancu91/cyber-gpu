# Fereastra următoare pe workstation (după I4 TRECE) — planul C-49

*1 nov 2026, sesiunea Cyber-GPU 30 (cloud, fără placă). Propunere, nu sarcină: o face Adrian, cu el de față.
Nimic de aici nu a rulat pe placă. Ordinea e fixă: un pas începe **doar** dacă cel dinainte a trecut; orice rezultat neașteptat = poză + oprire.*

**Unde suntem:** toate verificările de citire sunt dovedite pe placa reală — I0/I1 (identitatea), I2 (GFW_BOOT), I3 + I3b (VBIOS, FWSEC găsit,
29 oct), I4 (fișierele firmware 570.144 oficiale, antete citite 4/4, 1 nov). Niciuna n-a scris nimic în placă. **Pasul următor e, oricum am
alege, prima scriere în placă.**

## Recomandarea: întâi (a) proba tinygrad, apoi (b) driverul nostru

| | (a) proba tinygrad | (b) prima scriere a driverului nostru (I5g + I6) |
|---|---|---|
| Cine scrie în placă | tinygrad (driver open-source MIT, care pornește deja plăci RTX 30 cu același firmware 570.144) | codul nostru, care până azi a rulat **doar în simulare** |
| Ce aflăm | dacă placa, firmware-ul și secvența întreagă (FWSEC → Booter → GSP → o adunare pe GPU) merg pe workstation; plus „jurnalul de aur”: fiecare scriere în registre, în ordine | dacă primii noștri 18 registre (lista de mai jos) fac ce trebuie |
| Ce mai lipsește | nimic din codul nostru; trebuie stick Ubuntu 24.04 live, laptopul, cablu de rețea, internet | **cod nou în OS**: harta BAR0 de 16 MiB, Bus Master, o zonă de RAM pentru DMA, comanda care leagă totul (propunerea `idei/2026-11-01_cyber-gpu-i6-in-os.md` din hub) |
| Dacă nu merge | știm că nu e greșeala noastră → placa/firmware/BIOS | nu știm dacă greșeala e la noi sau la placă |

**De ce (a) întâi:** (1) un driver care merge deja pe plăci ca a noastră ne spune dacă drumul e liber, înainte să încercăm codul nostru netestat;
(2) jurnalul lui ne arată exact ce registre scrie și în ce ordine, iar noi comparăm cu lista noastră **înainte** de prima noastră scriere;
(3) (b) are nevoie oricum de cod nou în OS, care nu există încă — (a) se poate face imediat.
**Dacă (a) e prea greu de pregătit** (stick Ubuntu, rețea): (b) se poate face fără (a), dar atunci, la un eșec, nu vom ști a cui e vina.

## În aceeași fereastră, înainte de (a): verificarea I6 „doar citire” (dacă OS-ul o are)
Propunere pentru OS (comanda `gpu pregatire-i6`, cod deja scris la noi): citește, **fără nicio scriere**, ce îi trebuie lui I6:
siguranța „ucode version” a FWSEC (registrul `0x8241E0`, nou în sesiunea 30) și semnătura aleasă (din cele 3 ale VBIOS-ului), mărimea VRAM
(`0x1183A4`), dacă WPR2 e deja ridicat (`0x1FA828` trebuie 0 după o pornire rece), regiunea FRTS calculată. Cere ca OS-ul să mapeze BAR0
întreg (16 MiB), nu doar 4 MiB. Trece dacă: semnătura e aleasă (`index 0, 1 sau 2`), WPR2 = 0, „0 scrieri”. Poză.

## (a) Proba tinygrad — ⚠️ SCRIE ÎN PLACA VIDEO (prin tinygrad)
Neschimbată: `docs/FEREASTRA_C-36.md` §3 (pașii, comanda, oprirea sigură) și `tools/proba-tinygrad/proba.sh`.
- **Se scrie:** registrele plăcii și memoria video (VRAM), prin driverul tinygrad: pornirea FWSEC/Booter/GSP și o adunare de 64 de numere.
- **Nu se poate strica:** memoria permanentă a plăcii (VBIOS, pe cip separat) — tinygrad n-o scrie, iar placa rulează doar programe
  semnate de NVIDIA (le verifică singură). Totul e în memorie volatilă.
- **Rezultat bun:** `PROBA: OK` + `jurnal.txt` cu rândurile `wreg:` → în hub, `rezultate/<data>-workstation-tinygrad/` (fără fișierele firmware).

## (b) Prima scriere a driverului nostru — ⚠️ SCRIE ÎN PLACA VIDEO — fereastra de după (a)
**Înainte:** codul OS din propunerea din `idei/`; opțiunea `hw-write` pornită doar în stick-ul de test (oprită în stick-urile obișnuite);
comanda cere confirmare scrisă la tastatură (`DA`). Pornire **rece** (oprire completă înainte), ca WPR2 să fie 0.

**Pașii, în ordine, fiecare cu oprire la prima nepotrivire:**

| # | Pas | Ce se scrie, exact | Trece dacă |
|---|---|---|---|
| 1 | Verificarea I6 (doar citire, ca mai sus) | nimic | semnătură aleasă, WPR2 = 0 |
| 2 | Pregătirea în RAM (I5): FWSEC copiat din VBIOS în RAM, semnătura aleasă pusă în el, comanda FRTS pusă în zona lui de date | **nimic în placă** (doar memoria calculatorului) | imaginea are mărimea din descriptor (0xEA00 → 234 de blocuri de 256 B) |
| 3 | **I5g — Bus Master pornit** (placa are voie să citească RAM-ul) | **o singură scriere**: registrul de comandă PCI al plăcii (configurație, `0x04`), bitul 2; valoarea înainte/după în jurnal | bitul 2 = 1 la recitire; ecranul merge |
| 4 | **I6 — FWSEC-FRTS** pe falcon-ul GSP | **doar 18 registre**, toate în fereastra GSP `0x110000–0x113FFF` (lista de mai jos), ~720 de scrieri (majoritatea, 3 pe bloc la copierea celor 234 de blocuri), fiecare cu înainte/după în jurnal | `SW_SCRATCH(0xE)` (`0x1438`) biții 31:16 = 0 și WPR2 (`0x1FA824/828`) exact unde am cerut (regiunea FRTS de 1 MiB din capătul VRAM) |
| 5 | Oprire | — (Bus Master rămâne cum e; la oprirea completă se șterge oricum) | jurnalul e pe stick |

**Cele 18 registre ale lui I6** (fixate de testul `gsp::hs_load::tests::i6_writes_exactly_the_listed_gsp_registers` — orice registru nou face testul
să pice): `0x1103C0` reset motor (1, apoi 0) · `0x111668` alegerea nucleului falcon · `0x110624` FBIF_CTL · `0x11010C` DMACTL = 0 ·
`0x110600` TRANSCFG(0) (memorie sistem, adrese fizice) · `0x110110`, `0x110128` adresa sursei DMA · `0x110114`, `0x11011C`, `0x110118`
câte un bloc de 256 B (sursă, destinație, comandă) · `0x111210`, `0x11119C`, `0x111198`, `0x111180` parametrii verificării semnăturii
(boot ROM) · `0x110104` adresa de pornire · `0x110040`, `0x110044` cutiile poștale = 0 · `0x110100` pornirea procesorului falcon.
Ce face apoi programul FWSEC (semnat NVIDIA, nu al nostru): ridică zona protejată WPR2 în VRAM — **asta e tot ce schimbă în placă**.

**Ce nu se poate strica:**
- **VBIOS-ul nu se scrie:** codul nostru nu are nicio cale de scriere spre fereastra VBIOS (`0x300000`); se citește doar. Comanda dată lui
  FWSEC e „FRTS” (pregătește o zonă în VRAM), nu actualizare de VBIOS.
- **Siguranțele (fuses) doar se citesc.**
- Codul străin nu poate rula: boot ROM-ul plăcii verifică semnătura NVIDIA; dacă am greși semnătura sau un parametru, falcon-ul refuză și se
  oprește cu un cod de eroare (îl citim și ne oprim).
- Orice scriere în afara ferestrei GSP e refuzată de cod (`GuardedBar0`) **înainte** să ajungă la placă; FWSEC nu rulează de două ori (dacă
  WPR2 e deja ridicat, ne oprim).
- Totul e volatil: WPR2, falcon-ul, Bus Master dispar la oprirea completă. Discul, Windows și Kronos nu sunt atinse.

**Ce se poate întâmpla (și ce faci):**
- Ecranul se stinge sau se blochează (placa ține și ecranul) → aștepți 1 minut (jurnalul se scrie pe stick înainte și după fiecare pas),
  apoi oprire completă.
- **Revenirea, oricând:** butonul de pornire ținut 10 s → cablul de alimentare scos 30 s → pornire în Windows. Verificare: `nvidia-smi`
  vede placa, Kronos pornește normal. Dacă `nvidia-smi` nu vede placa: încă o oprire cu cablul scos 30 s; dacă tot nu, poză și ne scrii
  (nu repeta pasul).

**NU în (b):** Booter pe SEC2 și pornirea GSP (I7–I8), soneria, lansările pe GPU. Vin după ce I6 trece pe placă, cu o listă nouă.

## După fereastră
Pozele + `raport/*.txt` + (la a) folderul `proba-tinygrad/<data>/` → hub `rezultate/<data>-workstation-.../`.
**Fișierele firmware nu se pun niciodată în hub sau în repo.**
