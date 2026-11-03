# Primele scrieri în placă, direct din Cyber Universe — planul C-49 + C-51

*1 nov 2026, sesiunea Cyber-GPU 30 (cloud, fără placă). Propunere, nu sarcină: o face Adrian, cu el de față.
Nimic de aici nu a rulat pe placă. Ordinea e fixă: un pas începe **doar** dacă cel dinainte a trecut; orice rezultat neașteptat = poză + oprire,
fără reîncercări în aceeași fereastră.*

> **Schimbare (C-51, decizia lui Adrian din 1 nov 12:00):** pe placa reală rulează **doar Cyber Universe**. Proba tinygrad pe Ubuntu
> (C-20 A) **s-a anulat**. Prima versiune a acestui plan (care o recomanda întâi) e înlocuită de cea de mai jos.

**Unde suntem:** toate verificările de citire sunt dovedite pe placa reală — I0/I1, I2 (28 oct), I3 + I3b (29 oct), I4 (1 nov). Niciuna n-a
scris nimic în placă. Pasul următor e prima scriere a driverului nostru.

## Ce ține locul „jurnalului de aur” de la tinygrad
1. **Secvența de referință în simulare**, scrisă după sursele deschise (RM 570.144 și tinygrad, MIT, citite ca specificație): codul nostru
   I5g/I6 rulează cap-coadă pe placa simulată (`gpu-sim`), iar testele fixează **exact cele 18 registre** I6 și ordinea lor
   (`gsp::hs_load::tests::i6_writes_exactly_the_listed_gsp_registers`).
2. **Jurnalul fiecărei citiri/scrieri, pe stick, pe loc**: `Journal::with_sink` trimite fiecare acces OS-ului în clipa în care se face; OS-ul
   îl scrie cu ora lui (`[ 606.590] … bar0 w 0x110624: … <- … => …`). Dacă placa se blochează la jumătate, accesele de dinainte rămân pe stick.
3. **Verificarea după fereastră**: `nvidia_ga104::i6_log` citește rândurile de pe stick și verifică ordinea scrierilor (fiecare bloc copiat,
   nimic în afara ferestrei GSP, nimic după pornire); orice abatere apare cu numărul rândului.

## Pașii ferestrei (cu `gpu pregatire-i6`, apoi `gpu i6 --scrie` — comenzile noi propuse pentru OS în `idei/`)
**Înainte:** oprire completă a workstation-ului (cablul scos 30 s), ca placa să pornească „rece” (zona protejată WPR2 coborâtă).
Stick cu OS-ul de test (opțiunea `hw-write` pornită doar în el) și firmware-ul 570.144 (cum e deja).

| # | Pas | Ce se scrie, exact | Ce se citește înapoi ca dovadă | Trece dacă | Dacă nu trece |
|---|---|---|---|---|---|
| 1 | `gpu pregatire-i6` (**doar citire**) | nimic | siguranța FWSEC `0x8241E0`, mărimea VRAM `0x1183A4`, WPR2 `0x1FA828` | semnătura aleasă = `0`, `1` sau `2`; VRAM `8192 MiB`; `WPR2 coborat`; „0 scrieri” | poză; **STOP** (nu se scrie nimic) |
| 2 | Pregătirea în RAM (I5): FWSEC copiat din VBIOS, semnătura aleasă și comanda FRTS puse în el | **nimic în placă** | — (în memoria calculatorului) | imaginea are 0xEA00 octeți (234 de blocuri de 256 B) | poză; STOP |
| 3 | ⚠️ **I5g — Bus Master pornit** | **o scriere**: registrul COMMAND al plăcii (configurație `0x04`), doar bitul 2 | COMMAND recitit: bitul 2 = 1, ceilalți biți neschimbați; `BOOT_0` recitit = `0x174…` (placa răspunde) | rândul `cfg w 0x004: … <- … => …` cu bitul 2 pus; ecranul merge | codul pune la loc valoarea veche singur; poză; STOP |
| 4 | ⚠️ **I6 — FWSEC-FRTS** pe falcon-ul GSP | **18 registre, toate în fereastra GSP** `0x110000–0x113FFF` (lista de mai jos), ~720 de scrieri (3 pe fiecare bloc copiat), fiecare cu înainte/după | `CPUCTL` → HALTED (≤ 2 s); `SW_SCRATCH(0xE)` (`0x1438`) biții 31:16 = 0; WPR2 `0x1FA824/828` = exact regiunea FRTS de la pasul 1 | „pas I6 FWSEC-FRTS: ok” | driverul se oprește la prima eroare (nicio scriere în plus); poză cu ultimele rânduri; pasul 5 |
| 5 | ⚠️ Bus Master oprit la loc | **o scriere**: COMMAND înapoi la valoarea de la pasul 3 | COMMAND recitit = valoarea veche | rândul `cfg w 0x004` cu bitul 2 șters | poză; oprire completă |
| 6 | Oprire completă (cablul scos 30 s) → Windows | — | `nvidia-smi` vede placa; Kronos pornește normal | da | încă o oprire cu cablul scos; dacă tot nu, poză și ne scrii (nu repeta) |

**Cele 18 registre ale lui I6** (fixate de test — orice registru nou face testul să pice): `0x1103C0` reset motor (1, apoi 0) · `0x111668`
alegerea nucleului falcon · `0x110624` FBIF_CTL · `0x11010C` DMACTL = 0 · `0x110600` TRANSCFG(0) (memorie sistem, adrese fizice) ·
`0x110110`, `0x110128` adresa sursei DMA · `0x110114`, `0x11011C`, `0x110118` câte un bloc de 256 B (sursă, destinație, comandă) ·
`0x111210`, `0x11119C`, `0x111198`, `0x111180` parametrii verificării semnăturii (boot ROM) · `0x110104` adresa de pornire ·
`0x110040`, `0x110044` cutiile poștale = 0 · `0x110100` pornirea procesorului falcon.
Programul FWSEC (semnat NVIDIA, nu al nostru) ridică apoi zona protejată WPR2 în VRAM — **asta e tot ce schimbă în placă**.

## Ce nu se poate strica
- **VBIOS-ul nu se scrie:** codul nostru nu are nicio cale de scriere spre fereastra VBIOS (`0x300000`); comanda dată lui FWSEC e „FRTS”
  (pregătește o zonă în VRAM), nu actualizare de VBIOS.
- **Siguranțele (fuses) doar se citesc.**
- Cod străin nu poate rula: placa verifică semnătura NVIDIA; o semnătură sau un parametru greșit → falcon-ul refuză și se oprește cu un cod de
  eroare (îl citim și ne oprim).
- Orice scriere în afara ferestrei GSP e refuzată de cod (`GuardedBar0`) **înainte** să ajungă la placă; FWSEC nu rulează de două ori.
- Totul e volatil: WPR2, falcon-ul, Bus Master dispar la oprirea completă. Discul, Windows și Kronos nu sunt atinse.

## Ce se poate întâmpla
- Ecranul se stinge sau se blochează (placa ține și ecranul): aștepți 1 minut, apoi oprire completă. Jurnalul de până atunci e pe stick.
- **Revenirea, oricând:** butonul de pornire ținut 10 s → cablul scos 30 s → Windows → `nvidia-smi` + Kronos.

**NU în această fereastră:** Booter pe SEC2 și pornirea GSP (I7–I8), soneria, lansările pe GPU. Vin după ce I6 trece pe placă, cu o listă nouă.

## După fereastră
Pozele + `raport/*.txt` (cu jurnalul complet) → hub `rezultate/<data>-workstation-i6/`. Noi rulăm `i6_log::check_i6` pe jurnal și comparăm
cu simularea. **Fișierele firmware nu se pun niciodată în hub sau în repo.**

## Fereastra 4 nov 2026, 07:00 (C-63) — partea GPU, pe scurt

*Sesiunea 42 (cloud, fără placă). Pașii OS și Release-ul de folosit: în hub, `stare/cyber-universe.md` (acum `os-220`).*

1. **Înainte de orice pe placă:** copiezi `raport\gpu-jurnal-002.txt` de pe stick-ul de test **vechi** (cel din 3 nov) în hub `rezultate/`.
   E singura dovadă a capătului WPR2 (HI) scris de FWSEC pe 3 nov; stick-ul nou nu-l are.
2. Pornire rece (cablul scos 30 s) → stick-ul de test nou → `gpu pregatire-i6` → poză (trebuie „WPR2 coborat”, „TRECE”, „0 scrieri”).
   Altceva → STOP.
3. `gpu i6 --scrie` + `DA` → poză (rândul nou „WPR2 citit dupa I6: LO … HI … (capat: …)”), apoi `raport`.
4. Ecran blocat: 1 minut, oprire completă, **nu repeta azi**.
5. Pozele + `raport\*.txt` + `raport\gpu-jurnal-*.txt` → `rezultate/<data>-workstation-i6/`.

**Ce rulăm noi după:** `verifica_jurnal_i6` pe fiecare jurnal (ordinea celor ~720 de scrieri + LO/HI față de FRTS `0x1ffe00000..0x1fff00000`).
Dacă I6 trece: I7 (Booter pe SEC2) **nu** se încearcă în această fereastră — vine cu o comandă separată.
