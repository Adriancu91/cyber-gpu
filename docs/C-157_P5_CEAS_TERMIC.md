# C-157 punctul 5 (= C-151 punctul 2) — tabloul înainte/după viteza maximă și la N secunde, pragul de temperatură — SIMULARE

Sesiunea 169, 22 nov 2026 (cloud, fără placă). Comanda: C-151 punctul 2 („când tabloul merge pe placă: citirea lui înainte și după
comanda de viteză maximă și, mai târziu, la fiecare N secunde în lucrările lungi; propune un prag de temperatură la care OS-ul oprește
lucrul”). Tabloul merge pe placă din os-323 (21 nov 08:21: 35,8 °C, P8, 25,2 W, cele 3 cereri acceptate). OS-ul are de la os-356
managerul de lucrări lungi (fișa `lucrari\<nume>.txt`) — aici e partea care îi spune când să se oprească.

## 1. Planul (scris înainte de cod)
Doar adăugări, modul nou `nvidia_ga104::gsp::rusd_watch` (logică pură; nu trimite nimic, nu citește registre — citește doar pagina
tabloului, pe care OS-ul o are deja):
1. **Pragul (propunere, DEDUS):** `ThermalLimits::PROPOSED` = avertizare **80 °C**, oprire **85 °C**, pornire din nou doar sub
   **70 °C**, tablou „mut” (marca temperaturii neschimbată) mai mult de **10 s** = oprire. Plus oprire imediată la orice motiv de
   încetinire termic sau de urgență din pagină (biții 0x08 HW_SLOWDOWN, 0x20 / 0x40 termic, 0x80 frâna de putere).
2. **Ceasul de pază** `Watch`: `due(t)` (la fiecare N ms, implicit 5 000), `observe(t, &Dashboard, marca)` → `Continua` / `Avertizare`
   / `Opreste(motiv)`; oprirea se ține (nu revine singură); `may_start(&Dashboard)` pentru lucrarea următoare (sub 70 °C, fără motiv
   termic); rânduri `kw` puține: unul la fiecare schimbare de verdict și un rezumat la final (minim / maxim temperatură, putere maximă,
   câte citiri, câte mute) — nu un rând pe citire (o lucrare de 8 ore la 5 s ar fi 5 760 de rânduri).
3. **Înainte / după viteza maximă** `compare_boost(înainte, după)`: starea P și ceasul nucleului de o parte și de alta → `Urcata`
   (a ajuns în P0 sau nucleul a crescut ≥ 20 %), `DejaSus` (era deja P0 înainte — cazul os-323), `FaraEfect`, `Necunoscut` (secțiuni
   lipsă). Citirea „după” cere `≥ 2 ×` intervalul (1 s) după cerere, altfel pagina poate arăta încă valorile de dinainte.
4. Cititorul jurnalului `parse_kw` (ca `parse_kd`), pentru comparația de după un test pe placă.

**Ce e nou pentru placă când OS-ul îl leagă:** tabloul pornit **mai devreme** (înainte de viteza maximă, nu la sfârșitul lanțului) și
lăsat pornit pe durata lucrării; aceleași 3 comenzi deja acceptate pe placă (INIT, interval, mască) + oprirea citirilor la final
(fără `INIT(0)`, ca de la sesiunea 132). Nicio comandă nouă, niciun registru.

## 2. Predicții (înainte de cod)
- P1: pe modelul `SimRusd` (45,5 °C) ceasul dă `Continua` la fiecare citire; rezumatul: min = max = 45 °C.
- P2: temperatura urcată în model 70 → 79 → 80 → 84 → 85: verdictele `Continua, Continua, Avertizare, Avertizare, Opreste(Fierbinte 85)`;
  oprirea rămâne și dacă modelul coboară la 60 °C.
- P3: motivul 0x20 la 50 °C → `Opreste(Termic)` din prima citire; motivul 0x1 (GPU_IDLE, ce a văzut os-323) → nimic.
- P4: marca temperaturii neschimbată 10 s → `Continua` la 10 000 ms, `Opreste(Mut)` la prima citire de după.
- P5: înainte P8 / 210 MHz, după P0 / 1 770 MHz → `Urcata`; înainte P0 (os-323) → `DejaSus`; P8 → P8 → `FaraEfect`.
- P6: `may_start` refuză la 71 °C, primește la 70 °C.
- P7: rândurile `kw` citite înapoi de `parse_kw` dau aceleași cifre; 0 rânduri în plus pe citirile fără schimbare.

## 3. Sursele pragului (DEDUS — nu e o cifră măsurată pe placa noastră)
| Ce | Sursa | Cifra |
|---|---|---|
| temperatura maximă a plăcii | NVIDIA, pagina oficială GeForce RTX 3070 / 3070 Ti, „Founders Edition Thermal Power Specs”: „Maximum GPU Temperature (in C) 93” (pentru 3070 Ti; „specifications may vary by add-in-card manufacturer”) | 93 °C |
| cum arată pragurile NVIDIA pe o placă Ampere | `nvidia-smi -q` pe un RTX 3060 Laptop (forumul NVIDIA pentru dezvoltatori, 2023): Max Operating 87, Slowdown 102, Shutdown 105 | 87 / 102 / 105 °C |
| motivele de încetinire | `rusd::reason` (antetele NVIDIA 570.144, MIT): 0x08 HW_SLOWDOWN, 0x20 SW_THERMAL, 0x40 HW_THERMAL, 0x80 HW_POWER_BRAKES; 0x01 = GPU_IDLE | — |
| placa noastră | os-323: 35,8 °C în repaus, 25,2 W; workstation-ul nu e Founders Edition sigur (marca neștiută) | — |

Alegerea: oprirea la **85 °C** = 8 °C sub maximul NVIDIA (93) și sub „Max Operating” de 87 văzut la Ampere — OS-ul oprește lucrul
**înainte** ca placa să înceapă să se frâneze singură; avertizarea la 80 °C lasă loc de un rând în jurnal înainte de oprire; pornirea
din nou sub 70 °C (15 °C histerezis) ca să nu pornească-oprească la fiecare citire. Toate trei se pot schimba din fișa lucrării dacă
Adrian vrea (OS-ul trece alte cifre; `ThermalLimits::check` refuză cifre fără sens: oprire peste 90, avertizare ≥ oprire, pornire ≥
avertizare). Firmware-ul NVIDIA nu a fost deschis sau analizat.

Răspuns la întrebarea din os-323 („motivul 0x1, de tălmăcit”): bitul 0 = `GPU_IDLE` în antetele NVIDIA — placa era în repaus. Nu e
încetinire.

## 4. Rezultate — toate cele 7 predicții s-au potrivit la prima rulare
Codul: `crates/nvidia-ga104/src/gsp/rusd_watch.rs` (doar adăugări). Testele: T179, `crates/nvidia-ga104/tests/sim_c157_p5_ceas_termic.rs`
(7) + 2 în modul. Paginile vin din modelul `SimRusd` (SIMULARE).

| Test | Predicția | Rezultat |
|---|---|---|
| T179a | P1 + P7: 20 de citiri la 45 °C → `Continua`; **un singur** rând de verdict; rezumatul citit înapoi de `parse_kw` | PASS (SIMULARE) |
| T179b | P2: 70, 79, 80, 84, 85, 60, 40 → C, C, A, A, Oprire(85), Oprire(85), Oprire(85); 3 rânduri de verdict | PASS (SIMULARE) |
| T179c | P3: 0x20 / 0x08 / 0x40 / 0x80 la 50 °C → oprire din prima; 0x1 (repaus) și 0x4 (limită de putere software) → nimic | PASS (SIMULARE) |
| T179d | P4: aceeași marcă 10 000 ms → continuă, 10 001 ms → `Opreste(Mut)`; pagina nescrisă → `Opreste(FaraTemperatura)` după 10 s; o marcă nouă la fiecare citire → niciodată | PASS (SIMULARE) |
| T179e | P5: P8/210 → P0/1 770 = `Urcata`; P0 → P0 = `DejaSus`; P8 → P8 = `FaraEfect`; în P5 nucleul +20 % = `Urcata`, +19,9 % = `FaraEfect`; secțiuni lipsă = `Necunoscut` | PASS (SIMULARE) |
| T179f | P6: 71 °C nu pornește, 70,99 °C pornește; motiv termic sau fără temperatură → nu pornește | PASS (SIMULARE) |
| T179g | o lucrare de 8 ore la 5 s = 5 760 de citiri la 60–79 °C: niciun stop, **sub 12 rânduri** în jurnal | PASS (SIMULARE) |
| unități | limitele fără sens refuzate (oprire > 90, avertizare ≥ oprire, pornire ≥ avertizare, interval 0,5 s / 20 s); biții de oprire = antetele | PASS |
| greșeli plantate | 6 (oprirea la `>` în loc de `≥`, fără bitul 0x08, tabloul mut +5 s, pragul de 20 % mutat, oprirea care nu se ține, pornirea la 71 °C) — toate prinse | PASS |

`tools/check.sh`: format, clippy `-D warnings`, **855 de teste** (+7 ignorate), build no_std.

**NETESTAT pe placă:** cum urcă temperatura plăcii sub lucru (nicio citire sub sarcină încă: os-323 a citit-o în repaus); dacă marca
secțiunii de temperatură se schimbă la fiecare 500 ms (o presupunem — dacă GSP-RM o scrie doar la o schimbare de valoare, regula
„tablou mut” ar trebui să citească orice secțiune, nu doar temperatura: primul test cu ceasul pornit o arată); cât de repede reacționează
ceasurile în pagină după cererea de viteză maximă (de aceea 1 s de așteptare).
