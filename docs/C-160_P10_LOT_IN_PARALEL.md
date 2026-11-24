# C-160 P10 — ordinea 20:11, pasul 5: lotul următor pregătit în paralel cu pasul de antrenare (plan + predicții, scrise ÎNAINTE de cod)

**Data:** 24 noiembrie 2026, ~02:20 (ora României), sesiunea Cyber-GPU 196, în cloud, FĂRĂ placă. Totul de aici e **SIMULARE**
(modelul `gpu-sim`) sau **DEDUS**; nimic nu rulează pe RTX 3070 Ti.

**De ce acum:** ordinea lui Adrian din 23 nov 20:11, pasul 5 = „lotul următor în paralel + AdamW în RAM”. AdamW cu m, v în
RAM există deja în bibliotecă (`_mv_ram`, C-157) și rămâne **doar în simulare** (condiția lui Adrian din 20:36: nimic din
antrenare mutat în RAM pe placa reală). Rămâne partea „lotul următor în paralel”.

## Problema

Cu arena rezidentă (P8, `Plan::resident`), între doi pași procesorul face: scrie lotul nou (intrări, ținte, listele de
segmente ale simbolurilor, ~1,3 KiB) și citește pierderile + norma (260 B). Dar toate stau într-un **singur** loc al arenei:
- dacă procesorul scrie lotul N+1 cât timp placa încă lucrează la pasul N, pasul N citește (la `xent` și la `emb_back`,
  spre sfârșitul pasului) ținte / segmente ale altui lot → numere greșite;
- dacă procesorul trimite pasul N+1 înainte să citească pierderile pasului N, pasul N+1 le scrie peste → citește altceva.

Deci azi placa stă degeaba între pași: așteaptă procesorul să citească, să pregătească și să scrie lotul următor.

## Ce adaug (doar adăugări; nimic redenumit sau șters)

În `nvidia_ga104::train_step`:
1. `BatchVas` — locul unui lot pe placă: `tok`, `tgt`, `seg_tok`, `loss_rows`, `res` (norma + factorul de tăiere + 1/S).
2. `Plan::with_batch_slots(self) -> GpuResult<Plan>` — doar pentru un plan rezident: adaugă **un al doilea loc de lot** la
   capătul arenei (aceleași mărimi). Tot restul (greutăți, gradient, m, v, activări) rămâne comun și la aceleași adrese.
3. `Plan::slot(&self, k) -> GpuResult<Plan>` — planul care folosește locul `k` (0 sau 1): aceleași programe, aceleași
   lansări, doar adresele lotului altele. Refuză `k ≥ 2` și planurile fără două locuri.
4. `Plan::slot_uploads(&self)` — ce se scrie o dată pentru locul 1 (`res = [0, 1]`; pentru pasul mixt și `mixed_uploads`
   pe planul locului 1). `Plan::batch_slots()` = câte locuri are planul (1 sau 2).
5. `with_recompute` păstrează cele două locuri.

**Ordinea pe placă** (o coadă, pașii rulează unul după altul pe placă): scrie lotul 1 în locul 0 și lotul 2 în locul 1;
trimite pasul 1 (locul 0) și pasul 2 (locul 1); așteaptă pasul 1 → citește pierderile locului 0, scrie lotul 3 în locul 0,
trimite pasul 3; așteaptă pasul 2 → citește locul 1, scrie lotul 4 în locul 1, trimite pasul 4; … Placa are mereu un pas
în coadă, iar procesorul atinge doar locul pe care placa nu-l folosește.

## Predicțiile (înainte de cod)

- **L1 (SIMULARE):** locul 0 al planului cu două locuri are **exact** adresele planului rezident (toate câmpurile); locul 1
  stă după vechiul capăt al arenei; arena crește cu **3 840 de octeți** la lot 2 × 32 (6 bufere de 512 B + lista `off` de
  768 B, cu alinierea de 256 B); lista de lansări a fiecărui loc = a planului rezident, cu doar adresele lotului schimbate.
- **L2 (SIMULARE, forma lui `antrenare mini`, 10 pași, 10 loturi diferite, AdamW, F32):** drumul „în paralel” — lotul N+1
  scris **la mijlocul** pasului N (lansările pasului N trimise în două bucăți, scrierea între ele) și pierderile pasului N
  citite **după** ce pasul N+1 a rulat — dă **bit cu bit** aceleași pierderi, norme și greutăți ca drumul rezident de pas
  cu pas (T202); la fel cu recalcularea + ambele lipiri.
- **L3 (SIMULARE, greșeli plantate):** același drum „în paralel” cu un singur loc → pierderi și greutăți diferite (scrierea la
  mijloc strică pasul N; citirea târzie dă pierderile pasului N+1); `slot(2)` refuzat; `with_batch_slots` pe un plan
  nerezident refuzat.
- **L4 (DEDUS, pentru placă; NU se poate testa aici):** câștigul = timpul pe care placa îl stă azi degeaba între pași
  (așteptarea semaforului, citirea, segmentele lotului nou, scrierea, trimiterea): estimat **50–300 µs pe pas**. Cu arena
  în VRAM și pasul plăcii de 1–5 ms (predicția R4 din P8), asta înseamnă **~2–15 %** din pasul întreg. Mic, dar fără el
  orice pregătire mai grea a lotului (citirea textului de pe stick, amestecarea) s-ar adăuga direct la fiecare pas.

## Ce NU face

- Nu alocă nimic pe placă și nu trimite nimic: coada și semafoarele sunt ale OS-ului.
- Nu adaugă niciun program GPU și nu schimbă nicio lansare; nimic nou în registre.
- Nu mută nimic din antrenare în RAM-ul calculatorului pe placa reală.
