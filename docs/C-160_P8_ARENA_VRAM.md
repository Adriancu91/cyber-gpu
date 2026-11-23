# C-160 P8 — arena antrenării în VRAM: „arena rezidentă” (plan + predicții, scrise ÎNAINTE de cod)

**Data:** 24 noiembrie 2026, ~00:30 (ora României), sesiunea Cyber-GPU 194, în cloud, FĂRĂ placă.
**De ce acum:** nota centrului de comandă din 23 nov 23:36 — după os-380 (primul pas întreg de antrenare pe placa reală,
TRECE), ordinea pe placă e (1) `--antrenare-mini --mixt` și gen 4, (2) **arena în VRAM**, ca viteza antrenării să se poată
socoti, (3) #513. Pasul (2) e al OS-ului (alocarea în VRAM și copiile), dar el are nevoie de ceva ce biblioteca noastră
nu dă azi.

## Ce am găsit în os-380 (comparația, C-157 p.2)

- Rezultatul pe placă: 10 pași, ACELEAȘI NUMERE ca oracolul (pierdere 9,9e-8, greutăți 8,8e-6) — exact ce arăta
  `--placa-sim` (P6-162: greutăți ≤ 6,5e-5) și T199 în SIMULARE (F32 = oracolul). Nicio surpriză de corectitudine.
- Timpul: placa 25,6 ms pe pas (2 499 simboluri/s), pasul întreg **286 ms**.
- De ce pasul întreg e de 11 ori mai lung decât placa (citit în codul OS-ului, `train_mini_probe`, `gpu_i14.rs`):
  la **fiecare** pas OS-ul face un plan nou (`Plan::new`, pentru că listele de segmente ale lui `emb_back` depind de
  lot și mută adresele de după ele), urcă tot (`uploads`: greutăți, gradient zero, plus m, v ale lui AdamW =
  4 × 4,6 MB = 18,4 MB) și citește înapoi greutățile, m și v (13,8 MB). Cu arena în RAM asta merge prin procesor;
  cu arena în VRAM ar trebui copiat prin placă de două ori pe pas — ar strica tocmai viteza pe care vrem s-o măsurăm.

## Ce lipsește în bibliotecă

Un plan al cărui **aranjament nu depinde de lot**, ca arena să fie făcută o dată în VRAM și să rămână acolo:
greutățile, gradientul, m și v stau pe placă de la primul pas la ultimul; între pași procesorul scrie doar lotul nou
și citește doar pierderea și norma.

## Ce adaug (doar adăugări; nimic redenumit sau șters)

În `nvidia_ga104::train_step`:
1. `Plan::resident(self) -> GpuResult<Plan>` — același plan (aceleași programe, aceleași lipiri, aceeași rețetă), dar
   listele de segmente ale simbolurilor au loc pentru cel mai rău lot (`min(batch·len, vocab)` simboluri distincte),
   deci toate adresele sunt aceleași pentru orice lot de aceeași formă.
2. `Plan::rebatch(&self, inputs) -> GpuResult<Plan>` — planul pentru lotul următor: aceleași adrese (verificat), doar
   segmentele noi. Refuză un plan care nu e rezident.
3. `Plan::batch_uploads(&self, inputs, targets)` — ce scrie procesorul între pași: intrările, țintele și cele 3
   liste de segmente ale simbolurilor (~1,3 KiB la lot 2 × 32). Segmentele pozițiilor nu depind de lot (o dată).
4. `Plan::step_reads(&self)` — ce citește procesorul după fiecare pas: pierderile pe rând (`batch·len` f32) și norma
   (1 f32) = 260 de octeți la lot 2 × 32. Greutățile, m și v se citesc doar la sfârșit (sau la o salvare).
5. `Plan::traffic(&self)` — octeții pe legătura PCIe: o dată (la pornire), pe pas în sus, pe pas în jos; și cât face
   azi drumul OS-ului (plan nou + tot urcat + w, m, v citite), pentru raport.

## Predicțiile (înainte de cod)

- **R1 (SIMULARE):** planul rezident are aceleași adrese pentru 10 loturi diferite și pentru lotul cel mai rău (64 de
  simboluri distincte); arena lui e cel mult cu câteva sute de octeți mai mare decât cea mai mare arenă pe lot.
- **R2 (SIMULARE, forma lui `antrenare mini`, 10 pași, 10 loturi diferite, AdamW, F32):** drumul rezident (o urcare
  întreagă, apoi doar `batch_uploads` și `step_reads`, greutățile citite o dată la sfârșit) dă **bit cu bit** aceleași
  pierderi, norme și greutăți ca drumul de azi al OS-ului (plan nou + tot urcat + w, m, v citite la fiecare pas), și
  oracolul pe procesor în toleranța 1e-3 (cum a ieșit pe placă în os-380).
- **R3 (SIMULARE):** greșelile plantate se prind: lotul nou fără `batch_uploads` (segmentele vechi) → alte numere;
  `rebatch` pe un plan nerezident → refuzat; un simbol în afara vocabularului → refuzat.
- **R4 (DEDUS, pentru placă; NU se poate testa aici):** pe legătură, pe pas: de la ~32 MB (18,4 MB sus + 13,8 MB jos)
  la ~1,6 KB. Cu arena în VRAM: timpul plăcii **1–5 ms** pe pas în loc de 25,6 ms (≈ 13 000–64 000 simboluri/s la
  lot 2 × 32), pasul întreg cu câteva ms peste timpul plăcii (în loc de 286 ms). Socoteala: generarea #513, cu
  greutățile în VRAM, face 88 de lansări în 1,075 ms; pasul de antrenare are 73–75 de lansări, cu munca cea mai mare
  la capul de 16 384 de simboluri pe 64 de rânduri (~0,2 GFLOP pe pas cu tot cu „înapoi”; logits + gradientul lor
  8 MiB, citiți de câteva ori — sub 0,1 ms la viteza VRAM-ului). Dacă pe placă iese peste 5 ms, cauza e de căutat
  întâi în lansările mari (capul), apoi în așteptările driverului.

## Ce NU face

- Nu alocă nimic pe placă și nu copiază: alocarea arenei în VRAM și copiile (urcarea inițială, citirea finală) sunt ale
  OS-ului (are deja `map_vram`, copia prin placă `vram_copy` și proba RAM ↔ VRAM).
- Nu schimbă nicio listă de lansări: aceleași programe, aceleași argumente (doar adresele sunt cele ale planului).
- Nimic nu rulează pe RTX 3070 Ti în sesiunea asta.
