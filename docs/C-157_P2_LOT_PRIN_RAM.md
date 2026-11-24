# C-157 punctul 2, după os-390: lotul și rezultatele pasului prin RAM (plan + predicții, scrise înainte de cod)

**SIMULARE.** Nimic de aici nu rulează pe RTX 3070 Ti până nu-l pune OS-ul într-un Release și nu-l rulează centrul de comandă.

## De ce
os-390 (comparația în `docs/C-157_P2_COMPARATII.md`): pasul întreg 4 150 µs, placa 3 625. Din cele 525 µs din afara plăcii, cea mai
mare bucată e **lotul** (1 284 B = 321 de cuvinte) scris cuvânt cu cuvânt prin fereastra VRAM, fiecare recitit: ~245 µs la golire
+ 49 µs puse în coadă. Apoi **pierderile + norma** citite prin fereastră: 65 de cuvinte în 70 µs (~1,1 µs pe cuvânt).

## Ce
Programul de copiere `copy_b128` (pe placă de la I11; în os-390 a urcat arena RAM → VRAM și a adus-o înapoi VRAM → RAM, cu greutățile
= oracolul) mută lotul și rezultatele **în lista pasului**:
1. procesorul scrie lotul (aceleași 5 zone ca `Plan::batch_uploads`: intrări, ținte, cele 3 liste de segmente) într-o **zonă de
   tranzit din RAM** (fixată, ca zona de control) — o scriere obișnuită în memorie, fără fereastra VRAM;
2. lista pornește cu **5 copieri** RAM → arenă, câte una pe zonă (lungimea rotunjită în sus la 16 B, cât cere `copy_b128`);
3. după lansările pasului, **2 copieri** arenă → RAM: pierderile pe rânduri și norma (`Plan::step_reads`);
4. după așteptare, procesorul citește pierderile și norma **din RAM**.

Doar adăugări: modulul nou `nvidia_ga104::batch_ram` (`via_ram`, `pack_via_ram`, `results`). Nimic din `train_step` nu se schimbă.

## Predicții (scrise înainte de cod)
- **R1 (bit cu bit):** 10 pași `antrenare mini` pe 10 loturi: pierderile, normele și greutățile prin RAM = drumul de azi (T206 cu
  tăieturi) bit cu bit.
- **R2 (arena):** arena după fiecare pas e aceeași; singurele octeți care pot diferi sunt cozile rotunjirii la 16 B (≤ 12 B după
  listele de segmente de lungime variabilă, nefolosite de programe), iar acolo drumul prin RAM pune 0.
- **R3 (cuvinte prin fereastră pe pas, de la pasul 2):** scrise 397 → **83** (doar valorile înregistrărilor: 75 + 7 lansări + 1);
  citite 73 → **8** (doar prima + ultima înregistrare). Lotul: 0 cuvinte prin fereastră.
- **R4 (greșeli plantate prinse):** tranzitul nescris (lotul vechi rămas) → alte pierderi; copierile de la coadă lipsă → rezultatele
  vechi din RAM, prinse; adresă de tranzit nealiniată sau lot străin → refuz înainte de orice.
- **R5 (DEDUS pentru placă, aritmetică cu costurile MĂSURATE în os-390: 0,76 µs pe cuvânt scris, ~1,1 µs citit):** în afara plăcii
  −~310 µs (fereastra) − ~47 µs (lotul pus în coadă), pe placă +7 lansări mici ≈ +20–30 µs → pasul **~3,8 ms ≈ ~16 700 simb/s**
  (azi 4 150 µs, 15 421).

## Riscuri
- Ordinea: copierile sunt lansări ca toate celelalte (aceeași invalidare a cache-urilor înaintea fiecăreia), deci au aceeași garanție
  ca lanțul pasului de azi.
- Zona de tranzit nu se rescrie cât placa o citește: aici pas cu pas, deci sigur; cu lotul în paralel (P10) trebuie două zone de tranzit.
- Citirea din RAM după placă: același drum ca aducerea arenei înapoi la sfârșit (os-390, MĂSURAT corect).

## Rezultat (T209, SIMULARE, sesiunea 202)
- **R1 trece:** 10 pași, pierderile / normele / greutățile prin RAM = prin fereastră, bit cu bit.
- **R2 trece:** arena după pașii 1, 4, 9 identică; 0 octeți diferiți (cozile rotunjirii erau deja 0 în ambele drumuri).
- **R3 trece:** prin fereastră pe pas 397 → 83 cuvinte scrise, 73 → 8 citite; 82 de lansări (75 + 7); tranzitul 2 048 B în RAM.
- **R4 trece, cu o diferență față de predicție:** fără copierile de la coadă procesorul nu citește rezultatele vechi (cum scrisesem), ci
  **0** — zona de rezultate e scrisă pe zero odată cu lotul. Mai bine: pierderea 0 și norma 0 sunt imposibile, deci prinse sigur.
- **R5 (DEDUS, nemăsurat):** ~382 → ~72 µs prin fereastră (−~310 µs) + lotul pus în coadă −~47 µs, placa +~20–30 µs → **~3,8 ms pe pas**.
