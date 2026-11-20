# C-147 punctul 7 — estimarea actualizată pe etape, după cifrele de pe placă (os-313, os-322)

Sesiunea 125, 20 nov 2026, în cloud, **fără placă**. Sursele cifrelor: hub `rezultate/2026-11-20-workstation-os313-automat/`,
`rezultate/2026-11-20-workstation-os322-automat/`, jurnalele `rezultate/os-automat/2026-11-20/gpu-jurnal-015.txt` (os-313) și
`gpu-jurnal-017.txt` (os-322). Codul OS citit (doar citit): `cyber-universe` `7fad533`, `kernel/src/drivers/gpu_i14.rs`
(`BoardRunner::submit`), `gpu_i11.rs` (`run_records`), `gpu_golden.rs` (`flush`).

Etichete: **MĂSURAT** = cifră din jurnalul de pe placă; **DEDUS** = calcul sau interpretare a noastră, nedovedită pe placă;
**SIMULARE** = verificat doar pe `gpu-sim`.

## 1. Unde se duce un simbol azi (#513, viteza maximă, os-322)

Unealta nouă `timpul_driverului` (`nvidia_ga104::driver_split`) citește ceasul fiecărui rând din jurnalul de pe stick și împarte
timpul dintre două sonerii după fereastra VRAM deschisă (registrul `0x1700`). Pe `gpu-jurnal-017.txt`: 31 de simboluri în buclă,
25 „constante”; 6 lente lăsate deoparte (21–23 ms: simbolurile reci, când lansările se scriu întregi; 47 și 56 ms: scrierea
jurnalului pe stick).

| Parte (median, simbolurile constante) | µs | Ce e | Etichetă |
|---|---:|---|---|
| fereastra `0x1b2f` | 2 905 | așteptarea plăcii (lansările: 2 631) + citirea celor 197 de înregistrări (~270) | MĂSURAT; împărțirea DEDUS |
| fereastra `0x10e6` | 1 517 | **pushbuffer-ul rescris întreg** la fiecare simbol: 2 363 de cuvinte, fiecare recitit (~0,64 µs/cuvânt) | MĂSURAT; ce e în fereastră DEDUS |
| fereastra `0x1b2d` | 529 | cuvintele schimbate din lansări (umbra C-145) + **zerorizarea celor 197 de înregistrări** (788 de cuvinte, recitite) | MĂSURAT; DEDUS |
| fereastra `0x101f` | 16 | intrarea în inel + `GP_PUT` | MĂSURAT |
| fără fereastră (procesorul) | 840 | logit-urile citite din RAM (64 KiB), argmax, pregătirea simbolului următor, jurnalul | MĂSURAT; conținutul NECUNOSCUT |
| **total** | **5 788** | (`kt`: placa 2 631 + driver 3 081 + jurnal 61 = 5 802) | MĂSURAT |

De ce credem că `0x10e6` e pushbuffer-ul: ordinea scrierilor la `flush` (lansări → înregistrări → pushbuffer → inel), raportul
timp/cuvinte aproape egal cu al ferestrei `0x1b2d` (0,64 față de 0,61 µs/cuvânt), și faptul că `WorkQueue::place` pune fiecare
simbol la începutul zonei (simbolul de dinainte e retras) — DEDUS, ~85 %. În os-313 (P8) aceleași ferestre durau de ~2,3 ori
mai mult (3 632 și 1 149 µs): accesul prin fereastră depinde de ceasurile plăcii — MĂSURAT.

**Cifra raportată „94,84 simb/s cu driverul”** e media (cu simbolurile reci și scrierile pe stick); mediana e **172 simb/s**
(5 802 µs), iar ceasul plăcii de la primul la ultimul simbol dă 101,82. La generări lungi contează mediana.

### Placa (2 631 µs, `kp`, MĂSURAT)
| Grup | µs/simbol | % | Observație |
|---|---:|---:|---|
| înmulțirile cu matrice (qkv 368, out 144, fc 471, proj 400, logit 185) | 1 568 | 60 | 103,6 MB de greutăți q8 pe simbol → **~66 GB/s = 11 % din 608 GB/s** (DEDUS din `kp` + forma #513) |
| LayerNorm (ln1 262, ln2 259, ln_final 22) | 543 | 21 | **21,8 µs pe lansare pentru 768 de numere**: un singur grup de 32 de fire, 3 treceri cu citiri dependente — timp de așteptare, nu de calcul (DEDUS) |
| atenția | 255 | 10 | tot un grup de 32 de fire pe cap (DEDUS) |
| cele 9 calcule mărunte (bias, k, v, rezidual, gelu) | 232 | 9 | 2,2 µs fiecare — ce elimină P3 |
| embed + copierea logit-urilor | 15 | <1 | |

Plafonul memoriei: 103,6 MB / 563 GB/s (traficul măsurat la `kc`) ≈ **184 µs** pe simbol doar pentru greutăți.

## 2. Estimarea pe etape (#513, viteza maximă, median pe simbol; toate DEDUS)

Fiecare rând se adaugă peste cele de deasupra.

| # | Etapa | Ce schimbă | Câștig | Simbol după | simb/s (median) |
|---|---|---|---:|---:|---:|
| 0 | azi (os-322) | — | — | 5,80 ms | 172 |
| **D** | **dieta driverului (NOU)** — partea OS + ajutorul nostru `ShadowedMemory` | pushbuffer-ul scris prin umbră (doar cele ~200 de cuvinte schimbate: valorile înregistrărilor) + înregistrările nu se mai zerorizează (valorile sunt unice pe trimitere; T141) | −1,8 ms | ~4,0 ms | **~250** |
| P3 | calculele lipite (gata, sesiunea 120) | 196 → 88 lansări | placa −0,22; driver −0,2 | ~3,6 ms | ~280 |
| L | LayerNorm nou (NOU în listă) | o singură citire, reducerea în registre/`shuffle`; sau lipit în înmulțirea de după | −0,45 ms | ~3,1 ms | ~320 |
| 4 | etapa 4: înmulțirile la ~35–40 % din memorie (`q8_matvec32` e primul pas) | 1 568 → ~550 µs | −1,0 ms | ~2,1 ms | ~470 |
| A | atenția mai bună | 255 → ~80 µs | −0,17 ms | ~1,95 ms | ~510 |
| P7 | lanțul pe placă | după D, driverul e ~1,1 ms, din care ~0,84 procesorul (logit-uri, argmax) — P7 nu-l atinge | −0,1…0,3 ms | ~1,75 ms | ~570 |

**Ce s-a schimbat față de plan (concluzia punctului 7):**
1. **Frâna „driverul” e mai ales rescrierea acelorași cuvinte**, nu numărul de lansări în sine: ~1,5 ms pushbuffer + ~0,5 ms
   înregistrări din 3,1 ms. Dieta D ia cea mai mare bucată fără să schimbe nimic din ce execută placa (aceiași octeți în VRAM)
   — e primul pas, înaintea lui P3.
2. **P7 (lanțul pe placă) coboară** după etapa 4: după D, ce rămâne din driver e munca procesorului între simboluri.
3. **LayerNorm intră în listă** (21 % din timpul plăcii; nu era în ordinea propusă). E cel mai ieftin câștig pe placă după P3.
4. Câștigul mare pe placă rămâne la înmulțiri (60 %, la 11 % din viteza memoriei).
5. Cei ~840 µs ai procesorului trebuie măsurați înainte de orice altă schimbare (două mărci de timp în OS — nota pentru OS, pasul 4).
   Dacă citirea celor 64 KiB de logit-uri din RAM e partea mare, un argmax pe placă (8 octeți în loc de 64 KiB) e următorul pas.

### #651 (24 de straturi, 183 M), aceeași metodă (DEDUS)
Azi: placa ~5,0 ms + driver ~5,3 ms ≈ 10,4 ms → **~96 simb/s**; cu D: ~6,7 ms → **~150**; cu D + P3: ~5,8 ms → **~170**.

## 3. Ce s-a făcut în sesiunea 125 (doar adăugări)
- `nvidia_ga104::driver_split` (`split_by_bell`, `steady`, `token_loop`, `row_time_us`, `BellSplit`, `SteadySplit`) + exemplul
  `timpul_driverului` — împărțirea de mai sus, din orice jurnal de pe stick.
- `nvidia_ga104::kronos_fast::ShadowedMemory` — un `ChannelMemory` care trimite mai departe doar cuvintele pe care umbra nu le
  știe deja; pentru pushbuffer-ul canalului de lucru (pe care îl scrie doar procesorul).
- `tests/sim_c147_diet.rs` (T141, SIMULARE): un model de 3 straturi, 6 poziții, ambele liste (51 și 24 de lansări): cu dieta,
  logit-urile, cache-urile și duratele lansărilor **identice bit cu bit**; cuvinte scrise pe simbol 623 → 55 (lista lungă) și
  299 → 28 (lista lipită), zero octeți de înregistrări; înregistrările vechi nu trec niciodată drept noi (`analyze_sequence`,
  `records_seen`).

## 4. Ce NU e dovedit
Nimic pe RTX 3070 Ti. Câștigurile din tabelul 2 sunt calcule din cifrele os-322; atribuirea ferestrelor e DEDUS din codul OS.
Proba pe placă pentru D: aceleași 16 simboluri, logit-urile identice bit cu bit cu os-322 (aceleași programe, aceiași octeți
în pushbuffer), `kp` neschimbat, `kt median driver` de la ~3 081 spre ~1 300 µs.
