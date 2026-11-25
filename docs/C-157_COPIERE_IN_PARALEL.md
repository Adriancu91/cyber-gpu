# C-157 — copierea lotului în paralel cu calculul (nota centrului 25 nov 11:55, cererea 1) — plan + predicții, scrise ÎNAINTE de cod

**Data:** 25 noiembrie 2026, ~19:40 (ora României), sesiunea Cyber-GPU 235, în cloud, FĂRĂ placă. Totul de aici e
**SIMULARE** (modelul `gpu-sim`) sau **DEDUS**; nimic nu rulează pe RTX 3070 Ti.

**Cererea centrului (11:55, studiul `idei/2026-11-25-centru-franele-altora.md`, punctul 2):** „copierea pe motorul de copiere
al plăcii în paralel cu calculul (lotul următor sus, rezultatele jos)”. Ordinea: ocuparea → `mm_tc` → **copierea în
paralel + încărcarea dublă** → programul persistent. `mm_tc` rapid e gata în SIMULARE (sesiunile 229–234, os-438 așteaptă
placa), deci acum vine copierea.

## 1. Ce avem deja (citit în cod, nu presupus)

- **Două locuri de lot** în arena rezidentă: `Plan::with_batch_slots` / `slot(k)` (sesiunea 196, T204, nota
  `C-160_P10_lot-in-paralel.md`). Nelegate în OS.
- **Lotul prin RAM** cu programul de copiere în lista pasului: `batch_ram::{via_ram, pack_via_ram, results}` (sesiunea
  ~200, T209; pe placă os-397: −207 µs la `antrenare mini`). Merge doar cu lista F32 (`Launch`), **nu cu pasul mixt**.
- **Motorul de copiere** (`AMPERE_DMA_COPY_B`) pe canalul de lucru: `ram_vram` (os-376: corect, 5,97 GB/s sus / 6,76 jos =
  ~94 % din PCIe 3.0 x8).
- **Bucla OS de azi la #513** (`train_513_resident`, citită în `cyber-universe`): la fiecare pas procesorul face
  `rebatch` → scrie lotul **cuvânt cu cuvânt prin fereastra VRAM** (`put_bytes`) → construiește lista (483 de lansări) și o
  împachetează → 2 trimiteri, fiecare așteptată → citește pierderile + norma **prin fereastră** (`read_vram_f32s`). Placa
  stă degeaba în tot acest timp.

## 2. Cât costă azi „în afara plăcii” (MĂSURAT pe placă)

| rulare | placa pe pas | pasul întreg | în afara plăcii |
|---|---|---|---|
| #513 lot 2 (os-431, jurnalul 117: `0x3e798` / `0x4125f`) | 255,9 ms | 266,8 ms | **10,95 ms (4,1 %)** |
| #513 lot 6 (os-437) | 637,9 ms | 656,4 ms | **18,5 ms (2,8 %)** |

Cu pasul mixt rapid (prezis 115–136 ms pe înmulțiri la lot 2, sesiunea 234) cele ~11 ms ar deveni **~7–9 %** din pas, deci
merită ascunse acum.

## 3. Socoteala (DEDUS) — ce aduce motorul de copiere și ce nu

Octeții pe pas din `Plan::traffic` (formula din cod: sus `8n + 4(2·cap + 1 + n)`, jos `4n + 4`, `n` = lot × 512,
`cap = min(n, vocab)`):
- lot 2 (`n` = 1 024): sus **20 484 B** (5 121 cuvinte), jos **4 100 B** (1 025 cuvinte);
- lot 6 (`n` = 3 072): sus **61 444 B** (15 361 cuvinte), jos **12 292 B** (3 073 cuvinte).

Prin fereastră (costuri MĂSURATE în os-390: ~0,76 µs pe cuvânt scris, ~1,1 µs citit): lot 2 ≈ 3,9 + 1,1 = **~5,0 ms**,
lot 6 ≈ 11,7 + 3,4 = **~15,1 ms** — adică **cea mai mare parte** din cele 11 / 18,5 ms (restul: lista + împachetarea +
așteptarea între cele 2 trimiteri). Diferența lot 6 − lot 2 prezisă ~10 ms, măsurată 7,5 ms → costul real pe cuvânt e
probabil ~25 % mai mic decât în os-390 (OS-ul scrie acum fără recitire?) — de confirmat cu rândul cerut mai jos.

Pe legătura PCIe (5,97 GB/s), aceiași octeți: lot 2 ≈ **4 µs** sus + 1 µs jos, lot 6 ≈ 10 + 2 µs. **Deci motorul de
copiere nu are ce câștiga față de programul de copiere**: ambele sunt limitate de legătură, iar lotul are zeci de KiB.
Câștigul real vine din două lucruri care nu cer nimic nou pe placă:
1. **lotul și rezultatele prin RAM fixată**, mutate de programul de copiere în lista pasului (dovedit pe placă în os-397):
   procesorul scrie o zonă din RAM-ul lui (o copiere de memorie, ~µs), nu mii de cuvinte prin fereastră;
2. **două locuri** (lot, zonă de tranzit, zonă de control): procesorul pregătește pasul N+1 cât timp placa lucrează la
   pasul N, deci și lista + împachetarea (câteva ms) ies din pasul întreg.

Motorul de copiere rămâne util pentru transferurile mari (m, v în RAM — doar în SIMULARE, condiția lui Adrian din 20:36);
pentru lot îl lăsăm deoparte: ar cere încă o sincronizare între motoare pe același canal (DEDUS, nedovedit pe placă) fără
niciun câștig măsurabil.

## 4. Ce adaug (doar adăugări; nimic redenumit sau șters)

În `nvidia_ga104::batch_ram`:
- `pack_mixed_fast_via_ram(list: &[MixedLaunch], prog_va, progs: &TcPrograms, step_bytes, copy_va, zone_va, v: &ViaRam)`
  → zona de control + adresele QMD + alegerea fiecărei lansări: copiile RAM → arenă în față, lista mixtă rapidă
  (`pack_mixed_fast`, aceleași reguli: `mm_tc_v4` dacă se potrivește, altfel `mm_tc_pf`) la mijloc, copiile arenă → RAM
  la coadă. Cu `step_bytes = Some(s)`, și copiile primesc aceeași treaptă `s` (o singură treaptă pe toată trimiterea,
  ca regula din sesiunea 225: fără schimbări de treaptă în mijlocul listei).
- `extra_launches` rămâne cum e (7 la lista mică: 5 copii sus + 2 jos).

## 5. Predicțiile (înainte de cod)

- **Q1 (SIMULARE, `antrenare mini` rezident, 6 loturi diferite, pasul mixt rapid la S = 2¹⁰, treapta 64 KiB):** lotul prin
  RAM cu `pack_mixed_fast_via_ram` = drumul „prin fereastră” cu `pack_mixed_fast` **bit cu bit** (pierderile, normele,
  greutățile de la final); zona de control mai mare exact cu 7 sloturi de copiere; 0 cuvinte ale lotului prin fereastră.
- **Q2 (SIMULARE, aceeași formă):** cu **două locuri** (lot + tranzit + zonă de control pe loc), ordinea „în paralel” —
  pasul N trimis în două bucăți, iar **între ele** procesorul pregătește pasul N+1 în celălalt loc (tranzit + zonă), iar
  rezultatele pasului N citite din tranzit **după** ce pasul N+1 a rulat — dă **aceiași biți** ca Q1.
- **Q3 (SIMULARE, greșeli plantate):** aceeași ordine cu **un singur loc** (un tranzit, o zonă) → altfel (pasul N își
  citește la a doua bucată lotul pasului N+1 sau zona lui) — prinsă ca biți diferiți; o zonă de tranzit nealiniată,
  copierea nealiniată și un program `mm_tc*` neîncărcat refuzate înainte de orice trimitere.
- **Q4 (SIMULARE pe planul #513, fără rulare):** octeții pe pas din `Plan::traffic` pentru lot 2 / 6 / 8 = cifrele din §3;
  `pack_mixed_fast_via_ram` pe lista mixtă #513 lot 2 primește 147 de lansări `mm_tc_v4` și **exact 7 copii** în plus.
- **Q5 (DEDUS, pentru placă; NU se poate testa aici):** cu lotul prin RAM pe pasul mixt #513, în afara plăcii scade cu
  **~4–5 ms** la lot 2 și **~11–15 ms** la lot 6; placa +7 lansări mici (+20–40 µs). Cu cele două locuri legate în OS, ce
  rămâne în afara plăcii (lista, împachetarea, a doua trimitere) se ascunde și el: pasul întreg → **placa + < 0,5 ms**.

## 6. Ce NU face

- Nu trimite nimic la placă, nu alocă nimic; coada, semafoarele și RAM-ul fixat sunt ale OS-ului.
- Niciun program GPU nou, niciun registru nou; programul de copiere e cel din os-397.
- Nu mută nimic din antrenare în RAM-ul calculatorului (doar lotul și rezultatele trec prin el, ca în os-397).

## 7. Ce cere OS-ului (după os-438 și `pack_mixed_fast`)

Un singur lucru nou pe test: `--antrenare-513 --mixt --lot-prin-ram` = `pack_mixed_fast_via_ram` în locul
`pack_mixed_fast`, aceeași verificare față de oracol; în raport un rând nou cu **timpul în afara plăcii** (pasul întreg −
placa) și, dacă se poate, împărțit: lotul, lista + împachetarea, trimiterile, rezultatele. Cele două locuri — la testul
următor.

## 8. Rezultatul (sesiunea 235, ~20:00; cod + test `1ddf092`)

**Corectură la ordinea din Q2 (găsită la scrierea testului, înainte de rulare):** rezultatele pasului N−1 se citesc
*în timp ce rulează pasul N* (între cele două bucăți), **înainte** de a pregăti pasul N+1 în același loc — altfel
pregătirea (care rescrie tranzitul locului) le-ar șterge. Asta e și ordinea corectă pentru OS (vezi nota).

| Predicție | Ce a ieșit | Verdict |
|---|---|---|
| Q1 | 6 pași mixți rapizi (64 KiB), 6 loturi diferite: prin RAM = prin fereastră **bit cu bit** (pierderi, norme, greutăți); lansări 76 → 83 (+7 copii), zona +7 sloturi; cuvintele lotului prin fereastră pe pas **321 → 0** | PASS (SIMULARE) |
| Q2 | două locuri, pasul N+1 pregătit între bucățile pasului N, rezultatele N−1 citite în timpul pasului N = pas cu pas **bit cu bit** | PASS (SIMULARE) |
| Q3 | un singur loc în aceeași ordine: **5 / 6 pierderi** și greutățile alți biți (prinse); tranzit / program de copiere / zonă nealiniate, `mm_tc*` neîncărcat, treapta 12 KiB refuzate înainte de trimitere; fără treaptă, lista din mijloc = exact `pack_mixed_fast` | PASS (SIMULARE) |
| Q4 | #513 rezident: sus / jos pe pas **20 484 / 4 100 B** (lot 2), **61 444 / 12 292** (lot 6), **81 924 / 16 388** (lot 8) = §3; lista mixtă #513 lot 2: **484 lansări + 7 copii, 147 pe `mm_tc_v4`**, tranzitul 25 088 B | PASS (SIMULARE) |
| Q5 | în afara plăcii −4–5 ms (lot 2), −11–15 ms (lot 6); cu două locuri pasul întreg ≈ placa + < 0,5 ms | NOT TESTED (DEDUS) |

Test: `crates/nvidia-ga104/tests/sim_c157_copy_parallel.rs` (T241, 4 teste). Nota pentru OS:
`integration/cyber-universe/C-157_P2zg_lot-prin-ram-mixt.md`.
