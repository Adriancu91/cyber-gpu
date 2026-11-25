# C-165 — cursa de noapte a programelor: variantele lui `mm_train_tile`

**Sesiunea 211, 24 nov 2026, în cloud, FĂRĂ placă.** Planul și predicțiile sunt scrise **înainte** de compilare și de teste.

## Ce cere comanda (DECIZII 24 nov 17:35, H3)
Pentru formele de înmulțire ale pasului #513: mai multe variante ale aceluiași program (mărimea plăcii, fire pe bloc, mărimi
fixe), rulate într-o pornire pe placa reală; raportul scrie timpul median pe variantă și formă și păstrează cea mai rapidă.
Condiții: fiecare variantă = **aceiași biți în SIMULARE** ca un program deja trecut pe placă sau ca oracolul; SHA-256 verificat;
**niciun registru, nicio comandă nouă** (doar alt cod de program + alt QMD, ca la orice lansare de azi); paza de temperatură.
Prima cursă: abia după ce `mm_train_tile` a trecut o dată singur pe placă (os-410, `--mm-placi`).

## Partea Cyber-GPU (ce facem azi)
1. Un singur corp de program, `tools/nak/mm_train_tv.glsl`, cu geometria plăcii ca parametri (BM × BN bucata din C pe bloc,
   BK felia lui K, TM × TN rezultate pe fir; fire pe bloc = (BM/TM)·(BN/TN)); fiecare variantă e un fișier mic
   `tools/nak/mm_tv_<nume>.comp` care fixează cele 5 numere și include corpul. Același bufer constant ca `mm_train`.
2. Primele 4 variante (toate cu 256 de fire = 16 × 16, ca `mm_train_tile`):

| Variantă | Bucata din C | BK | Pe fir | Memorie partajată | Pentru ce formă |
|---|---|---|---|---|---|
| (`mm_train_tile`, martorul) | 64 × 64 | 16 | 4 × 4 | 8 KiB | toate |
| `mm_tv_128x64` | 128 × 64 | 16 | 8 × 4 | 12 KiB | multe rânduri, puține coloane |
| `mm_tv_64x128` | 64 × 128 | 16 | 4 × 8 | 12 KiB | multe coloane (capul, 16 384) |
| `mm_tv_128x128` | 128 × 128 | 8 | 8 × 8 | 8 KiB | formele mari (siboehm kernel 5) |
| `mm_tv_64x64k8` | 64 × 64 | 8 | 4 × 4 | 4 KiB | aceeași bucată, felie mai mică (mai multe blocuri pe SM) |

3. Modelul pe procesor: o singură funcție cu geometria ca parametru (`gpu_sim::cmdproc::mm_train_tv_on_cpu`), care urmează
   plăcile programului (încărcarea pe fire, lanțul de FMA pe fir) — și câte o funcție mică pe variantă pentru registrul de programe.
4. În `nvidia-ga104`: modulul nou `mm_train_variants` (doar adăugare): tabelul variantelor (nume, cod, SHA-256, registre,
   bloc, memorie partajată, grilă) + formele #513 ale pasului, ca OS-ul să aibă lista pentru `test-placa --cursa`.
5. „Mărimi fixe compilate” (K cunoscut la compilare, fără verificări de margine): **nu azi** — un program care primește altă
   formă ar da în tăcere alte numere; vine cu o pază pe procesor (`fits(m, n, k)`) într-un pas separat.

## Predicții (scrise înainte)
1. **Biții (SIMULARE):** fiecare variantă = `mm_train` = `mm_train_tile` = `reference::matmul_t_fma_ftz` **bit cu bit**, pe
   toate cele 8 combinații de opțiuni, pe margini (M, N, K care nu se împart la 128 / 16 / 8), cu −0,0 păstrat (nicio
   adunare de zerouri la ultima felie).
2. **Codul NAK:** câte TM·TN·BK `ffma.ftz` pe felie (desfăcute) + TM·TN `fadd.ftz` (bias), o barieră, **0 memorie locală**.
   Registre: `128x64` și `64x128` ~45–64; `128x128` ~80–110 (64 de sume); `64x64k8` ~34 (ca martorul). Dacă NAK pune
   sume în memoria locală → varianta iese din cursă (QMD-ul nostru nu pregătește memorie locală).
3. **Formele #513** (B 2 × T 512 = 1 024 de rânduri; 12 straturi, lățimea 768, MLP 3 072, vocabular 16 384): toate laturile se
   împart la 128 → nicio placă parțială în cursa #513.
4. **Viteza (DEDUS, NETESTAT):** `128x128` câștigă la capul (1 024 × 16 384 × 768 și gradienții lui) și la MLP; pe formele
   768 × 768 are doar 48 de blocuri = un bloc pe SM (48 de SM) → probabil pierde acolo în fața lui `mm_train_tile`
   (192 de blocuri). De aceea cursa alege **pe formă**, nu o variantă pentru tot. Câștig total estimat pe înmulțiri:
   ×1,2–1,8 față de `mm_train_tile` (siboehm: kernel 5 față de 4 × 4 ≈ ×1,5–2 pe A6000). NECUNOSCUT până la placă.

## Ce urmează după (nu azi)
- OS: rândul `test-placa --cursa <listă>` (partea sesiunii OS, nota din `integration/cyber-universe/`).
- Cyber-GPU: `Plan::with_mm_choice` — pentru fiecare formă, varianta câștigătoare din raportul cursei (aceiași biți → aceeași
  amprentă a pasului); apoi variantele pentru `mm_train_gelu` și atenție.

## Rezultat (sesiunea 211, după compilare și teste)
Mesa NAK 25.0.7 reconstruit în cloud; martorul `mm_train_tile` reconstruit = `f8acce21…` (neschimbat).

| Variantă | SHA-256 | Instrucțiuni / octeți | Registre (prezis) | Partajată | Bucla lui p | Memorie locală |
|---|---|---|---|---|---|---|
| `mm_tv_128x64` | `deb06f297f42fedca493d565010511987f7f413b3261d3b1ffb96857709c7a61` | 1 812 / 28 992 | **54** (45–64) | 12 KiB | păstrată (32 `ffma`) | 0 |
| `mm_tv_64x128` | `e5bc59e3ad9af1109c85cd3f895e4f2b26749b460d5bb6740dd12f71df27f582` | 1 788 / 28 608 | **54** (45–64) | 12 KiB | păstrată (32 `ffma`) | 0 |
| `mm_tv_128x128` | `5d3affd4fd8e9e63edd2719fc204cb7ad197f18d9c0c5c453a73cc0cb9e23ebc` | 2 762 / 44 192 | **90** (80–110) | 8 KiB | păstrată (64 `ffma`) | 0 |
| `mm_tv_64x64k8` | `880884afec4b4beb8f7dedbd7d2145e9c1981dc1c007c3ef98c0b238fbb8f903` | 1 023 / 16 368 | **34** (~34) | 4 KiB | desfăcută (128 `ffma`) | 0 |

**Predicția 2 parțial greșită:** NAK **nu** a desfăcut bucla lui p la cele trei variante mari (o singură trecere de TM·TN `ffma`
în cod, plus salt); la martor și la `64x64k8` a desfăcut-o. Biții nu se schimbă (aceeași ordine a FMA-urilor), doar viteza:
câteva instrucțiuni în plus pe pas al buclei (DEDUS: ~5–15 % pierdere la variantele mari). Dacă cursa le arată aproape de martor,
pasul următor e o variantă cu bucla desfăcută de mână (`BK` fix în cod).

| # | Predicție | Ieșit (T217, SIMULARE) | Stare |
|---|---|---|---|
| 1 | Biții = `mm_train` = martorul = oracolul | 5 programe × 8 combinații × 7 forme cu plăci parțiale (K 1, 9, 17, 33…) bit cu bit; −0,0 păstrat la K = 19 | PASS |
| 2 | 0 memorie locală, registre în intervale, `fadd` = TM·TN | toate; vezi mai sus pentru buclă | PASS (cu nota) |
| 3 | Formele #513 se împart la 128 | **15 forme, 147 lansări, 599,1 GFLOP pe pas**; toate laturile ÷ 128; copii micșorate ale fiecărei forme = oracolul pe toate variantele | PASS |
| — | Refuzuri | opțiune 8, partajată prea mică cu 256 B, fără barieră, blocuri 16 × 16 → refuzate, 0 scrieri | PASS |
| — | Greșeala plantată: grila altei variante | grilă prea mică → C cu goluri, prinsă de comparație; grilă prea mare → aceiași biți | PASS |

**Ce arată formele (DEDUS, pentru cursă):** 9 din 15 forme au N = 768 → `128x128` ar avea doar 36–48 de blocuri pe 48 de SM
(un val, câte un bloc pe SM); acolo martorul sau `128x64` au șanse mai bune. Capul (1 024 × 16 384 × 768 și gradientul lui
16 384 × 768 × 1 024: 51,5 GFLOP din 599) e locul unde `128x128` / `64x128` ar trebui să câștige. Forma 1 024 × 768 × 16 384
(dX al capului, K = 16 384) are puține blocuri la orice variantă → candidat pentru K împărțit (nu azi).

**Viteza pe placă: NOT TESTED.** Cursa o face OS-ul (`test-placa --cursa`), după ce os-410 arată `mm_train_tile` trecut pe placă.

## Pasul 2 — `Plan::with_mm_choice` (sesiunea 213, SIMULARE)
Predicții (scrise înainte de cod): (1) lista #513 cu o alegere = lista `with_mm_tile` cu doar formele alese schimbate, aceeași
arenă; (2) pasul de aur mic cu toate cele 4 variante în același pas = `with_mm_tile` bit cu bit (aceeași catenă `ffma.ftz`);
(3) alegerea goală = `with_mm_tile` exact. **Toate trei confirmate** (T219, 3/3 PASS, SIMULARE). Blocuri #513 cu alegerea
inventată (fiecare formă pe altă variantă): 64 960 → 34 016. Regula alegerii: cea mai rapidă cu biți „=”, doar dacă bate martorul
cu ≥ 3 %. Nimic rulat pe placă; ce variantă câștigă pe ce formă o spune doar cursa. Nota pentru OS:
`integration/cyber-universe/C-165_cursa-mm.md` §„Pasul 2”.

## Pasul 3 — cititorul raportului cursei: `parse_race` (sesiunea 215, pe procesor)
De ce: OS-ul scrie cursa în raport ca text (P6-190, `cyber-universe` `gpu_i14.rs`, `race_row`):
`…: cursa: <varianta> | MxNxK f<opțiuni> (x<de câte ori> in pas) | <µs cu o zecimală> us median | <x.yy> TFLOP/s | biti <=|DIFERA|NETERMINAT|SARIT>`.
Ca alegerea (`MmChoice::from_race`) să iasă **direct din raport**, fără rânduri copiate de mână, `nvidia_ga104::mm_train_variants::parse_race`
citește aceste rânduri în `RaceRow`. Aceeași funcție o poate folosi și OS-ul (nimic de schimbat la el; e doar o adăugare la noi).

Predicții (scrise înainte de cod):
1. Un raport făcut exact cu formatul OS-ului (aceleași șabloane `format!`, copiate în test) pentru cele **15 forme #513 × 5 variante** dă
   **75 de rânduri** citite, cu timpul = cel scris rotunjit la 0,1 µs (`ns` = µs·1000 + zecimea·100), biții „=” → `same_bits`.
2. Rândurile „- us” (SARIT / NETERMINAT), liniile „cea mai rapida cu aceiasi biti”, rezumatul „cursa mm: …”, tabloul de bord și orice alt rând
   **nu** devin `RaceRow`; un nume de variantă necunoscut nu devine rând (numărat separat).
3. Două curse în același jurnal: se păstrează **doar ultima** (antetul `cursa mm (C-165, …)` o ia de la capăt).
4. `from_race(parse_race(raport))` = `from_race(rândurile adevărate)` atunci când timpii diferă între variante cu mai mult de 0,1 µs
   (rotunjirea OS-ului nu schimbă alegerea), iar `Plan::with_mm_choice` cu ea schimbă exact formele alese (ca T219).
5. Rânduri stricate (număr lipsă, „x” lipsă, opțiuni ne-numerice, timp cu două puncte) → ignorate, fără panică.

**Rezultat (sesiunea 215, pe procesor, T221 3/3 + 1/1):** toate 5 predicțiile confirmate din prima: 73 de rânduri citite (75 − SARIT −
NETERMINAT), timpii = cei scriși rotunjiți la 0,1 µs, formele și lansările = ale pasului #513, alegerea = cea din timpii adevărați,
`with_mm_choice` mută exact formele alese; o a doua cursă înlocuiește prima; rândurile stricate (9 feluri) și străine sunt lăsate
deoparte fără panică. Nimic nou pentru placă. Nota pentru OS: `integration/cyber-universe/C-165_cursa-mm.md` §„Pasul 3”.

## os-413 — de ce `mm_tv_64x64k8` nu a terminat (sesiunea 216, 24 nov 23:12–23:40; în cloud, FĂRĂ placă)
**Ce s-a văzut (placa reală, prin OS):** pe prima formă (qkv, 1 024 × 2 304 × 768, f6) martorul și 3 variante au mers, cu aceiași
biți; `mm_tv_64x64k8` **nu a scris nicio înregistrare** (ultima = 0x0, nici măcar lansarea de încălzire), apoi placa și-a revenit și
generarea de după a fost corectă. Câștigătoarea: `mm_tv_128x128`, 1 889,8 µs față de 2 838,3 (×1,50).

**Ce NU e cauza (verificat în cod):** programul în sine are aceeași structură ca martorul (`mm_train_tile`, trecut pe placă de multe
ori): aceleași feluri de instrucțiuni, 34 de registre ca martorul, aceeași barieră, bucla lui p desfăcută ca la martor, 0 memorie
locală, 4 KiB partajată (sub configurația de 8 KiB, ca `mm_train` cu 2 KiB, trecut pe placă). Biții sunt corecți în SIMULARE (T217).

**Cauza cea mai probabilă (DEDUS, nedovedit pe placă):** citirea în avans a instrucțiunilor. Unitatea care aduce instrucțiunile
citește puțin **după** instrucțiunea pe care o rulează. OS-ul mapează fiecare program rotunjit doar la pagini de 4 KiB
(`map_vram(cod.len())`), iar programele cursei sunt mapate unul după altul — `mm_tv_64x64k8` **ultimul**. Codul lui are 16 368 de
octeți: se termină la **16 octeți** de capătul paginilor, iar după el nu mai e nimic mapat. Citirea în avans cade în memorie
nemapată → greșeală de pagină → canalul se oprește înainte de prima înregistrare. Toate celelalte variante au după cod fie restul
paginii, fie programul următor (T222: 17–45 KiB mapați după cod). **Singura** cu sub 2 KiB după cod e exact cea care s-a oprit.
Același lucru l-au pățit alții: Mesa nouveau/nvc0, commit-ul „increase overallocation on shader bo” (2024): „read page faults at
the end of the shader allocation”, iar comentariul: memoria programelor are nevoie de 2 KiB (cifra NVIDIA) în plus la capăt
„to avoid prefetch bugs”; NVK lasă o pagină întreagă. (Citit ca idee, MIT; nimic copiat.)

**Ce am adăugat (doar adăugări, nimic schimbat):**
- `qmd::PROGRAM_TAIL_BYTES` (4 096), `qmd::program_map_bytes(len)` = codul rotunjit la pagini **+ o pagină**,
  `qmd::program_page_slack(len)`.
- `mm_train_variants::RACE_HOLD = [4]`, `in_race(i)`, `race_list()` — lista cursei fără `64x64k8`; `VARIANTS` rămâne neschimbat
  (indicii 0–4, `Prog` 25–28, `parse_race`, `MmChoice` la fel).
- T222 (pe procesor): mărimile, așezarea din os-413 (doar `64x64k8` sub 2 KiB), lista cursei, și `parse_race` pe rândurile
  **adevărate** din os-413 → alegerea pe qkv = `mm_tv_128x128`, pasul ar scădea cu ~11,4 ms din cele 12 lansări qkv (DEDUS).

**Cum se lămurește pe placă (un singur lucru nou):** OS-ul mapează programele cu `program_map_bytes` (o pagină în plus după fiecare;
nicio scriere nouă în registre, nicio comandă nouă), apoi rulează `mm_tv_64x64k8` **singur** pe o formă. Dacă termină cu aceiași
biți: cauza confirmată și varianta se întoarce în cursă. Dacă tot nu termină: cauza e în program (rămâne scoasă; o recompilăm cu
altă geometrie). Până atunci, cursa se face cu `race_list()`.

## os-416 — cursa completă (15 forme × 4 variante) pe raportul adevărat (sesiunea 219, 25 nov 02:14; în cloud, FĂRĂ placă)
**Ce a venit de pe placă (prin OS, `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0135.txt`, rândurile 214–292):** 60 de
rânduri, toate cu aceiași biți ca martorul; `mm_tv_128x128` cea mai rapidă pe toate 15 formele; OS-ul a socotit că pasul ar scădea
cu ~141 387 µs (DEDUS, din timpii izolați); tabloul: 1 785 MHz, P0, 77 W, nicio încetinire.

**Predicții (scrise înainte de cod, T225, pe procesor):**
1. `parse_race` pe rândurile adevărate: 60 de rânduri cu timp, 0 fără timp, 0 necunoscute, 15 forme cu 147 de lansări în total, iar
   formele și numerele de lansări = cele ale pasului #513 (`Plan::new_tiles3` + `with_mm_tile`), în ordinea pasului.
2. `MmChoice::from_race`: toate 15 formele pe `mm_tv_128x128` (indicele 3), chiar și cele mici (768×768×1024: −11 %;
   2304×768×1024: −12 %, peste pragul de 3 %); un singur program în plus: `mm_tv_128x128`.
3. Economia socotită de noi din timpii rotunjiți la 0,1 µs = cea a OS-ului (141 387 µs) la cel mult ±20 µs.
4. Suma timpilor martorului (izolat) × lansări ≈ 420 ms ≤ timpul plăcii pe pas (463,7 ms): înmulțirile sunt ~90 % din pas,
   restul (~40 ms) e atenția, LayerNorm, GELU, AdamW, colsum.
5. `with_mm_choice` cu alegerea mută exact 147 de lansări, același număr de lansări, aceeași arenă (ca T219/T221).
6. Pe placă (de verificat de OS, nu aici): pasul #513 cu `mm_tv_128x128` peste tot ≈ 474,2 − 141,4 ≈ **333 ms**, interval
   325–345 ms (≈ 2 970–3 150 simb/s), aceiași biți ca oracolul (verdictul ca la os-416).

**Rezultat (sesiunea 219, pe procesor, T225 3/3 PASS):**
- **P1 corect:** 60 de rânduri cu timp, 0 fără timp, 0 necunoscute, toate „=”; 15 forme, 147 de lansări, aceleași (și în aceeași
  ordine) ca formele pasului #513 din `Plan::new_tiles3` + `with_mm_tile`.
- **P2 corect:** `from_race` → toate 15 formele pe `mm_tv_128x128`; câștigul cel mai mic pe o formă = **11,2 %** (768×768×1024),
  departe de pragul de 3 %; un singur program în plus. Alegerea = `MmChoice::all(forme, RACE_WINNER_513)` (nou, doar adăugări).
- **P3 corect:** economia noastră din timpii rotunjiți = **141 388 µs**, a OS-ului 141 387 (1 µs diferență).
- **P4 GREȘITĂ:** suma timpilor martorului izolați × lansări = **456,2 ms = 98 %** din timpul plăcii pe pas (463,7), nu ~420.
  În pas, profilul os-412 dă `mm_train_tile` = **430,5 ms**. Diferența (+6 %) se potrivește cu ceasul: cursa a mers la **1 785 MHz**
  (tabloul în timpul cursei, 77 W), pașii la **1 950 MHz** (os-414) — lansările scurte și izolate nu urcă placa la ceasul maxim (DEDUS).
  Lecția: timpii izolați ai cursei bun pentru *alegere* (raportul dintre variante), nu pentru *cifra absolută* a pasului.
- **P5 corect:** `with_mm_choice` mută exact cele 147 de înmulțiri pe `mm_tv_128x128`, același număr de lansări, aceeași arenă; nicio
  lansare nu mai rămâne pe `mm_train` / `mm_train_tile`.
- **P6 rafinată (de verificat pe placă):** raportul 128×128 / martor din cursă = **0,690**; aplicat pe înmulțirile din pas (430,5 ms)
  → economia **~133 ms** (nu 141), pasul întreg **~341 ms ≈ 3 000 simb/s** (OS-ul a socotit ~333 ms din timpii izolați).
  Intervalul 325–345 ms rămâne. Biții: aceiași ca oracolul (ca la os-416, unde toate variantele au dat biții martorului).
- **Ce rămâne după:** și `mm_tv_128x128` e doar ~2,0 TFLOP/s la 1 785 MHz (~9 % din vârful F32) → frâna e tot în citiri; următorul pas
  mare e `mm_tc` (os-352 / os-384 încă nerulate) pe aceeași formă de plăci, apoi pasul mixt pe #513 (`LOSS_SCALE_513`).

## `mm_tc128` — `mm_tc` pe plăci 128×128 (sesiunea 220, 25 nov 03:14; în cloud, FĂRĂ placă)

**De ce:** cursa os-416 a ales pe toate 15 formele placa de 128×128 (×1,45 față de 64×64 în F32). `mm_tc` (circuitele speciale, f16 → f32)
are azi doar plăci de 64×64 (128 de fire). Pregătim varianta lui de 128×128, ca să intre în aceeași cursă imediat după ce `mm_tc` trece
o dată singur pe placă (os-352 / os-384, încă nerulate — regula „primul de felul lui” rămâne).

**Ce e:** `tools/nak/mm_tc128.comp` — aceeași treabă, aceleași constante (cbuf 0 + 0x700), aceeași aritmetică ca `mm_tc` (op(A), op(B)
rotunjite la f16 la copiere; produsele adunate în f32 de circuitele speciale de la 0, K în pași de 16 în ordine; bias o dată la final).
Se schimbă doar lansarea: 256 de fire (8 subgrupuri), 128×128 din C pe bloc, fiecare subgrup 32×64 = 2×4 acumulatori; ieșirea prin câte
o zonă mică de 16×16 pe subgrup (16 + 8 = 24 KiB de memorie comună, ca `mm_tc`; un bloc de 128×128 f32 nu încape în 48 KiB).

**Predicții (scrise înainte de cod, T226):**
1. NAK 26.3.1 compilează programul; registre ≤ 128 pe fir (8 acumulatori × 8 + fragmentele), fără scurgeri în memoria locală.
2. În listare: 16 `HMMA.16816.F32` pe felie de 32 (8 acumulatori × 2 pași de 16) — de 4 ori mai multe decât `mm_tc` (4), două bariere de bloc.
3. În SIMULARE: aceiași biți ca `mm_tc` pe toate 15 formele pasului #513 și pe forme cu margini (M, N, K care nu se împart la 128 / 32),
   cu și fără TA / TB / BIAS; grila = ⌈N/128⌉ × ⌈M/128⌉; modelul refuză blocuri de 128 sau grila prea mică.
4. Pe placă (ESTIMARE, de verificat după os-352): între ×1,3 și ×2 față de `mm_tc` 64×64 pe formele mari ale pasului #513
   (aceeași cauză ca în F32: de 2 ori mai puține citiri din memorie pe operație).

**Rezultat (sesiunea 220, T226 4/4 PASS în SIMULARE):**
- **P1 corect, la limită:** NAK 26.3.1 îl compilează; **128 de registre** (exact pragul), 0 scurgeri (`ldl`/`stl`); 58 688 B de cod;
  SHA-256 `df5041bd…68f79`, aceeași la două compilări (și `mm_tc` recompilat azi = `f60cd944…`, suma înregistrată).
- **P2 corect ca raport, greșit ca cifră:** 32 de `HMMA` în listare față de 16 la `mm_tc` (×2, cum am zis), dar un `coopMatMulAdd` de
  16×16×16 = **două** `HMMA.16816`, deci 32 pe felie de 32, nu 16. Bariere de bloc: 2 pe felie, ca `mm_tc`.
- **P3 corect:** aceiași biți ca `mm_tc` pe 6 forme cu margini × 8 combinații de opțiuni și pe toate 15 formele pasului #513 (M, N
  tăiate la 160 / 288, K întreg); modelul refuză blocuri de 128 și grila scurtă. Limita testului: modelul aritmetic e același pentru
  amândouă, deci testul confirmă lansarea, grila și acoperirea; ordinea adunărilor pe element e DEDUSĂ din cod — placa o confirmă.
- **P4:** de verificat pe placă, după os-352. Risc nou văzut: 128 de registre × 256 de fire → cel mult 2 blocuri pe unitate de calcul.

## os-417 — proba `mm_tv_64x64k8` singură, cu pagina de pază (sesiunea 221, 25 nov 04:14; în cloud, FĂRĂ placă)
**Ce a venit de pe placă (prin OS, `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0348.txt`, rândurile 214–222, copiate
neschimbat în `crates/nvidia-ga104/tests/rapoarte/os417-proba64k8.txt`):** după cei 10 pași #513, martorul `mm_train_tile` și
`mm_tv_64x64k8` pe forma qkv (1 024 × 2 304 × 768, f6), `64x64k8` mapată tot **ultima** (ca la os-413), dar cu `program_map_bytes`
(o pagină în plus după cod). Un singur lucru schimbat față de os-413.

**Ce prezisese sesiunea 216 (scris înainte, nota `C-165_os413_64x64k8.md`):** termină cu aceiași biți → cauza (citirea în avans)
confirmată și varianta se întoarce în cursă. Nu prezisesem un timp pentru `64x64k8`.

**Predicții pentru verificarea de azi (scrise înainte de cod, T227):**
1. `parse_race` citește din os-417 exact 2 rânduri cu timp, amândouă „=”, 0 fără timp, 0 necunoscute; o formă, 12 lansări.
2. Alegerea din os-417 singur: qkv → `mm_tv_64x64k8` (indicele 4); economia = 12 × (2 839,2 − 1 768,0) µs ≈ 12 854 µs, cât a scris OS-ul.
3. Rândurile os-416 + os-417 puse împreună (martorul măsurat de două ori: 2 839,4 / 2 839,2 µs, deci același ceas al cursei,
   1 785 MHz): qkv → `64x64k8` (1 768,0 < 1 890,2 la `128x128`), celelalte 14 forme → `128x128`; programele alegerii: 2.
4. Pasul #513 din os-417 (`--placi3 --mm-placi --colsum-impartit`, fără alegere) = os-414 (aceeași listă): placa 463,7 ms față de
   463,6, sub 1 ‰ — pasul e repetabil, deci comparația os-418 cu alegerea `128x128` (prezis ~341 ms) rămâne curată.
5. `program_page_slack` pentru `mm_tv_64x64k8` cu `program_map_bytes` ≥ 4 096 (față de 16 la os-413).

**Rezultat (sesiunea 221, T227 4/4 PASS pe procesor; timpii = ai plăcii reale, prin OS):**
- **Cauza de la os-413 confirmată pe placă:** cu pagina de pază după cod (singurul lucru schimbat), `mm_tv_64x64k8` a terminat cu
  **aceiași biți** ca martorul. `program_page_slack` = 16 B la os-413, ≥ 4 096 B la os-417. Varianta se întoarce în cursă:
  `RACE_CLEARED_OS417 = [4]`, `race_list_full()` (toate 5; `RACE_HOLD` și `race_list()` rămân neschimbate — doar adăugări).
- **P1, P2 corecte:** 2 rânduri „=”, 12 lansări; qkv → `64x64k8`, **1 768,0 µs față de 2 839,2** (0,622 din martor = ×1,61);
  economia 12 854 µs = cât a scris OS-ul.
- **Surpriza (n-o prezisesem):** pe qkv, `64x64k8` e **mai rapidă și decât `128x128`** (1 768,0 față de 1 890,2 µs, −6,5 %),
  la același ceas al cursei (martorul 2 839,2 / 2 839,4 µs în cele două rulări).
- **P3 corectă:** os-416 + os-417 împreună → qkv pe `64x64k8`, celelalte 14 pe `128x128`, 2 programe; economia 142 852 µs față de
  141 385 (+1,5 ms pe pas, DEDUS din timpii izolați). Mic — dar `64x64k8` n-a fost încă măsurată pe celelalte 14 forme.
- **P4 corectă:** pasul os-417 (463 732 µs) = os-414 (463 594 µs), sub 1 ‰ → pasul e repetabil; înmulțirile 431,2 ms = 93 % din placă.
- **P5 corectă.**

**De ce poate conta mai mult decât 1,5 ms (DEDUS, de verificat cu cursa completă):** `128x128` face puține blocuri pe formele
înguste: 1 024 × 768 → 8 × 6 = **48 de blocuri** = un singur bloc pe fiecare din cele 48 de unități de calcul; `64x64k8` face 192.
Exact pe formele cu 768 de coloane `128x128` a câștigat cel mai puțin la os-416 (768×768×1024: 0,888 din martor; 2304×768×1024:
0,883; 1024×768×768: 0,75). **Predicție pentru cursa completă cu 5 variante:** `64x64k8` câștigă cel puțin cele 3 forme de mai sus
(și probabil alte forme cu N = 768); `128x128` păstrează capul (16 384) și formele cu N ≥ 2 304, unde are destule blocuri. Dacă
`64x64k8` ține ~0,62 din martor peste tot, pasul ar mai scădea cu ~15–40 ms față de alegerea os-416 (interval larg: un singur punct
măsurat).

## os-418 — pasul #513 cu alegerea cursei (`--mm-alegere`, toate 147 de înmulțiri pe `mm_tv_128x128`) (sesiunea 222, 25 nov 05:15; în cloud, FĂRĂ placă)
**Ce a venit de pe placă (prin OS, `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0433.txt`, rândurile 158–218, copiate
neschimbat în `crates/nvidia-ga104/tests/rapoarte/os418-pas.txt`):** lista os-414 (`--placi3 --mm-placi --colsum-impartit`) + o
singură schimbare: `Plan::with_mm_choice` cu alegerea citită din cursa os-416 (`parse_race`) → toate 15 formele pe `mm_tv_128x128`.
Verdictul OS-ului: TRECE (= oracolul în pașii 1–9).

**Ce prezisese sesiunea 219 (scris înainte, §„os-416”, P6 rafinată):** raportul cursei 0,690 aplicat pe înmulțirile din pas →
înmulțirile ~297 ms, pasul întreg **~341 ms** (interval 325–345 ms, ≈ 3 000 simb/s), aceiași biți ca oracolul. Cinste: rezultatul
(`REZULTAT.md` din hub) l-am văzut înainte să scriu verificările de mai jos; predicția care se judecă e cea a sesiunii 219.

**Verificări pentru azi (scrise înainte de cod, T228, pe procesor, pe rândurile copiate):**
1. `parse_profiles` citește un profil: 327 581 µs, 483 de lansări (aceleași ca os-414/417); pe programe `mm_tv_128x128` 147 de
   lansări, iar `mm_train_tile` / `mm_train` nu mai apar.
2. Pierderile și normele celor 10 pași (rândul `banc antrenare`) = cele din os-414, cifră cu cifră (aceeași ordine a adunărilor pe
   element în toate variantele, deci aceiași biți).
3. Raportul înmulțirilor în pas, os-418 / os-417 (294 946 / 431 180 µs), e la cel mult 1 % de raportul cursei (0,690) — adică
   raportul măsurat izolat (la 1 785 MHz) se păstrează în pas (la 1 950 MHz).
4. Restul plăcii (fără înmulțiri) neschimbat față de os-417, la cel mult 0,5 ms; partea din afara plăcii (pasul întreg − placa)
   neschimbată față de os-414, la cel mult 1 ms — singura schimbare a fost programul înmulțirilor.
5. Pasul întreg (median) în intervalul 325–345 ms și la cel mult 2 % de ~341.
6. **Înainte, pentru cursa cu 5 variante (`race_list_full()`, de rulat de OS):** cu alegerea nouă, pasul #513 **300–330 ms**
   (−8 … −38 ms față de os-418), aceiași biți. Cifra pe care o urmărim: câte forme trec pe `64x64k8`.
