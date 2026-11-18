# Analiza I14 după `os-305` — de ce iese alt simbol, deși I9–I13 au trecut pe placă (comanda C-139)

Sesiunea Cyber-GPU 112 (18 nov 2026, cloud, **fără placă**). Tot ce e cod aici e testat **doar în SIMULARE**. Nimic nu a rulat pe RTX 3070 Ti.
Date: hub `rezultate/2026-11-18-workstation-os305/REZULTAT.md` + `rezultate/os-automat/2026-11-18/gpu-jurnal-007.txt` (`gpu lant --i9 c135`, cyber-gpu `f732365`).
Etichete: **OBSERVAT** = scris în jurnalul de pe placă; **CITIT** = scris în documentația NVIDIA sau în surse deschise (MIT), citite ca specificație;
**DEDUS** = concluzia noastră; **IPOTEZĂ** = explicație posibilă, nedovedită; **NECUNOSCUT** = nu se poate afla fără placă.
Nota pentru OS: `integration/cyber-universe/C-139_i14-diagnostic-si-reparatie.md`.

## 0. Pe scurt (pentru Adrian)

- **Ce s-a întâmplat la `os-305`:** placa a făcut pentru prima dată calcule adevărate și corecte: o adunare (I10), o copiere de 8 MiB (I11), o
  înmulțire matrice × vector (I12) și un strat întreg de transformer (I13). La I14 — un **simbol întreg** al modelului mic de control — placa a
  terminat tot ce i-am cerut (nicio blocare, nicio eroare), dar a ales **alt simbol** decât Kronos. Atât știm din jurnal: numerele plăcii nu au ajuns pe stick.
- **Nu e rotunjire.** Pe procesor, primul simbol e ales cu o distanță de 0,28 față de al doilea; toleranța noastră e 0,001. Ca să iasă alt simbol,
  cel puțin un număr al plăcii a fost greșit de peste o sută de ori mai mult decât toleranța.
- **Cauza cea mai probabilă (cam 55 %): placa a citit numere vechi.** Un simbol trece prin două straturi care folosesc **aceeași zonă de lucru**.
  Stratul 2 trebuia să citească ce a scris stratul 1, dar placa ține minte ce a citit mai devreme (o memorie rapidă, „cache”) și noi îi spuneam să uite
  o singură dată, la începutul simbolului. I14 e primul pas în care aceeași adresă e citită, rescrisă și citită din nou în aceeași trimitere — de aceea
  I13 (un singur strat) a trecut. Documentația NVIDIA spune că driverul trebuie să golească această memorie între calcule care depind unul de altul;
  driverul deschis tinygrad o face la fiecare lansare.
- **Reparația:** îi spunem plăcii să uite **înaintea fiecărui calcul**, cu aceeași comandă pe care placa a primit-o deja la fiecare trimitere de la
  I10 încoace (niciun registru nou). În simulare, pe această ipoteză, felul vechi greșește ca pe placă, iar cel reparat dă simbolurile lui Kronos.
- **Fiindcă fără placă nu putem fi siguri**, testul următor nu pariază pe o singură explicație. Înaintea textelor rulează primul simbol în mai multe
  feluri și **scrie în jurnal unde apare primul număr greșit**: ce calcul, în ce strat, cu cât, și dacă e „număr vechi” sau calcul greșit. Dacă un fel
  trece, textele rulează așa mai departe, în aceeași pornire. Dacă e greșit un singur calcul care poate fi ocolit (citirea simbolului sau copierea
  rezultatului), textele rulează ocolindu-l, dar pasul rămâne „ESEC” până îl reparăm — nu numim „reușit” ceva ce a trecut pe ocolite.
- **Șanse la testul următor (`os-306`):** cam **50 %** ca I14 să treacă de tot; peste **90 %** ca, dacă nu trece, jurnalul să spună exact ce calcul
  e greșit — adică să nu mai avem nevoie de o pornire doar pentru a afla unde e problema.

## 1. Ce s-a observat pe placă (OBSERVAT, `gpu-jurnal-007.txt`)

| Pas | Rezultat | Ce a însemnat pe placă |
|---|---|---|
| I0–I8, golden, I9 (`c135`, varianta N) | ok | canalul de lucru primit de la GSP-RM |
| `i10-add64`, `i10-cost-lansare` | ok (119,92 s; 120,16 s) | o lansare; apoi 16 lansări ale aceluiași program într-o trimitere |
| `i11-copiere` | ok (121,15 s) | 8 MiB RAM → VRAM → RAM, bit cu bit |
| `i12-q8-matvec` | ok (122,10 s) | copiere sus, W·x (3072 × 768, Q8_0), copiere jos — 3 lansări într-o trimitere |
| `i13-strat` | ok (126,35 s) | un strat (E 768, 12 capete), 4 poziții, **16 lansări pe trimitere**, x scris de driver |
| `i14-simbol` | **ESEC** (127,07 s): „I14: argmax diferit de vectorii lui Kronos” | vezi mai jos |
| Bus Master pus la loc, turul de citire | ok | placa lăsată curată |

În segmentul I14 sunt **două** sonerii ale canalului de lucru (126,73 s și 126,99 s) și nicio eroare a canalului, nicio așteptare depășită.

**Ce rezultă de aici, cu codul OS-ului de la `os-305` în față** (`kernel/src/drivers/gpu_i14.rs`; DEDUS):
1. Prima sonerie = urcarea modelului (greutăți + zona de lucru + logit-uri, 692 480 B, o singură bucată). A doua = **un singur simbol**: primul simbol
   al primului text de control (poziția 0; `vectori.json` spune ce simbol trebuie să iasă după el).
2. Toate cele 37 de înregistrări ale simbolului au sosit, în ordine (`analyze_sequence` a trecut înainte să se citească logit-urile): cele 36 de
   lansări (35 ale modelului + copierea logit-urilor în RAM) **s-au executat și s-au terminat**.
3. Mesajul „argmax diferit” apare doar dacă `argmax` există, adică **toate cele 16 384 de logit-uri citite din RAM erau numere finite** (cu un singur
   NaN verificarea aceasta e sărită și mesajul ar fi fost altul). Simbolul plăcii a fost altul decât cel cerut.
4. **Ordinea verificărilor (întrebarea din C-139):** da — argmax-ul față de `vectori.json` e verificat în `run_token`, **înaintea** judecății
   logit-urilor față de procesor (`Generation::accept` → `kronos_model::judge`, toleranță 1e-3). De aceea știm doar „alt simbol”, nu și cât de departe
   sunt numerele. Rândul cu cele două valori (`argmax … pe placa, vectori.json …`) a mers doar pe ecran (`lines`), nu în jurnalul de pe stick.

**NECUNOSCUT din `os-305`:** simbolul ales de placă, logit-urile ei, abaterea față de procesor, etapa la care a apărut greșeala.

## 2. Ce nu poate fi: rotunjirea

Marja top1 − top2 a motorului nostru de pe procesor, la toate cele 29 de poziții ale celor trei texte de control (unealta `analiza_i14`, cu vectorii
din hub pregătiți local; testul `kgr1_margins_rule_out_rounding`):

| Poziție | Marja | Față de toleranța logit-urilor (1e-3) |
|---|---|---|
| primul text, poziția 0 (cea care a căzut) | **0,28** | de peste 280 de ori |
| cea mai mică dintre cele 29 | 0,028 | de 28 de ori |

Două logit-uri mișcate unul spre altul cu toată toleranța (2 × (1e-3 + 1e-3 · 8,4)) nu ajung nici la cea mai mică marjă. Deci pe placă cel puțin un
logit a fost greșit cu peste 0,14 — o **eroare de calcul sau de date**, nu o diferență de rotunjire (MUFU, FMA, ordinea adunărilor). (DEDUS)

## 3. Ce face I14 și nu făcuse niciun pas dinainte

Lista din C-139, punct cu punct. „Poate da simptomul?” = toate logit-urile finite, alt argmax, toate lansările terminate.

| Ce diferă | La I13 și înainte (trecut pe placă) | La I14 | Poate da simptomul? |
|---|---|---|---|
| Citirea simbolului (embedding) | x scris de driver prin fereastră | programul `embed` — **prima lui rulare pe placă** | da → ipoteza I2 |
| Al doilea strat | un singur strat pe trimitere | stratul 2 **refolosește zona de lucru** a stratului 1 în aceeași trimitere: x, h, qkv, … sunt citite, rescrise și citite din nou | da → **ipoteza I1** |
| Norma finală | — | același program `layernorm`, dar citește x a cincea oară (x a fost scris de 3 ori) | doar prin I1 (programul a rulat la I13) |
| Capul de ieșire 16 384 × 32 | `q8_matvec` cu 3072×768, 2304×768, 768×768, 768×3072 | 16 384 de rânduri × **un singur bloc Q8** pe rând (doar firul 0 al fiecărui grup lucrează) | da → I3 (formă nerulată) |
| Copierea logit-urilor în RAM (arena DMA) | I11: 8 MiB; I12: y | 64 KiB, a 36-a lansare, sursa scrisă de lansarea dinainte | da → I4: tamponul rămâne cu umplutura `0xEE` = −3,7·10²⁸ la fiecare logit, finit, argmax 0 |
| Ordinea `run_records` / sonerie | cel mult 16 lansări pe trimitere | 36 de lansări + 37 de înregistrări într-o trimitere, **o singură golire de cache** la început | ordinea e bună (înregistrările au sosit); golirea unică = I1 |
| Cache-ul K/V între poziții | I13: 4 poziții, fiecare în trimiterea ei | neatins: a căzut la poziția 0 (rândul 0 e scris la lansările 3–4 și citit prima dată la 5) | nu la această cădere |
| Umplutura `0xEE` | — | vezi „copierea” | doar prin I4 |
| Alinierea și bucățile blobului (8 MiB) | — | 692 480 B = **o bucată**; toate părțile aliniate la 16; zona de lucru la 623 360 (multiplu de 256), finalul la 626 816 | nu singură (granița dintre bucăți nu e atinsă de modelul mic; va fi la #513) |
| Unde stau logit-urile | — | în blob, după vectorul final; scrise de cap, citite doar de copiere | nu |
| Formele mici | E 768, 12 capete (d 64) | E 32, 4 capete (d 8): `layernorm` cu un element pe fir, atenția cu d 8, matricile cu 1–4 blocuri pe rând | da → I3 |

Un fapt care restrânge mult: OS-ul rulează **același cod** al pasului I14, cu același model și aceiași vectori, pe modelul nostru de GPU în QEMU, și
acolo trece. Deci greșeala e în ceva ce simularea nu modelează: cum se poartă **memoria plăcii** între lansări (I1), ce face **codul SASS** al unui
program pe procesorul plăcii (I2, I3 — simularea rulează echivalentele de pe procesor, nu SASS-ul) sau **scrierea plăcii în RAM** (I4).

## 4. Ipotezele, în ordine

Procentele sunt estimări, nu măsurători.

### I1 — date vechi citite în aceeași trimitere (cam 55 %)
- **Ce spune:** programele noastre citesc intrările cu `ld.global.a64.constant.b32` (NAK le compilează așa pentru bufferele declarate `readonly`).
  O asemenea citire poate fi servită din cache-ul L1 al procesorului plăcii. Scrierile (`st.global.a64.strong.sys`) ajung în memorie, dar nu
  împrospătează liniile L1 deja citite. `transfer::encode_sequence` (C-83) golește cache-urile **o dată**, la începutul trimiterii.
- **Pentru:** (a) e exact ce are I14 nou — vezi §3: I10–I13 nu recitesc niciodată o adresă rescrisă în aceeași trimitere (verificat lansare cu
  lansare, și în SIMULARE: cu modelul „date vechi” pornit, secvențele I11, I12 și I13 trec neschimbate); (b) simptomul se potrivește: stratul 2 lucrează pe
  numere reale, dar vechi → logit-uri finite, alt simbol; (c) documentația NVIDIA cere driverului să golească aceste cache-uri între lansări
  dependente, iar tinygrad o face la fiecare lansare (§5).
- **Contra:** (a) QMD-ul nostru nu setează `SM_GLOBAL_CACHING_ENABLE` (Mesa și tinygrad îl pun pe 1); nu știm dacă, cu el pe 0, citirile
  `.constant` mai trec prin L1 — NECUNOSCUT; (b) ipoteza cere linii de cache de cel mult 128 B: cu linii de 256 B doi vectori vecini ai zonei de
  lucru ar împărți o linie, ar apărea NaN (zona pornește cu `0xFF`) și mesajul plăcii ar fi fost altul (SIMULARE, §6); (c) textul NVIDIA despre
  L1 e scris pentru plăci mai vechi; nicio sursă citită nu spune cum se poartă Ampere (SM 8.6) — NECUNOSCUT.
- **Cum o confirmă testul:** proba de control (trimiterea ca la `os-305`) dă „DATE VECHI” la prima normalizare a stratului 1, iar proba cu golirea
  înaintea fiecărei lansări trece.

### I2 — programul `embed` greșit pe placă (cam 15 %)
- **Pentru:** singurul program al unui simbol care nu a mai rulat pe placă; folosește trei forme de instrucțiuni pe care programele rulate până acum
  nu le au (`lop3`, `shf.r.w.u32.hi` și `shf.l.w.i32` cu numărul de biți luat dintr-un registru — `embed.sm86.txt`); un x greșit dă toate logit-urile
  finite și greșite.
- **Contra:** e cel mai simplu program (un element pe fir); codul intermediar al lui NAK a fost comparat cu sursa GLSL.
- **Cum o arată testul:** proba „doar embedding-ul” (o lansare) iese „LANSARE GREȘITĂ”; simbolul trece cu embedding-ul scris de driver.

### I3 — o formă mică nerulată încă (cam 10 %)
- Capul de ieșire (16 384 de grupuri, un bloc pe rând), atenția cu d = 8, `layernorm` cu E = 32. **Contra:** aceleași programe, generice în formă,
  au trecut la I12/I13. **Cum o arată testul:** o etapă „LANSARE GREȘITĂ” (ieșirea nu urmează din intrările aflate în memorie), cu numele ei.

### I4 — logit-urile nu au ajuns în RAM (cam 5 %)
- **Pentru:** umplutura `0xEE` dă exact mesajul de pe placă (argmax 0, altul decât cel cerut). **Contra:** I11 și I12 au dovedit pe placă scrierea în aceleași
  pagini DMA. **Cum o arată testul:** scara trece (ea citește VRAM), primul simbol al textului nu; explicația scrie câte logit-uri diferă între RAM
  și VRAM și câte au rămas `eeeeeeee`.

### I5 — rotunjire sau egalitate apropiată (sub 1 %) — exclusă de marje (§2).

### I6 — altceva (cam 15 %)
Ceva ce nu am gândit: de aceea diagnosticul nu depinde de nicio ipoteză — compară fiecare etapă cu ce trebuia să iasă și cu ce iese din intrările plăcii.

## 5. Sursele pe care se sprijină I1 (CITIT, 18 nov 2026; nimic copiat — `docs/PROVENANCE.md`)

- **NVIDIA, „PTX ISA” 8.8, §9.7.9.1 „Cache Operators”, tabelul 31, rândul `.ca`** (operatorul implicit al instrucțiunii de citire): datele globale
  sunt coerente la nivelul L2, dar cache-urile L1 **nu** sunt coerente între ele pentru date globale; driverul trebuie să invalideze liniile L1
  globale între grile (lansări) care depind una de alta. Rândul `.wb` al scrierilor repetă cerința.
- **tinygrad `runtime/ops_nv.py` (MIT, ramura principală):** `NVComputeQueue.memory_barrier` trimite `INVALIDATE_SHADER_CACHES_NO_WFI` cu
  instrucțiuni + date globale + constante — exact metoda și valoarea noastră. În plus, șablonul QMD-ului (`NVProgramData`) are
  `sm_global_caching_enable = 1`, `invalidate_shader_data_cache = 1`, `constant_buffer_invalidate_0 = 1`, `cwd_membar_type = L1_SYSMEMBAR` și cele trei
  invalidări de texturi; șablonul e copiat la fiecare lansare, deci **fiecare lansare a lui tinygrad cere golirea**. (Unde anume e chemat
  `memory_barrier` nu e în acest fișier — necitit.)
- **Mesa NAK (MIT), commit `ad98b76a147` (MR 35217), pentru plăcile Kepler:** o citire globală din memorie „constantă” primește operatorul „cache la
  toate nivelurile”, celelalte citiri globale „doar L2”. Arată cum privesc autorii compilatorului citirile „constante” (ale noastre sunt așa); nu
  spune nimic direct despre SM 8.6.
- **Mesa NVK (MIT), commit `c39bf7e7f98` (MR 35246):** NVK are bariere numite `NVK_BARRIER_INVALIDATE_TEX_DATA` și
  `NVK_BARRIER_INVALIDATE_SHADER_DATA`; am citit doar partea despre texturi (codul barierei pentru datele de shader nu l-am citit).
- **mesa-dev, oct. 2016 (driverul vechi nvc0, patch 4/7):** aceeași metodă pe plăci mai vechi: bitul 0 = cod, bitul 12 = buffere constante;
  driverul NVIDIA trimite de obicei `0x1011`.
- **Forumul NVIDIA pentru dezvoltatori, iunie 2012:** un inginer citează aceleași rânduri din manualul PTX și conchide că între lansări dependente
  trebuie golite doar cache-urile L1; tot el spune că plăcile viitoare ar putea avea L1 coerent — de aici „contra (c)” de mai sus.

Valoarea pe care o trimitem (`0x1011`: instrucțiuni + date globale + constante) e cea pe care placa a primit-o deja la începutul fiecărei trimiteri
de la I10 încoace. **Nu apare nicio metodă și niciun registru nou.**

## 6. Ce arată modelul (SIMULARE — o ipoteză pusă în model, nu o măsurătoare)

`gpu_sim::cmdproc::CommandProcessor::model_stale_reads(linie)` (oprit implicit): citirile unui program trec printr-un cache pe care nimic nu-l ține
la zi până la următoarea invalidare. Pe **modelul de control real** (`mic_q8_0.kgr`, vectorii din hub pregătiți local), primul simbol al primului text, la poziția 0:

| Cum rulează (SIMULARE) | argmax | Logit-uri nefinite | Abaterea maximă față de procesor | Ce ar fi spus pasul |
|---|---|---|---|---|
| memorie coerentă (modelul de până acum) | cel cerut | 0 | 3,3·10⁻⁶ | trece |
| **I1, linii de 4…128 B, trimis ca la `os-305`** | **altul** | **0** | 7,17; 16 369 din 16 384 în afara toleranței | „argmax diferit” — ca pe placă |
| I1, linii de 256 B | niciunul | 16 384 | — | „logit-uri NaN/infinit” — nu ca pe placă |
| I1 + reparația (invalidare înaintea fiecărei lansări) | cel cerut | 0 | 3,3·10⁻⁶ | trece |
| I1 + o trimitere pe strat | cel cerut | 0 | 3,3·10⁻⁶ | trece |
| I4: logit-urile necopiate în RAM | 0 | 0 | 3,7·10²⁸ | „argmax diferit” |
| I2: embedding fără rândul poziției | altul | 0 | 8,04 | „argmax diferit” |
| I2: embedding cu numerele fără semn | altul | 0 | 10,4 | „argmax diferit” |

Cele patru feluri greșite dau **patru simboluri diferite** și logit-uri diferite; valorile (simbolul și primele 8 logit-uri pentru fiecare) le afișează
`analiza_i14` din vectorii pregătiți local — nu le scriem aici, fiindcă vectorii de control nu se pun în acest depozit.
Aceste semnături se compară cu numerele plăcii din jurnalul următor (`analiza_i14 <gpu-jurnal.txt>` o face singură).
Pe placă, ce valori rămân vechi depinde de ce procesor (SM) rulează fiecare grup, deci numerele plăcii pot diferi de ale modelului chiar dacă I1 e adevărată;
ce contează e **verdictul pe etape**, nu potrivirea cifrelor.

## 7. Reparația și plasele de siguranță (cod nou, doar adăugări)

| Fel de a trimite un simbol | Ce face | Când e ales |
|---|---|---|
| `SeqMode::Once` (`transfer::encode_sequence`) | o golire de cache pe trimitere — ce a rulat la `os-305` | niciodată: e doar proba de control |
| **`SeqMode::EachLaunch`** (`transfer::encode_sequence_fresh`) | aceleași cuvinte + `INVALIDATE_SHADER_CACHES_NO_WFI` înaintea **fiecărei** lansări | **reparația**: primul încercat |
| `SeqMode::PerLayer` | o trimitere (și o sonerie) pe strat — regimul în care I13 a trecut pe placă | dacă `EachLaunch` nu trece |
| `EmbedBy::Driver` (`kronos_diag::driver_embedding`) | embedding-ul calculat de driver (E înmulțiri) și scris la x, cum a scris I13 x-ul lui; `embed` nu mai e lansat | dacă embedding-ul singur e cel greșit — **ocolire**: textele merg mai departe, pasul raportează **ESEC** cu motivul |
| logit-urile citite din VRAM (`Explained::vram_pass`) | fără copierea în RAM | dacă doar copierea e greșită — **ocolire**: textele merg mai departe, pasul raportează **ESEC** cu motivul |

Regula: un fel de a **trimite** aceleași lansări (`EachLaunch`, `PerLayer`) e o reparație — pasul poate fi `ok`. O **ocolire** a unei lansări greșite
pe placă (embedding-ul sau copierea făcute altfel) nu e: textele rulează până la capăt doar ca să aflăm în aceeași pornire dacă restul e bun.

Costuri: `EachLaunch` adaugă 2 cuvinte pe lansare — un simbol al modelului de control cu copierea are 1 760 B de comenzi (zona de 16 KiB ajunge);
#513 (196 de lansări) are 9 440 B, deci zona de comenzi trebuie mărită la 32 KiB (64 KiB pentru 24 de straturi). `PerLayer` cere 712 B, dar 3 sonerii pe
simbol la modelul de control (13 la #513).

De ce nu am schimbat și QMD-ul (biții de invalidare, `SM_GLOBAL_CACHING_ENABLE`): ar fi câmpuri pe care placa nu le-a primit încă de la noi. Reparația
aleasă folosește doar ce placa a acceptat deja. Dacă pe placă `EachLaunch` nu trece și `PerLayer` trece, QMD-ul e pasul următor de studiat.

## 8. Diagnosticul din jurnal (`nvidia_ga104::kronos_diag`)

Totul trece prin patru lucruri pe care le dă OS-ul (`TokenIo`: trimite lansările; citește memoria plăcii; scrie câteva cuvinte; jurnalul).

- Rândurile se scriu în același jurnal în care OS-ul scrie accesele lui (`TokenIo::journal`), deci pe stick apar în ordine, imediat după
  trimiterile fiecărei probe — și rămân acolo chiar dacă placa se blochează la proba următoare.
- **`ladder`** (scara), o dată, înaintea textelor, pe primul simbol al primului text: (1) proba de control, ca la `os-305`; (2) reparația; (3) doar
  dacă nu trece: o trimitere pe strat; (4) doar dacă nici așa: embedding-ul singur, iar dacă el e greșit, simbolul cu embedding-ul driverului;
  (5) dacă tot nu trece: fiecare strat până la a 15-a lansare a lui (intrarea lui încă în memorie), până e numită lansarea. O trimitere care nu se
  termină oprește totul. Răspunsul: felul în care rulează textele (`Ladder::chosen`).
- După fiecare probă, memoria e citită înapoi (zona de lucru, rândurile noi K/V, vectorul final, logit-urile) și **fiecare lansare e judecată de
  trei ori**: față de ce trebuia să scrie; față de ce iese din **intrările pe care le are placa în memorie**; iar dacă nu urmează din ele, față de
  **valorile vechi** ale intrărilor (semnătura unei citiri vechi). Verdictul arată prima etapă care **dovedește** ceva.
- „Date vechi” e recunoscut și când doar **o parte** din ieșirea unei lansări vine din valori vechi (unele rânduri, elemente sau capete), iar restul
  urmează din intrările de acum — așa ar arăta o placă pe care fiecare procesor (SM) are cache-ul lui și doar unele au ținut linia veche.
- **`explain`**, când un simbol al unui text nu trece: memoria așa cum a lăsat-o acel simbol, judecată la fel, fără să trimită nimic; spune și dacă
  doar copierea în RAM e greșită. **`locate`**: dacă greșeala a intrat într-un strat ale cărui valori au fost deja rescrise, o găsește cu prefixe.
- **Rândurile din jurnal** (toate încep cu `i14d`, numele lor sunt în `kronos_diag::rows`): felul trimiterii, cine a scris embedding-ul, câte lansări,
  verdictul (1 trece, 2 date vechi, 3 lansare greșită, 4 moștenit, 5 placa nu a terminat, 6 doar logit-urile), simbolul și poziția, argmax-ul plăcii /
  al procesorului / din `vectori.json`, câte logit-uri sunt nefinite și câte au rămas `eeeeeeee`, **primele 8 logit-uri ale plăcii și ale
  procesorului**, logit-ul fiecăruia la argmax-ul celuilalt, **abaterea maximă** și indexul ei, câte sunt în afara toleranței, marja pe procesor;
  apoi, pentru fiecare etapă în afara toleranței: stratul și lansarea, numele, abaterea, dacă urmează din intrările plăcii, dacă e „date vechi” și de
  unde; un rezumat într-o propoziție; la sfârșit felul ales. Numerele reale sunt scrise ca biții lor (exact).
- **Cât costă:** pe I1, scara scrie 148 de rânduri (cam 2,5 s la ritmul jurnalului de pe stick); o probă care trece costă vreo 14 rânduri.
- **Citirea:** `verifica_jurnal_lant <gpu-jurnal.txt> …` afișează diagnosticul în română simplă; `analiza_i14 <gpu-jurnal.txt>` îl compară și cu
  semnăturile din §6.

## 9. Ce înseamnă fiecare rezultat al testului următor

| Ce scrie jurnalul | Ce înseamnă | Ce urmează |
|---|---|---|
| control: DATE VECHI (în stratul 1; întreg sau „o parte din ieșire”); reparația: TRECE | I1 confirmată | textele rulează cu reparația în aceeași pornire; dacă trec, I14 e trecut pe placă și urmează #513 |
| control: LANSARE GREȘITĂ sau EROARE MOȘTENITĂ într-o lansare a stratului 1; reparația: TRECE | tot I1, cu un amestec de valori vechi pe care diagnosticul nu l-a recunoscut (aceeași lansare nu poate fi greșită doar când lipsește golirea) | la fel ca mai sus |
| control: TRECE; reparația: TRECE; apoi textul cade la poziția 0 cu „logit-uri diferite între RAM și VRAM” | I4: doar copierea în RAM | textele merg mai departe cu logit-urile din VRAM (pasul rămâne ESEC); de reparat copierea de 64 KiB |
| control și reparația nu trec, „pe straturi” trece | date vechi pe care golirea dintre lansări nu le curăță | textele rulează pe straturi (mai lent); de studiat câmpurile QMD (§7) |
| niciun fel nu trece; embedding-ul singur: LANSARE GREȘITĂ; cu embedding-ul driverului: TRECE | I2: programul `embed` | textele rulează cu embedding-ul driverului (pasul rămâne ESEC); de reparat `embed` (jurnalul are abaterea) |
| o etapă LANSARE GREȘITĂ, cu nume (de ex. „logit-uri = Emb * h” sau „atentie (stratul 0, lansarea 5)”) | I3: acel program, la acea formă | pasul se oprește; reparăm programul numit, în SIMULARE, apoi alt test |
| control: TRECE și textele trec | greșeala de la `os-305` nu s-a repetat | trecut; comparăm jurnalele ca să vedem ce a fost diferit |
| PLACA NU A TERMINAT | o trimitere nu și-a scris ultima înregistrare | pasul se oprește imediat, nimic nu mai e trimis; analiză cu jurnalul |
| scara trece, un text cade la o poziție ≥ 1 | ceva legat de pozițiile dinainte (cache-ul K/V, atenția cu mai multe poziții) | `explain` (+ `locate`) numesc etapa în același jurnal |

## 10. Ce nu știm și ce nu am făcut

- Nu știm nimic nou despre placă: tot ce e mai sus în afară de §1 e SIMULARE, CITIT sau DEDUS. I1 poate fi falsă.
- Nu am dezasamblat și nu am analizat nimic din firmware; nu am scris niciun registru nou; nu am ocolit nicio verificare (pasul rămâne ESEC dacă
  embedding-ul sau copierea în RAM sunt greșite pe placă, chiar dacă textele trec ocolindu-le; felul ales de scară e scris în jurnal).
- `SM_GLOBAL_CACHING_ENABLE` și biții de invalidare din QMD rămân nesetați (ca la `os-305`); îi atingem doar dacă placa arată că e nevoie.
- Modelul „date vechi” ține un singur cache pentru toată placa; placa reală are câte unul pe fiecare procesor (SM). E destul pentru a arăta
  mecanismul, nu pentru a prezice cifrele plăcii. Diagnosticul recunoaște și amestecul (o parte veche, o parte nouă), dar numai față de valorile
  vechi pe care le poate calcula (cele ale primului strat, care a rulat pe date proaspete).

## 11. Teste și unelte (toate SIMULARE)

- `crates/nvidia-ga104/tests/sim_i14_diag.rs` (16 teste, 2 cu vectorii din hub): tabloul plăcii pe ipoteză (I13 trece, I14 cade cu logit-uri finite);
  scara confirmă „date vechi” și alege reparația; un strat ascuns arată citirea veche în rândurile lui K/V; căderea pe „o trimitere pe strat”; o
  lansare greșită e numită (în primul strat, în al doilea, după straturi — norma finală și capul de ieșire —, la orice poziție sau doar de la
  poziția 1); o lansare doar pe jumătate „veche” e tot „date vechi”, una pe jumătate greșită nu; embedding greșit → al driverului; o trimitere
  care nu se termină oprește scara; explicația unui simbol deja rulat (și cazul „doar copierea în RAM”); **pasul întreg, cum îl descrie nota pentru OS**;
  memorie citită ca gunoi (NaN, infinit, zero, biți la întâmplare, umplutură) — diagnosticul rulează în nucleu, deci nu are voie să se oprească
  cu panică: dă un verdict, nu alege niciun fel, iar rândurile se citesc înapoi (rulat și cu verificarea depășirilor aritmetice);
  modelul de control real pe ipoteză (cele trei texte dau exact vectorii lui Kronos cu reparația și cu embedding-ul driverului); marjele.
- `sim_i11_copy.rs`, `sim_i12_q8.rs`: secvențele I11 și I12 trec și cu modelul „date vechi” pornit (ca pe placă).
- `transfer` (teste de unitate): secvența reparată = cea veche + o invalidare pe lansare, cuvânt cu cuvânt.
- Unelte pe gazdă: `cargo run --release -p nvidia-ga104 --example analiza_i14 [-- <gpu-jurnal.txt>]` (cu `KGR1_VECTORI`), `verifica_jurnal_lant`.
- Total la sfârșitul sesiunii: **531 PASS + 6 ignorate** (cele 6 cer vectorii din hubul privat; trec local), `tools/check.sh` verde.
