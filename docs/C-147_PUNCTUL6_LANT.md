# C-147 punctul 6 — etapa 3b: „lanțul pe placă” (DOAR studiu)

Sesiunea 127, 20 noiembrie 2026, ~22:30 (ora României), în cloud, fără placă. **Nimic din acest document nu e cod și nimic nu schimbă
ce trimitem azi plăcii.** Etichete: CITIT (scris așa în sursă), DEDUS (concluzia noastră din surse), NOT TESTED (nimic pe placă).

## 0. Pe scurt (pentru Adrian)
- Azi, între oricare două calcule, procesorul pune în coada plăcii trei lucruri: „golește memoria rapidă”, „pornește calculul”,
  „scrie ora și așteaptă să se termine tot”. Placa stă puțin la fiecare din cele 88–196 de calcule ale unui simbol.
- „Lanțul pe placă” înseamnă că fiecare calcul poartă în el adresa următorului, iar placa îl pornește singură, fără procesor.
  tinygrad face așa la **fiecare** lansare; Mesa are teste pe placă reală exact pentru asta.
- Câștigul așteptat e **mic** după dieta driverului (−0,1…0,3 ms pe simbol, DEDUS, ca în estimarea punctului 7), iar riscul e real
  (felul de trimitere care a trecut I14 se schimbă). Recomandare: rămâne după dietă, P3, LayerNorm și etapa 4; când vine rândul lui,
  se face în 3 trepte mici, fiecare cu proba „aceleași 16 simboluri ca procesorul” și întoarcere automată la felul de azi.
- **Pe placă: NOT TESTED. Intră doar cu „da”-ul separat al lui Adrian (P7).**

## 1. Ce trimitem azi (`SeqMode::EachLaunch`, `transfer::encode_sequence_fresh`, a trecut I14 pe placă)
Pentru fiecare lansare `k` a simbolului, în pushbuffer (CITIT din codul nostru):

| Cuvinte | Metoda | Rol |
|---:|---|---|
| 2 | `NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI` (instrucțiuni + date globale + constante) | reparația I14: lansarea `k` nu mai citește date vechi |
| 2 | `NVC7C0_SEND_PCAS_A` (adresa QMD-ului >> 8) | care lansare |
| 2 | `NVC7C0_SEND_SIGNALING_PCAS2_B` = `PREFETCH_SCHEDULE` (9) | pornește-o |
| 6 | `NVC56F_SEM_ADDR_LO…SEM_EXECUTE` cu `RELEASE_WFI` + `RELEASE_TIMESTAMP` | înregistrarea `k` (16 B: valoarea, apoi ora la +8) — **așteaptă ca placa să fie liberă** |

Deci 12 cuvinte pe lansare + 8 la început; ordinea între lansări e ținută de `RELEASE_WFI` (canalul nu trece mai departe până nu
se termină tot). QMD-ul nostru (`qmd.rs`) pune doar: versiunea, mărimile grilei și ale grupului, adresa programului, registrele,
memoria partajată + barierele, bufferul constant 0 și (la măsurători) `RELEASE0`. **Nu pune** `SM_GLOBAL_CACHING_ENABLE`,
`API_VISIBLE_CALL_LIMIT`, biții de golire din QMD, `CWD_MEMBAR_TYPE`, nimic din `DEPENDENT_QMD0_*`.

## 2. Regula „trei surse” — ce spune fiecare

### Sursa 1: NVIDIA, antetele MIT (forma câmpurilor) — CITIT
`open-gpu-doc` commit `9fdf5c40` (același depozit fixat în `tools/gen-rm570/generate.sh`), `classes/compute/clc6c0qmd.h` și
`clc7c0qmd.h` (GA104 = `AMPERE_COMPUTE_B` = C7C0; câmpurile de mai jos sunt **identice** în V03_00 din cele două fișiere — am comparat):

| Câmp V03_00 | Biți | Valori |
|---|---|---|
| `SM_GLOBAL_CACHING_ENABLE` | 134 | 0 / 1 |
| `QMD_GROUP_ID` | 133:128 | |
| `DEPENDENCE_COUNTER` | 157:142 | |
| `INVALIDATE_TEXTURE_{HEADER,SAMPLER,DATA}_CACHE` | 186, 187, 188 | |
| `INVALIDATE_SHADER_DATA_CACHE` | 189 | |
| `INVALIDATE_INSTRUCTION_CACHE` | 190 | |
| `INVALIDATE_SHADER_CONSTANT_CACHE` | 191 | |
| `CWD_MEMBAR_TYPE` | 369:368 | `L1_NONE` 0, `L1_SYSMEMBAR` 1, `L1_MEMBAR` 3 |
| `API_VISIBLE_CALL_LIMIT` | 378 | `_32` 0, `NO_CHECK` 1 |
| `DEPENDENT_QMD0_POINTER` | 511:480 | adresa QMD-ului următor >> 8 |
| `DEPENDENT_QMD0_ENABLE` | 512 | |
| `DEPENDENT_QMD0_ACTION` | 515:513 | `INCREMENT_PUT` 0, `SCHEDULE` 1, `INVALIDATE_COPY_SCHEDULE` 3, `DECREMENT_DEPENDENCE` 4 |
| `DEPENDENT_QMD0_PREFETCH` | 516 | |
| `RELEASE0_MEMBAR_TYPE` | 819 | `FE_NONE` 0, `FE_SYSMEMBAR` 1 |
| `RELEASE0_STRUCTURE_SIZE` | 831:830 | `FOUR_WORDS` 0 (cu ora), `ONE_WORD` 1, `TWO_WORDS` 2 |
| `CONSTANT_BUFFER_INVALIDATE(i)` | 1074 + 64·i | |
| `HW_ONLY_*` (`INNER_GET`, `SKED_NEXT_QMD_POINTER`, `HW_ONLY_DEPENDENCE_COUNTER` …) | 1824–1958 | scrise de placă (DEDUS din nume) |

Atenție: codurile `DEPENDENT_QMD0_ACTION` din QMD **nu** sunt aceleași cu cele din metoda `SEND_SIGNALING_PCAS2_B`
(`SCHEDULE` = 2, `INVALIDATE_COPY_SCHEDULE` = 3, `PREFETCH_SCHEDULE` = 9 în `clc7c0.h`). Toate câmpurile sunt deja generate în
`nvidia-rm570::regs::qmd_v03_00` (verificat). Antetele dau **doar forma**, nu și ce face placa cu ele.

### Sursa 2: tinygrad `runtime/ops_nv.py` (MIT), ramura principală, commit `5e8b5da9` (20 nov 2026) — CITIT
- Șablonul QMD al fiecărui program (scris o dată): `sm_global_caching_enable = 1`, `qmd_group_id = 0x3f`, cele trei
  `invalidate_texture_*` = 1, **`invalidate_shader_data_cache = 1`**, `api_visible_call_limit = 1`, `sampler_index = 1`,
  `barrier_count = 1`, **`cwd_membar_type = L1_SYSMEMBAR`**, `constant_buffer_invalidate_0 = 1`, mărimile memoriei partajate,
  `program_prefetch_size`, `sass_version`.
- `NVComputeQueue.exec`: **prima** lansare dintr-o coadă se pornește cu `SEND_PCAS_A` + `SEND_SIGNALING_PCAS2_B = PREFETCH_SCHEDULE`;
  **fiecare lansare următoare** e legată de cea dinainte: `dependent_qmd0_pointer = adresa >> 8`, `dependent_qmd0_action = 1`
  (`SCHEDULE`), `dependent_qmd0_prefetch = 1`, `dependent_qmd0_enable = 1`. Nimic de la procesor între ele.
- Semnalele și orele: dacă ultima lansare are un `RELEASE0/1` liber, semnalul se pune **în QMD** (`release_structure_size` 0 =
  patru cuvinte, cu ora; 2 = fără oră; `payload64b = 1`); altfel o eliberare obișnuită de canal. O barieră (`memory_barrier`) sau o
  așteptare **rupe** lanțul (următoarea lansare pornește iar cu `SEND_PCAS_A`).
- tinygrad rescrie toate QMD-urile cozii la fiecare trimitere (`patch(self.qmd_buf, …)`).

### Sursa 3: Mesa 26.3.1 (MIT), NAK `compiler/nak/qmd.rs` + teste pe placă `hw_tests.rs`, `hw_runner.rs` — CITIT
- `Qmd3_0::new` pune în **orice** QMD V03_00: `API_VISIBLE_CALL_LIMIT = NO_CHECK`, `SAMPLER_INDEX = INDEPENDENTLY` (0; tinygrad pune 1),
  **`SM_GLOBAL_CACHING_ENABLE = 1`**.
- `set_dependent_qmd(addr, schedule)` pentru V03_00: `DEPENDENT_QMD0_ENABLE = 1`, acțiunea **`INVALIDATE_COPY_SCHEDULE` (3)** dacă
  „schedule”, altfel `DECREMENT_DEPENDENCE` (4). **Fără** `PREFETCH`. (tinygrad: `SCHEDULE` 1 + `PREFETCH`.)
- Testele pe placă reală (rulează pe orice clasă ≥ Pascal; nu scrie pe ce plăci au trecut):
  - `test_qmd_dependencies_simple`: o lansare cu una dependentă — ordinea.
  - `test_qmd_dependencies`: întâi un program **lent**, cu unul **rapid** dependent; comentariul testului: verifică „ordinea așteptată”
    și că **„golirea memoriei rapide cerută în QMD face scrierile vizibile peste dependențe”** — în QMD-ul dependent se pune
    `INVALIDATE_SHADER_CONSTANT_CACHE` (programul rapid citește prin calea constantelor, `ldcg`); fără ea, comentariul spune că
    rezultatul ar fi greșit.
  - `test_qmd_hw_decrement_dependence`: două lansări care scad contorul unei a treia (arbore „mai mulți → unul”) — nu ne trebuie.
- Pornirea primei lansări în runner: `SEND_PCAS_A` + `SEND_SIGNALING_PCAS2_B = INVALIDATE_COPY_SCHEDULE` (Ampere și după).
- NVK (driverul Vulkan) **nu** folosește lanțul la lansările obișnuite (`nvk_cmd_dispatch.c`: `dependence_counter = 0`), doar testele
  NAK — DEDUS: Vulkan cere bariere explicite între lansări, deci NVK nu are nevoie de el.

### Verdictul regulii — DEDUS
- **Forma câmpurilor:** îndeplinită (antet NVIDIA + două programe care le scriu, cu aceleași poziții).
- **Ce face placa** (lansarea dependentă pornește abia după ce se termină cea dinainte; golirea din QMD face vizibile scrierile):
  două surse care se bazează pe asta (tinygrad în producție, Mesa în teste pe placă) + numele câmpurilor. Nicio documentație NVIDIA
  care să descrie comportamentul. Suficient pentru o **probă**, nu pentru a-l considera dovedit pe GA104.
- **Diferențe între surse** de lămurit pe placă: acțiunea (`SCHEDULE` + `PREFETCH` la tinygrad, `INVALIDATE_COPY_SCHEDULE` la Mesa)
  și ce golire (tinygrad: date + `CWD_MEMBAR_TYPE = L1_SYSMEMBAR`; Mesa: constante, în test).

## 3. Câmpurile pe care tinygrad le pune și noi nu — ce ar face și dacă ne trebuie
| Câmp | tinygrad | Mesa | Noi | Ne trebuie la lanț? (DEDUS) |
|---|---|---|---|---|
| `DEPENDENT_QMD0_*` | da, la fiecare | da (teste) | nu | **da** — e lanțul însuși |
| `INVALIDATE_SHADER_DATA_CACHE` | 1 | — | 0 | **da** — înlocuiește comanda separată de golire (reparația I14) |
| `INVALIDATE_SHADER_CONSTANT_CACHE` | 0 | 1 (test) | 0 | probabil da: bufferele constante se rescriu de procesor între simboluri |
| `CWD_MEMBAR_TYPE = L1_SYSMEMBAR` | da | — | 0 | probabil da: scrierile unei lansări ies din L1 înainte să pornească următoarea |
| `SM_GLOBAL_CACHING_ENABLE` | 1 | 1 | **0** | nu pentru lanț; poate aduce viteză la citiri (L1 pentru date globale) — **pas separat**, are nevoie de golire corectă |
| `API_VISIBLE_CALL_LIMIT = NO_CHECK` | 1 | 1 | 0 | nu (programele noastre nu fac apeluri); aliniere cu ambele surse, fără efect așteptat |
| `INVALIDATE_TEXTURE_*`, `SAMPLER_INDEX` | 1 | 0 / — | 0 | nu (nu folosim texturi) |
| `QMD_GROUP_ID = 0x3f` | da | — | 0 | necunoscut; nu-l punem fără motiv |
| `PROGRAM_PREFETCH_*`, `SASS_VERSION` | da | — | 0 | nu pentru corectitudine; poate puțină viteză la pornirea programului — nemăsurat |
| `RELEASE0` cu `STRUCTURE_SIZE = FOUR_WORDS` | da (ora) | — | doar la măsurători | **da**, ca să păstrăm rândurile `kp` (ora fiecărei lansări) fără procesor |

## 4. Ce s-ar schimba față de `EachLaunch` (DEDUS; nimic scris)
**Pushbuffer-ul unui simbol** ar deveni: golirea (2) + ora de început (6) + `SEND_PCAS_A` + `PCAS2_B` pentru **prima** lansare (4) +
o eliberare finală cu `RELEASE_WFI` (6) ≈ **18 cuvinte**, în loc de 8 + 12·n (1 064 la 88 de lansări, 2 360 la 196).
**QMD-urile** primesc, o dată, la pregătirea lanțului: adresa următorului + acțiunea + biții de golire + `RELEASE0` cu ora într-o
înregistrare proprie (același format de 16 B ca azi: valoarea la 0, ora la +8 — DEDUS din tinygrad, ca la `launch_cost`).
**Lanțul se construiește din lista de lansări a modelului** (care vine din fișa KGR1): doar un pointer pe QMD — se potrivește cu
decizia lui Adrian din 20 nov 19:41 (nimic legat de #513 scris în cod).

**Unde e câștigul și cât** (DEDUS): după dieta D, pushbuffer-ul costă deja puțin (doar cuvintele schimbate). Rămâne timpul plăcii
între lansări: azi fiecare lansare așteaptă să se golească tot canalul (`RELEASE_WFI`) și abia apoi citește metodele următoare.
`kl` (lansarea goală, os-313) = 3,90 µs median, din care o parte e chiar această așteptare. Dacă lanțul scoate ~1–3 µs pe lansare:
88 × (1…3) µs ≈ **0,1–0,26 ms** pe simbol — ca în estimarea punctului 7 (rândul P7). Cifra reală se vede doar pe placă: aceleași
rânduri `kp`, înainte / după.

## 5. Riscuri (DEDUS) și cum le prinde proba
1. **Date vechi** (ca la I14), dacă golirea din QMD nu e echivalentă cu comanda de azi → proba „aceleași 16 simboluri” + diagnosticul
   existent (`kronos_diag`, „unde apare primul număr greșit”) rămân pornite; la prima diferență, lanțul se oprește și simbolul se
   reface cu `EachLaunch` (felul de azi).
2. **Placa nu pornește lansarea următoare** (acțiune greșită) → ultima înregistrare nu vine; `records_seen` (C-87) spune până unde a
   ajuns; așteptarea are deja limită de timp; refacerea plăcii: calea de recuperare existentă.
3. **QMD-uri refolosite de la un simbol la altul.** Ambele surse scriu QMD-urile din nou la fiecare trimitere; noi le refolosim
   (iar cu dieta D le scriem doar dacă s-au schimbat față de **copia noastră**). Câmpurile `HW_ONLY_*` (biții 1824–1958) par scrise
   de placă. Nelămurit: dacă placa lasă acolo o stare care strică a doua pornire. Prevenție propusă: acțiunea `INVALIDATE_COPY_SCHEDULE`
   (Mesa; „copy” = placa lucrează pe o copie, DEDUS din nume) la prima treaptă, și o **citire înapoi** a cuvintelor 57–61 ale
   câtorva QMD-uri după un simbol (doar citire, rând în jurnal), înainte de orice altă treaptă.
4. **Fără ora fiecărei lansări** dacă `RELEASE0` din QMD nu se comportă ca eliberarea de canal → rândurile `kp` ar lipsi; treapta 1
   de mai jos le verifică întâi, separat de lanț.
5. **Câmpuri noi pentru placă** (`SM_GLOBAL_CACHING_ENABLE`, biții de golire, `CWD_MEMBAR_TYPE`): nu sunt registre, ci octeți în
   memoria noastră pe care placa îi citește; un câmp greșit poate da un calcul greșit sau o eroare a canalului (prinsă de
   `ErrorNotifier`, ca la I9), nu o scriere în registre necunoscute.

## 6. Ordinea propusă când vine rândul lui P7 (fiecare treaptă: opțiune oprită implicit, „da” separat, proba „identic cu procesorul”)
| Treaptă | Ce se schimbă | Ce aflăm |
|---|---|---|
| 0 (oricând, doar citire) | după un simbol, citim înapoi cuvintele 57–61 ale primelor QMD-uri → rânduri în jurnal | dacă placa scrie în QMD (riscul 3) |
| 1 | QMD-urile primesc biții de golire + `CWD_MEMBAR_TYPE` + `RELEASE0` cu ora; trimiterea rămâne `EachLaunch` | că placa acceptă câmpurile și că orele din QMD = orele de azi |
| 2 | se scoate comanda separată de golire (golirea doar din QMD); trimiterea rămâne lansare cu lansare + `RELEASE_WFI` | dacă golirea din QMD ajunge (I14 nu revine) |
| 3 | lanțul: `DEPENDENT_QMD0` cu `INVALIDATE_COPY_SCHEDULE` (forma Mesa), doar prima lansare din pushbuffer | câștigul real, cu `kp` înainte / după |
| 3b | dacă 3 trece: forma tinygrad (`SCHEDULE` + `PREFETCH`) într-o cursă A B B A, ca la `q8_race` | care formă e mai rapidă |
| separat | `SM_GLOBAL_CACHING_ENABLE = 1` (ambele surse) | viteza citirilor; doar după ce golirea e lămurită |

## 7. Anexă: mărimi Kronos scrise în cod (decizia din 20 nov 19:41) — inventar rapid, DEDUS din căutare
- Venite din fișa KGR1 la rulare: `Shape` (n_emb, capete, eps), straturile, vocabularul, contextul — lanțul, bugetul VRAM și lista de
  lansări se calculează din ele (`kronos_model`, `kronos_gen::VramBudget::of_model`).
- Limite ale **programelor** (nu ale lui #513): `attention.comp` — context ≤ 512 (`shared float sc[512]`, `gpu_compute::layer::MAX_POS`),
  dimensiunea capului ≤ 128; înmulțirile Q8_0 — intrările multiplu de 32; `q8_matvec32` e gândit pentru 768 / 3 072 de intrări
  (merge pentru orice multiplu de 32, dar la alte mărimi nu mai folosește toate firele). Un Kronos cu context > 512 sau cap > 128 cere
  programe noi = un „pachet de calcul” nou (decizia din 19:45).
- Constante numite după model, folosite doar la verificări, viteză și cursă: `kronos_speed::{K513_*, K651_*}`, `q8_race::K513_LANES`.
  Nu intră în calcul. Pasul următor (când se cere): `ModelLimits` direct din fișă, nu din aceste constante.

## Surse
- NVIDIA `open-gpu-doc` (MIT), commit `9fdf5c4062007929d9f4e6cbad9c9771fe61b880`: `classes/compute/clc6c0qmd.h`, `clc7c0qmd.h`, `clc7c0.h`.
- tinygrad (MIT), commit `5e8b5da9699721a3613b5a36e376506ae5aded85`: `tinygrad/runtime/ops_nv.py` (`QMD`, `NVComputeQueue.exec`,
  `.release`, `.memory_barrier`, `NVProgram.__init__`).
- Mesa 26.3.1 (MIT), tarball-ul de la `archive.mesa3d.org` (aceeași versiune ca a doua unealtă NAK): `src/nouveau/compiler/nak/qmd.rs`,
  `hw_tests.rs` (`test_qmd_dependencies_simple`, `test_qmd_dependencies`, `test_qmd_hw_decrement_dependence`), `hw_runner.rs`
  (`dispatch_qmd`), `src/nouveau/vulkan/nvk_cmd_dispatch.c`.
- Ale noastre: `transfer::encode_sequence_fresh`, `gsp::work_queue::{invalidate_caches, launch}`, `launch_cost`, `qmd.rs`,
  `docs/os-305_ANALIZA_I14.md`, `docs/C-147_PUNCTUL7_ESTIMARE.md`.
Citite doar ca specificație; niciun rând copiat în cod (acest punct nu are cod).
