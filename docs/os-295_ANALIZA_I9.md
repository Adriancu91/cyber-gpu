# os-295 (12 noiembrie 2026): I8 TRECE pe placă; I9 se oprește la al 11-lea mesaj RM — fără motiv în jurnal

Sesiunea Cyber-GPU 93, 12 noiembrie 2026, ~18:00–18:40 (ora României), în cloud. Sursa: `cyber-hub/rezultate/2026-11-12-workstation-os295/REZULTAT.md`
și jurnalul `rezultate/os-automat/2026-11-12/gpu-jurnal-010.txt` (2425 rânduri, cyber-gpu `49e6685`, `gpu lant --varianta 1`, stick `os-295`).
Tot ce scrie aici e **citit din jurnal** (OBSERVAT) sau **dedus din codul nostru** (DEDUS); nimic nu e verificat altfel pe placă.

## 1. Ce s-a întâmplat (OBSERVAT, din jurnal)

| Pas | Rezultat pe placă | Ce spune jurnalul |
|---|---|---|
| I6, I7 | ok | ca la os-283; RPC-urile `SetSystemInfo` + `SetRegistry` + soneria **înainte de Booter** (varianta 1, C-108) |
| **I8** | **ok** | `GSP_INIT_DONE: rpc_result = 0x00000000`, răspuns după **11 169 710 µs** (11,17 s) de la sonerie; `pas i8-gsp-init: ok` |
| I8, pe drum | — | **doar 2 NOCAT** (os-283: 844): 1 ASSERT (adresa `0x0132d3b2`, alta decât cele 16 din os-283) + 1 **ENGINE `NV_ERR_INVALID_DMA_SPECIFIER` (0x27), source „Display Subsystem”, faultingEngine „Display Engine”**; 2 `UCODE_LIBOS_PRINT` de la ENG_PMU (16 și 24 de octeți, conținutul în jurnal) |
| **I8e golden** | **ok** | 11 sonerii GSP = cele 11 mesaje RM (7 alocări + 4 comenzi), toate acceptate; tabelele de pagini scrise prin fereastra `0x1700` |
| **I9** | **ESEC** | 11 sonerii GSP, apoi `pas i9-canal-lucru: ESEC` — **fără niciun rând despre motiv** |
| curățare | ok | `bus-master-oprit: ok`, turul după: WPR2 ridicat (normal după I6), `tur-dupa: ok`; „OPRIT CURAT la i9-canal-lucru” (`verifica_jurnal_lant`) |

`compara_variante 0x9c66000 gpu-jurnal-009.txt gpu-jurnal-010.txt` (unealta din sesiunea 89): „au trecut de I8: varianta 1 (rpc-inainte)”;
față de os-283: 51 rânduri doar în referință (asserturile), 40 doar aici (golden + I9).

**Concluzia pentru C-108:** varianta 1 (`rpc-inainte`) era cea corectă — analiza din `docs/os-283_ANALIZA_ORDINE_RPC.md` s-a confirmat pe placă.
Variantele 2–10 nu mai sunt necesare.

## 2. Unde s-a oprit I9 (DEDUS din numărarea soneriilor)

Fiecare mesaj RM = o scriere la soneria GSP (`0x110c00`, `QueueTransportBell`), iar OS-ul scrie cuvintele VRAM strânse (tabelele de pagini) prin
fereastra `0x1700` **imediat înainte** de mesajul următor (`gpu_golden::OsRm::call` → `flush`). Jurnalul după `pas i8e-golden: ok`:

| Rânduri | Scrieri | Mesajele RM din `user_space` (ordinea din cod) |
|---|---|---|
| 2368–2380 | 7 sonerii | `open_user_space`: ROOT, DEVICE, SUBDEVICE, VASPACE, `SET_PAGE_DIRECTORY` (5) + `open_compute_group`: CHANNEL_GROUP, CONTEXT_SHARE (2) — **toate acceptate** |
| 2382–2384 | fereastra 0x1700 (3) | tabelele de pagini pentru inel + USERD (`alloc_mapped`), USERD pus pe 0 |
| 2386, 2388 | 2 sonerii | `open_channel`: CHANNEL_GPFIFO, COMPUTE — **acceptate** |
| 2390–2394 | fereastra 0x1700 (5) | tabelele de pagini pentru cele 3 buffere de context (MAIN, PM, PATCH) |
| 2396 | sonerie | `PROMOTE_CTX` jumătatea **fizică** (gpuPhysAddr, size, physAttr 0x4, bInitialize, bNonmapped) — **acceptată** (altfel n-ar fi urmat a doua) |
| 2398 | sonerie | `PROMOTE_CTX` jumătatea **virtuală** (gpuVirtAddr, restul 0) — **aici s-a oprit**: nu mai urmează nicio sonerie (a 12-a ar fi fost `WORK_SUBMIT_TOKEN`) |
| 2399 | — | `pas i9-canal-lucru: ESEC` |

Deci GSP-RM a acceptat 10 mesaje ale clientului de lucru și **al 11-lea, `PROMOTE_CTX` cu adresele virtuale ale bufferelor MAIN/PM/PATCH, a eșuat**.
Cum: **nu se știe** — eroarea poate fi (a) un refuz cu cod `NV_STATUS` (`eroare hardware 0x..`), (b) limita de 5 s (`timeout dupa 5000000 us`),
(c) un răspuns nepotrivit (`RM_CONTROL: raspunsul e pentru alta comanda` / `de alta lungime`). Marcajele de timp de pe stick **nu ajută**:
sunt ora scrierii pe stick (~13 ms pe rând, în serie), nu ora evenimentului — așteptarea de 11 s a lui I8 apare între două rânduri la 13 ms distanță.

**De ce nu e motivul în jurnal:** `chain::run_chain_ready` scria doar `Event::Step { ok: false }`; textul erorii mergea în `ChainReport` (ecranul) și în
`lines` din OS (ecranul). Ecranul a înghețat la I8 (vezi §4), deci motivul s-a pierdut. **Reparat în `45dfc77`** (§3).

## 3. Ce s-a schimbat în driver (sesiunea 93, `45dfc77`; doar adăugări)

1. `gpu_core::diag::Journal::push_text(what, more, text)` — un text oricât de lung, în bucăți de 32 de octeți.
2. `chain::run_chain_ready`: după `pas X: ESEC` scrie `text motivul esecului = "…"` (+ `motivul esecului (urmare)` dacă e lung) — și pentru pașii opriți de
   o poartă (blocat în plan, mod doar-citire). **OS-ul nu are nimic de schimbat**: leagă `45dfc77` sau mai nou și rândul apare pe stick.
3. `gsp::golden::QueueTransportBellKeep` — ca `QueueTransportBell`, dar păstrează **întregi** mesajele GSP-RM sărite în așteptarea răspunsului (`skipped`,
   cel mult 64), și `gsp::golden::journal_skipped(j, "I9", &skipped)` le scrie în jurnal (NOCAT decodat ca la I8, LibOS print, altele numite). Opțional
   pentru OS (o schimbare în `OsRm::call`; vezi `integration/cyber-universe/os-295_i9-motiv-in-jurnal.md`). `QueueTransportBell` rămâne neschimbat.
4. `variant_log` / `compara_variante`: citește motivul și îl arată („motivul la i9-canal-lucru: …”, sau „fara motiv in jurnal” pentru jurnalele vechi).

Teste: T124 a–c (SIMULARE), 455 PASS.

## 4. Ecranul înghețat (OBSERVAT + ipoteză)

Adrian a văzut ultimul rând `SCRATCH_14 = 0x17100000` — rândul 2274 din jurnal, **starea după predarea etapei 3 a lui CORE_RESUME, exact când RISC-V-ul GSP
devine activ** (2278) și GSP-RM începe să ruleze. Jurnalul a continuat pe stick încă ~9 s de evenimente (I8 ok, golden ok, I9) și s-a închis curat.
Singurul NOCAT de tip ENGINE spune „Display Subsystem / Display Engine / `NV_ERR_INVALID_DMA_SPECIFIER`”.

Ipoteza (DEDUS, neverificată): la inițializare GSP-RM preia motorul de afișare (încearcă să-l configureze, se plânge de un descriptor DMA al afișajului) și
framebuffer-ul UEFI nu mai e scanat → ecranul rămâne pe ultimul cadru. **Nu e o blocare a calculatorului**: OS-ul a mers mai departe, a scris jurnalul, a
oprit Bus Master și a făcut turul de citire.

Ce putem face noi (cyber-gpu): nimic sigur deocamdată — nu avem nicio cheie de registru GSP sau setare publică, cu sursă, care să țină GSP-RM departe de
afișaj (în `ip.py` al tinygrad, citit azi ca specificație, nu există așa ceva; nici în ce am citit din nouveau/nova). Nu inventăm valori. Ce poate face OS-ul (domeniul sesiunii OS,
propunere): să considere că după pornirea GSP-RM ecranul e pierdut și să nu mai aștepte confirmări vizuale — jurnalul de pe stick + `trimite` la pornirea
următoare sunt raportul; eventual portul serial, dacă placa de bază îl are.

## 5. Următorul pas

1. **Un singur test pe placă**, când vine Adrian: același stick cu un Release OS care leagă `cyber-gpu` `45dfc77` (sau mai nou), `gpu lant --varianta 1`
   → jurnalul va avea `text motivul esecului = "…"` după `pas i9-canal-lucru: ESEC` (și, dacă OS-ul adoptă §3.3, mesajele GSP-RM sărite în I9).
2. Cu motivul în mână: dacă e un cod `NV_STATUS` → `nvidia_rm570::names` îl numește; se compară cu ce acceptă GSP-RM pentru `PROMOTE_CTX` (tinygrad face
   exact aceleași două jumătăți, `ip.py promote_ctx`: fizic `bInitialize=1, bNonmapped=1, physAttr=0x4`, apoi virtual `gpuVirtAddr`, restul 0 — codul nostru
   e la fel; diferențele posibile sunt în **ce** promovăm: mărimile/alinierea bufferelor, spațiul de adrese al canalului prin context share, `engineType`).
   Dacă e timeout → GSP-RM s-a blocat pe promovarea virtuală (atunci intră în discuție tabelele de pagini: adresele virtuale alese, PD-ul cu `SET_PAGE_DIRECTORY`).
3. Nimic altceva pe driver până atunci (regula: fără jurnal nou, fără lucru inventat).
