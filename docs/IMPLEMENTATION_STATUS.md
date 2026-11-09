# Starea implementării — cyber-gpu

## Sesiunea 75 — 9 noiembrie 2026, ~21:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă (testul os-276 mutat mâine 07:15)

Pornită de programare. Nicio comandă nouă pentru cyber-gpu (C-94 e pentru Kronos): Adrian nu vine diseară, **testul `os-276` e mâine, 10 nov, la 07:15**.
Niciun jurnal nou de pe placă. Codul neschimbat față de `bbbdb2f` (cel din `os-276`). **Nimic testat pe RTX 3070 Ti.** CI GitHub verde pe `8f2af33`.
Verificare locală (`sh tools/check.sh`; ținta `x86_64-unknown-none` reinstalată): format, clippy, **417 teste PASS**, build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă; reparația C-91 așteaptă proba de mâine |
| CI GitHub | PASS (pe `8f2af33`) |

**Următorul pas:** după testul de mâine 07:15: `verifica_jurnal_lant` pe jurnalul din `rezultate/`; dacă I8 ajunge la `GSP_INIT_DONE`, urmărim primul pas care se oprește după el.

---
## Sesiunea 74 — 9 noiembrie 2026, ~13:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă (os-276 așteaptă testul)

Pornită de programare. Nicio comandă nouă pentru cyber-gpu (C-92/C-93 sunt pentru Kronos); C-91 e gata la ambele proiecte: OS-ul e fixat pe `bbbdb2f`
și a scos Release-ul **`os-276`**. Testul pe placă e **diseară la 20:00 sau mâine la 08:30** (C-93) — niciun jurnal nou de pe placă.
**Nimic testat pe RTX 3070 Ti.** CI GitHub verde pe `7f162a5`.
Verificare locală (`sh tools/check.sh`, Rust 1.97.0; ținta `x86_64-unknown-none` reinstalată): format, clippy, **417 teste PASS** (4 ignorate), build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă; reparația C-91 așteaptă proba pe placă |
| CI GitHub | PASS (pe `7f162a5`) |

**Următorul pas:** după testul `os-276`: `verifica_jurnal_lant` pe jurnalul din `rezultate/`; dacă I8 ajunge la `GSP_INIT_DONE`, urmărim primul pas care se oprește după el.

---
## Sesiunea 73 — 9 noiembrie 2026, ~08:10–08:30 (ora României), în cloud, FĂRĂ placă — C-91 URGENT: I8 oprit de secvențiator

Pornită de programare. Testul de azi (os-275) pe workstation: I6 + I7 TREC; **i8-gsp-init ESUAT** „secventiator: marime sau index in afara bufferului”.
Cauza (găsită din cod; aceeași și în diagnosticul sesiunii OS): `decode_sequencer` lua `bufferSizeDWord` (capacitatea bufferului GSP) drept numărul de
comenzi; GSP trimite doar `cmdIndex` cuvinte, deci pe placă capacitatea > cuvintele primite → refuz. Simularea punea capacitatea = cuvintele trimise.
**Vestea bună:** mesajul a trecut de semnătură, versiune, lungime, sumă de control și numărul de secvență — adică primul mesaj real al GSP-RM a fost citit corect.
Jurnalul complet (`gpu-jurnal-003.txt`, urcat la 08:30) confirmă: în I8 singura scriere a fost soneria GSP (`0x110c00`), apoi ~1,8 s mai târziu oprirea, fără nicio comandă de secvențiator executată — GSP-RM a răspuns, noi am refuzat mesajul. Valorile `bufferSizeDWord`/`cmdIndex` nu sunt scrise în jurnal (doar textul erorii).

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Reparația | comenzile = `0..cmdIndex`; refuz la capacitate 0, `cmdIndex` > capacitate sau > cuvinte primite | PASS (SIMULARE) |
| Teste | test nou cu forma de pe placă (capacitate 0x4000, mesaj scurt); testul de pornire până la `GSP_INIT_DONE` folosește aceeași formă | PASS (SIMULARE), 417 PASS, 4 ignorate |
| Aceeași greșeală în I8–I14 | citit: `rpc::decode` (lungime/elemente — a trecut pe placă), răspunsurile RM (`device info`, `GR info`, `PROMOTE_CTX`: număr de intrări ≤ maximul din antet, lungime = structura trimisă), `RegStore` (index < 8). Niciun alt câmp „capacitate citită ca lungime” | PASS (citire de cod) |
| Pe RTX 3070 Ti | — | NOT TESTED |

**Următorul pas:** OS-ul (P6-68) leagă noul commit și scoate Release-ul; la reîncercare se urmărește dacă I8 ajunge la `GSP_INIT_DONE`. Nota: `integration/cyber-universe/C-91_secventiator.md`.

---
## Sesiunea 72 — 9 noiembrie 2026, ~07:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă (fereastra de luni începe ~08:50)

Pornită de programare. Nicio comandă nouă pentru cyber-gpu (C-90 e pentru Kronos: nimic pe placă cât Adrian e la workstation), niciun jurnal nou de pe placă
(fereastra n-a început). **Nimic testat pe RTX 3070 Ti.** CI GitHub verde pe `d2223d0`.
Verificare locală (`sh tools/check.sh`, Rust 1.97.0): format, clippy, **416 teste PASS** (4 ignorate), build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă; lista `docs/FEREASTRA_C-83.md` rămâne valabilă |
| CI GitHub | PASS (pe `d2223d0`) |

**Următorul pas:** după fereastră: `verifica_jurnal_lant` pe jurnalul din `rezultate/` + rândul `kronos --gpu:`.

---
## Sesiunea 71 — 9 noiembrie 2026, ~00:00 (ora României), în cloud, FĂRĂ placă — lista de luni potrivită cu lista OS

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă. Lista OS pentru luni (P6-63, Release `os-275`) e scrisă
(`stare/cyber-universe.md` 20:50) → recitită și potrivită cu `docs/FEREASTRA_C-83.md`. **Nimic testat pe RTX 3070 Ti.** CI GitHub verde pe `f6d9f57`.
Verificare locală (`sh tools/check.sh`, Rust 1.97.0; ținta `x86_64-unknown-none` reinstalată în sesiunea nouă): format, clippy, **416 teste PASS** (4 ignorate), build no_std PASS.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Lista GPU = lista OS | aceiași pași și aceeași ordine; potrivite: 4 fișiere firmware (și `booter_unload`), #513 pe al doilea stick (`/usb2/kronos/`), pornire rece separată înainte de `kronos --gpu` | scris (`docs/FEREASTRA_C-83.md`) |
| Jurnalul lui `kronos --gpu` | citit în OS (`gpu_i14.rs`): #513 rulează în pasul `i14-simbol` cu aceleași accese ca modelul de control (fereastra VRAM + soneria de lucru, fără apeluri RM) → `verifica_jurnal_lant` îl judecă corect; un eșec #513 nu face I14 „ESEC”, deci se citește și rândul `kronos --gpu:` | PASS (citire de cod, nu rulare) |
| Cod | neschimbat (`34a9793` în OS rămâne bun) | — |

**Următorul pas:** luni, după fereastră: `verifica_jurnal_lant` pe jurnalul din `rezultate/` + rândul `kronos --gpu:`.

---
## Sesiunea 70 — 8 noiembrie 2026, ~19:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă; lista OS pentru luni (P6-63) încă nu e scrisă
(`stare/cyber-universe.md` 17:50: „P6-63 diseară”), deci lista finală GPU tot așteaptă. **Nimic testat pe RTX 3070 Ti.**
**CI GitHub: din nou VERDE** — rularea pe `bf2a00d` (repornită la 15:08) a trecut; blocajul de ieri (joburi fără mașină) a dispărut.
Verificat (doar citire) în `cyber-universe` `c86281c`: OS-ul e fixat pe cyber-gpu `34a9793` și rulează `WINDOW_CHAIN_C83` — aceiași 17 pași pe care îi
știe `verifica_jurnal_lant` (sesiunea 68); `docs/FEREASTRA_C-83.md` rămâne valabil, nimic de schimbat.
Verificare locală (`sh tools/check.sh`, Rust 1.97.0): format, clippy, **416 teste PASS** (4 ignorate), build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă și fără lista OS |
| CI GitHub | PASS (pe `bf2a00d`) |
| Lanțul din OS = lanțul verificat de unealtă | PASS (citire de cod, nu rulare) |

**Următorul pas:** după P6-63 al OS-ului — lista finală în hub (`stare/cyber-gpu.md`); luni, după fereastră: `verifica_jurnal_lant` pe jurnal.

---
## Sesiunea 69 — 8 noiembrie 2026, ~13:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă; lista OS pentru luni (P6-63) se face abia diseară,
deci lista finală GPU așteaptă. **Nimic testat pe RTX 3070 Ti.**
CI GitHub: rularea pe `c186f79` repornită acum → pică iar în 3 s, fără niciun pas rulat și fără jurnal (joburile nu primesc mașină: `runner_id 0`).
E o problemă a contului/GitHub, nu a codului (codul e același care a trecut pe `25566e5`). Motivul exact se vede doar pe pagina Actions a repo-ului.
Verificare locală (`sh tools/check.sh`, Rust 1.97.0): format, clippy, **416 teste PASS** (4 ignorate), build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă și fără lista OS |
| CI GitHub | BLOCKED (joburile nu pornesc; local PASS) |

**Următorul pas:** diseară, după P6-63 al OS-ului — lista finală în hub (`stare/cyber-gpu.md`); luni, după fereastră: `verifica_jurnal_lant` pe jurnal.

---
## Sesiunea 68 — 8 noiembrie 2026, ~07:30–08:00 (ora României), în cloud, FĂRĂ placă — C-83: verificarea jurnalului pentru fereastra de luni

Pornită de programare. Nicio comandă nouă, niciun jurnal nou de pe placă. **Nimic testat pe RTX 3070 Ti.**
Golul găsit: unealta `verifica_jurnal_lant` știa doar lanțul până la I10 (13 pași); luni OS-ul rulează lanțul C-83 (17 pași, cu I11–I14) → jurnalul
de luni ar fi ieșit „DE ANALIZAT: pas necunoscut” fără să fie ceva greșit. CI: rularea pe `0e86d1a` picase în 2 s fără jurnal (mașina GitHub, nu codul); repornită → la fel; și rularea pe `84d3685` pică la fel (joburile nu pornesc). Testele locale trec.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Verificarea jurnalului C-83 | `chain_log::check_chain_journal_for(lant, …)`, `ChainJournal::verdict_for(lant)`, `chain_for_journal(text)` (alege C-83 dacă jurnalul are pași I11–I14), `step_ids_for`; mesajul pentru o scriere străină numește pasul (nu mai spune „I10” la I11–I14) | PASS (T107, SIMULARE — text) |
| Unealta | `verifica_jurnal_lant` tipărește „lantul: C-83 (I6 -> I14), 17 pasi” și judecă după lanțul ales; pe jurnalul real din 5 nov dă același verdict ca înainte | PASS |
| Lista pentru luni (partea GPU) | `docs/FEREASTRA_C-83.md`: cei 17 pași, scrierile noi la fiecare, ce înseamnă „trece”, unealta de după | scris |
| Compatibilitate OS | doar adăugări (`check_chain_journal`, `verdict`, `step_ids` neschimbate) | PASS |

Verificări (`sh tools/check.sh`, Rust 1.97.0): format, clippy `-D warnings`, **416 teste PASS** (4 ignorate), build no_std PASS.

**Următorul pas:** duminică seara — lista finală în hub (`stare/cyber-gpu.md`), după ce OS-ul o scrie pe a lui (P6-63); luni, după fereastră: `verifica_jurnal_lant` pe jurnal.

---
## Sesiunea 67 — 8 noiembrie 2026, ~01:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Pornită de programare. Nicio comandă nouă pentru cyber-gpu (ultima: C-88 la Kronos), niciun jurnal nou de pe placă, nicio întrebare nouă de la OS
(`stare/cyber-universe.md` 23:50: „cyber-gpu: nimic urgent”; zona de 32 KiB pentru 24 de straturi o fac ei când vine modelul). CI verde pe `25566e5`. **Nimic testat pe RTX 3070 Ti.**
Verificare locală (`sh tools/check.sh`, Rust 1.97.0): **415 teste PASS** (4 ignorate), clippy, format, build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă |

**Următorul pas:** duminică seara (8 nov) lista exactă pentru fereastra de luni (C-83/C-87).

---
## Sesiunea 66 — 7 noiembrie 2026, ~19:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Pornită de programare. C-87 (partea GPU) era gata din sesiunea 65; nicio comandă nouă pentru cyber-gpu (ultima nouă: C-88 la Kronos), niciun jurnal nou de pe placă,
nicio întrebare nouă de la OS (`stare/cyber-universe.md`: „nimic nou de cerut acum”). CI verde pe `ed75615`. **Nimic testat pe RTX 3070 Ti.**
Verificare locală (`sh tools/check.sh`, Rust 1.97.0): **415 teste PASS** (4 ignorate), clippy, format, build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă |

**Următorul pas:** duminică seara (8 nov) lista exactă pentru fereastra de luni (C-83/C-87).

---
## Sesiunea 65 — 7 noiembrie 2026, ~18:00–18:30 (ora României), în cloud, FĂRĂ placă — C-87 pasul 2: un singur program pe simbol

Pornită manual (aceeași programare), cu C-87 ca sarcină. Pașii 1, 3, 4 erau gata din sesiunea 64. **Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Unealta NAK | Mesa 25.0.7 reconstruit în cloud; `add64` recompilat identic (`a890b75c…`) | PASS |
| Programul `token.comp` | toată lista unui simbol într-o lansare: G grupuri × 32 fire, barieră pe toată placa între operații; NAK: 43 232 B, 35 registre, 0 memorie locală | COMPILAT, NU RULAT |
| Partea driverului | `kronos_token`: lista de lucru (256 B pe operație = bufferul constant al lansării separate + operația + grila), bufferul constant, QMD-ul, `groups_for` (SM-urile din GR info) | PASS (teste de unitate) |
| Comparația cu lansările separate | `gpu-sim::token_on_cpu`; 12 poziții (E 64) și 2 poziții la E 768: logit-uri **bit cu bit** egale cu cele 35 de lansări; generare cu verificarea procesorului; listă stricată refuzată | PASS (SIMULARE — nu arată bariera pe placă) |
| Compatibilitate OS | doar adăugări | PASS |

Verificări (`sh tools/check.sh`): **415 teste PASS** (4 ignorate), clippy, format, build no_std.

**C-87 (partea GPU) e gata în simulare.** Următorul pas: duminică seara (8 nov) lista exactă pentru fereastra de luni; „un singur program” abia după
ce lansările separate trec pe placă.

---
## Sesiunea 64 — 7 noiembrie 2026, ~17:45–18:15 (ora României), în cloud, FĂRĂ placă — C-87: generare continuă, recuperare, modelul mare

Pornită de programare. **Comandă nouă: C-87** (Adrian, prin centrul de comandă: „mai mulți pași în simulare până luni”). **Nimic testat pe RTX 3070 Ti.**

| Pas C-87 | Ce s-a făcut | Rezultat |
|---|---|---|
| 1. Generare continuă | `kronos_gen::Generation` (+ `GenPlan`, `TokenRunner`, `TokenCheck`, `Stop`, `GenStats`): N simboluri la rând, o trimitere `submit_fenced` pe poziție, cache-urile cresc; **oprire la prima eroare**, nimic trimis după; viteza doar din marcajele de timp ale plăcii (cu și fără pauzele driverului) | PASS (SIMULARE): 40 de simboluri ca procesorul (logit-uri ≤ 3,2e-7); defect plantat la poziția 5 → oprire acolo, 6 trimiteri, 3 simboluri păstrate; context plin → oprire curată |
| 2. Un singur program pe simbol | neînceput: cere un program NAK nou („megakernel” cu barieră pe toată placa) + Mesa reconstruit în cloud | NOT TESTED — rămâne următorul pas |
| 3. Recuperare fără repornire | `recovery::recover` (ping GSP doar citire → `user_space::stop_compute_group` prin RM, niciun registru scris de noi → GPGet/GPPut → **Bus Master oprit mereu**); paginile RAM ale plăcii eliberate doar dacă e dovedit că nimic nu le mai poate scrie, altfel carantină până la repornire; `classify` + `transfer::records_seen` pentru jurnal | PASS (SIMULARE): cu `FakeRm` + configurare simulată, GSP viu → RAM eliberat; GSP tăcut → carantină, Bus Master tot oprit |
| 4. Modelul mare (24 straturi, 183M) | `kronos_gen::VramBudget` (calculat înainte de urcare, verificat față de `GpuMemory::vram_free()`), `push_bytes_with_copy` | PASS: 24 straturi E 128 pe canal (387 lansări/poziție) ca procesorul (SIMULARE); bugetul 183M ≈ 270 MiB, #513 ≈ 142 MiB (calcul); **pushbuffer 32 KiB pentru 24 de straturi** (16 KiB nu mai ajung) |
| Compatibilitate OS | doar adăugări (`GpuMemory::vram_free`, `RangeAllocator::largest_free`, module noi, `stop_compute_group`, `records_seen`); `FakeRm` acceptă și `bEnable = 0` | PASS |

Verificări (`sh tools/check.sh`, Rust 1.97.0): format, clippy `-D warnings`, **410 teste PASS** (4 ignorate), build no_std PASS; local cu vectorii KGR1: 3/3 PASS.
Nota pentru OS: `integration/cyber-universe/C-87_generare.md`.

**Următorul pas:** C-87 pasul 2 („un singur program pe simbol”: programul NAK persistent + modelul lui în `gpu-sim`, comparat cu cele 195 de lansări);
duminică seara (8 nov) lista exactă pentru fereastra de luni.

---
## Sesiunea 63 — 7 noiembrie 2026, ~12:00 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Pornită de programare. Nicio comandă nouă pentru cyber-gpu (ultimele: C-84 la OS, C-85/C-86 la Kronos), niciun jurnal nou de pe placă,
nicio întrebare nouă de la OS (`stare/cyber-universe.md`: „nimic nou de cerut acum”, copia `d706cbb` e în OS). CI verde. **Nimic testat pe RTX 3070 Ti.**
Verificare locală (`sh tools/check.sh`, Rust 1.97.0): **394 teste PASS** (4 ignorate), build no_std PASS.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă |

**Următorul pas (C-83):** duminică seara (8 nov) lista exactă pentru fereastra de luni; „un singur program pe simbol” după măsurătoarea de pe placă.

---
## Sesiunea 62 — 7 noiembrie 2026, ~07:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă, nicio întrebare nouă de la OS
(`stare/cyber-universe.md`: „Nimic nou de cerut acum”; P6-60 și P6-62 gata la ei, fixați pe `d706cbb`). CI verde pe `d706cbb`. **Nimic testat pe RTX 3070 Ti.**
Observație: OS-ul a reparat o depășire de stivă la `kronos --gpu` (lanțul nostru rula pe 16 KiB; acum 256 KiB, folosește ~16 KiB) — reparată în OS, nimic de schimbat la noi.

| Pas | Rezultat |
|---|---|
| Lucru nou | NOT TESTED — nu există pas nou fără placă |

**Următorul pas (C-83):** duminică seara (8 nov) lista exactă pentru fereastra de luni; „un singur program pe simbol” după măsurătoarea de pe placă.

---
## Sesiunea 61 — 7 noiembrie 2026, ~01:30 (ora României), în cloud, FĂRĂ placă — C-83: răspuns la întrebarea OS despre zona de pushbuffer

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă. **Nimic testat pe RTX 3070 Ti.**
Golul găsit: în `stare/cyber-universe.md` OS-ul întreabă dacă o zonă de pushbuffer de 16 KiB pentru #513 e bună.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Mărimea pe simbol | `kronos_model::push_bytes_per_token(layers)` (adăugare): 7 832 B pentru 12 straturi | PASS |
| Zona de 16 KiB | testul `a_16_kib_area_carries_513_tokens`: 4 KiB nu încape, 8 KiB un simbol, 16 KiB două în zbor; 512 simboluri la rând fără să atingă o bucată în zbor; al treilea refuzat | PASS (SIMULARE, aritmetica zonei) |
| Pentru OS | răspuns în `integration/cyber-universe/C-83_i14.md`: 16 KiB e bine | propus |
| Compatibilitate OS | doar adăugare | PASS |

Verificări (`sh tools/check.sh`, Rust 1.99.1): format, clippy `-D warnings`, **394 teste PASS** (4 ignorate), build no_std PASS.

**Următorul pas (C-83):** neschimbat — duminică seara (8 nov) lista exactă pentru luni; „un singur program pe simbol” după măsurătoarea de pe placă.
Fără comandă nouă, jurnal nou sau întrebare nouă, sesiunile următoare: oprire scurtă.

---
## Sesiunea 60 — 6 noiembrie 2026, ~20:05 (ora României), în cloud, FĂRĂ placă — C-34: copia din hub reîmprospătată pentru OS

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă; CI verde pe `0974ecd`. **Nimic testat pe RTX 3070 Ti.**
Golul găsit: sesiunile 57–59 n-au reîmprospătat copia din hub (C-34), iar sesiunea OS (P6-62, partea ei din C-83) aștepta `e681bea` în copie.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Copia din hub | `git bundle create` pe `g0-g2-fundatie` (clona locală era superficială → `git fetch --unshallow` întâi, altfel copia ieșea incompletă), `git bundle verify` OK | PASS |
| Clonare de probă | clonă din copie + `sh tools/check.sh` pe ea (Rust 1.99.1): **393 teste PASS** (4 ignorate), clippy, format, build no_std | PASS |
| Firmware | în istoric doar cod/documente cu „fwsec” în nume, niciun fișier binar de firmware | PASS |
| Compatibilitate OS | nicio schimbare de cod | PASS |

**Următorul pas (C-83):** neschimbat — duminică seara lista exactă pentru luni; „un singur program pe simbol” după măsurătoarea de pe placă.
**Regulă pentru sesiunile următoare:** după orice push, copia din hub (C-34) se reîmprospătează în aceeași sesiune; clona trebuie să fie completă (nu superficială).

---
## Sesiunea 59 — 6 noiembrie 2026, ~19:30–20:00 (ora României), în cloud, FĂRĂ placă — C-83 I14: vectorii de control KGR1 ai lui Kronos

Pornită de programare. Nicio comandă nouă pentru cyber-gpu, niciun jurnal nou de pe placă; CI verde pe `e681bea`. **Nimic testat pe RTX 3070 Ti.**
Golul găsit: I14 fusese verificat doar pe un model sintetic, comparat cu oracolul nostru; oracolul însuși nu fusese legat de motorul de referință al
lui Kronos.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Oracolul față de Kronos | vectorii de control KGR1 din hub (`integrare/kronos/vectori`, model minuscul cu greutăți la întâmplare, 2 straturi, 4 capete, n_emb 32, 16 384 simboluri, 3 texte) prin `gpu_compute::model::step_cpu`: argmax pe fiecare poziție și 5 simboluri greedy identice; logit-uri ≤ 4,6e-6 (pragul hub-ului 1e-3) | PASS (procesor, local): T106 |
| Același model pe canalul de lucru | `kronos_model::token_launches` (35 de lansări pe simbol) pe `gpu-sim`: aceleași simboluri, logit-uri 2,4e-6 de motorul lui Kronos | PASS (SIMULARE, local): T106 |
| Testul poate pica | o scară din rândul de încorporare schimbată cu 1 % → logit-uri peste 1e-3 | PASS (local): T106 |
| Date private | hub-ul e privat, repo-ul public: vectorii NU intră în repo; `tools/vectori-kgr1/pregateste.py` îi pregătește local (`target/`), testele îi citesc din `KGR1_VECTORI`, sunt `ignored` fără ei și PICĂ dacă sunt cerute fără fișiere | — |
| Pentru OS | `integration/cyber-universe/C-83_i14_vectori.md`: opțional, la fereastră, modelul minuscul înaintea lui #513 | propus |
| Compatibilitate OS | nicio schimbare în crate-uri (doar teste + o unealtă de gazdă) | PASS |

Verificări (`sh tools/check.sh`, Rust 1.99.1): format, clippy `-D warnings`, **393 teste PASS** (4 ignorate: cele 3 noi fără vectori + 1 vechi), build
no_std PASS. Cu vectorii (`--include-ignored`): cele 6 teste I14 PASS. **În CI testele noi nu rulează** (n-au vectorii).

**Următorul pas (C-83):** neschimbat — duminică seara lista exactă pentru luni; „un singur program pe simbol” după măsurătoarea de pe placă.
Fără comandă nouă sau jurnal nou, sesiunile următoare: oprire scurtă.

---
## Sesiunea 58 — 6 noiembrie 2026, ~18:15–19:30 (ora României), în cloud, FĂRĂ placă — C-83 pașii 3 și 4: I13 și I14 în SIMULARE

Pornită manual cu amintirea comenzii C-83. Niciun jurnal nou de pe placă, nicio comandă nouă pentru cyber-gpu (C-85 e pentru kronos).
**Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Programele | 4 programe noi compilate cu NAK: `vadd`, `layernorm`, `attention`, `gelu` (fișele și sumele în `tools/nak/README.md`) | COMPILAT |
| Referința | `gpu_compute::layer::step_cpu` (stratul în ordinea motorului procesorului, o poziție nouă + cache-uri) și modelele celor 4 programe; toleranța 1e-4 + 1e-4·\|x\| | PASS: T100 |
| Stratul pe canal | `kronos_layer::token_launches` (16 lansări pe poziție), `Launch::qmd`, `LayerVas`, `test_layer`, `judge`; 768/12 pe 2 poziții și 64/2 pe 6: în toleranță (768: < 1e-5) | PASS (SIMULARE): T101, T102 |
| Lanțul | `i13-strat` în `WINDOW_CHAIN_C83`, I13 în `GA104_PLAN` și `READY_FOR_WINDOW_C83`; `chain_log` îi permite doar soneria + fereastra VRAM | PASS (SIMULARE): T103 |
| I14 — un simbol întreg | programul `embed` (NAK); `gpu_compute::model` (simbolul în ordinea motorului, argmax, toleranța logit-urilor 1e-3); `kronos_model::token_launches` (`16·L + 3` lansări), `judge`; generare greedy pe un model sintetic mic: aceleași simboluri ca procesorul | PASS (SIMULARE): T104, T105 |
| Pushbuffer | `WorkQueue::submit_fenced`: o secvență lungă pe simbol eliberează zona după ultima înregistrare | PASS (SIMULARE): T104 |
| Lanțul | `i14-simbol` în `WINDOW_CHAIN_C83`; I14 în `GA104_PLAN` | PASS (SIMULARE) |
| Pentru OS | `integration/cyber-universe/C-83_i13.md`, `C-83_i14.md` | propus |
| Compatibilitate OS | doar adăugări | PASS |

Verificări (`sh tools/check.sh`, Rust 1.99.1): format, clippy `-D warnings`, **393 teste PASS** (1 ignorat), build no_std PASS; clippy cu `hw-write` și nightly PASS.

**Următorul pas (C-83):** „un singur program pe simbol” (idee aprobată 3 nov; `docs/TINTA_UN_SINGUR_PROGRAM.md`) — doar după ce măsurăm pe placă
costul celor 195 de lansări (I14 dă durata fiecăreia). Duminică seara: lista exactă pentru fereastra de luni. #513 însuși se încearcă doar pe placă (OS).

---
## Sesiunea 57 — 6 noiembrie 2026, ~17:45–19:00 (ora României), în cloud, FĂRĂ placă — C-83: I11 și I12 în SIMULARE

Comandă nouă de la Adrian în hub: **C-83** („până luni, cât mai mulți pași în simulare, și după I10”). Fereastra de azi (C-79) s-a închis fără
test (C-80), deci niciun jurnal nou. **Nimic testat pe RTX 3070 Ti.** Oprirea de siguranță nu se aplică (pași terminați în această sesiune).

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| I11 — RAM văzut de GPU | `mmu::encode_pte_sys` / `map_sys` / `alloc_sys_mapped` (aperture coerent, `ADDRESS_SYS`, `VOL`; pagini DMA împrăștiate); în `gpu-sim` un al doilea cititor `translate_any` + RAM simulat | PASS (SIMULARE): T93 |
| I11 — copierea mare | programul `copy_b128` (NAK, 288 B); `transfer`: cbuf, QMD, o intrare GPFIFO cu 3 mărci de timp, analiza vitezei; RAM → VRAM → RAM bit cu bit pe tot lanțul golden → canal de lucru | PASS (SIMULARE): T94, T95 |
| I12 — oracolul | `gpu_compute::q8`: ordinea motorului OS, ordinea programului (citită în fișa NAK: FMUL + 31 FFMA + FFMA + 5 FADD, toate FTZ), marja scrisă dinainte | PASS: T96 |
| I12 — pe canalul de lucru | `q8_matvec` (NAK, 8 048 B, 128 B partajată, fără barieră hardware); `q8_matvec::judge`; W/x în RAM → copiere sus → înmulțire → copiere jos, 3072 × 768: **identic** cu oracolul | PASS (SIMULARE): T97, T98 |
| Lanțul | `WINDOW_CHAIN_C83` + `READY_FOR_WINDOW_C83`; I11/I12 în `GA104_PLAN` (blocate până la placă); `chain_log` le permite doar soneria + fereastra VRAM | PASS (SIMULARE): T99 |
| CI | clippy 1.99 are o regulă nouă (`chunks_exact_to_as_chunks`): primul push a picat, reparat în `4ac1646` | PASS |
| Pentru OS | `integration/cyber-universe/C-83_i11-i12.md` (pașii, scrierile noi, ce așteptăm) | propus |
| Compatibilitate OS | doar adăugări; `nvidia-ga104` depinde acum normal de `gpu-compute` (era doar la teste) | PASS |

Verificări (`sh tools/check.sh`, Rust 1.99.1): format, clippy `-D warnings`, **380 teste PASS** (1 ignorat), build no_std PASS; clippy cu
`--features hw-write` PASS; clippy nightly PASS.

**Următorul pas (C-83):** I13 — un strat întreg al lui Kronos pe placă (LayerNorm, qkv, atenția cu softmax, proiecția, GELU erf, MLP), fiecare
program NAK cu oracolul lui ca la I12, comparat cu motorul KGR1 din OS. Apoi I14 (un simbol #513). Duminică seara: lista pentru fereastra de luni.

---
## Sesiunea 56 — 6 noiembrie 2026, ~13:30 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Nicio comandă nouă în hub pentru cyber-gpu (nimic `nou`/`preluat`), niciun jurnal nou de pe placă (ultimul: `rezultate/2026-11-05-workstation-os242`;
jurnalele ferestrei de azi, C-79 cu `os-250`/`os-251`, nu sunt încă în hub). Nicio observație nouă de la OS. CI verde pe `24c0304`. **Nimic testat pe RTX 3070 Ti.**
Niciun pas nou: fără placă nu mai e nimic cu plus cinstit (regula 8). Sesiuni consecutive fără pas: 2 (55, 56). Dacă și sesiunea 57 nu are pas,
sesiunea 58 scrie „OPRIT: necesită analiza lui Adrian”.

**Următorul pas:** jurnalele ferestrei din 6 nov în `rezultate/2026-11-06-workstation-os250` (sau `-os251`), apoi `verifica_jurnal_lant` pe ele.

---
## Sesiunea 55 — 6 noiembrie 2026, ~09:00 (ora României), în cloud, FĂRĂ placă — oprire scurtă

Nicio comandă nouă în hub pentru cyber-gpu (nimic `nou`/`preluat`), niciun jurnal nou de pe placă (ultimul: `rezultate/2026-11-05-workstation-os242`),
nicio observație nouă de la OS (`os-251` folosește deja `wpr_meta_mem`). CI verde pe `657e209`. **Nimic testat pe RTX 3070 Ti.**
Niciun pas nou: fără placă nu mai e nimic cu plus cinstit (regula 8). Sesiuni consecutive fără pas: 1 (oprirea de siguranță nu se aplică).

**Următorul pas:** fereastra pe workstation cu `os-251` (`gpu lant` + DA), apoi `verifica_jurnal_lant` pe jurnalele din `rezultate/`.

---
## Sesiunea 54 — 6 noiembrie 2026, ~01:30 (ora României), în cloud, FĂRĂ placă — C-71 (lucru continuu): Booter-ul simulat lărgește WPR2 ca placa

Nicio comandă nouă în hub pentru cyber-gpu și niciun jurnal nou de pe placă după sesiunea 53. Sesiunea OS (`os-250`, folosește `free_after_i7`) a
lăsat o observație pentru noi: „modelul `FakeGsp` nu lărgește WPR2 după Booter (placa da)”. **Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Modelul Booter-ului | `gpu-sim`: câmp nou `FakeGsp::wpr_meta_mem` (memoria de sistem simulată cu `GspFwWprMeta`); Booter-ul simulat care reușește coboară WPR2 LO la `gspFwWprStart` din meta, HI rămâne cel de la I6 (OBSERVAT pe placă la 5 nov); magic greșit / început nepotrivit → refuz, cod SIM `0xbad2`. Implicit `None` = modelul vechi, testele vechi neschimbate | PASS (SIMULARE): T92 |
| Pe valorile plăcii | aranjamentul T91 + meta-ul dat de `wpr_meta::encode`: după I6 `0x1ffe00000..0x1ffee0000`, după I7 `0x1f4100000..0x1ffee0000` = exact placa; `free_after_i7` trece fără valori puse de mână | PASS (SIMULARE): T92 |
| Pentru OS | `integration/cyber-universe/C-71_wpr2-largit-in-sim.md` (opțional: meta-ul în model în loc de LO-ul scris de mână în `gpu lant --sim`) | propus |
| Compatibilitate OS | doar adăugare (un câmp public nou în `gpu-sim`); nimic redenumit sau șters | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **359 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** fără placă, nimic cu plus cinstit în cyber-gpu. Așteptăm fereastra pe workstation cu `os-250` (`gpu lant` + DA), apoi
`verifica_jurnal_lant` pe jurnalele din `rezultate/`. Sesiunile următoare fără comandă nouă sau jurnal nou: oprire scurtă.

---
## Sesiunea 53 — 5 noiembrie 2026, ~19:30 (ora României), în cloud, FĂRĂ placă — C-71 (lucru continuu): zona VRAM după I7

Nicio comandă nouă în hub pentru cyber-gpu și niciun jurnal nou de pe placă după sesiunea 52. Sesiunea OS a scos `os-249` (tot lanțul I6 → I10
în `gpu lant`) și a cerut în `stare/cyber-universe.md` „confirmarea zonei VRAM pentru golden/I9/I10 (peste 256 MiB, sub GSP-RM)”.
**Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Zona VRAM pe valorile reale | WPR2 după I7 de pe placă (5 nov) = începutul pe care îl dă `FbLayout` (cu `booter_load` real și `.fwimage` DEDUS din WPR2); zona liberă se termină la `0x1f4000000`, sub heap-ul non-WPR și sub WPR2 | PASS (valori reale): T91 |
| Verificare nouă, doar aritmetică | `FbLayout::free_after_i7(wpr2_citit)`: aceeași zonă ca `free_below()` doar dacă WPR2 recitit după I7 începe unde a cerut `GspFwWprMeta` și se termină în ultimul bloc FRTS; altfel eroare înainte de orice scriere în VRAM | PASS: T91 |
| Pentru OS | `integration/cyber-universe/C-71_zona-vram-dupa-i7.md`: zona OS e bună; propunere opțională: `free_after_i7` în loc de `free_below` (nicio scriere sau citire nouă) | propus |
| Compatibilitate OS | doar adăugare (`FbLayout::free_after_i7`); nimic redenumit sau șters | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **358 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** fără placă, nimic cu plus cinstit în cyber-gpu. Așteptăm fereastra pe workstation cu `os-249` (`gpu lant` + DA), apoi
`verifica_jurnal_lant` pe jurnalele din `rezultate/`. Dacă I8 trece pe placă: compararea zonei VRAM cu regiunile FB raportate de GSP-RM
(`GET_GSP_STATIC_INFO`). Sesiunile următoare fără comandă nouă sau jurnal nou: oprire scurtă.

---
## Sesiunea 52 — 5 noiembrie 2026, ~13:45 (ora României), în cloud, FĂRĂ placă — C-71 (lucru continuu): soneria canalului de lucru din antetele MIT

Nicio comandă nouă în hub pentru cyber-gpu și niciun jurnal nou de pe placă după sesiunea 51. Rămânea întrebarea 3 a sesiunii OS (I10): adresa
soneriei canalului de lucru `0xBB0090` era doar DEDUSĂ (nouveau/tinygrad). **Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Sursa adresei | antetul MIT 570.144 `tu102/dev_vm.h`: `NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090`; RM 570.144 (citit ca specificație): GA104 → `kfifoUpdateUsermodeDoorbell_GA100` → `_TU102` (RM pe procesor) → `GPU_VREG_WR32`, baza `0xB80000` (`gpuGetVirtRegPhysOffset_TU102`) | DOCUMENTAT (nu OBSERVAT) |
| Generatorul | `tools/gen-rm570/gen_regs.py`: prefix nou în `tu102_dev_vm`; `regs.rs` regenerat (verificat întâi că generatorul reproduce exact fișierul vechi) — 1 constantă nouă | PASS |
| Codul | `push::DOORBELL` = baza + `NV_VIRTUAL_FUNCTION_DOORBELL` din antet; verificare la compilare că e egal cu `DOORBELL_INFERRED` și `chain_log::WORK_DOORBELL` | PASS: T90 |
| Pentru OS | `integration/cyber-universe/C-72_soneria-i10.md`: răspunsul la întrebarea 3 (adresa, doar în pasul I10 al lanțului, citirile prin `VramWindow::read_u32`, `invalidate_tlb` înainte de prima sonerie) | propus |
| Compatibilitate OS | doar adăugare (`push::DOORBELL`); nimic redenumit sau șters | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **357 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** fără placă, nimic cu plus cinstit în cyber-gpu. Așteptăm: (1) sesiunea OS — I7 cu `booter_load_dma_libos` (nota C-75) și I10 (nota
C-72), Release nou; (2) fereastra următoare pe workstation (`gpu lant` + DA); (3) apoi `verifica_jurnal_lant` pe jurnalul din `rezultate/`.
Sesiunile următoare fără comandă nouă sau jurnal nou: oprire scurtă.

---
## Sesiunea 51 — 5 noiembrie 2026, ~13:00 (ora României), în cloud, FĂRĂ placă — C-75 (comanda nouă, jurnalul real): de ce I8 s-a oprit la cozile GSP

În hub a apărut jurnalul ferestrei de azi cu `os-246` (`rezultate/2026-11-05-workstation-os242/os246-gpu-jurnal-002.txt`): I6 și **I7 TREC pe placa
reală**, I8 se oprește („antetul cozii de stare nu are forma 570.144”). Jurnalul arată cauza: `bar0 w 0x110040: 0x00000000 <- 0x09c71000 => 0x00000000`
— cutia GSP **nu a păstrat** adresa LibOS, deci GSP-RM a pornit fără ea și nu și-a creat cozile. Cauza DEDUS: după FWSEC, GSP trebuie **resetat în
modul RISC-V** înainte de cutii (ordinea nouveau / RM 570.144) — exact întrebarea 1 a sesiunii OS. **Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Diagnosticul pe jurnalul real | `verifica_jurnal_lant` pe jurnalul os-246: `i7-booter-gsp` → „cutia GSP nu a pastrat adresa LibOS: GSP nu a fost resetat in RISC-V dupa FWSEC” | PASS (jurnal real): T88 |
| I7 în ordinea plăcii | `falcon::booter_load_dma_libos`: resetare GSP + BCR_CTRL = 0x111 (RISC-V) → cutiile LibOS recitite (oprire înainte de Booter dacă nu se păstrează) → Booter → `FALCON_OS` = 570144 → RISC-V activ | PASS (SIMULARE): T86, T87 |
| Simularea ca placa | `gpu-sim`: după FWSEC cutiile GSP ignoră scrierile până la resetarea GSP (OBSERVAT 5 nov); I8 simulat pornește doar dacă GSP-RM a primit adresa LibOS. Ordinea veche reproduce placa (I7 TRECE, I8 oprit), ordinea nouă trece tot lanțul | PASS (SIMULARE): T86 |
| Regula jurnalului | `i6_log::check_i7_libos` (5 scrieri GSP înainte, Booter ca `check_i7`, 1 scriere `FALCON_OS` după, fiecare recitită egal); `chain_log` o alege singur | PASS (SIMULARE + jurnal real): T86, T88 |
| Turul de după I7 | faptul nou `i6.wpr2.etapa` (`I6Check::wpr2_stage`): coborât / FRTS (după I6) / lărgit pentru GSP-RM (după I7); `i6.trece` (= „I6 se poate încerca”) neschimbat | PASS (valori reale): T89 |
| Răspunsuri C-75 | (1) `0xa0` la `0x111388` = bitul 7 ACTIVE_STAT = 1 → RISC-V **activ** (bitul 4 HALTED = 0); (2) cele ~2 s nu sunt cauza: GSP-RM rula, dar fără adresa LibOS nu avea unde să-și facă cozile; (3) formatul cozii nu se poate judeca până nu există coada — rămâne cel din 570.144 | analiză |
| Pentru OS | `integration/cyber-universe/C-75_i7-reset-gsp-libos.md`: un apel `booter_load_dma_libos` în locul scrierilor proprii la I7 (3 scrieri noi, toate în fereastra GSP) | propus |
| Compatibilitate OS | doar adăugări; câmp nou `ChainJournal::i7_libos`; `booter_load_dma` / `check_i7` neschimbate | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **356 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** sesiunea OS cheamă `booter_load_dma_libos` la I7 (Release nou) → fereastra următoare: `gpu lant`, așteptat rândul
`0x110040 … => 0x09c71000` (păstrat) și I8 TRECE; dacă cutia tot nu se păstrează, oprire curată înainte de Booter. Apoi `verifica_jurnal_lant`.

---
## Sesiunea 50 — 5 noiembrie 2026 (ora României), în cloud, FĂRĂ placă — C-71 (lucru continuu): fereastra VRAM în jurnalul lanțului

Nicio comandă nouă în hub (C-18 și C-20 sunt de mult `gata`; ultimele: C-66, C-69, C-71 — gata pentru noi). Pasul rămas din sesiunea 49
(„citirea înapoi a tabelelor de pagini din jurnal, dacă aduce un plus cinstit”) s-a redus la ce se vede cinstit în jurnal: cuvintele din VRAM
nu sunt în jurnal (driverul le recitește singur și oprește la nepotrivire), dar **mutările ferestrei VRAM** sunt. **Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Fereastra VRAM în jurnal | `chain_log::check_chain_journal`: fiecare scriere în `0x1700` (`NV_PBUS_BAR0_WINDOW`) recitită egal; la sfârșitul jurnalului fereastra pusă înapoi la valoarea dinaintea primei mutări (stare a plăcii, ca Bus Master) | PASS (SIMULARE): T85 |
| Jurnal real, nu inventat | jurnalul `VramWindow` (tabelele de pagini + invalidarea + punerea la loc, pe modelul PRAMIN din `gpu-sim`) pus în pasul I9 al jurnalului T82: lanțul tot TRECE; 2 greșeli plantate prinse (fereastra nepusă la loc, mutare recitită diferit) | PASS (SIMULARE): T85 |
| Pentru OS | `integration/cyber-universe/C-71_fereastra-vram.md` (informare: `restore()` și pe drumul de eroare, când vor exista I9/I10 în OS) | propus |
| Compatibilitate OS | doar adăugare: constanta `chain_log::VRAM_WINDOW` | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **352 teste PASS** (1 ignorat; T85 e în testul T82), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** fără placă, nu mai rămâne nimic cu plus cinstit în cyber-gpu: I6–I10 sunt în SIMULARE și în lanț, jurnalul se verifică pas cu pas.
Totul așteaptă: (1) partea OS din C-72 (Adrian a spus „da I7–I10” la 07:27; P6-56 în OS), (2) fereastra pe workstation (azi de la ~08:30),
(3) soneria canalului de lucru `0xBB0090` e DEDUSĂ: prima probă reală la I10, în lanț. Sesiunile următoare fără comandă nouă sau jurnal nou: oprire scurtă.

---
## Sesiunea 49 — 5 noiembrie 2026, ~01:50 (ora României), în cloud, FĂRĂ placă — C-71 (lucru continuu): verificarea jurnalului pentru tot lanțul

Pasul următor din sesiunea 48, făcut fără placă: jurnalul unei ferestre `gpu lant` complete se verifică **pas cu pas**, dintr-o comandă.
**Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Verificarea jurnalului lanțului | `chain_log::check_chain_journal`: jurnalul se împarte după liniile `pas <id>: ok/ESEC`; fiecare pas cu regula lui: doar-citire = 0 scrieri; Bus Master = 1 scriere în Command, pus exact la loc; I6/I7 = `check_i6`/`check_i7`; WPR2 după I6 și RISC-V activ după I7; **I8** = doar ferestrele GSP/SEC2, prima scriere = soneria; **golden + I9** = doar sonerii (valoarea 0) și cele 4 registre ale ferestrei VRAM, ≥ 1 sonerie, iar cu numărul de apeluri RM dat de OS: exact o sonerie pe apel; **I10** = fereastra VRAM + soneria canalului (0xbb0090, DEDUS, numărată). Verdict: TRECE / OPRIT CURAT la pas / DE ANALIZAT | PASS (SIMULARE): T82 |
| Simularea mai aproape de placă | lanțul T78 încarcă acum FWSEC și Booter prin DMA, ca pe placă (înainte, simulatorul scurta încărcarea): jurnalul are scrierile I6/I7 reale | PASS (SIMULARE): T83 |
| Greșeli plantate | 7 greșeli în textul jurnalului, toate prinse; lanțul fără sonerii → „OPRIT CURAT la i8e-golden” | PASS (SIMULARE): T82 |
| Unealta | exemplul `verifica_jurnal_lant` (ieșire 0 TRECE / 3 OPRIT CURAT / 2 DE ANALIZAT) | PASS: T84 (jurnal fără pași → DE ANALIZAT) |
| Pentru OS | `integration/cyber-universe/C-71_jurnal-lant.md`: ce trebuie să conțină jurnalul de pe stick (liniile `pas`, opțional numărul apelurilor RM) | propus |
| Compatibilitate OS | doar adăugări (modul nou `chain_log`); nimic redenumit sau șters | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **352 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** sesiunea OS leagă `gpu lant` de `WINDOW_CHAIN_FULL` și copiază liniile `pas` în jurnalul de pe stick; după fereastră:
`verifica_jurnal_lant`. Fără placă, pentru noi: verificarea citirii înapoi a tabelelor de pagini din jurnal (I9) — de văzut dacă aduce un plus
cinstit; altfel, așteptăm fereastra. Înainte de I10 pe placă: aprobarea soneriei canalului de lucru (0xbb0090, încă nu e pe lista permisă).

---
## Sesiunea 48 — 4 noiembrie 2026, ~23:35 (ora României), în cloud, FĂRĂ placă — C-71 (lucru continuu: I7 → I10 în lanț)

Decizia lui Adrian (23:13): pașii avansează în simulare fără să aștepte fiecare o fereastră; testele pe placă se fac în grup, cu `gpu lant`.
Comanda **C-71**: I7 → I8 → I9 → I10 în simulare, cu `gpu lant` pregătit să-i ruleze pe toți la fereastră. **Nimic testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Lanțul complet | `chain::WINDOW_CHAIN_FULL` (13 pași: pregătire → Bus Master → I6 → WPR2 citit → I7 → I7 citit → I8 → golden → I9 → I10 add64 → costul lansării → Bus Master la loc → tur), `run_chain_ready` + `READY_FOR_WINDOW`; `WINDOW_CHAIN` / `run_chain` neschimbate | PASS (SIMULARE): T80 |
| Lanțul rulat cap-coadă | toți cei 13 pași pe placa simulată, cu codul real: `FakeGsp` (I6/I7), cozile reale + `SimGsp` + `FakeRm` (I8, golden, I9), tabelele noastre în VRAM simulat, `add64` bit cu bit, costul lansării (mediana = ceasul simulat) | PASS (SIMULARE): T78 |
| Defect găsit în simulare | `QueueTransport` nu suna soneria GSP după fiecare mesaj RM (pe placă GSP-RM n-ar fi răspuns la golden image); nou `QueueTransportBell` (soneria după fiecare mesaj); GSP-ul simulat răspunde acum doar după sonerie | PASS (SIMULARE): T78; fără sonerie lanțul se oprește la golden (timeout), Bus Master pus la loc: T79 |
| Planul | textul de blocare I7 actualizat (scrie în SEC2, doar cu Adrian, după I6 pe placă) | — |
| Compatibilitate OS | doar adăugări; `QueueTransport` are același comportament | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **349 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.

**Următorul pas:** sesiunea OS leagă `gpu lant` de `WINDOW_CHAIN_FULL` (nota `integration/cyber-universe/C-71_gpu-lant-complet.md`); la fereastră,
un singur `gpu lant` + „DA” rulează tot până la primul eșec. Pentru noi, fără placă: verificarea jurnalului I8 + golden (doar ferestrele GSP/SEC2,
o sonerie pe mesaj) și extinderea `verifica_jurnal_i7` la tot lanțul.

---
## Sesiunea 47 — 4 noiembrie 2026, ~23:00 (ora României), în cloud, FĂRĂ placă — C-69 (comandă nouă: I7–I10 + ținta „un singur program”)

În hub: planul independenței OS aprobat de Adrian („da la tot”) → pentru noi **C-69** (F5: I6 TRECE → I7–I10 → primul calcul → Kronos pe placă).
I7–I10 aveau deja codul în SIMULARE (sesiunile 7–30); ce lipsea: verificarea I7 la fel de strictă ca I6, lista ferestrelor și primul pas al țintei
„un singur program”. **Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Lista exactă a scrierilor I7 | `I7_WRITE_SET`: aceleași 18 registre ca la I6, pe SEC2 (`0x840000–0x843FFF`); cutiile 0/1 = adresa `GspFwWprMeta` | PASS (SIMULARE): T72 |
| Jurnalul I6 + I7 | `i6_log::check_i7`, `check_i6_i7_journal` (verificarea comună cu I6, rescrisă o dată, `check_i6` neschimbat ca rezultat); 8 greșeli plantate prinse | PASS (SIMULARE): T73 |
| Rezultatul I7 fără permis | `falcon::read_after_i7` + `I7Status`: 3 citiri (cutiile SEC2, RISC-V GSP), jurnalizate, fără scriere — lecția C-66 | PASS (SIMULARE): T74 |
| Unealta de jurnal | exemplul `verifica_jurnal_i7` (TRECE / DE ANALIZAT) | PASS (rulat pe un jurnal gol: „DE ANALIZAT”, ieșire 2) |
| Măsurătoarea 1 „un singur program” | `launch_cost`: marcaj de start + N lansări + eliberare cu marcă de timp după fiecare; mediana/min/max; `gpu-sim` modelează marca de timp (ceas simulat) | PASS (SIMULARE): T75 — cifrele sunt ale testului, doar metoda e verificată |
| Lista ferestrelor | `docs/FEREASTRA_C-69.md`: fereastra 0 (I6) → 1 (I7) → 2 (I8) → 3 (golden + I9) → 4 (I10 + măsurătoarea 1) | doar plan |
| Pentru OS | `integration/cyber-universe/C-69_i7-fereastra.md` (propunere, doar după I6 TRECE și comandă nouă) | propus |
| Compatibilitate OS | doar adăugări; `gpu-sim` acceptă acum eliberarea cu marcă de timp (înainte o refuza) | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **346 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pași terminați în această sesiune).

**Următorul pas:** fereastra 0 (I6 o dată, os-224) → `verifica_jurnal_i6`; între timp, fără placă: lista exactă a scrierilor I8 (soneria +
secvențiatorul) și verificarea ei în jurnal, ca la I6/I7.

---
## Sesiunea 46 — 4 noiembrie 2026, ~19:40 (ora României), în cloud, FĂRĂ placă — C-66 (comandă nouă)

În hub: rezultatul de pe placă `rezultate/2026-11-04-workstation-os222/` și comanda **C-66**. Pe placă (os-222), I6 a rulat FWSEC-FRTS și a dat
„NU TRECE … la 0x1fa828: asteptat 0x1fff00, citit 0x1ffee0”: LO = exact începutul FRTS, HI = începutul ultimului bloc de **128 KiB** al zonei →
FWSEC-FRTS a reușit pe placă, regula noastră nu cunoștea această formă. Al doilea defect: citirea WPR2 de după I6 era refuzată (cerea permis de scriere).
**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Capătul de 128 KiB | `compare_frts_full` / `check_frts_full` / `Wpr2EndForm` (3 forme, cu `label()` în română); `check_frts` (folosit de I6) acceptă acum forma plăcii | PASS (SIMULARE): T70 |
| Citirea după I6 | `read_frts_after_i6` + `StatusReadBar0`: doar citiri, 3 registre de stare, fără permis, fără funcție de scriere, fiecare citire în jurnal | PASS (SIMULARE): T71 |
| `gpu-sim` | `FakeGsp::wpr2_hi_back` (implicit 128 KiB, ca placa) | PASS (SIMULARE) |
| Jurnalul real din 3 nov | `verifica_jurnal_i6` pe `gpu-jurnal-002.txt`: ordinea celor 720 de scrieri I6 TRECE, Bus Master pornit/pus la loc; HI nu era în jurnal (codul vechi nu jurnaliza citirile) | PARȚIAL: T68 |
| Unealta de jurnal | `wpr2_after_i6_full`, câmpul `StickJournal::wpr2_full`; `verifica_jurnal_i6` folosește regula nouă | PASS (SIMULARE) |
| Pentru OS | `integration/cyber-universe/C-66_wpr2-bloc-128k.md`: fixarea ≥ `9ac89b8` + înlocuirea a 1 funcție în `gpu_i6.rs` | propus |
| Compatibilitate OS | doar adăugări; `check_frts_detail` / `Wpr2End` neschimbate (refuză forma de 128 KiB — de aceea OS-ul trece pe funcția nouă) | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **338 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.
Oprirea de siguranță nu se aplică (pas terminat în această sesiune).

**Următorul pas:** sesiunea OS face partea ei din C-66 (fixare + Release) → fereastra următoare: I6 o dată, pornire rece → așteptat TRECE →
jurnalul de pe stick în hub → `verifica_jurnal_i6` → plan pentru I7 (doar cu comandă nouă).

---
## Sesiunea 45 — 4 noiembrie 2026, ~13:30 (ora României), în cloud, FĂRĂ placă — fără lucru nou (fereastra C-63 s-a încheiat, jurnalele lipsesc)

Verificat: CI verde (rularea 97, `70fc8e3`); în hub nicio comandă nouă/preluată pentru `cyber-gpu`; singura decizie nouă (11:33, „super-școala de
română”) e doar notată, plan abia după ce OS + GPU ajung la viteză. Fereastra C-63 (07:00–~09:30) s-a încheiat, dar în `rezultate/` **nu a apărut
niciun raport/jurnal** (nici `gpu-jurnal-002.txt`, nici jurnalul noului I6). Cod neschimbat: NOT TESTED pe placă. Nu am început lucru nou.
Atenție: sesiunile 43, 44 și 45 n-au avut niciun pas → dacă nici următoarea nu are, se aplică oprirea de siguranță („OPRIT”).
**Următorul pas:** neschimbat — jurnalele ferestrei în `rezultate/` → `verifica_jurnal_i6` → T68 (HI real).

---
## Sesiunea 44 — 4 noiembrie 2026, ~07:30 (ora României), în cloud, FĂRĂ placă — fără lucru nou (fereastra C-63 e în desfășurare)

Verificat: CI verde (rularea 96, `f180b6a`); în hub de la sesiunea 43 nicio comandă nouă/preluată pentru `cyber-gpu`, nicio decizie nouă,
niciun raport nou în `rezultate/` (fereastra lui Adrian a început la 07:00). Cod neschimbat: NOT TESTED pe placă. Nu am început lucru nou.
**Următorul pas:** neschimbat — jurnalele ferestrei în `rezultate/` → `verifica_jurnal_i6` → T68 (HI real).

---
## Sesiunea 43 — 4 noiembrie 2026, ~01:30 (ora României), în cloud, FĂRĂ placă — fără lucru nou (așteaptă fereastra C-63, 07:00)

Verificat: CI verde (rularea 95, `9a6e9e5`); în hub de la sesiunea 42 nicio comandă nouă/preluată pentru `cyber-gpu`, nicio decizie nouă,
niciun raport nou în `rezultate/`. Cod neschimbat: NOT TESTED pe placă. Nu am început lucru nou (nu inventăm lucru fără hardware).
**Următorul pas:** neschimbat — după fereastra de azi, jurnalele de pe stick în `rezultate/` → `verifica_jurnal_i6` → T68 (HI real).

---
## Sesiunea 42 — 3 noiembrie 2026, ~19:30 (ora României), în cloud, FĂRĂ placă — pregătire pentru fereastra C-63 (4 nov 07:00)

În hub: nicio comandă nouă pentru `cyber-gpu`. Nou: **C-61 gata** (OS fixat pe `3286a3b`, Release `os-219`, apoi `os-220`) și **C-63**: fereastra lui
Adrian pe workstation **4 nov de la 07:00** — printre teste: copierea `gpu-jurnal-002.txt` și I6 repetat (pornire rece). CI verde (rularea 94, `772c1e3`).
**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Verificarea jurnalului dintr-o comandă | `i6_log::check_stick_journal` (nou, doar adăugare): desparte scrierile în configurație (Bus Master), verifică ordinea I6 și WPR2 după FWSEC; exemplul `verifica_jurnal_i6` (`cargo run --release -p nvidia-ga104 --example verifica_jurnal_i6 -- gpu-jurnal-002.txt`) dă „TRECE” sau „DE ANALIZAT” (ieșire 2) | PASS (SIMULARE): T69 |
| Fereastra C-63 în docs | `docs/FEREASTRA_C-49.md`: secțiunea „Fereastra 4 nov (C-63)” — ordinea pentru partea GPU și ce rulăm noi după | doar plan |
| Compatibilitate OS | doar adăugări (`StickJournal`, `check_stick_journal`) | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **336 teste PASS** (1 ignorat), build no_std PASS; clippy cu `--features hw-write` PASS.

**Următorul pas:** după fereastră, cu `gpu-jurnal-002.txt` (și jurnalul noului I6) în `rezultate/`: `verifica_jurnal_i6` pe fiecare → T68 (HI real);
dacă I6 trece pe placă → plan pentru I7 (rămâne blocat până la o comandă separată).

---
## Sesiunea 41 — 3 noiembrie 2026, ~13:30 (ora României), în cloud, FĂRĂ placă — doar plan (ideea „un singur program”)

În hub, nicio comandă nouă/preluată pentru `cyber-gpu` după C-58 (gata), niciun raport nou în `rezultate/`. CI verde (rularea 93, `3286a3b`).
Nou: ideea `idei/2026-11-03_cyber-gpu-un-singur-bloc.md`, aprobată de Adrian (13:25) **ca direcție de plan**; ne cere s-o păstrăm în `docs/`.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Ținta în docs | `docs/TINTA_UN_SINGUR_PROGRAM.md`: ce avem deja în simulare (semafoare QMD/coadă, inel GPFIFO), ce lipsește (atomice în NAK, toate blocurile rezidente, ceas de siguranță, tabel de calcule), ordinea măsurătorilor (întâi costul unei lansări pe placă) | doar plan — NOT TESTED |
| Cod | neschimbat; `sh tools/check.sh`: 335 teste PASS (1 ignorat), build no_std PASS | NOT TESTED pe placă |

**Următorul pas:** neschimbat față de sesiunea 40 — sesiunea OS mută fixarea (≥ `af02bea`) → la fereastra următoare `raport/gpu-jurnal-002.txt` în hub
→ confirmăm HI cu `wpr2_after_i6` → I6 se repetă pe placă (pornire rece).

---
## Sesiunea 40 — 3 noiembrie 2026, ~12:40 (ora României), în cloud, FĂRĂ placă — C-58 (OPRIT ridicat: comandă nouă)

Starea „OPRIT” se încheie: în hub au apărut rezultatul de pe placă (`rezultate/2026-11-03-workstation-os214-i6/`) și comanda **C-58** pentru noi.
Pe placă, I6 (`gpu i6 --scrie`) a dat „NU TRECE”: „verificare esuata la 0x1fa824: asteptat 0x1ffe00, citit 0x1ffe00” — începutul WPR2 era exact
cel cerut, dar mesajul numea registrul greșit; diferea capătul (HI). **Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| (1) Eroarea numește HI | `check_frts` (același nume): start greșit → eroare la LO `0x1fa824`; capăt greșit → eroare la HI `0x1fa828` cu așteptat/citit. Nou: `compare_frts`, `check_frts_detail`, `FrtsCheck`, `Wpr2End` (valorile brute LO/HI) | PASS (SIMULARE): T65 |
| (2) Sursele publice | RM 570.144 (MIT) verifică după FWSEC doar HI ≠ 0 și LO = începutul FRTS; nova-core (GPL, doar citit) la fel — comentariul lui spune „HI exclusiv”, dar nu-l verifică. Pe placă HI ≠ sfârșitul exclusiv → cel mai probabil ultima pagină de 4 KiB (capăt inclusiv). Regula nouă: LO exact; HI = sfârșitul exclusiv **sau** ultima pagină; altceva = eroare | DE CONFIRMAT cu HI brut din `gpu-jurnal-002.txt` |
| (2) `gpu-sim` | `FakeGsp::wpr2_hi_last_page` (implicit: ultima pagină, ca pe placă; `false` = exclusiv) | PASS (SIMULARE): T66 |
| (3) Test cu valorile reale | FRTS `0x1ffe00000..0x1fff00000`: ambele forme trec, 5 capete greșite prinse cu mesaj la HI; `i6_log::wpr2_after_i6` citește LO/HI din jurnalul de pe stick | PASS (SIMULARE): T65–T67 |
| Propunerea pentru OS | `integration/cyber-universe/C-58_wpr2-capat.md` + hub `idei/2026-11-03_cyber-gpu-fixare-c58.md`: fixarea ≥ `af02bea` | propus |
| Compatibilitate OS | doar adăugări; `check_frts` păstrat (aceleași citiri, în aceeași ordine) | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **335 teste PASS** (1 ignorat), build no_std PASS; `--features hw-write` PASS.

**Următorul pas:** sesiunea OS mută fixarea (C-52) → la fereastra următoare Adrian copiază `raport/gpu-jurnal-002.txt` în hub → noi confirmăm HI cu
`wpr2_after_i6` (T68) → abia apoi I6 se repetă pe placă (pornire rece).

---
## Sesiunea 39 — 3 noiembrie 2026, ~07:28 (ora României), în cloud, FĂRĂ placă — OPRIT rămâne

**OPRIT: necesită analiza lui Adrian** (neschimbat). Verificat: CI verde (vârful `a8d72c3`); în hub de la sesiunea 38 nicio comandă nouă/preluată
pentru `cyber-gpu`, nicio decizie nouă, niciun raport nou în `rezultate/`. Singura noutate: C-57 (pentru Kronos) — Adrian e la workstation azi
09:00–11:30, cu „primele scrieri în placă” în plan; pașii noștri pentru asta sunt în `docs/FEREASTRA_C-49.md` (Release OS `os-214`).
Cod neschimbat: NOT TESTED. Nu am început lucru nou. După fereastră: raportul/jurnalul în `rezultate/` → `compara_raport` + `i6_log`.

---
## Sesiunea 38 — 3 noiembrie 2026, ~01:15 (ora României), în cloud, FĂRĂ placă — OPRIT rămâne

**OPRIT: necesită analiza lui Adrian** (neschimbat). Verificat: CI verde (vârful `3e77f9a`); în hub de la sesiunea 37 nicio comandă nouă/preluată
pentru `cyber-gpu`, nicio decizie nouă, niciun raport nou în `rezultate/`. Cod neschimbat: NOT TESTED. Nu am început lucru nou.

---
## Sesiunea 37 — 2 noiembrie 2026, ~19:40 (ora României), în cloud, FĂRĂ placă — OPRIT rămâne

**OPRIT: necesită analiza lui Adrian** (neschimbat). Verificat din nou: CI verde (vârful `d95b07b`); în hub de la sesiunea 36 nu a apărut nicio
comandă nouă/preluată pentru `cyber-gpu`, nicio decizie nouă, niciun raport nou de pe placă în `rezultate/`. Cod neschimbat: NOT TESTED. Nu am început lucru nou.
Ce ar debloca: la fel ca în sesiunea 36.

---
## Sesiunea 36 — 2 noiembrie 2026, ~13:27 (ora României), în cloud, FĂRĂ placă — OPRIT: necesită analiza lui Adrian

**OPRIT: necesită analiza lui Adrian.** Regula de oprire de siguranță se aplică: ultimele 3 sesiuni (33b, 34, 35) n-au terminat niciun pas,
iar nici acum nu există lucru nou. Verificat: CI verde (rularea 87, `86480ef`); în hub nicio comandă nouă/preluată pentru `cyber-gpu`
(ultima: C-54, gata), niciun raport nou de pe placă în `rezultate/` (ultimul: `2026-11-01-workstation-477-i4`), nicio decizie nouă pentru noi.
Cod neschimbat: NOT TESTED (nimic de testat). Nu am început lucru nou.

**Ce ar debloca:** fereastra I6 cu Adrian (`docs/FEREASTRA_C-49.md`, Release `os-214`), un raport `gpu tur` al buclei automate în `rezultate/`,
sau o comandă nouă de la Adrian / centrul de comandă în `COMENZI.md`. Până atunci, sesiunile următoare doar verifică și se opresc.

---
## Sesiunea 35 — 2 noiembrie 2026, ~07:25 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat: CI verde (rularea 85, `5426b8e`); în hub **nicio comandă nouă/preluată pentru `cyber-gpu`**, niciun raport nou în `rezultate/`
de la placa reală (ultimul: `2026-11-01-workstation-477-i4`), nicio decizie nouă. Cod neschimbat: NOT TESTED (nimic de testat). Nu am inventat lucru nou.
Atenție: sesiunile 33b, 34 și 35 n-au avut niciun pas → dacă nici următoarea nu are, regula de oprire de siguranță se aplică (scrie „OPRIT”).

**Următorul pas:** neschimbat — fereastra I6 cu Adrian (`docs/FEREASTRA_C-49.md`, Release `os-214`) sau prima tură automată `gpu tur`
→ raportul în `rezultate/` → `compara_raport` / `i6_log`.

---
## Sesiunea 34 — 2 noiembrie 2026, ~01:30 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat: CI verde (rularea 84, `d37cd37`); în hub **nicio comandă nouă/preluată pentru `cyber-gpu`** (C-54 era ultima, gata). Nou de la sesiunea 33:
sesiunea OS a terminat **C-52, C-53 și C-55** (Release `os-214`, fixat pe `4e6f747`): `gpu pregatire-i6`, `gpu i6 --scrie`, bucla automată, `gpu tur`,
`gpu lant` — toate doar în QEMU/simulare. Tot ce mai urmează pentru noi cere placa reală (fereastra cu Adrian sau un raport `gpu tur` în `rezultate/`).
Cod neschimbat: NOT TESTED (nimic de testat). Nu am inventat lucru nou.

**Următorul pas:** fereastra I6 cu Adrian (`docs/FEREASTRA_C-49.md`, Release `os-214`) și/sau prima tură a buclei automate → raportul în `rezultate/`
→ noi rulăm `compara_raport` / citim jurnalul cu `i6_log`.

---
## Sesiunea 33b — 1 noiembrie 2026, ~20:10 (ora României), trecere extra pentru C-54 — fără lucru nou

Trecerea extra cerea C-54, dar comanda era deja terminată în sesiunea 33 (`14a0b57`…`4e6f747`, CI verde la rulările 80–83). În hub nu există
nimic nou pentru noi. Cod neschimbat: NOT TESTED (nimic de testat). **Următorul pas:** neschimbat (un „da” pentru `idei/2026-11-01_cyber-gpu-tur-si-lant.md`).

---
## Sesiunea 33 — 1 noiembrie 2026, ~19:45–20:05 (ora României), în cloud, FĂRĂ placă — C-54

Comandă nouă în hub: **C-54** (din decizia lui Adrian „da bucla automată”, 1 nov 19:45): pași în lanț la ferestre, lista citirilor fără risc
pentru bucla automată + unealta de comparare, `gpu-sim` cu datele reale. **Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| (3) Date reale în `gpu-sim` | `gpu_sim::observed`: din rapoartele OS de pe placă (28 oct, 29 oct, 1 nov): Command `0x0003`, D0, Resizable BAR (3 intrări: 16/256/32 MiB), BOOT_0 `0xb74000a1`, BOOT_42 `0x174a1000`, PRIV_LEVEL_MASK `0x8b8f`, GFW `0x3ff`, VBIOS: 4 imagini la `0x0/0xfe00/0x24600/0x29c00` cu mărimile reale, BIT la `0x1b0` cu cele 17 jetoane, tabela PMU `0x87ecb`, descriptorul FWSEC `0x41208` (v3, antet `0x4ac`, ucode `0xea00`, 3 semnături, mască `0x7`, `ucode_id` 9). Doar poziții, mărimi, id-uri — **fără conținut VBIOS/firmware**. Ce încă ghicește: `observed::STILL_SIM` | PASS (SIMULARE): 2 teste |
| Ce era greșit înainte | modelul vechi ghicea BOOT_0 `0x174000a1` (placa: `0xb74000a1`), Command `0x0007` (placa: Bus Master oprit, `0x0003`), a doua imagine VBIOS la `0x12000` (placa: `0xfe00`), FWSEC `ucode_id` 10 (placa: 9). Modelele vechi rămân (testele vechi le folosesc); cel nou e `observed::*` | — |
| (2) Turul de citire | `nvidia_ga104::read_tour`: `SAFE_READS` (6 grupe; doar „i6” încă necitită pe placă), `NOT_YET` (temperatura: registrul nu e în antetele NVIDIA MIT 570.144 → nu intră), `run_tour` → fapte `cheie = valoare` prin aceleași funcții doar-citire pe care le folosește OS-ul; respinge rezultatul dacă se jurnalizează vreo scriere | PASS (SIMULARE): 4 teste; pe modelul nou dă exact valorile de pe placă |
| (2) Unealta de comparare | `nvidia_ga104::board_compare` + exemplul `compara_raport`: citește rapoartele OS (formatul de azi al `gpu`/`gpu gfw`/`gpu vbios`/`gpu fwsec`/`gpu firmware` și liniile noi `fapt …`), compară cu turul pe simulare: LA FEL / DIFERIT / diferit intenționat / doar pe placă / doar în simulare; ieșire 2 la o diferență reală | PASS (SIMULARE): 5 teste. **Pe cele 3 rapoarte reale: 33 la fel, 0 diferit**, 5 diferit intenționat (primul cuvânt VBIOS, cele 4 fișiere firmware), 5 doar în simulare (legătura PCIe, registrele I6) |
| (1) Pași în lanț | `nvidia_ga104::chain`: `WINDOW_CHAIN` (pregătire-i6 → Bus Master pornit → I6 → I7 → Bus Master oprit → tur-după) + `run_chain`: oprire la primul eșec, curățarea rulează oricum, niciun pas care scrie fără permisiune, I7 (blocat în plan) nu se încearcă, pas doar-citire care scrie = eșec | PASS (SIMULARE): 6 teste, 4 greșeli plantate în cod prinse |
| Propunerea pentru OS | `integration/cyber-universe/C-54_gpu-tur-si-lant.md` + hub `idei/2026-11-01_cyber-gpu-tur-si-lant.md`: `gpu tur` (lista albă a buclei automate), `gpu lant` (doar cu „DA”, nu în modul automat) | propus |
| Compatibilitate OS | doar adăugări: `gpu_sim::observed`, `SimConfig::{read_raw, write_raw}`, `nvidia_ga104::{read_tour, board_compare, chain}`; textul blocajului I6 din `init::GA104_PLAN` adus la zi (doar textul) | PASS |

Verificări (`sh tools/check.sh`): format, clippy `-D warnings`, **332 teste PASS** (1 ignorat), build no_std PASS; `--features hw-write` PASS.
CI: rularea 80 verde; 81 pornită la push.

**Următorul pas:** „da” pentru propunerea `gpu tur` / `gpu lant` → sesiunea OS (după C-52/C-53). După fiecare tur automat: `compara_raport`
pe raportul nou din `rezultate/`; orice „DIFERIT” → reparăm modelul. Fereastra cu Adrian rămâne cea din `docs/FEREASTRA_C-49.md`.

---
## Sesiunea 32 — 1 noiembrie 2026, ~19:25–19:35 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat la pornire: CI rularea pentru `9597863` verde; în hub **nicio comandă nouă pentru `cyber-gpu`** după C-51. Nou de la sesiunea 31:
**C-52** (pentru `cyber-universe`) — propunerea noastră `idei/2026-11-01_cyber-gpu-i6-in-os.md` a fost aprobată prin delegare, iar sesiunea OS
a preluat-o (P6-40, 17:24 UTC): `gpu pregatire-i6`, jurnalul pe stick, `gpu i6 --scrie`, cu fixarea cyber-gpu ≥ `d879742`. Ce le trebuie
din driver e deja gata și publicat (doar adăugări față de fixarea lor). Nu am început lucru nou: pasul următor e al sesiunii OS, apoi
fereastra cu Adrian (`docs/FEREASTRA_C-49.md`). Cod neschimbat: NOT TESTED (nimic de testat). Copia locală a hub-ului era iar
desincronizată: aliniată la `origin/main` (copia veche păstrată local, ramura `old-local-main`).

**Următorul pas:** sesiunea OS termină C-52 (Release nou) → fereastra pe workstation cu Adrian → citim jurnalul de pe stick cu
`nvidia_ga104::i6_log::check_i6`. Dacă sesiunea OS ne cere ceva (în `stare/cyber-universe.md` sau `idei/`), are prioritate.
Oprirea de siguranță: a doua sesiune fără pas după 30b (la a treia se scrie „OPRIT”).

---
## Sesiunea 31 — 1 noiembrie 2026, ~13:25–13:35 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat la pornire: CI rularea 77 (vârful `d879742`) verde; în hub nicio comandă nouă pentru `cyber-gpu` după C-51 (C-49, C-51 gata),
nicio decizie nouă după 1 nov 12:10, niciun rezultat nou de pe placă (ultimul: `2026-11-01-workstation-477-i4`). Propunerea pentru OS
(`idei/2026-11-01_cyber-gpu-i6-in-os.md`) e încă „propus” — așteaptă „da”-ul centrului de comandă; sesiunea OS e oprită (coada goală).
Tot ce se putea pregăti în simulare pentru fereastra primelor scrieri e gata (sesiunile 30/30b); pașii următori cer comenzile OS și placa
→ nu am început lucru nou (nu inventăm lucru). Cod neschimbat: NOT TESTED (nimic de testat). Copia locală a hub-ului era desincronizată
(istorie rescrisă pe GitHub): aliniată la `origin/main`, copia veche păstrată local pe ramura `backup-local-2026-11-01`.

**Următorul pas:** neschimbat — „da” pentru propunerea OS → sesiunea OS face `gpu pregatire-i6` + `gpu i6 --scrie` → fereastra cu Adrian
(`docs/FEREASTRA_C-49.md`). Oprirea de siguranță: prima sesiune fără pas după 30b.

---
## Sesiunea 30b — 1 noiembrie 2026, ~12:25–12:55 (ora României), în cloud, FĂRĂ placă — C-51 (trecere extra)

**Decizia lui Adrian (DECIZII 1 nov 12:00):** pe placa reală rulează doar Cyber Universe; proba tinygrad pe Ubuntu se anulează. Comanda **C-51**
(completează C-49): primele scrieri direct din OS, pas cu pas; în locul jurnalului tinygrad, referința din simulare + jurnalul nostru complet.
Recomandarea din sesiunea 30 („întâi tinygrad”) e înlocuită. **Nimic nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Jurnal pe loc | `gpu_core::diag::Journal::with_sink`: fiecare eveniment trimis imediat OS-ului (ora + stick) | PASS (SIMULARE): 1 test |
| Verificarea jurnalului I6 | `nvidia_ga104::i6_log`: `parse_line` (cu/fără prefixul OS) + `check_i6` (ordinea scrierilor I6, blocuri consecutive, nimic în afara GSP, nimic după pornire) | PASS (SIMULARE): 4 teste; 8 greșeli plantate în jurnal, fiecare cu motivul exact; 6 greșeli plantate în verificator prinse |
| **I5g** Bus Master | `gpu_pci::bus_master::{set_bus_master, restore_command, BusMasterPermit}`: o scriere în COMMAND, recitită; refuz fără decodarea memoriei/fără placă; nimic dacă bitul e deja bun; recitire greșită → valoarea veche la loc | PASS (SIMULARE): 5 teste, 6 greșeli plantate prinse |
| **I6 pasul 2** | `fwsec::read_ucode_and_signature`: codul FWSEC și semnătura aleasă copiate neatinse din PROM (doar citire, peste granița dintre imagini) | PASS (SIMULARE): 1 test |
| Planul | `docs/FEREASTRA_C-49.md` rescris: 6 pași, fiecare cu ce se scrie, ce se recitește ca dovadă, criteriul, oprirea sigură; propunerea OS completată (`integration/cyber-universe/C-49_gpu-pregatire-i6.md`, partea C-51) | gata |
| Compatibilitate OS | doar adăugări (`Journal::with_sink`, `gpu_pci::bus_master`, `nvidia_ga104::{i6_log, fwsec::read_ucode_and_signature}`); câmp privat nou în `Journal` (nicio schimbare vizibilă) | PASS |

Verificări (`tools/check.sh`): format, clippy `-D warnings`, **315 teste PASS** (1 ignorat), build no_std PASS; `--features hw-write` PASS.

**Următorul pas:** sesiunea OS (cu „da”-ul centrului de comandă): `gpu pregatire-i6`, jurnalul pe loc, apoi `gpu i6 --scrie`; fereastra cu Adrian.

---
## Sesiunea 30 — 1 noiembrie 2026, ~11:55–12:15 (ora României), în cloud, FĂRĂ placă — OPRIT ridicat (comandă nouă: C-49)

**Oprirea de siguranță s-a ridicat** pe condiția scrisă la sesiunea 28: în hub au apărut **rezultatul ferestrei din 1 nov**
(`rezultate/2026-11-01-workstation-477-i4/`: `gpu firmware` → **criteriul I4: TRECE**, 4/4 oficiale, antete 4/4, 0 accese la placă) și
**comanda C-49** pentru noi (planul ferestrei următoare). **Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.**

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | `tools/check.sh` verde înainte de orice schimbare (294 teste; ținta `x86_64-unknown-none` instalată în sesiune). C-49 → `preluat` în hub | PASS |
| **I3c** siguranța „ucode version” (`fuse.rs`, nou) | registrul era NECUNOSCUT pe GA10x, deci semnătura FWSEC/Booter nu se putea alege pentru placa reală. Din RM 570.144 (MIT): GSP `0x8241C0`, SEC2 `0x824140`, + 4·(id−1); versiune = bitul cel mai înalt + 1. Pe placa noastră FWSEC are `ucode_id` 9 → `0x8241E0`; cere BAR0 întreg (16 MiB). O citire jurnalizată, 0 scrieri; biții 31:16, `0xbadf…`, `0xffffffff` refuzate | PASS (SIMULARE): 6 teste, 6 greșeli plantate prinse |
| **Verificarea I6 doar citire** (`i6_check.rs`, nou) | siguranța + semnătura aleasă, mărimea VRAM (`0x1183A4`), WPR2 (`0x1FA828`; `LO` doar dacă e ridicat), regiunea FRTS; WPR2 ridicat = „nu trece” (trebuie pornire rece) | PASS (SIMULARE): 3 teste, 4 greșeli plantate prinse (a 5-a era o verificare dublă, scoasă) |
| Lista exactă a scrierilor I6 | testul `gsp::hs_load::tests::i6_writes_exactly_the_listed_gsp_registers` fixează cele **18 registre GSP** (`0x1103C0` … `0x110100`) pe care le scrie FWSEC-FRTS; pe imaginea reală (234 de blocuri) ~720 de scrieri | PASS (SIMULARE) |
| **C-49: planul ferestrei** | `docs/FEREASTRA_C-49.md`: recomandare **întâi (a) proba tinygrad** (jurnalul de aur, nu depinde de codul nostru), **apoi (b)** I5g Bus Master (o scriere PCI, `0x04` bitul 2) + I6 FWSEC-FRTS (18 registre, doar fereastra GSP); ce nu se poate strica (VBIOS-ul nu are cale de scriere, siguranțele doar se citesc, cod nesemnat refuzat de boot ROM, totul volatil); revenirea = oprire completă cu cablul scos 30 s. Propunere OS: `integration/cyber-universe/C-49_gpu-pregatire-i6.md` (+ în hub `idei/`) | gata |
| Compatibilitate OS | doar adăugări în interfața publică (`nvidia_ga104::{fuse, i6_check}`); nicio linie `pub` ștearsă; OS-ul nu trebuie schimbat | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **304 teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fereastra următoare cu Adrian: (dacă OS-ul o adaugă) `gpu pregatire-i6` (doar citire), apoi **proba tinygrad**.
Fără placă: sesiunile următoare nu încep lucru nou dacă nu apare un rezultat sau o comandă nouă (contorul opririi pornește de la 0 după sesiunea 30).

---
## Sesiunea 29 — 1 noiembrie 2026, ~07:30 (ora României), în cloud, FĂRĂ placă — OPRIT (continuă)

**OPRIT: necesită analiza lui Adrian** (rămâne valabil din sesiunea 28). Verificat la pornire: CI verde (rularea pe `2916cb5`); în hub nicio comandă nouă
pentru `cyber-gpu`, nicio decizie nouă, niciun rezultat nou de pe placă (ultimul: `2026-10-29-workstation-kronos-gpu`). Nu am început lucru nou.
Cod neschimbat: NOT TESTED (nimic de testat). Deblocare: la fel ca la sesiunea 28 (fereastra pe workstation, o comandă nouă sau decizia de a rări sesiunile).

---
## Sesiunea 28 — 31 octombrie 2026, ~19:20–19:35 (ora României), în cloud, FĂRĂ placă — OPRIT

**OPRIT: necesită analiza lui Adrian.**

Motiv: oprirea de siguranță — sesiunile 25, 26 și 27 n-au terminat niciun pas, iar de atunci n-a apărut nimic nou pentru noi.
Verificat la pornire: CI rularea 73 (vârful `e40e98e`) verde; în hub nicio comandă nouă pentru `cyber-gpu` (ultimele: C-44, C-45 pentru Kronos,
C-46 pentru OS), nicio decizie nouă, niciun rezultat nou de pe placă (ultimul: `2026-10-29-workstation-kronos-gpu`). `os-210` (C-46) păstrează
fixarea `aee06c4`, deci e bun și el pentru fereastra următoare. Nu am început lucru nou. Cod neschimbat: NOT TESTED (nimic de testat).

**Ce deblochează:** (1) fereastra pe workstation cu Adrian (`docs/FEREASTRA_C-39.md`: `gpu firmware` → `criteriul I4: TRECE`) și rezultatul ei
în hub, sau (2) o comandă nouă pentru `cyber-gpu` în `COMENZI.md`, sau (3) decizia lui Adrian să rărească/oprească sesiunea programată până la fereastră.
Sesiunile următoare: dacă nu există nimic din (1)–(3), nu încep lucru nou.

---
## Sesiunea 27 — 31 octombrie 2026, ~13:20–13:35 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat la pornire: CI rularea 72 (vârful `11d48ac`) verde; în hub nicio comandă nouă pentru `cyber-gpu` (C-18, C-20, C-39 gata), nicio
decizie nouă, niciun rezultat nou de pe placă (ultimul: `2026-10-29-workstation-kronos-gpu`), sesiunea OS oprită cu coada goală.
**Fără placă nu e niciun pas mic de făcut** → nu am început lucru nou. Cod neschimbat: NOT TESTED (nimic de testat).

**Următorul pas:** neschimbat — fereastra pe workstation cu Adrian (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`.
Notă pentru oprirea de siguranță: a treia sesiune la rând fără pas terminat (25, 26, 27; motiv: lipsa plăcii). **Sesiunea 28 scrie aici
„OPRIT: necesită analiza lui Adrian” și nu începe lucru nou**, dacă între timp nu apare un rezultat de pe placă sau o comandă nouă.

---
## Sesiunea 26 — 31 octombrie 2026, ~07:20–07:30 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat la pornire: CI rularea 71 (vârful `e74a53a`) verde; în hub nicio comandă nouă pentru `cyber-gpu` (ultima comandă din `COMENZI.md` e C-43,
pentru Kronos; C-18, C-20, C-39 gata), niciun rezultat nou de pe placă (ultimul: `2026-10-29-workstation-kronos-gpu`), nicio decizie nouă,
sesiunea OS oprită cu coada goală. **Fără placă nu e niciun pas mic de făcut** → nu am început lucru nou. Cod neschimbat: NOT TESTED (nimic de testat).

**Următorul pas:** neschimbat — fereastra pe workstation cu Adrian (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`.
Notă pentru oprirea de siguranță: a doua sesiune la rând fără pas terminat (25, 26; motiv: lipsa plăcii). Dacă și sesiunea 27 nu are nimic nou,
sesiunea 28 scrie „OPRIT: necesită analiza lui Adrian”.

---
## Sesiunea 25 — 31 octombrie 2026, ~01:20–01:30 (ora României), în cloud, FĂRĂ placă — fără lucru nou

Verificat la pornire: CI rularea 70 (vârful `647398c`) verde; în hub nicio comandă nouă pentru `cyber-gpu` (C-18, C-20, C-39 gata), niciun
rezultat nou de pe placă (ultimul: `2026-10-29-workstation-kronos-gpu`, deja citit), sesiunea OS oprită cu coada goală. Cum scrie la sesiunea 24,
**fără placă nu mai e niciun pas mic de făcut** → nu am început lucru nou (nu inventăm lucru). Cod neschimbat; teste nerulate (nimic de testat): NOT TESTED.

**Următorul pas:** neschimbat — fereastra pe workstation cu Adrian (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`.
Notă pentru oprirea de siguranță: aceasta e prima sesiune fără pas terminat (motiv: lipsa plăcii, nu un blocaj).

---

## Sesiunea 24 — 30 octombrie 2026, ~19:25–19:45 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-18, C-20, C-39 gata; în hub nu sunt
rezultate noi de pe placă; sesiunea OS e oprită, coada ei e goală; `os-209` e gata pentru fereastra următoare). Am făcut ultimul pas mic rămas
„fără placă”: **treapta 6 matmul** (Mesa NAK pe gazdă). Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 69 (vârful `3222669`) verde; `tools/check.sh` verde înainte de orice schimbare (290 teste; ținta `x86_64-unknown-none` instalată în sesiune); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță. Copiile locale ale hub-ului și `cyber-universe` erau desincronizate de GitHub (istorie diferită): aliniate la `origin/main`, copia veche păstrată pe ramura locală `vechi-local` (nimic de-al nostru pierdut) | PASS |
| Unealta NAK | Mesa 25.0.7 reconstruit în cloud (~10 min; sursa verificată cu SHA-256); `add64` și `matmul_2d` au ieșit **identice** (aceleași sume) | PASS |
| **Treapta 6** matmul | `tools/nak/matmul_vec4.comp` (ideea siboehm kernel 6 „vectorize”, MIT, scris de noi): treapta 5 cu citiri/scrieri de 128 de biți și placa lui A transpusă în memoria partajată. NAK: **821 instrucțiuni** (treapta 5: 1 953), 13 136 B, 90 registre, memorie locală 0, 4 KiB partajată, 1 barieră, 64 `ffma.ftz`; SHA-256 `18efe1da…55f7`, identic la a doua compilare. **Condiții pe care programul nu le verifică:** K și N multipli de 4, a/b/c aliniate la 16 | PASS (SIMULARE): `gpu-sim::cmdproc::matmul_vec4_on_cpu` urmează algoritmul; 6 forme identice bit cu bit cu `matmul_fma_ftz` și cu treapta 5; −0,0; K/N nemultipli de 4 și C nealiniat refuzate fără nicio scriere; **8 greșeli plantate, toate prinse** |
| Compatibilitate OS | `git diff 3222669` pe crate-uri: nicio linie `pub` ștearsă; doar adăugări (`gpu_sim::cmdproc::{matmul_vec4_on_cpu, MATMUL_VEC4_NEEDS}`). OS-ul nu folosește nimic din ele | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **294 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă. OS-ul nu a fost reconstruit (nicio schimbare de interfață pentru el): NOT TESTED.

**Următorul pas:** la fereastra următoare (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`, apoi (opțional, cu Adrian) proba tinygrad.
**Fără placă nu mai e niciun pas mic de făcut:** tot ce urmează (I5+ pe placă: FWSEC, Booter, GSP, soneria, lansările reale, viteza treptelor matmul)
cere fereastra pe workstation cu Adrian. Sesiunile următoare fără rezultate noi de pe placă nu au ce începe (nu inventăm lucru).

---

## Sesiunea 23 — 30 octombrie 2026, ~13:20–13:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-18, C-20, C-39 gata; în hub nu sunt
rezultate noi de pe placă; sesiunea OS e oprită, coada ei e goală; `os-209` e gata pentru fereastra următoare). Am făcut unul din cei doi pași
mici rămași „fără placă”: **I10d**, refolosirea bufferului de comenzi după semafor. Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 68 (vârful `01bc248`) verde; `tools/check.sh` verde înainte de orice schimbare (288 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată în sesiune); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță | PASS |
| **I10d** zona de comenzi ca cerc | `nvidia-ga104::gsp::work_queue::WorkQueue`: zona nu mai e liniară. Fiecare trimitere ține minte porțiunea ei; o porțiune fără semafor propriu (de ex. configurarea) e legată de semaforul următoarei lansări (intrările rulează în ordine). O porțiune nouă se pune după cea mai nouă porțiune în lucru sau, dacă nu încape la capăt, de la începutul zonei — **niciodată peste o porțiune în lucru**; altfel refuz, fără nicio scriere. `WorkQueue::retire(sem_va, valoare)` eliberează, de la cea mai veche, porțiunile al căror semafor a ajuns la valoarea citită; o valoare mai mare decât ultima cerută = defect, nimic eliberat. `launch_and_signal` cere acum ca valoarea să crească pe același semafor. `push_in_flight()` spune câți octeți sunt încă la placă | PASS (SIMULARE): 12 lansări `add64` într-o zonă de 128 B (încap doar configurarea + 2 lansări), cu trecere peste capăt; rezultat identic bit cu bit la fiecare; **7 greșeli plantate, toate prinse** |
| Alegeri notate | eliberarea doar după semafor, nu după `GPGet` (mai prudent) | `COMPATIBILITATE_FIRMWARE.md` („De verificat”) |
| Compatibilitate OS | `git diff 01bc248` pe crate-uri: nicio linie `pub` ștearsă; doar adăugări (`WorkQueue::{retire, push_in_flight}`). Comportament schimbat (intenționat): `submit` poate refolosi zona după `retire`; `launch_and_signal` refuză o valoare de semafor care nu crește (pe același semafor). OS-ul nu folosește `WorkQueue` | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **290 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă. OS-ul nu a fost reconstruit (nicio schimbare de interfață pentru el): NOT TESTED.

**Următorul pas:** la fereastra următoare (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`, apoi (opțional, cu Adrian) proba tinygrad.
Fără placă mai rămâne un singur pas mic: treapta 6 matmul (Mesa NAK pe gazdă). După el, tot ce urmează (I5+ pe placă) cere fereastra cu Adrian.

---

## Sesiunea 22 — 30 octombrie 2026, ~07:05–07:40 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-18, C-20, C-39 gata; în hub nu sunt
rezultate noi de pe placă; sesiunea OS e oprită, coada ei e goală; `os-209` e gata pentru fereastra următoare). Am făcut următorul pas
„fără placă”: **I10c**, forma GR cerută de la GSP-RM. Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea pe vârful `ab5a918` verde; `tools/check.sh` verde înainte de orice schimbare (285 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată în sesiune); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță | PASS |
| Antete | generatorul `tools/gen-rm570` (bindgen-cli 0.71.1 instalat în sesiune) produce în plus `NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS` (3712 B = 8 motoare × 58 perechi), comanda `0x20800a2a` și indicii `NV0080_CTRL_GR_INFO_INDEX_*` (12, 13, 20, 23, 32); `rmapi.rs` doar cu adăugări | PASS (fixate în `rm_api_layouts_570_144`) |
| **I10c** forma GR | `nvidia-ga104::gsp::user_space::query_gr_info`: o comandă pe subdispozitivul clientului de lucru, ca tinygrad pe calea GSP (`_query_gpu_info`, ramura `is_nvd()`: comanda internă, fiindcă `GR_GET_INFO` public poartă un pointer). `rmapi::parse_gr_info` citește GR0 după poziție: GPC-uri, TPC/GPC, SM/TPC, warp-uri/SM, versiunea SM; orice valoare 0, lungime greșită sau `index` care nu-și numește poziția = refuz. `GrInfo::check_sm86`: o placă care nu e `sm_86` e refuzată (programele noastre NAK sunt pentru GA104). Testul cap-coadă al lansării `add64` o cere acum înainte de lansare | PASS (SIMULARE): 3 teste noi + T41 extins; **8 greșeli plantate în răspuns + 4 în cod, toate prinse** |
| Modelul | `gpu-sim::fake_rm`: comanda `KGR_GET_INFO` pe subdispozitiv (mărime exactă), GR0 cu `SIM_GR_INFO` (forma unui GA104 complet: 6 × 4 × 2 = 48 SM, 48 warp-uri/SM, `sm_86`) — **valori SIM**, alese de noi, nu ale plăcii | PASS |
| Alegeri notate | citirea după poziție; `index` = 0 acceptat (nu știm dacă GSP-RM îl completează); valorile „litter” = maximele cipului | `COMPATIBILITATE_FIRMWARE.md` („De verificat”) |
| Compatibilitate OS | `git diff ab5a918` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`, `gpu-sim`: nicio linie `pub` ștearsă; doar adăugări (`user_space::query_gr_info`, `rmapi::{gr_info_request, parse_gr_info, GrInfo, SM_VERSION_SM86}`, `cmd::KGR_GET_INFO`; în `gpu-sim::fake_rm`: `cmd::KGR_GET_INFO`, `size::GR_INFO`, `SIM_GR_INFO`) | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **288 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă. OS-ul nu a fost reconstruit (nicio schimbare de interfață pentru el): NOT TESTED.

**Următorul pas:** la fereastra următoare (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`, apoi (opțional, cu Adrian) proba tinygrad.
Fără placă mai rămân: refolosirea bufferului de comenzi (acum liniar) după semafor; treapta 6 matmul. Restul drumului (I5+ pe placă) cere fereastra cu Adrian.

---

## Sesiunea 21 — 30 octombrie 2026, ~01:25–01:45 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna. Sesiunea OS a făcut C-42 (fixarea
pe `aee06c4`, Release `os-209`); în hub nu sunt rezultate noi de pe placă. Am făcut următorul pas „fără placă”: **I10b**, prima lansare pe
canalul de lucru. Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 65 (vârful `5366ed3`) verde; `tools/check.sh` verde înainte de orice schimbare (280 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță | PASS |
| **I10b** lansarea | `nvidia-ga104::gsp::work_queue`: `invalidate_caches` (`INVALIDATE_SHADER_CACHES_NO_WFI`: instrucțiuni + date globale + constante), `launch` (`SEND_PCAS_A` = QMD >> 8, `SEND_SIGNALING_PCAS2_B` = `PREFETCH_SCHEDULE`; QMD aliniat la 256 și în VA-ul nostru), `release` (semaforul de terminare: `RELEASE` + `RELEASE_WFI_EN` + 64 de biți, fără timestamp, fără întrerupere), `read_semaphore` (două citiri pe 32 de biți, recitire dacă partea de sus se schimbă), `WorkQueue::launch_and_signal`. Ordinea și valorile după tinygrad `ops_nv.py` (MIT, `f5b17be`), câmpurile din antetele generate | PASS (SIMULARE): cuvintele = exact cele din tinygrad |
| **I10b** cap-coadă | golden image → I9c → I9d → I9e → configurarea I10a → lansarea `add64` (codul real compilat cu Mesa NAK, 240 B): driverul scrie programul, argumentele, intrările, QMD-ul și semaforul în VRAM după adresa fizică; modelul `gpu-sim::cmdproc` le citește **doar** prin tabelele de pagini, „rulează” programul pe procesor, scrie rezultatul, semaforul și `GPGet` înapoi; driverul vede `GPGet = 2`, semaforul `0x1_0000_0007` și rezultatul **identic bit cu bit** cu `reference::add_ftz` | PASS (SIMULARE); **8 greșeli plantate, toate prinse**; 5 refuzuri în driver fără nicio scriere |
| Modelul | `gpu-sim::cmdproc`: `INVALIDATE_SHADER_CACHES_NO_WFI` (câmpuri verificate, numărate în `cache_invalidations`); `SEM_EXECUTE` refuză acum biții nemodelați (de ex. timestamp) și semaforul pe 64 de biți nealiniat; `RELEASE_WFI` acceptat | PASS (testele vechi `add64`/matmul neschimbate și verzi) |
| Alegeri notate | `PREFETCH_SCHEDULE` (tinygrad) vs `INVALIDATE_COPY_SCHEDULE` (NAK); fără `NON_STALL_INTERRUPT`; invalidarea la fiecare lansare | `COMPATIBILITATE_FIRMWARE.md` („De verificat”) |
| Compatibilitate OS | `git diff 5366ed3` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`, `gpu-sim`: nicio linie `pub` ștearsă; doar adăugări (`work_queue::{invalidate_caches, launch, release, read_semaphore}`, `WorkQueue::launch_and_signal`, `CommandProcessor::cache_invalidations`). Comportament schimbat doar în modelul de test (`SEM_EXECUTE` mai strict) | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **285 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă. OS-ul nu a fost reconstruit (nicio schimbare de interfață pentru el): NOT TESTED.

**Următorul pas:** la fereastra următoare (cu `os-209`): `gpu firmware` → `criteriul I4: TRECE`, apoi (opțional, cu Adrian) proba tinygrad.
Fără placă: cererea `GR_GET_INFO` (forma GR pentru memoria locală); refolosirea bufferului de comenzi (acum liniar) după semafor; treapta 6 matmul.

---

## Sesiunea 20 — 29 octombrie 2026, ~19:20–19:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-39 gata; fereastra următoare
așteaptă Release-ul OS cu fixarea nouă; în hub nu sunt rezultate noi de pe placă). Am făcut următorul pas „fără placă”: **I10a**, primul
pushbuffer pe canalul de lucru. Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 64 (vârful `9d671fd`) verde; `tools/check.sh` verde înainte de orice schimbare (268 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță | PASS |
| Antete | generatorul `tools/gen-rm570` (bindgen-cli 0.71.1 instalat în sesiune) a reprodus întâi **identic** fișierele; apoi produce și USERD (`AmpereAControlGPFifo` din `clc56f.h`: 512 B, `GPGet` 0x88, `GPPut` 0x8c); `rmapi.rs` doar cu adăugări | PASS (dimensiuni fixate în `rm_api_layouts_570_144`) |
| **I10a** configurarea calculului | `nvidia-ga104::gsp::work_queue`: `compute_setup` = `SET_OBJECT` (AMPERE_COMPUTE_B) + `SET_SHADER_LOCAL_MEMORY_WINDOW` + `SET_SHADER_SHARED_MEMORY_WINDOW` pe subcanalul 1, cu adresele tinygrad (`0x7293…`/`0x7294…`, verificate în afara VA-ului nostru); `local_memory_size` + `local_memory_setup` (aritmetica tinygrad; forma GR intră ca parametru, nu e presupusă) | PASS (SIMULARE): cuvintele = exact ce dă `nvm(1, …)` din tinygrad |
| **I10a** trimiterea | `WorkQueue::submit`: cuvintele în bufferul de comenzi, intrarea GPFIFO în slotul următor (ca tinygrad: `LEVEL = SUBROUTINE`), apoi `GPPut` în USERD — în ordinea asta; întoarce **descrierea** soneriei (BAR0 + 0xbb0090 = jetonul), **fără s-o scrie**; `update_get` citește `GPGet` și refuză orice valoare în afara [get, put]. Testul a găsit o problemă reală: USERD-ul neșters conține „gunoi” → `open_work_channel_cleared` șterge USERD-ul **înainte** de a cere canalul (RM 570.144 o face singur doar pe vGPU) | PASS (SIMULARE): cap-coadă golden image → I9c → I9d → I9e → I10a; modelul `gpu-sim::cmdproc` citește inelul, USERD și pushbuffer-ul **doar** prin rădăcina primită de RM-ul fals, execută, scrie `GPGet`; inelul trece peste capăt (1023 de intrări, a 1024-a refuzată fără nicio scriere); **5 greșeli plantate în driver + 3 în octeți, toate prinse** |
| Modelul | `gpu-sim::cmdproc`: metodele de configurare (perechi `_A`/`_B`, biți în afara câmpului refuzați) în `ComputeState`; `run_channel` (inelul văzut prin USERD); `GpuMem::{load_from_vram, store_to_vram}` prin cititorul independent de tabele | PASS: 3 teste noi |
| Alegeri notate | ferestrele, `LEVEL`, ștergerea USERD, memoria locală | `COMPATIBILITATE_FIRMWARE.md` („De verificat”) |
| Compatibilitate OS | `git diff 9d671fd` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute` și `gpu-sim`: nicio linie `pub` ștearsă; doar adăugări (`gsp::work_queue`, `user_space::{open_work_channel_cleared, USERD_SIZE}`, `ChannelMemory` pentru `mmu::GpuMemory`; în `gpu-sim::cmdproc`: `ComputeState`, `run_channel`, câmpuri noi, `USERD_GP_GET/PUT`) | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **280 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă. OS-ul nu a fost reconstruit (nicio schimbare de interfață pentru el): NOT TESTED.

**Următorul pas:** la fereastra următoare (după Release-ul OS cu fixarea ≥ `ef97110`): `gpu firmware` → `criteriul I4: TRECE`, apoi
(opțional, cu Adrian) proba tinygrad. Fără placă: cererea `GR_GET_INFO` (forma GR pentru memoria locală) și prima lansare pe canalul de lucru
(QMD + `SEND_PCAS` + semaforul de terminare, prin `WorkQueue`, în model); treapta 6 matmul (Mesa NAK pe gazdă).

---

## Sesiunea 19 — 29 octombrie 2026, ~13:30–13:55 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-39 gata; fereastra următoare
așteaptă Release-ul OS cu fixarea nouă). Am făcut următorul pas „fără placă”: I9e. Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 63 (vârful `aee06c4`) verde; `tools/check.sh` verde înainte de orice schimbare (262 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță | PASS |
| Antete | generatorul `tools/gen-rm570` (bindgen-cli 0.71.1 instalat în sesiune) a reprodus întâi **identic** fișierele; apoi produce și `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS` (4 B), `NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS` (2 B), comenzile `0xc36f0108` și `0xa06c0101`; `rmapi.rs` doar cu adăugări | PASS (dimensiuni fixate în `rm_api_layouts_570_144`) |
| **I9e** canalul de lucru | `nvidia-ga104::gsp::user_space::open_work_channel`: inel GPFIFO (1024 intrări) + USERD (0x400, imediat după inel) într-un buffer mapat; canalul `AMPERE_CHANNEL_GPFIFO_A` cu **părinte = TSG** și context share-ul din I9d (fără VA space, `engineType` 0 = motorul grupului); obiectul compute; `PROMOTE_CTX` pentru MAIN/PM/PATCH pe buffere noi ale canalului, întâi jumătatea fizică, apoi cea virtuală (ca tinygrad); jetonul soneriei = canalul din răspunsul GSP-RM (doar biții 11:0) + runlist-ul lui GR0 `<< 16` (`dev_ctrl.h`); programarea grupului (`GPFIFO_SCHEDULE`). `rmapi::{WorkChannelSetup, work_channel_params, work_submit_token_request, parse_work_submit_token, doorbell_token, schedule_params}`. Regulile citite ca specificație în RM 570.144 (MIT): `kernel_channel.c`, `kernel_fifo_ga100.c`, `kernel_channel_group_api.c` | PASS (SIMULARE): 6 teste noi + testul cap-coadă extins (golden image → I9c → I9d → I9e prin cozile reale; inelul, USERD și bufferele de context găsite prin rădăcina primită de model); `gpu-sim::fake_rm` știe canale în TSG, jetonul și programarea (poziții scrise de mână); 7 refuzuri RM întorc codul RM; **4 greșeli plantate în codificator + 4 în secvență, toate prinse** |
| Alegeri notate | handle-urile `hObjectError`/`hObjectBuffer`/`hUserdMemory` = cei 32 de biți de jos ai adreselor fizice, ca tinygrad (RM spune că pe GSP nu sunt căutate); `flags`/`cid`/`internalFlags` = 0; soneria **nu** e sunată (ar fi o scriere în registru) | notat în `COMPATIBILITATE_FIRMWARE.md` („De verificat”) |
| Compatibilitate OS | `git diff aee06c4` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: nicio linie `pub` ștearsă; doar adăugări (`gsp::user_space::{WorkChannel, open_work_channel, WORK_CTX_BUFFERS}`, funcții și constante noi în `rmapi`, `cmd::{WORK_SUBMIT_TOKEN, TSG_SCHEDULE}`). În `gpu-sim::fake_rm`: `chid`, `promoted`, `is_scheduled`, `parent_of`, `status::OBJECT_NOT_FOUND`, comenzi și mărimi noi | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **268 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă. OS-ul nu a fost reconstruit (nicio schimbare de interfață pentru el): NOT TESTED.

**Următorul pas:** la fereastra următoare (după Release-ul OS cu fixarea ≥ `ef97110`): `gpu firmware` → `criteriul I4: TRECE`, apoi
(opțional, cu Adrian) proba tinygrad. Fără placă: primul pushbuffer pe canalul de lucru (`SET_OBJECT` compute + ferestrele de memorie, ca
tinygrad) scris în inel și verificat în modelul de comenzi din `gpu-sim`; treapta 6 matmul (Mesa NAK pe gazdă).

---

## Sesiunea 18 — 29 octombrie 2026, ~12:45–13:20 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comandă nouă în hub: **C-39** (la fereastra din 29 oct, `gpu firmware`: 4/4
fișiere oficiale, dar antetele a 3 fișiere refuzate de parserul nostru). Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | `tools/check.sh` verde înainte de orice schimbare (260 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță; C-39 preluat în hub | PASS |
| Cauza | am citit **doar cuvintele de antet** ale celor 3 fișiere oficiale (descărcate temporar, SHA-256 = oficial, șterse imediat; codul, datele și semnăturile necitite): `bin_size` e mai mare decât fișierul (`booter_load`: 0xf000 față de 61 304 B; `booter_unload`: 0xa100 / 41 080; `bootloader`: 0x6100 / 24 684); datele se termină exact la capătul fișierului. A doua greșeală, ascunsă de prima: cuvântul 6 din antetul „HS v2” e **adresa** numărului de semnături (0x350 → 2), nu numărul (nouveau face la fel) | găsit |
| **C-39** reparația | `fw_hdr::BinHdr::parse`: `bin_size` păstrat doar pentru raport, nu mai e comparat; în schimb datele trebuie să stea în fișier și după antetul exterior. `HsV2::parse`: numărul de semnături citit prin adresă. `gpu-sim::fw_stick::{booter_570_144, bootloader_570_144}`: fișiere de probă cu **exact** valorile de antet ale celor oficiale (restul umplutură); `files()` le folosește. Teste noi: antetele cu forma oficială (`fw_hdr`), Booter-ul cu forma oficială încărcat cap-coadă (`hs_load::from_booter` + `booter_image`) | PASS (SIMULARE): 262 teste; 2 greșeli plantate prinse (vechea verificare: 4 teste pică; numărul citit direct: 9 teste pică) |
| OS | nota `integration/cyber-universe/C-39_gpu-firmware-antete.md` (doar mutarea fixării pe `ef97110`+); copie de lucru a OS-ului cu cyber-gpu local: build normal și `test-gpu-sim` PASS | PASS (compilare); QEMU: NOT TESTED |
| Pasul următor pe placă | `docs/FEREASTRA_C-39.md`: 1 `gpu firmware` (doar citire) → 2 proba tinygrad (**scrie în placă**, ca la C-36 §3) → I5+ cu driverul nostru **nu** în fereastra următoare (I6 = prima scriere a noastră, doar după jurnalul tinygrad) | propunere |
| Compatibilitate OS | `git diff 2c9c90c` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: nicio linie `pub` ștearsă; doar adăugări. Comportament schimbat (intenționat): `BinHdr::parse` acceptă `bin_size` ≠ mărime; `HsV2::parse` citește numărul de semnături prin adresă (fișierele sintetice vechi cu numărul direct în antet nu mai trec — erau greșite) | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **262 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** sesiunea OS mută fixarea pe `ef97110`+ și face Release; la fereastra următoare `gpu firmware` → `criteriul I4: TRECE`, apoi
(opțional, cu Adrian) proba tinygrad. Fără placă: canalul GPFIFO în grupul de canale, promovarea contextului GR, jetonul soneriei; treapta 6 matmul.

---

## Sesiunea 17 — 29 octombrie 2026, ~07:10–07:40 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-36 gata; fereastra I3 → I4 așteaptă
pe Adrian; în hub nu există încă rezultate de la ea). Am făcut următorul pas „fără placă”: I9d. Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 61 (vârful `695e537`) verde; `tools/check.sh` verde înainte de orice schimbare (256 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată); ultimele 3 sesiuni au terminat pași → fără oprire de siguranță | PASS |
| Antete | generatorul `tools/gen-rm570` (bindgen-cli 0.71.1) produce și `NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS` (20 B), `NV_CTXSHARE_ALLOCATION_PARAMETERS` (12 B), clasele `KEPLER_CHANNEL_GROUP_A` (0xa06c), `FERMI_CONTEXT_SHARE_A` (0x9067), `NV2080_ENGINE_TYPE_GRAPHICS`, `NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_*`; `regs.rs`, `structs.rs`, `sysinfo.rs` au ieșit **identic**, `rmapi.rs` doar cu adăugări | PASS (dimensiuni fixate în `rm_api_layouts_570_144`) |
| **I9d** grupul de calcul | `nvidia-ga104::gsp::user_space::open_compute_group`: după clientul de lucru (I9c), grupul de canale (TSG) pentru GR0 pe spațiul de adrese cu tabelele noastre, apoi un context share (subcontext `ASYNC`, ca tinygrad); subcontextul se ia **doar** din răspunsul lui GSP-RM și trebuie să fie < 64, iar restul parametrilor trebuie să vină înapoi neschimbați. Spre deosebire de tinygrad, TSG-ul numește explicit spațiul de adrese (altfel RM ar folosi spațiul implicit al dispozitivului, fără tabelele noastre). `rmapi::{channel_group_params, ctxshare_params, parse_ctxshare_reply, subcontext, MAX_SUBCONTEXTS}`; `Rm::alloc_reply` (intern). Regulile citite ca specificație în RM 570.144 (MIT): `kernel_channel_group_api.c`, `kernel_ctxshare.c`, `vaspace.c` | PASS (SIMULARE): 4 teste noi + testul cap-coadă extins (golden image → I9c → I9d prin cozile reale); `gpu-sim::fake_rm` știe acum TSG și context share (poziții scrise de mână; subcontextele `SYNC`/`ASYNC`/`PREFER_LOWER`/`SPECIFIED`); 9 refuzuri RM întorc codul RM; **4 greșeli plantate în codificator + 9 în octeți, toate prinse** |
| Limită a modelului | `hVASpace = 0` (spațiul implicit al dispozitivului) nu e modelat: modelul îl refuză cu `NOT_SUPPORTED` (alegerea modelului, nu răspunsul RM); driverul nu-l trimite niciodată | notat în `fake_rm` |
| Compatibilitate OS | `git diff 695e537` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: nicio linie `pub` ștearsă; doar adăugări (`gsp::user_space::{ComputeGroup, open_compute_group}`, funcții noi în `rmapi`, `class::{CHANNEL_GROUP, CONTEXT_SHARE}`). În `gpu-sim::fake_rm`: `bound_vaspace`, `subcontexts`, `SUBCONTEXTS`, clase și mărimi noi | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **260 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fereastra C-36 (cu Adrian): pașii 1 → 2 → (3) din `docs/FEREASTRA_C-36.md`. Fără placă: canalul GPFIFO al clientului de
lucru în grupul de canale (părinte = TSG, cu context share-ul), promovarea contextului GR pentru obiectul compute, jetonul soneriei; treapta 6
matmul (Mesa NAK pe gazdă).

---

## Sesiunea 16 — 29 octombrie 2026, ~01:25–01:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub pentru noi: niciuna (C-36 gata; fereastra I3 → I4 așteaptă
pe Adrian). Am făcut un pas „fără placă” din listă: `SET_PAGE_DIRECTORY` (I9c). Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 60 (vârful `0d06f4d`) verde; `tools/check.sh` verde înainte de orice schimbare (251 teste; rustc 1.97.0, ținta `x86_64-unknown-none` instalată); generatorul `tools/gen-rm570` (bindgen-cli 0.71.1 instalat în sesiune) a reprodus **identic** fișierele generate | PASS |
| Antete | generatorul produce și `rpc_set_page_directory_v1E_05` (48 B) + `NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05` (32 B), comanda `0x801813` și valorile `FLAGS` (`ctrl0080dma.h`); restul `rmapi.rs` neschimbat | PASS (dimensiuni fixate în `rm_api_layouts_570_144`) |
| **I9c** clientul de lucru | `nvidia-ga104::gsp::user_space::open_user_space`: după „golden image”, al doilea client (`0xc1000000`, ca tinygrad), dispozitiv, subdispozitiv, spațiu de adrese **deținut din afară** (`IS_EXTERNALLY_OWNED \| ENABLE_PAGE_FAULTING`, `vaBase 0x1000`), apoi RPC-ul `SET_PAGE_DIRECTORY` (funcția 54) cu rădăcina PD3 a tabelelor noastre (4 intrări, VRAM, toate canalele). Rădăcină greșită = refuz înainte de primul mesaj. `rmapi::{user_vaspace_params, set_page_directory_rpc, check_root}`; `GoldenImage::next_handle`. Regulile din RM 570.144 (`dma.c`, `gpu_vaspace.c`, MIT, citite ca specificație): doar spațiu de adrese deținut din afară / administrare comună, o singură rădăcină, intrările trebuie să acopere limita | PASS (SIMULARE): 5 teste noi; cap-coadă prin cozile reale către `gpu-sim::fake_rm` (acum cu mai mulți clienți și cu `SET_PAGE_DIRECTORY`, poziții scrise de mână); din rădăcina primită de model, cititorul independent găsește bufferele „golden” și unul mapat după RPC; 6 refuzuri RM întorc codul RM; **5 greșeli plantate în octeți + 4 în codificator, toate prinse** |
| Constatare | RM spune că „copiază mereu PDE-urile administrate de RM” în directorul nou → GSP-RM ar putea scrie singur în rădăcina noastră; tinygrad nu vede probleme | notat în `COMPATIBILITATE_FIRMWARE.md` („De verificat”) |
| Compatibilitate OS | `git diff 0d06f4d HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: nicio linie `pub` ștearsă; doar adăugări (`gsp::user_space`, funcții noi în `rmapi`, `rpc::function::SET_PAGE_DIRECTORY`, `GoldenImage::next_handle`). În `gpu-sim::fake_rm` (model de test): câmpul nou `RmLog::page_dirs` | PASS |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **256 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fereastra C-36 (cu Adrian): pașii 1 → 2 → (3) din `docs/FEREASTRA_C-36.md`. Fără placă: restul I9 pentru clientul de lucru
(grupul de canale `KEPLER_CHANNEL_GROUP_A`, `FERMI_CONTEXT_SHARE_A`, canalul GPFIFO, promovarea contextului GR pentru obiectul compute, jetonul
soneriei); treapta 6 matmul (Mesa NAK pe gazdă).

---

## Sesiunea 15 — 28 octombrie 2026, ~22:30–22:55 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti sau pe fișierele firmware reale.** Comandă nouă în hub: **C-36** (firmware-ul
570.144 e la Adrian; pregătirea ferestrei I3 → I4, posibil 29 oct). Tot ce scrie „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | `tools/check.sh` verde înainte de orice schimbare (243 teste; rustc 1.97.0 stable în acest container, ținta `x86_64-unknown-none` instalată); C-36 preluat în hub | PASS |
| **C-36 (2)** I4 pe stick | `nvidia_ga104::fw_check::check_stick`: citește pe rând cele 4 fișiere + `LICENCE.nvidia`, **nu se oprește** la primul greșit, raportează pentru fiecare: lipsă / necitibil / altă mărime / alt SHA-256 / oficial; antetul (`fw_hdr`, `gsp::elf`) se citește **doar** la fișierele oficiale și se raportează separat („fișier greșit” ≠ „parserul nostru greșit”); `FwCheck::lines()` cu `criteriul I4`; 0 accese la placă. `gpu_sim::fw_stick`: fișiere de probă scrise independent (NU NVIDIA). Commit `2c9c90c` | PASS (SIMULARE): 7 teste; 3 greșeli plantate (fără SHA, fără mărime, antet necitit) prinse; fișierele de probă nu trec niciodată drept cele reale |
| **C-36 (1c)** I3b pe ecran | `FwsecLocation::lines()`: 5 rânduri (imaginile FWSEC, tabela PMU, descriptorul, semnăturile, `criteriul I3b`); doar poziții și mărimi, ucode-ul necitit | PASS (SIMULARE, lanțul de tip ASUS) |
| OS | `integration/cyber-universe/C-36_gpu-firmware-fwsec.md` (schița `gpu firmware [--sim]` + `gpu fwsec [--sim]`); compilată într-o copie de lucru a OS-ului (repo-ul OS neatins) cu cyber-gpu local: build normal și `test-gpu-sim` PASS; în hub, comanda-propunere `idei/2026-10-28_cyber-universe_gpu-firmware-fwsec.md` | PASS (compilare); QEMU: NOT TESTED |
| **C-36 (1–3)** pașii ferestrei | `docs/FEREASTRA_C-36.md`: 0 pregătire (sumele pe Windows), 1 I3 (`gpu vbios`, `--detalii`, `gpu fwsec`), 2 I4 (`gpu firmware` sau varianta fără comandă), 3 doar dacă 1+2 trec: proba tinygrad (**scrie în placă**, oprire sigură descrisă); pornirea cu driverul nostru NU în fereastra asta | gata (document) |
| Compatibilitate OS | `git diff 8313973 HEAD -- crates/*/src`: nicio linie `pub` ștearsă; doar adăugări | PASS |
| CI GitHub | nu pornește (repo privat, fără minute; sesiunea 14) | BLOCKED (nu e o eroare de cod) |

Verificări (`tools/check.sh`): format PASS, clippy `-D warnings` PASS, **251 de teste PASS** (1 ignorat: viteza), build no_std PASS;
`nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fereastra C-36 (29 oct, cu Adrian): pașii 1 → 2 → (3). Sesiunea OS: comanda-propunere din `idei/` (dacă e aprobată).
Fără placă: treapta 6 matmul (Mesa NAK pe gazdă); `SET_PAGE_DIRECTORY`.

---

## Sesiunea 14 — 28 octombrie 2026, ~19:30–19:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Comenzi noi în hub: **C-33** (licența) și **C-34** (copia din hub). Codul driverului
nu s-a schimbat (doar licență, metadate și documente).

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | `tools/check.sh` verde înainte de orice schimbare (243 teste, rustc 1.99.1); hub-ul avea istoria rescrisă pe GitHub → clona locală mutată pe `origin/main` (nimic de-al nostru pierdut); clona cyber-gpu era „shallow” → completată | PASS |
| **C-33** licența | `LICENSE` proprietar („toate drepturile rezervate”) cu nota: versiunile până la `8313973` inclusiv au fost MIT; fișierele generate din antetele NVIDIA rămân MIT NVIDIA (`crates/nvidia-rm570/LICENSE-NVIDIA`); firmware-ul NVIDIA nu e distribuit de noi. `Cargo.toml`: `license = "MIT"` → `license-file = "LICENSE"`. Actualizate: `README.md`, `PROVENANCE.md`, `RESEARCH_I4_I8.md` §5.1 (argumentul „OS cu licență OSI” nu mai ține → Adrian descarcă singur firmware-ul; varianta initrd exclusă), `nvidia-rm570/README.md`, `tools/nak/README.md`, `integration/cyber-universe/UTILIZARE.md`. Commit `f440f6f` | PASS (local: format, clippy, 243 teste, build no_std, `hw-write`) |
| CI GitHub | rularea 57 (commit `f440f6f`): ambele joburi „failure” în 3 secunde, fără jurnal (404) = nu au pornit deloc. Cauza probabilă: repo-ul e **privat** din 28 oct 14:33 și GitHub Actions nu mai are minute gratuite (la fel la OS, C-32). Conform C-32, nu se repară prin plată; ne bazăm pe testele din sesiune | BLOCKED (nu e o eroare de cod) |
| **C-34** copia din hub | `git bundle create` pe `g0-g2-fundatie`, `git bundle verify` OK, clonare de probă din bundle + `tools/check.sh` pe clonă: verde; fără firmware (doar cod/documente cu „firmware” în nume) | PASS |
| Treapta 6 matmul | instalarea uneltelor pentru Mesa NAK (apt/pip/cargo + construirea Mesa) a fost **refuzată de sistemul de permisiuni** al sesiunii programate | NOT TESTED (amânat) |
| Compatibilitate OS | `git diff 8313973 HEAD -- crates/*/src`: gol (nicio linie de cod schimbată) | PASS |

**Următorul pas:** fereastra pe workstation cu Release-ul OS care conține C-31 (`gpu vbios` + `gpu vbios --detalii`): dacă I3 trece, I3b
(`fwsec::locate`) și fișierele de pe stick (I4). Fără placă: treapta 6 matmul (cere construirea Mesa NAK, ~15 min, dacă permisiunile o lasă);
`SET_PAGE_DIRECTORY`. **De la Adrian:** fereastra pe workstation; fișierele firmware pe stick; decizia lui pentru CI (repo privat = fără teste automate GitHub).

---

## Sesiunea 13 — 28 octombrie 2026, ~13:35–13:50 (ora României), în cloud, FĂRĂ placă

**Prima dată cu date de pe placa reală** (fereastra lui Adrian de azi, 10:00–10:35, os-195, hub `rezultate/2026-10-28-workstation-kronos-in-os/`):
**I0, I1, I2 = PASS pe RTX 3070 Ti** (`BOOT_0 = b74000a1`, `BOOT_42 = 174a1000`, GFW `0xff` în 3 µs, 0 scrieri); **I3 = FAIL**: la `0x24600`
parserul aștepta `0xAA55` și a citit `0x4E56` („NV”). Comanda nouă din hub: **C-30**. Tot ce scrie mai jos „PASS” e pe gazdă: SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 50 verde; `tools/check.sh` verde înainte de orice schimbare (229 teste, rustc 1.99.1); C-30 preluat în hub | PASS |
| **C-30 (1)** diagnostic | `nvidia-ga104::vbios_diag::diagnose` + `VbiosDiag::lines()`: pentru fiecare imagine offset, semnătură, PCIR (vendor, device, clasă, tip, lungime, blocuri, indicator), NPDE; la rupturi primii 16 octeți + căutare înainte (pas 512 B) și reluare; semnăturile de după ultima imagine; tabela BIT; rezumat cu „0 scrieri”. Fără conținutul imaginilor. Limitat: ≤ 16 imagini, ≤ 4 rupturi, ≤ 16 semnături (sub 200 de rânduri) | PASS (SIMULARE): 9 teste (lanț curat, cazul workstation, ruptură + reluare, PCIR lipsă, mărime 0, semnături după „ultima”, nicio semnătură, IFR v3, lanț lung, placă absentă); 0 scrieri, doar citiri în fereastra PROM |
| **C-30 (2)** parserul | `vbios::read_layout` acceptă ca nouveau `image.c`/`pcir.c` (MIT) semnăturile de imagine `AA55`/`BB77`/`4E56` și PCIR „PCIR”/„RGIS”/„NPDS” (nova-core la fel, citit doar ca specificație); parserul și diagnosticul folosesc **aceeași** funcție de citire a unei imagini. `gpu-sim::rom::asus_like_chain` reproduce placa (imaginile „NV” la `0x24600`; restul mărimilor = presupuneri SIM) | PASS (SIMULARE): lanțul trece I3, FWSEC găsit în imaginile „NV”; o semnătură necunoscută tot refuzată; „NPDS”/„RGIS” acceptate, altele nu. **Greșeală plantată** (doar `AA55` acceptat) → 3 teste pică. Pe placă: **NOT TESTED** — de confirmat la fereastra următoare |
| Ecranul | raportul OS: framebuffer la `0xe0000000` = începutul BAR1 → `screen_vram_range` dă VRAM `0x0–0x870000` (protejată la I9b); test cu valorile reale | PASS (aritmetică) |
| OS (C-31) | nota `integration/cyber-universe/C-31_gpu-vbios-detalii.md`; OS-ul construit (copie de lucru, repo-ul OS neatins) cu fixarea mutată pe `4538f90`: build normal și `test-gpu-sim` PASS | PASS (compilare) |
| CI | rulările 51, 52, 54 verzi; **53 roșie** (format uitat la commit-ul cu docs, reparat imediat în 54) | PASS la final |
| Compatibilitate OS | `git diff 41e44ba HEAD`: tot singura linie `pub` = `GSP_VERSION` (sesiunea 5). Nou: `vbios_diag`; în `gpu-sim::rom`: `set_signature`, `asus_like_chain(_with_fwsec)`, `ASUS_LIKE_OFFSETS` | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **243 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** sesiunea OS face C-31 (`gpu vbios --detalii`, fixare pe `4538f90` sau mai nou). La fereastra următoare: `gpu vbios` +
`gpu vbios --detalii` (poze) — dacă I3 trece, urmează I3b (`fwsec::locate`, încă neafișat în OS) și fișierele de pe stick (I4).
Fără placă: treapta 6 matmul; `SET_PAGE_DIRECTORY`.
**De la Adrian:** fereastra pe workstation cu Release-ul care conține C-31; fișierele firmware pe stick.

---

## Sesiunea 12 — 28 octombrie 2026, ~07:15–07:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software,
imagini sintetice — niciun fișier firmware NVIDIA real). Comenzi noi în hub: niciuna (C-18 și C-20 gata); am continuat pașii „fără placă” din sesiunea 11.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea de la sesiunea 11 verde; `tools/check.sh` verde înainte de orice schimbare (220 teste, rustc 1.99.1) | PASS |
| **I6/I7** încărcătorul DMA | `nvidia-ga104::gsp::hs_load`: `HsUcode` (descriptorii FWSEC v3 și Booter „BOOT_FROM_HS”, cu verificările RM + cele cerute de blocurile DMA de 256 B), `fwsec_image` / `booter_image` (semnătura aleasă copiată la locul ei; Booter: numărată de la coadă, ca RM), `dma_load` (FBIF_CTL, DMACTL=0, TRANSCFG = RAM coerent fizic, blocuri `DMATRF*` cu așteptare FULL/IDLE, IMEM cu SEC=1, apoi BROM: semnătura, motorul, ucode-ul, RSA-3K, și BOOTVEC). **Toate verificările înainte de prima scriere.** `falcon::{fwsec_frts_dma, booter_load_dma}` îl folosesc (funcțiile vechi rămân). Sursa: RM 570.144 (MIT) `kernel_gsp_falcon_ga102.c`, `kernel_gsp_fwsec.c`, `kernel_gsp_booter.c` | PASS (SIMULARE): 6 teste; în `gpu-sim::fake_gsp` am adăugat un motor DMA și un „boot ROM” simulate (acceptă programul doar dacă octeții, blocurile sigure, registrele BROM și BOOTVEC sunt exact cum trebuie): FWSEC → Booter cap-coadă prin DMA, RISC-V pornit; semnătura pentru altă siguranță și 4 valori BROM greșite refuzate; 5 intrări greșite = 0 scrieri; coadă plină / DMA blocat = limită de timp; RAM negătit = eroare. **7 greșeli plantate în încărcător, toate prinse.** **DEDUS:** `meta_data` din fișierul Booter = `patchMeta` din RM; alinierea 256 B |
| **I6** comanda FRTS | `hs_load::{frts_command, patch_fwsec_command, fwsec_frts_image}`: FWSEC primește comanda 0x15 (FRTS) în „DMEM mapper”-ul din tabela lui de interfață, cu regiunea de 1 MiB din `fb_layout` (RM `kernel_gsp_frts_tu102.c`). Mai strict decât RM: buffer prea mic sau suprapus peste semnătură = refuzat. Se ating doar tabela de interfață, `init_cmd` și bufferul de intrare (interfața documentată de NVIDIA); codul FWSEC rămâne opac | PASS (SIMULARE): 3 teste; FWSEC-ul simulat face WPR2 exact unde cere comanda; comanda pentru altă regiune = `VerifyFailed`; fără comandă = cod de eroare; 7 tabele stricate refuzate; **5 greșeli plantate în comandă prinse** |
| Constatare | RM 570.144 pe GA102 cere ca imaginile FWSEC/Booter să fie în **RAM** (nu în VRAM, cum face tinygrad) → pe placă e nevoie de Bus Master (I5g, nescris intenționat) și de o zonă de RAM pusă deoparte de OS. Versiunea din siguranțe (care semnătură) nu e citită încă: e parametru | notate în `COMPATIBILITATE_FIRMWARE.md` și `RISKS.md` |
| CI | rularea pentru încărcător; a doua (comanda FRTS) verificată la final | PASS / la final |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot singura linie `pub` = valoarea `GSP_VERSION` (sesiunea 5). Nou: `gsp::hs_load`, `falcon::{fwsec_frts_dma, booter_load_dma}`; în `gpu-sim`: variante noi în `SimFault` (model de test) | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **229 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fără placă au mai rămas puține: treapta 6 matmul (încărcări vectoriale); `SET_PAGE_DIRECTORY` pentru canal; citirea
versiunii din siguranțe (doar citire, după B0). Restul I0–I10 cere placa. Cu placa (B0, os-192): I0–I3 + FWSEC + fișierele de pe stick +
rândul „ecran:”, 0 scrieri; apoi, cu Adrian: proba P (tinygrad).
**De la Adrian:** fereastra B0 (lista în `TEST_MATRIX.md`); fișierele firmware pe stick.

---

## Sesiunea 11 — 28 octombrie 2026, ~01:25–01:45 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software)
sau comparație cu antetele NVIDIA. Comenzi noi în hub: niciuna (C-18 și C-20 gata); am continuat pașii „fără placă” din sesiunea 10.
Din hub: C-26 e gata în OS (os-192 afișează în `hw` rândul „ecran: framebuffer la adresa fizica …”) — folosit mai jos.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 42 verde; `tools/check.sh` verde înainte de orice schimbare (211 teste, rustc 1.99.1) | PASS |
| **I9b** registrele | generatorul `tools/gen-rm570` produce și `tu102_dev_ram` (`NV_PRAMIN` = `0x700000–0x7fffff`), `gm107_dev_bus` (`NV_PBUS_BAR0_WINDOW` = `0x1700`; publicat doar în antetul Maxwell, dar RM 570.144 îl folosește pe GA104: `kbusSetBAR0WindowVidOffset_GM107`), `tu102_dev_vm` (invalidarea TLB, baza PF `0xb80000`); restul `regs.rs` a ieșit **identic** | PASS |
| **I9b** fereastra VRAM + TLB | `nvidia-ga104::vram_window::VramWindow` (implementează `mmu::VramWrite`): mută fereastra de 1 MiB în pași de 64 KiB, scrie VRAM, **recitește fiecare cuvânt**, refuză zonele protejate (ecranul, GSP-RM) înainte de orice acces; `invalidate_tlb` (PDB = rădăcina, ALL_VA, așteptare până la TRIGGER = 0, limită de timp); `restore` pune fereastra la loc. Scrie **doar** 4 registre, fiecare în jurnal cu înainte/după; pe placa reală cere `hw-write`. Model independent: `gpu-sim::pramin` (pozițiile scrise de mână, 3 defecte) | PASS (SIMULARE): 8 teste noi; tabelele I9a scrise prin fereastră și citite de cititorul independent; „golden image” cap-coadă prin fereastră (10 buffere GR verificate, ecranul și zona GSP-RM neatinse, o invalidare după toate scrierile). 3 greșeli plantate (jumătățile inversate, împărțirea PDB greșită, fără recitire) **prinse**. **DEDUS:** valoarea invalidării; că fereastra merge după UEFI |
| Ecranul (C-26) | `vram_window::screen_vram_range`: rândul „ecran:” din `hw` (os-192) + BAR1 → zona din VRAM a ecranului, rotunjită la 64 KiB; `TEST_MATRIX.md` B0 folosește os-192 și pasul 9 cere poza acestui rând | PASS (aritmetică). **DEDUS:** BAR1 = vedere liniară a VRAM-ului după UEFI |
| CI | rulările 43–45 | 43, 44 PASS; 45 verificată la final |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot singura linie `pub` = valoarea `GSP_VERSION` (sesiunea 5). Nou: `vram_window`, `GpuMemory::writer_mut` | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **220 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fără placă: încărcătorul DMA al ucode-ului (I6/I7, în spatele `hw-write`); treapta 6 matmul (încărcări vectoriale);
`SET_PAGE_DIRECTORY` pentru canal. Cu placa (B0, os-192): I0–I3 + FWSEC + fișierele de pe stick + rândul „ecran:”, 0 scrieri;
apoi, cu Adrian: proba P (tinygrad). Pașii „fără placă” se împuținează: după încărcătorul DMA, restul I0–I10 cere placa.
**De la Adrian:** fereastra B0 (lista în `TEST_MATRIX.md`); fișierele firmware pe stick.

---

## Sesiunea 10 — 27 octombrie 2026, ~19:05–19:55 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software),
compilare sau comparație cu antetele NVIDIA. Comenzi noi în hub: niciuna (C-18 și C-20 gata); am continuat pașii „fără placă” din sesiunea 9.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 38 verde; `tools/check.sh` verde înainte de orice schimbare (rustc 1.99.1); Mesa 25.0.7 reconstruit: `add64`, `matmul_smem`, `matmul_1d` **identice** (aceleași sume) | PASS |
| **I9a** formatele | generatorul `tools/gen-rm570` produce și `nvidia-rm570::regs::tu102_dev_mmu_ver2`: câmpurile PDE / PDE dublu / PTE ale MMU v2 din `dev_mmu.h` (NVIDIA 570.144, MIT; copia completă e cea din `kernel-open/nvidia-uvm/hwref`, cea din `swref/published` e un ciot), tipul nou `EntryField` (câmpuri până la bitul 127, valorile prea mari refuzate, nu tăiate); restul `regs.rs` a ieșit **identic** | PASS (test nou: pozițiile = citirea de mână) |
| **I9a** tabelele de pagini | `nvidia-ga104::mmu::GpuMemory`: VRAM și adresele GPU prin `gpu-memory::RangeAllocator` (fără suprapuneri), tabelele PD3→PD2→PD1→PD0 (dublu)→PT (pagini de 4 KiB) după nivelurile NVIDIA (`kern_gmmu_fmt_gp10x.c`), fiecare tabelă **golită înainte de legare**, mapare dublă refuzată fără nicio scriere; implementează `GoldenMemory` (înlocuiește alocatorul de test al sesiunii 9). **Verificare independentă:** `gpu-sim::vram` (VRAM cu „otravă” la octeții nescriși + un cititor al tabelelor cu pozițiile scrise de mână) | PASS (SIMULARE): 10 teste noi; „golden image” cap-coadă prin cozi cu tabelele reale: toate cele 10 buffere GR (3,3 MiB) verificate pagină cu pagină; nicio tabelă în interiorul unui buffer; o greșeală plantată într-un PTE e prinsă. **DEDUS** (tinygrad): `NO_ATS=1`, `KIND=6`, fereastra VA de la 64 GiB |
| **E** treapta 5 matmul | `matmul_2d.comp` (siboehm „2D blocktiling”: 64 de fire pe grup, 8×8 rezultate pe fir): 1 953 instrucțiuni, 31 248 B, 94 registre, 4 KiB partajată, o barieră, **0 memorie locală**; `cmdproc::matmul_2d_on_cpu` urmează algoritmul. **Constatare:** fără `[[unroll]]` NAK ține sumele în memoria locală a firului (256 B), pe care lansatorul nostru nu o pregătește | PASS (SIMULARE): 5 forme **identice bit cu bit** cu `matmul_fma_ftz`; cazul −0,0; lansări greșite refuzate, 0 scrieri în C; o greșeală plantată prinsă |
| Constatări | tabelul din `COMPATIBILITATE_FIRMWARE.md` („De verificat”): mărimea rădăcinii trimise lui RM (32 B la noi, 16 B la tinygrad); unde e imaginea ecranului UEFI în VRAM (de aflat la B0); invalidarea TLB după scrierea tabelelor (I9b, nescrisă) | notate |
| CI | rulările 39 (I9a) și 40 (treapta 5) | 39 PASS; 40 verificată la final |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot singura linie `pub` = valoarea `GSP_VERSION` (sesiunea 5). Nou: modulul `mmu`; `nvidia-ga104` depinde acum și de `gpu-memory` (din același repo) | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **211 teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fără placă: încărcătorul DMA al ucode-ului (I6/I7, în spatele `hw-write`); I9b = scrierea VRAM printr-o fereastră BAR +
invalidarea TLB (în spatele `hw-write`) și `SET_PAGE_DIRECTORY`; treapta 6 matmul (încărcări vectoriale). Cu placa (B0): I0–I3 + FWSEC + fișierele
de pe stick, 0 scrieri, plus locul imaginii ecranului în VRAM; apoi, cu Adrian: proba P (tinygrad).
**De la Adrian:** fereastra B0 (lista în `TEST_MATRIX.md`); fișierele firmware pe stick.

---

## Sesiunea 9 — 27 octombrie 2026, ~13:05–13:55 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software),
compilare sau comparație cu antetele NVIDIA. Comenzi noi în hub: niciuna (C-18 și C-20 gata); am continuat pașii „fără placă” notați în sesiunea 8.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 35 verde; `tools/check.sh` verde înainte de orice schimbare (rustc 1.99.1); Mesa 25.0.7 reconstruit: `add64`, `matmul_naive`, `matmul_smem` **identice** (aceleași sume) | PASS |
| **I8e** structurile | generatorul `tools/gen-rm570` produce `nvidia-rm570::rmapi` din antetele NVIDIA 570.144 (MIT): parametrii pentru client/dispozitiv/subdispozitiv/spațiu de adrese/canal, tabela motoarelor, PDE rezervate, bufferele de context GR, PROMOTE_CTX, numerele claselor și comenzilor; antetele `rpc_gsp_rm_alloc/control_v03_00` (tablou flexibil) → poziții calculate de clang, emise ca constante (fără `unsafe`). `regs.rs`, `structs.rs`, `sysinfo.rs` au ieșit **identice** la regenerare | PASS (verificările bindgen la compilare + `rm_api_layouts_570_144`) |
| **I8e** „golden image” | `gsp::rmapi` (mesajele `GSP_RM_ALLOC`/`GSP_RM_CONTROL`, toate pozițiile prin `offset_of!` pe structurile generate) + `gsp::golden` (`init_golden_image`, ordinea după tinygrad `ip.py`: client → dispozitiv → subdispozitiv → spațiu de adrese → tabela motoarelor → PDE rezervate 512 MiB → canal 32 intrări GR0 → mărimile bufferelor GR → 10 buffere, 9 promovate → compute + copiere); `QueueTransport` (cozile reale, evenimentele GSP păstrate deoparte, limită de timp). **Verificare independentă:** `gpu-sim::fake_rm`, cu pozițiile scrise de mână din antete, care cere părinți corecți, mărimi 570.144, VA space existent, și refuză obiectul compute fără context promovat | PASS (SIMULARE): 10 teste noi; secvența cap-coadă prin cozi: 7 alocări + 4 comenzi acceptate, alocările VRAM nu se suprapun; refuzurile RM întorc codul RM (niciodată succes); GSP tăcut = limită de timp. Două greșeli plantate (un buffer nepromovat, `hVASpace` pus în alt câmp) **sunt prinse**. **DEDUS:** valorile empirice tinygrad (`flags=0x200320`, `internalFlags=0x1a`, `cid=3`, clientul `0xc1e00004`) |
| Constatări | tinygrad trimite `bIsPassthru=1` în `GspSystemInfo` (noi 0); la clientul RM, `hObject` diferă între tinygrad și nouveau | notate în `COMPATIBILITATE_FIRMWARE.md` („De verificat”), de lămurit cu jurnalul probei P |
| **E** treapta 4 matmul | `matmul_1d.comp` (siboehm „1D blocktiling”: 512 fire pe grup, 8 rezultate pe fir, plăci 64×8 și 8×64): 478 instrucțiuni, 7 648 B, 24 registre, 4 KiB partajată, o barieră, `ffma.ftz`, determinist; `cmdproc::matmul_1d_on_cpu` urmează algoritmul | PASS (SIMULARE): 4 forme **identice bit cu bit** cu `matmul_fma_ftz`; cazul −0,0; lansări greșite refuzate, 0 scrieri în C; două greșeli plantate prinse |
| CI | rulările 36 (I8e) și 37 (treapta 4) | PASS |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot singura linie `pub` = valoarea `GSP_VERSION` (sesiunea 5) | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **196 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fără placă: încărcătorul DMA al ucode-ului (I6/I7, în spatele `hw-write`); alocatorul VRAM + tabelele de pagini Ampere
(I9, care vor implementa `GoldenMemory`); treapta 5 matmul (2D blocktiling). Cu placa (B0): I0–I3 + FWSEC + fișierele de pe stick, 0 scrieri;
apoi, cu Adrian: proba P (tinygrad), al cărei jurnal se compară cu GSP-ul fals, cu RM-ul fals (golden image) și cu câmpurile QMD.
**De la Adrian:** fereastra B0 (lista în `TEST_MATRIX.md`); fișierele firmware pe stick.

---

## Sesiunea 8 — 27 octombrie 2026, ~08:05–08:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software),
compilare sau comparație cu antetele NVIDIA. Comenzi noi în hub: niciuna (C-18 și C-20 erau gata); am continuat pașii „fără placă” notați în sesiunea 7.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| Pornire | CI rularea 32 verde; `tools/check.sh` verde înainte de orice schimbare; rustc actualizat la 1.99.1 | PASS |
| **I8c'** `GspSystemInfo` | generatorul `tools/gen-rm570` produce și `nvidia-rm570::sysinfo` (`gsp_static_config.h`@570.144, MIT, 928 B, cu verificările bindgen); `regs.rs`/`structs.rs` au ieșit **identice** la regenerare. `nvidia-ga104::gsp::sysinfo::encode`: BAR0/BAR1/BAR3, `domain<<32 \| bus<<8 \| slot`, ID-urile PCI ca dword-urile citite, oglinda PCFG 0x88000/0x1000, pagina 4 KiB, limita VA; ACPI/SR-IOV/hypervisor = 0 („absent”); intrările imposibile (alt producător, BAR0 lipsă/nealiniat…) refuzate | PASS (SIMULARE): 3 teste + verificarea față de antet; pornirea simulată cap-coadă trimite acum conținutul real `SET_SYSTEM_INFO` (plăcuța workstation-ului) și tabela reală `SET_REGISTRY`. **DEDUS:** că GSP-RM acceptă zerourile pe placa reală |
| **E** treapta 3 matmul | Mesa 25.0.7 reconstruit (add64 și coalesced: aceleași sume); `matmul_smem.comp` (plăci 16×16 în memoria partajată, o barieră): 218 instrucțiuni, 3 488 B, 19 registre, 2 KiB partajată, `ffma.ftz`; `qmd::encode_with_shared` (câmpurile pentru memoria partajată + bariere, după Mesa NAK `qmd.rs`); `gpu-sim::cmdproc::register_program_with_needs` | PASS (SIMULARE): 4 forme (inclusiv plăci parțiale) **identice bit cu bit** cu `matmul_fma_ftz`; cazul −0,0 prinde o variantă care ar aduna zerouri (verificat: varianta greșită pică); QMD fără memorie partajată/barieră = refuzat, 0 scrieri în C |
| Constatare | Mesa NAK pune mereu în QMD trei câmpuri pe care noi nu le punem (`API_VISIBLE_CALL_LIMIT`, `SAMPLER_INDEX`, `SM_GLOBAL_CACHING_ENABLE`) | notat în `COMPATIBILITATE_FIRMWARE.md` („De verificat”), de aliniat cu jurnalul probei P |
| CI | rulările 33 (I8c') și 34 (treapta 3) | 33 PASS; 34 verificată la final |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot singura linie `pub` = valoarea `GSP_VERSION` (sesiunea 5) | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **184 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**Următorul pas:** fără placă: treapta 4 matmul (mai multe rezultate pe fir, 1D blocktiling); „golden image” (I8e, după tinygrad, în SIMULARE
pe GSP-ul fals); încărcătorul DMA al ucode-ului (I6/I7, în spatele `hw-write`). Cu placa (B0): I0–I3 + FWSEC + fișierele de pe stick, 0 scrieri;
apoi, cu Adrian: proba P (tinygrad), care dă jurnalul de comparat cu GSP-ul fals (și cu câmpurile QMD de mai sus).
**De la Adrian:** fereastra B0 (lista în `TEST_MATRIX.md`); fișierele firmware pe stick.

---

## Sesiunea 7 — 27 octombrie 2026, ~01:25–01:45 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software),
aritmetică sau compilare. Comanda lucrată: **C-20**, pașii rămași din sesiunea 6: D partea 2, I4c, E.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| C-20 reluat | hub `COMENZI.md` C-20 → „reluat 01:28, sesiunea 7” | gata |
| B (completare) | generatorul `tools/gen-rm570` + `dev_bus.h` (`NV_PBUS_SW_SCRATCH`), `rmgspseq.h` (codurile secvențiatorului), `rmRiscvUcode.h` (`RM_RISCV_UCODE_DESC`, 84 B), toate NVIDIA MIT 570.144; eticheta sursei din `regs.rs` corectată (scria „open-gpu-doc” la antetele din open-gpu-kernel-modules) | PASS (verificările bindgen la compilare) |
| **D partea 2a** GSP fals pe registre | `gpu-sim::fake_gsp`: falcon-ul GSP + SEC2 (reset, „scrubbing”, nucleu falcon/RISC-V, start/oprire, mailbox), WPR2 scris după FWSEC, Booter care pornește RISC-V-ul, soneria; 5 defecte de simulat. `nvidia-ga104::gsp::falcon`: I6 (`fwsec_frts`) și I7 (`booter_load`) pe registre, prin `GuardedBar0` (scrieri **doar** în ferestrele GSP/SEC2, fiecare în jurnal cu valoarea înainte/după; pe placa reală cere `HardwareWritePermit` = opțiunea `hw-write`) | PASS (SIMULARE): 5 teste — secvența întreagă; fiecare defect raportat (nu sărit); ordine greșită, regiune FRTS greșită, FWSEC de două ori, încărcătorul lipsă = eroare, niciodată succes fictiv; BAR „real” fără permis = refuzat, 0 scrieri |
| **D partea 2b** secvențiator + INIT_DONE | `gsp::sequencer`: cele 9 comenzi (din antet), executate doar prin `GuardedBar0`; `wait_init_done` (sare peste evenimentele străine, rulează cererile secvențiatorului, limită de timp) | PASS (SIMULARE): 4 teste — **pornire simulată cap-coadă: I6 → I7 → cozi → SET_SYSTEM_INFO/SET_REGISTRY + sonerie → cerere de secvențiator (6 comenzi, inclusiv CORE_RESET/CORE_RESUME) → GSP_INIT_DONE**; scrieri cerute în afara GSP/SEC2 refuzate; GSP tăcut = limită de timp |
| **I4c** antetele fișierelor | `nvidia-ga104::fw_hdr`: `nvfw_bin_hdr` (0x10de), descriptorul RISC-V (generat), „HS v2” al Booter-ului (semnături, `patch_loc`/`patch_sig`, aplicații); totul verificat să stea în fișier; conținutul rămâne opac | PASS (SIMULARE): 3 teste pe fișiere sintetice; pe fișierele reale: NOT TESTED |
| **E** oracolul FMA | `gpu-compute::math::fma_f32` (o rotunjire, no_std) — **identic bit cu bit cu `std` pe 2 000 000 de cazuri** (testul prinde o implementare greșită); `ftz`, `reference::add_ftz`, `matmul_fma`, `matmul_fma_ftz` | PASS (gazdă) |
| **E** NAK + matmul | Mesa 25.0.7 reconstruit în cloud (`add64` iese **identic**, aceeași sumă); `build.sh kernel <nume>`; `matmul_naive.comp` (treapta 1) și `matmul_coalesced.comp` (treapta 2): fiecare 51 de instrucțiuni, 816 B, 19 registre. **Constatare:** NAK folosește `ffma.ftz` / `fadd.ftz` (subnormalele aduse la 0) → oracolele programelor GPU sunt variantele `_ftz`; baza grupurilor stă în cbuf0 0x0 (trebuie 0) | PASS (SIMULARE): treptele 1 și 2 lansate prin pushbuffer + QMD în procesorul de comenzi simulat, **identice bit cu bit** cu `matmul_fma_ftz`; nimic scris după C; grilă prea mică = C parțial; program schimbat = refuzat. Programul „rulează” pe procesor, nu pe placă |
| CI | rulările 28–31 | verificate la final |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot singura linie `pub` = valoarea `GSP_VERSION` (sesiunea 5). Nou: `Mmio::is_simulation` (metodă cu valoare implicită, compatibilă) | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **176 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; `nvidia-ga104 --features hw-write`: clippy + teste PASS. Nicio scriere în placă.

**C-20:** B, C, D, E (treptele 1–2) gata în SIMULARE; A rămâne doar pregătită (se rulează doar cu Adrian de față).
**Următorul pas:** fără placă: treapta 3 matmul (memorie partajată) și 2D blocktiling; încărcătorul DMA al ucode-ului (I6/I7, în spatele
`hw-write`); `GspSystemInfo` (I8c'); „golden image” (I8e). Cu placa (B0, 27–28 oct): I0–I3 + FWSEC + fișierele de pe stick, 0 scrieri;
apoi, cu Adrian: proba P (tinygrad), care dă jurnalul de comparat cu GSP-ul fals.
**De la Adrian:** fereastra B0 (lista în `TEST_MATRIX.md`).

---

## Sesiunea 6 — 26 octombrie 2026, ~19:25–19:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (modele software),
aritmetică sau comparație cu antetele NVIDIA. Comanda lucrată: **C-20** (hub), în ordinea B → C → A (pregătire) → D. C-18 era deja gata.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| C-20 preluat | hub `COMENZI.md` C-20 → `preluat (19:27)` | gata |
| **B** antete generate | crate nou `nvidia-rm570` (doar date, no_std): **1 419 valori** generate din antetele MIT NVIDIA (open-gpu-kernel-modules **570.144**, commit fixat; open-gpu-doc, commit fixat): registrele GA102 falcon/GSP/SEC2/RISC-V/FBIF/GC6, `WPR2_ADDR_*`, metodele C56F/C7C0, QMD V03_00; structurile `GspFwWprMeta`, `msgqTxHeader`, `GSP_ARGUMENTS_CACHED` prin bindgen-cli 0.71.1 (cu verificări de mărime/poziție la compilare). Unealta: `tools/gen-rm570/` (doar pe gazdă). Notificarea NVIDIA în `LICENSE-NVIDIA` | PASS: 7 teste + verificările bindgen |
| B: verificarea codului vechi | offset-urile scrise de mână în sesiunea 5 (WPR meta, cozi, argumentele GSP, GFW_BOOT, pushbuffer/GPFIFO) comparate automat cu antetele | **PASS: toate egale** (`tests/rm570_crosscheck.rs`, 5 teste) |
| **C** RPC mari | „continuation records” (funcția 71) după `rpc.c` 570.144: `rpc::encode_split` / `join_split` | PASS (SIMULARE): 1 test (20 de elemente → 16 + continuare; ordine/secvență greșită refuzată) |
| C: SET_REGISTRY | `gsp::registry` (`PACKED_REGISTRY_TABLE`): cele 2 intrări folosite pe GA102 (`RMForcePcieConfigSave=1`, `RMSecBusResetEnable=1`), 82 de octeți | PASS (SIMULARE): 3 teste |
| C: „golden image” | pas nou I8e notat în `COMPATIBILITATE_FIRMWARE.md`; nu e încă în `init.rs` (o variantă nouă în enumerarea publică ar putea strica OS-ul) | gata (documentat) |
| **A** pregătire | `tools/proba-tinygrad/proba.sh` (tinygrad `9cf7829a6` + tinymesa/NAK, jurnalul `wreg:` pe stick; se oprește fără `CONFIRM_ADRIAN=da`, fără placă, cu nouveau încărcat sau cu sume greșite); instrucțiuni pe minute în `TEST_MATRIX.md` partea **P**; parserul `gpu-sim::golden_log` | script: **NERULAT** (verificat doar că refuză fără confirmare); parser PASS (SIMULARE): 3 teste pe un jurnal scris de mână |
| **D** partea 1 | `gpu-sim::cmdproc`: procesorul de comenzi simulat (GPFIFO, pushbuffer, semafoare, `SET_OBJECT`, `SEND_PCAS_A` + `PCAS2_B`, QMD V03_00 citit cu valorile din antete); `nvidia-ga104::qmd` (construiește QMD-ul) | PASS (SIMULARE): `add64` (codul NAK real, 240 B) lansat cap-coadă, rezultatul **identic bit cu bit** cu `reference::add`, semaforul `release0` și cel de canal la valorile așteptate; lansări greșite refuzate. Programul „rulează” pe procesor, nu pe GPU |
| D partea 2 (GSP fals la nivel de registre) | — | NOT TESTED (nescris; sesiunea următoare) |
| E (NAK + matmul) | — | NOT TESTED (nescris; Mesa trebuie reconstruit în cloud; tratarea FMA de decis) |
| CI | rulările 22 (B) și următoarele | PASS la 22; restul verificate la final |
| Compatibilitate OS | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: tot o singură linie `pub` (valoarea `GSP_VERSION`, din sesiunea 5). **Nou:** `nvidia-ga104` depinde de `nvidia-rm570` (același repo, MIT) | PASS (doar adăugări) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **157 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; și `--features hw-write`: clippy PASS. Nicio scriere în placă, nicăieri.

**Următorul pas:** fără placă: D partea 2 (GSP fals pe registre: reset falcon, mailbox, WPR2 după FWSEC, INIT_DONE + o cerere de
secvențiator), apoi I4c (antetele `bootloader`/`booter_load`, structura `nvfw_bin_hdr` în generator), apoi E (Mesa NAK reconstruit, matmul naiv f32 +
oracol FMA). Cu placa (B0, 27–28 oct): I0–I3 + FWSEC + fișierele de pe stick; dacă rămâne timp și Adrian vrea: proba P (tinygrad).
**De la Adrian:** fereastra B0; pentru proba P: un al doilea stick cu Ubuntu 24.04 live și laptopul pentru SSH.

---

## Sesiunea 5 — 26 octombrie 2026, ~17:15–17:45 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Tot ce scrie „PASS” e pe calculatorul gazdă: SIMULARE (GSP simulat, imagini
sintetice), aritmetică sau compilare. Deciziile lui Adrian din 26 oct 17:10 (hub C-18): firmware NVIDIA **570.144**, separat pe stick; compilator **Mesa NAK**.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| a. C-18 preluat | hub `COMENZI.md` C-18 → `preluat`; deciziile notate în `RESEARCH_I4_I8.md` §5, `RISKS.md` (B3, B4, B5, B9 rezolvate; riscuri noi), `TOOLCHAIN.md`; `.gitignore` pentru `*.bin`/`firmware/` | gata |
| b. Tabelul de compatibilitate | `docs/COMPATIBILITATE_FIRMWARE.md`: 23 de rânduri I0–I10 (sursă, ABI 570.144, fișier Rust, test, stare) + cele 4 fișiere de pus pe stick, cu mărimi și SHA-256 | gata |
| c. I4 | `firmware::GSP_570_144` (mărimi + SHA-256 calculate din fișierele oficiale linux-firmware, apoi șterse), `load_gsp_set` (citire de pe stick prin `FirmwareSource`), `sha256` scris după FIPS 180-4; planul se oprește acum la **I5**; backend-ul refuză I4 fără firmware | PASS (SIMULARE): 5 + 3 teste (vectori NIST, un octet schimbat, fișier lipsă/trunchiat, manifestul real refuză orice alt conținut) |
| c. I5–I8 | `gsp::wpr_meta` (GspFwWprMeta 256 B), `gsp::libos` (regiunile LibOS, GSP_ARGUMENTS_CACHED 72 B), `gsp::msgq` (memoria partajată: tabela de pagini + 2 cozi × 63 elemente, SWAP_RX), `gsp::rpc` (element + antet RPC v3 + suma de control XOR, numerele funcțiilor 570.144, antetul secvențiatorului), `gsp::elf` (secțiunile `.fwimage`/`.fwsignature_ga10x`) | PASS (SIMULARE): 20 de teste, inclusiv un „GSP simulat” scris separat (pornire, INIT_DONE, 300 de mesaje cu ocolirea inelului, coadă plină, mesaje corupte/în afara ordinii) |
| c. Siguranță | nicio scriere în placă nicăieri; `gsp::HardwareWritePermit` există doar cu opțiunea de build `hw-write` (oprită implicit) | PASS (test cu și fără opțiune) |
| d. NAK pe gazdă | `tools/nak/`: Mesa 25.0.7 (NVK + NAK + drm-shim) construit în cloud; `add64.comp` compilat pentru GA104 (SM 8.6) fără placă: **15 instrucțiuni, 240 octeți** (`tools/nak/out/add64.sm86.sass`, determinist) | PASS (compilare); rularea pe placă: NOT TESTED (vine după I8–I10) |
| e. Fereastra pe workstation | `TEST_MATRIX.md` partea **B0**: pregătirea stick-ului (unde se pun fișierele, verificarea sumelor pe Windows), 10 pași, 0 scrieri în placă | gata |
| Integrare OS | `integration/cyber-universe/UTILIZARE.md` (pentru C-19: **nu există `main`**, fixarea pe un commit din `g0-g2-fundatie`), `I4_gpu-firmware.md` (propunere: comanda `gpu firmware`, doar citire) | gata (propunere) |
| CI | rularea 14 roșie: lint nou în clippy 1.99 (`chunks_exact` → `as_chunks`); reprodus local după `rustup update` (1.99.1), reparat | PASS: rulările 16–18 verzi |
| Compatibilitate | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: o singură linie `pub` „ștearsă” = **valoarea** `GSP_VERSION` schimbată din `"535.113.01"` în `"570.144"` (forma e aceeași; OS-ul nu o folosește) | PASS (doar adăugări + o valoare, notată) |

Verificări (`tools/check.sh`, rustc 1.99.1 stable): format PASS, clippy `-D warnings` PASS, **134 de teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS; și `--features hw-write`: clippy + teste PASS.

**Următorul pas:** fără placă: (1) antetele fișierelor `bootloader`/`booter_load` (offset-urile cod/date/manifest, după antetele MIT nouveau —
lipsesc pentru WPR meta, rândul I4c); (2) conținutul `GspSystemInfo` și al tabelei de registru (I8c); (3) registrele falcon GA10x pentru I6–I7,
scrise dar oprite. Cu placa (fereastra B0, 27–28 oct): I0–I3, FWSEC și verificarea fișierelor de pe stick.
**De la Adrian:** fereastra pe workstation și fișierele firmware pe stick (B0); urcarea tag-ului `stabil-2026-10-23` (tot 403 din cloud).

---

## Sesiunea 4 — 24 octombrie 2026, ~01:20–01:40 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Doar calcul pe procesorul gazdei.
C-16 (integrarea în OS) a fost făcut între timp de sesiunea OS, pe commit-ul nostru `41e44ba` (Release OS `os-189`).

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| 0. Tag-ul `stabil-2026-10-23` | există local, adnotat, pe `41e44ba`; push reîncercat o dată | **BLOCKED**: HTTP 403 (politica proxy-ului cloud: tag-urile nu se pot urca). OS-ul e fixat pe commit `41e44ba`, deci nu blochează nimic |
| Compatibilitate | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: nicio linie `pub` ștearsă | PASS (doar adăugări) |
| c. Backend CPU, mai multe nuclee | `gpu-compute::rows`: `split_rows`, `softmax_rows_range` (cauzal, cu rândul absolut), `causal_attention_rows`, `cross_entropy_terms` + `cross_entropy_mean`; fără stare globală, ieșiri disjuncte | PASS (gazdă): 4 teste, rezultatul unit **identic bit cu bit** cu `reference` pentru 1, 2, 3, 4, 7, 16, 40 de părți; pe nucleele OS-ului NOT TESTED |
| b. Sprijin integrare | `integration/cyber-universe/UTILIZARE.md`: fixarea actuală, promisiunea de compatibilitate, ce e nou după `41e44ba`, exemple (identificare doar-citire, SIMULARE cu configurația workstation-ului, operație pe n nuclee cu toleranța 0); README-ul `gpu-compute` explică ce se împarte prin felii și ce cere `rows` | gata |
| CI | rularea 11 (codul nou) verde | PASS |
| Placă reală | — | NOT TESTED (Adrian plecat) |

Verificări (`tools/check.sh`, rustc 1.97.0 stable): format PASS, clippy `-D warnings` PASS, **106 teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS. Nicio scriere nouă în registre.

**Următorul pas:** fără placă și fără decizia de licență **nu mai e nimic util de făcut** (pașii a–d sunt acoperiți; I4+ depind de
firmware-ul NVIDIA, adică de decizia lui Adrian). Cu placa: testul I0–I3 (+5c FWSEC) din `TEST_MATRIX.md` partea B.
**Blocate de Adrian:** licența firmware-ului, compilatorul (NAK vs ptxas), fereastra pe workstation, urcarea tag-ului.

---

## Sesiunea 3 — 23 octombrie 2026, ~19:25–19:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Testele pe imagini ROM sunt SIMULARE; restul e aritmetică pură.
Pașii a–e din prompt erau deja gata din sesiunea 2; am făcut pașii următori notați atunci.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| C-16 (hub): punct de revenire | tag-ul adnotat `stabil-2026-10-23` creat local pe `41e44ba` | **BLOCKED**: push-ul tag-ului refuzat (HTTP 403, politica proxy-ului sesiunii cloud); ramura se urcă normal. OS-ul poate fixa dependența pe commit-ul `41e44ba492c606ba5abf15126210caf1313b9e6e` (C-16 permite „tag / commit”) |
| CI | repo-ul public (decizia lui Adrian, 19:10) → GitHub Actions merge din nou | PASS: rulările 7 și 8 verzi |
| I3b: descriptorul FWSEC | `nvidia-ga104/src/fwsec.rs`: jetonul BIT 0x70 → tabela PMU → aplicația 0x85 (FWSEC_PROD) → `FalconUCodeDescV3`; unde stau semnăturile și ucode-ul (și dacă trece dintr-o imagine FWSEC în următoarea); **ucode-ul nu e citit niciodată** (test dedicat); alegerea semnăturii după siguranță (formulă DEDUS, nefolosită pe placă) | PASS (SIMULARE), 11 teste |
| I5 pregătire: regiunile din VRAM | `nvidia-ga104/src/fb_layout.rs`: VGA, FRTS, încărcător, ELF, heap WPR2 (LIBOS3), WPR2, heap 1 MiB; pentru 8 GiB GSP ia ultimii 162 MiB (cu mărimi SIM ale firmware-ului) | PASS (aritmetică), 6 teste |
| I5 pregătire: radix3 | `nvidia-ga104/src/radix3.rs`: lista de pagini prin care GSP își găsește imaginea în RAM; verificată prin parcurgere, octet cu octet | PASS (aritmetică), 3 teste |
| Placă reală | — | NOT TESTED (Adrian plecat) |

Verificări (`tools/check.sh`, rustc 1.97.0 stable, Linux x86_64): format PASS, clippy `-D warnings` PASS, **102 teste PASS** (1 ignorat:
măsurătoarea de viteză), build `x86_64-unknown-none` PASS. Nicio scriere nouă în registre; niciun registru nou pe lista permisă.

**Următorul pas:** fără placă și fără decizia de licență **nu mai e nimic util de făcut**: următorul (mesajele RPC GSP-RM) depinde de
versiunea firmware-ului NVIDIA, deci de decizia lui Adrian. Cu placa: testul I0–I3 din `TEST_MATRIX.md` partea B (acum și pasul 5c, FWSEC).
**Blocate de Adrian:** licența firmware-ului (I4 → tot ce urmează), compilatorul (NAK vs ptxas), fereastra pe workstation, tag-ul de revenire.

---

## Sesiunea 2 — 23 octombrie 2026, ~13:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Toate testele hardware sunt pe modelul `gpu-sim` = SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| a. Datele workstation-ului | `docs/HARDWARE_WORKSTATION.md`; `gpu-sim::workstation` (FAPT din K0) + `SimConfig::workstation_rtx3070ti()`; 2 teste noi | PASS (SIMULARE) |
| a'. Dependența de ReBAR / Above-4G | analiză: I0–I8 **nu** depind de ele; BAR1 256 MiB ajunge; VT-d oprit simplifică DMA, dar fără plasă | gata (DEDUS) |
| b. CI | mutat în `.github/workflows/ci.yml` — push acceptat; jobul `kernel-integration` scos (repo OS privat + patch depășit) | PASS: rulările 2–4 verzi pe GitHub; **rularea 5 (doar documente schimbate) BLOCKED**: joburile nu primesc calculator de la GitHub (0 pași, ~5 s, de 2 ori) — cauză probabilă: limita de minute/plată Actions pentru depozite private (neverificat) |
| c. I2 GFW_BOOT | `nvidia-ga104/src/gfw.rs`: 2 registre noi pe lista permisă (0x118128, 0x118234), limită 4 s, zero scrieri; 5 teste | PASS (SIMULARE) |
| c. I3 VBIOS | `nvidia-ga104/src/vbios.rs`: fereastra PROM (BAR0+0x300000), IFR/RFRD, lanț PCIR/NPDE, tabela BIT; limite 2 s / 300 000 citiri; 8 teste pe imagini sintetice | PASS (SIMULARE) |
| c. Planul | I2, I3 marcate implementate (doar citire); `initialize()` se oprește acum la **I4** (licența); plan de test pe placă în `TEST_MATRIX.md` | PASS (SIMULARE) |
| d. Backend CPU | `gpu-compute/src/kernels.rs`: matmul pe blocuri + matmul cu greutăți bf16 (op 9), **identice bit cu bit** cu referința; bf16 verificat exhaustiv (65 536 valori) și față de un oracol de rotunjire | PASS (gazdă) |
| e. Cercetare I4–I8 | `docs/RESEARCH_I4_I8.md` + argumentele pentru deciziile lui Adrian; fără cod care scrie în placă | gata |
| Placă reală | — | NOT TESTED (Adrian plecat; B1) |

Verificări (`tools/check.sh`, rustc 1.97.0 stable, Linux x86_64): format PASS, clippy `-D warnings` PASS, **82 de teste PASS** (1 ignorat:
măsurătoarea de viteză), build `x86_64-unknown-none` PASS.

**Următorul pas:** fără placă — extragerea descriptorului FWSEC din VBIOS (doar citire, pe imagine sintetică) și modelul aritmetic al
regiunilor WPR2/`GspFwWprMeta`; cu placa — testul I0–I3 din `TEST_MATRIX.md` (partea B), după ce OS-ul are comenzile `gpu gfw` / `gpu vbios`.
**Blocaj nou:** GitHub Actions nu mai pornește joburile (vezi rândul b) — *rezolvat în sesiunea 3: repo-ul a devenit public.*
**Blocate de Adrian:** licența firmware-ului (I4 → tot ce urmează), compilatorul (NAK vs ptxas), fereastra pe workstation.

---

*Sesiunea 1 (mai jos), 23 octombrie 2026. Etichete: FAPT (verificat), SIMULARE (doar pe modele software), ESTIMARE, NECUNOSCUT.*

## G0 — Audit și inventar

### Ce exista la început (FAPT)
- `Adriancu91/cyber-gpu`: repository **gol** (niciun commit, niciun ref; `git ls-remote` fără ieșire, cod 0).
- Workspace local `C:\Users\krono\Desktop\gpu`: gol (doar `.claude/`).
- Calculatorul folosit în seara asta: laptop Lenovo, i5-1135G7, Intel Iris Xe (`8086:9a49`), 8 GB. **Nu are placă NVIDIA**,
  deci **nu e workstation-ul**. Nimic nu s-a putut testa pe RTX 3070 Ti.
- Toolchain: nu exista Rust. Instalat `rustup` (winget, sursa oficială) + `stable-x86_64-pc-windows-gnu` (rustc 1.99.1)
  + ținta `x86_64-unknown-none`. Fără MSVC Build Tools (de aceea toolchain-ul GNU). Fără QEMU, fără `gh`.

### API-urile cyber-universe verificate în cod (commit `2d038de`, 22 oct)
| Afirmație din ARHITECTURA_KERNEL | Verificat în cod | Observație importantă pentru GPU |
|---|---|---|
| enumerare PCIe, BAR-uri 64 biți | DA: `pci::Device`, `bar(i)`, `first_memory_bar()` | **nu există dimensionarea BAR-urilor**; doar `read32/write32` (nu 8/16 biți) |
| `enable()` memorie + Bus Master | DA: `enable_as()`, bit cu bit, jurnalizat | nivelul 0 al GPU-ului **nu îl cheamă** |
| `map_mmio` | DA: UC (PCD+PWT) | **sare peste paginile deja mapate** → o zonă deja mapată poate rămâne cache-abilă; glue-ul nostru refuză în acest caz |
| DMA contiguu sub 4 GB | DA: `alloc_contiguous_between(n, 16<<20, 1<<32)`, `pin` | nu e zeroizată; limita 4 GB e politică, nu a GPU-ului |
| jurnal RAM + serial | DA: `info!/warn!/error!`, inel 64 KiB | |
| fără IOMMU, SMP, MSI | DA (absente) | |
| fără salvare SSE/AVX | DA | crate-urile noastre folosesc f32 în soft-float pe ținta kernelului |
| crate-uri separate, teste pe gazdă cu model | DA: tiparul `rtl8168/` (`trait Hw` + `model.rs`) | refolosit: `ConfigSpace`/`Mmio` + `gpu-sim` |

### Necunoscute hardware (rămân pentru K0 pe workstation)
BAR0/BAR1/BAR3 reale, Above-4G Decoding, Resizable BAR, starea IOMMU, Secure Boot, ID-ul subsistemului, versiunea VBIOS.
Instrumentul: `tools/k0-inventar-workstation.ps1` (Windows, doar citire; testat pe laptop).

## Stadiul pe etape

| Etapă | Stare | Dovadă |
|---|---|---|
| G0 audit | **gata** | acest document |
| G1 nucleul generic | **gata (gazdă)** | `gpu-core`: erori, I1, stări, jurnal; teste PASS |
| G2 descoperire GPU | **implementat, testat doar în SIMULARE** | `gpu-pci` + `nvidia-ga104::probe`; zero scrieri, verificat de teste; patch kernel `gpu identifica` |
| G3 cercetare/inițializare | **cercetare gata; sesiunea 2: I2–I3 implementate (doar citire, SIMULARE), cod oprit la I4** | `docs/GA104_INITIALIZATION.md`, `init.rs` (plan I0–I10) |
| G4 memoria GPU | **parțial** | invarianți + allocator + DMA policy + poarta IOMMU (logică); sesiunea 3: regiunile GSP din VRAM + radix3 (aritmetică); nimic pe VRAM real |
| G5 primul calcul GPU | **NU** | blocat la I4–I8 (vezi mai jos) |
| G6 toolchain | **cercetare** | `docs/TOOLCHAIN.md`; niciun binar SM86 produs |
| G7 operații tensoriale | **referință CPU gata** | `gpu-compute::reference` (add, bias, matmul, layernorm, GELU, softmax cauzal, embedding, atenție, cross-entropy), teste diferențiale f64 PASS |
| G8 pregătire Kronos | **parțial** | `TensorDesc` (f32/bf16), `OpCode` cu convenție de argumente, `CpuBackend` prin I1 |

### Componente terminate / parțiale / blocate
- **Terminate (pe gazdă):** erori structurate, contractul I1, mașina de stări, jurnalul limitat, decodarea BAR, capabilități
  PCI/PCIe/ReBAR, raportul nivel 0, ID-uri GA104, decodarea BOOT_0/BOOT_42, lista de registre permise, allocatorul de intervale,
  tabela de buffere, inelul PUT/GET, fence-urile, math `no_std`, operatorii de referință, backend-ul CPU.
- **Parțiale:** dimensionarea BAR (scrisă și testată în simulare; pe hardware e opțiune explicită `gpu dimensiuni da`),
  codificarea pushbuffer/GPFIFO (după `clc56f.h`, niciodată executată), inventarul firmware (listă + verificare structurală).
- **Blocate:** I4 (licența și transportul firmware-ului), I5 (IOMMU + DMA), I6–I8 (falcon, GSP, RPC), I9–I10 (canal, QMD, program).

## Progres (ESTIMARE argumentată)
- **Până la primul calcul GPU real: ~10 %.** Din cei 11 pași I0–I10, primii doi (cei mai ieftini, doar citire) sunt scriși;
  pașii I6–I8 (falcon + GSP-RM + RPC) sunt, după volumul de cod din nouveau/nova-core, partea cea mai mare (ESTIMARE: 70 % din efort).
  Fără I8 nu există canal de comenzi, deci nimic nu rulează.
- **Implementarea totală (driver + toolchain + runtime util lui Kronos pe GPU): ~5 %.** Runtime-ul CPU de referință e
  un câștig real pentru K6-CPU, dar motorul de calcul GPU (compilator/assembler SM86, kernels matmul) nu a început.
- **Următorul prag tehnic:** rularea `gpu identifica` pe workstation, într-o fereastră aprobată de Adrian: confirmă BOOT_0 = 0x174xxxxx,
  BAR-urile și ReBAR, fără nicio scriere. Apoi I2 (GFW_BOOT) și I3 (citirea VBIOS), ambele doar citire.
