# Analiza I9 după `os-303` — de ce răspunde placa `0x57` și ce încearcă `--i9 c133` (comanda C-133)

Sesiunea Cyber-GPU 107 (17 nov 2026, cloud, **fără placă**). Tot ce e cod aici e testat **doar în SIMULARE**. Nimic nu a rulat pe RTX 3070 Ti.
Surse de date: hub `rezultate/2026-11-16-workstation-os302/` și `rezultate/2026-11-17-workstation-os303/` + `rezultate/os-automat/2026-11-17/gpu-jurnal-005.txt`.
Etichete: **OBSERVAT** = scris în jurnalul de pe placă; **CITIT** = scris în sursele deschise (MIT), citite ca specificație; **DEDUS** = concluzia noastră;
**NECUNOSCUT** = nu se poate afla fără placă (partea „fizică” a GSP-RM e în firmware, pe care nu avem voie să-l analizăm).

## 0. Pe scurt (pentru Adrian)

- **Ce s-a întâmplat:** la I9, placa primește bufferele de lucru ale canalului în **două** mesaje: întâi „unde sunt în memorie” (trece), apoi „la ce adresă
  virtuală le vede canalul” (refuzat cu `0x57` = „nu găsesc obiectul”). La fel în `os-302` și `os-303`, cu aceleași două semnale interne ale plăcii.
- **Ce am găsit comparând cu cine merge pe exact firmware-ul nostru (570.144):** driverul Linux deschis (nouveau) și driverul oficial NVIDIA trimit totul
  **într-un singur mesaj** (adresa din memorie + adresa virtuală împreună) și **înainte** să creeze obiectul de calcul. Noi trimitem în două mesaje și
  **după** obiect. Mai mult: **imaginea „golden” (I8e), care trece pe placa noastră, folosește deja forma cu un singur mesaj.** Singura referință care
  face ca noi (tinygrad) rulează cu alt firmware (570.190).
- **Cauza cea mai probabilă:** forma și ordinea promovării (ipotezele (a) + (d) de mai jos). **Cât de sigur: mediu — cam 50–60 % că `--i9 c133` trece.**
  Nu e o certitudine: canalul de lucru mai diferă de golden prin grupul de canale și prin spațiul de adrese „al nostru”, pe care încercarea nu le schimbă.
- **Ce testează `--i9 c133`** (o singură pornire, două încercări una după alta): **A** = același canal, dar promovarea într-un singur mesaj înaintea
  obiectului; dacă placa o refuză, **B** = într-un grup nou, cu setul de buffere al lui nouveau (fără bufferul PM, plus adresele a două buffere globale).
  Jurnalul spune la fiecare ce s-a trimis și ce a răspuns placa — și dacă pică amândouă.

## 1. Ce s-a observat pe placă (OBSERVAT)

| | `os-302` (16 nov, cyber-gpu `835f24d`, `gpu lant`) | `os-303` (17 nov, `239f986`, `gpu lant --i9 c125`) |
|---|---|---|
| I0–I8 + golden (I8e) | trec | trec |
| alocarea canalului de lucru | trece; 2 ASSERT `0x01ab9bda`, `0x01ab9c56` | la fel, aceleași 2 adrese |
| obiectul compute | trece | trece |
| V3 `GR_GET_CTX_BUFFER_INFO` | — (netrimis) | **refuzat `0x31` `NV_ERR_INVALID_OBJECT`**, fără ASSERT |
| `PROMOTE_CTX` fizic (MAIN, PM, PATCH) | trece | trece |
| `PROMOTE_CTX` virtual (aceleași 3) | **`0x57` `NV_ERR_OBJECT_NOT_FOUND`**; 2 ASSERT `0x01443f48`, `0x01418a2a` | **`0x57`**, aceleași 2 ASSERT; bufferele mapate privilegiat (V2) |
| curățarea | ok | ok |

Aceleași două adrese de ASSERT în ambele rulări ⇒ **aceeași cauză**; maparea privilegiată (V2) nu a schimbat nimic ⇒ V2 singur e eliminat.
Imaginea golden a produs și ea 3 ASSERT (`0x01a9068a`, `0x01a90cf4`, `0x01394270`) și a trecut ⇒ un ASSERT în jurnal nu înseamnă, singur, eșec.

## 2. Cele două refuzuri mici, explicate

**Refuzul V3 (`0x31`, fără `ROUTE_TO_PHYSICAL`).** CITIT în RM 570.144, `kernel_graphics.c` `subdeviceCtrlCmdKGrGetCtxBufferInfo` →
`kernel_graphics_context.c` `kgrctxGetCtxBuffers`: comanda găsește clientul și canalul, apoi cere descriptorul bufferului MAIN ținut de **RM-ul de pe CPU**
(`pKernelChannelGroup->ppEngCtxDesc`); dacă lipsește, întoarce exact `NV_ERR_INVALID_OBJECT`, printr-o verificare care **nu** produce ASSERT — cum arată și
jurnalul (niciun NOCAT la V3). DEDUS: la noi bufferele sunt alocate de driver, nu de un RM de pe CPU, deci GSP-RM singur nu are ce lista. Refuzul nu arată
un handle greșit și **nu** e același „obiect lipsă” ca la promovare; arată doar că V3 nu poate merge fără RM-ul de pe CPU. `--i9 c133` nu-l mai trimite.

**Cele 2 ASSERT de la alocarea canalului.** CITIT în `kernel_channel.c` `kchannelConstruct`: două `NV_ASSERT` alăturate, care nu opresc nimic, cer ca tipul
notificatorului de erori și al celui ECC să nu fie „necunoscut”; pe GSP tipurile vin din `internalFlags` trimise de client. Noi (ca tinygrad) trimitem
`internalFlags = 0` = „necunoscut” pentru amândouă; nouveau trimite „NONE”. DEDUS (se potrivește: două ASSERT, aceeași funcție, la `0x7c` distanță): sunt
inofensive pentru I9 și apar și la tinygrad. Nu le schimbăm acum (o schimbare pe rând); notat ca posibilă curățenie.

## 3. Pas cu pas față de cine merge (C-133 pasul 1)

Coloane: **noi** (`os-302`/`os-303`); **golden-ul nostru** (I8e — trece pe placă); **tinygrad** (`runtime/ops_nv.py`, `runtime/support/nv/ip.py`, commit
`9ecf871`, MIT; firmware **570.190**); **nouveau** (`nvkm/subdev/gsp/rm/r535/{vmm,fifo,gr}.c` + `r570/{fifo,gr}.c`, MIT; `ga102.c` leagă GA104 de
**570.144**); **NVIDIA** (open-gpu-kernel-modules **570.144**, MIT: `kernel_channel*.c`, `kernel_ctxshare.c`, `kernel_graphics_object.c`,
`kernel_graphics_context.c`, `nv_gpu_ops.c`). Totul CITIT ca specificație; nimic copiat.

| Pas / câmp | noi, canalul de lucru | golden-ul nostru | tinygrad (570.190) | nouveau (570.144) | NVIDIA 570.144 |
|---|---|---|---|---|---|
| client, device, subdevice | client `0xc1000000`, `hClientShare` = client | client separat | la fel ca noi (+ `vaMode` 0) | client + device **per spațiu de adrese** | — |
| `FERMI_VASPACE_A` | `EXTERNALLY_OWNED` + `ENABLE_PAGE_FAULTING`, `vaBase 0x1000` | implicit (al RM), cu PDE-uri rezervate | la fel ca noi | doar `EXTERNALLY_OWNED`, bază/mărime 0 | UVM: `EXTERNALLY_OWNED` |
| rădăcina tabelelor | RPC `SET_PAGE_DIRECTORY`, `ALL_CHANNELS`, VIDMEM | — (`COPY_SERVER_RESERVED_PDES`, zona la `0x10_0000_0000`) | la fel ca noi | control `DMA_SET_PAGE_DIRECTORY`, fără `ALL_CHANNELS`; zona rezervată a golden-ului la 4 GiB | — |
| grup de canale (TSG) | da, `engineType` GR, **`hVASpace` = spațiul nostru** | **nu** | da, **`hVASpace` = 0** (RM ia spațiul implicit al device-ului) | **nu** | permis (`kernel_channel_group_api.c`) |
| context share | da, pe spațiul nostru, `ASYNC` | **nu** | la fel ca noi | **nu** | permis (`kernel_ctxshare.c`) |
| canal GPFIFO | părinte = TSG, `hContextShare`, `flags` 0, **`internalFlags` 0** | părinte = device, `hVASpace`, `internalFlags 0x1a` | la fel ca noi | părinte = device, `hVASpace`, `internalFlags` = privilegiu + notificatoare `NONE` | — |
| **ordinea** | canal → **obiect compute** → promovare | canal → **promovare** → obiect | canal → obiect → promovare | canal → **promovare** → obiect | **promovare** (din constructorul obiectului) → RPC-ul obiectului |
| **forma promovării** | **2 mesaje**: fizic (`bInitialize`, `bNonmapped`), apoi doar-VA | **1 mesaj**, PA + VA în aceeași intrare | 2 mesaje (ca noi) | **1 mesaj**, PA + VA în aceeași intrare | fără UVM: **1 mesaj** PA + VA; cu UVM: RM trimite fizicul, UVM trimite doar-VA |
| buffere per canal | MAIN, **PM**, PATCH | MAIN, PATCH (PM nepromovat) | MAIN, PM, PATCH | MAIN, PATCH (**fără PM**) | MAIN, PATCH, PM doar dacă e alocat |
| buffere globale (VA în spațiul canalului) | **niciunul** | 3–6, FECS_EVENT (9), UNRESTRICTED (11); PRIV_ACCESS_MAP (10) doar fizic | niciunul | BUNDLE/PAGEPOOL/ATTRIBUTE/RTV, **FECS_EVENT, PRIV_ACCESS_MAP** (doar VA) | compute: **FECS_EVENT, PRIV_ACCESS_MAP, UNRESTRICTED**; UVM: toate cele listate de RM |
| cine trimite doar-VA | clientul canalului | — | clientul canalului | — | UVM, din clientul lui intern, după ce RM-ul de pe CPU i-a „reținut” bufferele |
| maparea bufferelor | 4 KiB, neprivilegiat (`os-303`: privilegiat) | 4 KiB, neprivilegiat | 4 KiB | privilegiat, pagină după mărime | după lista RM |
| `physAttr`, `hChanClient`, `hObject`, `engineType` | 4, clientul, canalul, GR0 | la fel | la fel | la fel | la fel |
| token + programare | `GET_WORK_SUBMIT_TOKEN`, `TSG_SCHEDULE` | — | la fel | `BIND` + `GPFIFO_SCHEDULE` pe canal | — |

**Unde diferim de tot ce merge pe 570.144** (nouveau, NVIDIA, golden-ul nostru): (1) promovarea în două mesaje în loc de unul; (2) promovarea după obiectul
compute în loc de înainte; (3) niciun VA de buffer global în spațiul canalului; (4) bufferul PM promovat. Cu tinygrad nu diferim în (1)–(4) — dar tinygrad
rulează 570.190 și nu avem nicio dovadă că merge pe 570.144.

## 4. Ipotezele, în ordinea probabilității (C-133 pasul 2)

| # | Ipoteza | Ce o susține | Ce o contrazice | Probabilitate |
|---|---|---|---|---|
| (a) | **Un singur mesaj PA + VA**: GSP-RM 570.144 nu primește o promovare doar-VA de la clientul canalului | golden-ul nostru (1 mesaj) trece pe placă, canalul de lucru (2 mesaje) pică exact la mesajul doar-VA, în ambele jurnale; nouveau și NVIDIA fără UVM: 1 mesaj; singurul doar-VA oficial e al UVM, din client intern, după „reținerea” de către RM-ul de pe CPU (pe care nu-l avem — vezi refuzul V3) | tinygrad trimite 2 mesaje (dar pe 570.190) | **mare** |
| (d) | **Ordinea**: promovarea trebuie să ajungă înaintea obiectului compute | toate cele trei referințe de pe 570.144 și golden-ul promovează înainte; la noi obiectul există deja când sosește fizicul | tinygrad promovează după (570.190); fizicul nostru trece | **mare** (nu se poate despărți de (a) din jurnale) |
| (c) | **Lipsesc VA-urile bufferelor globale** (FECS_EVENT, PRIV_ACCESS_MAP) în spațiul canalului; sau **bufferul PM** e cel refuzat | `OBJECT_NOT_FOUND` e exact ce întoarce lista de adrese a RM când spațiul nu e în listă (`vaddr_list.c`); nouveau și NVIDIA promovează globalele lângă MAIN; nouveau nu promovează PM deloc | tinygrad nu trimite globalele și trimite PM (570.190) | **medie** |
| (b) | **Spațiul de adrese / grupul**: GSP-RM nu găsește spațiul canalului așa cum îl caută (TSG cu `hVASpace`, context share, spațiu „deținut din afară”) | golden-ul și nouveau nu au TSG, nici spațiu extern; noi punem `hVASpace` în TSG, tinygrad nu | CITIT: `hVASpace` în TSG **nu** pornește „modul legacy” (acela pornește doar la un canal fără context share) și e singura cale ca TSG-ul să nu ia spațiul implicit al device-ului; alocările TSG, context share și canal trec; fizicul găsește canalul | **mică–medie** |
| (e) | **V1**: pagina / alinierea mapării | UVM și nouveau mapează după alinierea cerută | toate bufferele noastre sunt aliniate la 4 KiB cum cere `KGR_GET_CONTEXT_BUFFERS_INFO`; golden-ul trece cu aceeași mapare | **mică** |
| (f) | **570.144 față de 570.190** | tinygrad merge cu secvența noastră doar pe 570.190 | nouveau merge pe 570.144 cu altă secvență — deci 570.144 poate porni canale | nu e o cauză de sine stătătoare: explică de ce „ca tinygrad” nu e o garanție; firmware-ul rămâne 570.144 (decizia din 26 oct) |

De ce `0x57` și nu alt cod (DEDUS, NECUNOSCUT în detaliu): la mesajul doar-VA, partea fizică a GSP-RM trebuie să găsească ceva pus de un pas anterior
(descriptorul bufferului, contextul GR al grupului sau adresa unui buffer în spațiul canalului). În (a)/(d) acel „ceva” nu a fost creat pentru că fizicul a
venit separat și după obiect; în (c) lipsește adresa unui buffer global. Nu se poate alege între ele fără placă.

## 5. Reparația aleasă: `--i9 c133` (C-133 pasul 3) — SIMULARE, implicit oprită

`nvidia_ga104::gsp::user_space::open_work_channel_c133` (cod nou, doar adăugări; `gpu lant` fără opțiune și `--i9 c125` trimit exact ce trimiteau).
Fiecare încercare își face **grupul ei nou** de canale (context GR curat); o încercare refuzată lasă un grup care nu a fost niciodată programat, deci nu
rulează nimic. A doua pornește **doar** dacă placa refuză `PROMOTE_CTX` al primei; orice alt refuz sau o eroare de transport oprește imediat.

| | Încercarea A | Încercarea B (doar dacă A e refuzată la `PROMOTE_CTX`) |
|---|---|---|
| ce schimbă față de `os-302` | **un mesaj** PA + VA, **înaintea** obiectului compute | ca A, plus setul de buffere al lui nouveau |
| buffere | MAIN, PM, PATCH (aceleași trei) | MAIN, PATCH cu PA + VA; FECS_EVENT și PRIV_ACCESS_MAP **doar VA** (adresele din golden — aceleași tabele de pagini) |
| ipoteze acoperite | (a), (d) | (a), (d), (c), PM |
| mesaje | TSG, context share, canal, **1 × `PROMOTE_CTX`**, compute, token, programare | la fel, în grup nou |

De ce acestea două: A e schimbarea cea mai mică spre forma **dovedită pe placa noastră** (golden) și folosită de ambele referințe de pe 570.144; B adaugă
singurele alte două diferențe față de nouveau care țin de promovare. Amândouă încap în aceeași pornire: sunt doar mesaje RM de același fel cu cele deja
trimise (niciun registru nou, nicio comandă nouă), iar memoria în plus e de câțiva MiB. Nu am pus FREE între încercări (ar fi un tip de mesaj nou, neîncercat).

Ce **nu** face: nu schimbă TSG-ul, context share-ul sau spațiul de adrese (ipoteza (b)), nu trimite V3, nu mapează privilegiat (golden-ul trece fără).

## 6. Ce spune fiecare rezultat de pe placă (C-133 pasul 4)

| Rezultat în jurnal (`i9 c133 …`) | Ce confirmă / infirmă | Ce urmează |
|---|---|---|
| **A trece** (`varianta reusita = 1`) | (a) și/sau (d) confirmate; (c), PM, (b), (e) infirmate | I9 gata → **I10 (add64)**, primul calcul pe placă; forma A devine cea implicită după „da”-ul lui Adrian |
| **A `0x57`, B trece** (`= 2`) | (c) sau PM confirmat; (a)/(d) singure infirmate | I10 cu canalul B (fără PM — destul pentru calcul; nouveau merge așa) |
| **A și B `0x57`** (aceleași 2 ASSERT) | (a), (c), (d), PM infirmate împreună → rămâne (b) | următoarea variantă: canal ca nouveau — direct pe device, cu `hVASpace`, fără TSG / context share (cere „da” nou); abia apoi (e) |
| **A sau B pică cu alt cod** la `PROMOTE_CTX` | forma nouă ajunge mai departe în GSP-RM | codul și ASSERT-urile noi din jurnal decid |
| `PROMOTE_CTX` trece, **obiectul compute** refuzat | ordinea nouă cere altceva la obiect | se oprește (fără B); jurnalul are pasul și codul |
| `i9 c133 incercari facute = 0` sau oprire înainte de canal | problemă înainte de promovare (nu ține de C-133) | se compară cu `os-303` |

## 7. Ce rămâne NECUNOSCUT / ce nu s-a putut face

- Care dintre (a) și (d) e cauza: A le schimbă pe amândouă deodată (scopul e reparația; despărțirea lor ar cere încă o încercare).
- Dacă tinygrad chiar merge pe 570.144 cu secvența noastră: nu am cum verifica din cloud.
- Ce caută exact GSP-RM la `0x01443f48` / `0x01418a2a`: adresele sunt în firmware, pe care nu îl analizăm; rămân amprente de comparat între rulări.
- Modelul `gpu_sim::fake_rm::PromoteRule` joacă **ipoteze**, nu comportamentul plăcii: testele arată că încercările fac ce scrie aici în fiecare caz, nu că
  placa va răspunde așa.

## 8. Pe placă — doar în fereastra cu Adrian

După ce OS-ul leagă opțiunea (nota `integration/cyber-universe/C-133_i9-un-singur-mesaj.md`): stick nou + `github.token` → **pornire la rece** →
`gpu lant --i9 c133` → DA → `trimite`. O singură rulare ajunge.
