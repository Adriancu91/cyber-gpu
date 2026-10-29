# Starea implementării — cyber-gpu

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

## Sesiunea 7 — 27 octombrie 2026, ~01:25–02:05 (ora României), în cloud, FĂRĂ placă

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
