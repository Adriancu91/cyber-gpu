# Matricea de teste

## Rezultate — sesiunea 30 (1 nov 2026, cloud, fără placă) — C-49: pregătirea I6

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T56 | Siguranța „ucode version”: adrese (GSP 0x8241C0, SEC2 0x824140, +4·(id−1)), decodare (bitul cel mai înalt + 1), refuzuri (biți 31:16, 0xbadf…, 0xffffffff, fereastră < 16 MiB, id 0/17) | `cargo test -p nvidia-ga104 fuse` | 6 teste; o citire, 0 scrieri | conform; 6 greșeli plantate prinse | PASS (SIMULARE) |
| T57 | Alegerea semnăturii FWSEC cu forma de pe placă (ucode_id 9, 3 semnături, masca 0x0007): versiunea 0/1/2 → indexul 0/1/2; versiunea 3 → `Blocked` | `cargo test -p nvidia-ga104 fuse::tests::signature_choice` | conform | conform | PASS (SIMULARE: valorile siguranței sunt ale noastre) |
| T58 | Verificarea I6 doar citire (`i6_check`): 3 citiri pe placa rece, FRTS sub zona VGA, WPR2 ridicat = nu trece, 0xbadf…/0xffffffff/VRAM implauzibil = eroare | `cargo test -p nvidia-ga104 i6_check` | 3 teste, 0 scrieri | conform; 4 greșeli plantate prinse | PASS (SIMULARE) |
| T59 | I6 scrie exact cele 18 registre GSP din `docs/FEREASTRA_C-49.md` (78 de scrieri pe imaginea SIM de 20 de blocuri) | `cargo test -p nvidia-ga104 i6_writes_exactly` | lista fixă, nimic în afara GSP | conform | PASS (SIMULARE) |
| T60 | `gpu pregatire-i6` pe placa reală | — (propunere pentru OS) | semnătura 0/1/2, VRAM 8192 MiB, WPR2 coborât, 0 scrieri | — | NOT TESTED |

## Rezultate — sesiunea 24 (30 oct 2026, cloud, fără placă) — treapta 6 matmul

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T51 | Treapta 6 (`matmul_vec4`, NAK, 821 instr., 90 registre) lansată în modelul de comenzi | `cargo test -p nvidia-ga104 --test sim_matmul vec4_step_matches` | 6 forme identice bit cu bit cu `matmul_fma_ftz`; nimic scris după C; semaforul 1 | conform | PASS (SIMULARE) — programul „rulează” pe procesor |
| T52 | Treapta 6 = treapta 5, aceiași biți (70×28×68) | `cargo test -p nvidia-ga104 --test sim_matmul vec4_step_gives_the_same_bits` | identic | conform | PASS (SIMULARE) |
| T53 | −0,0 păstrat; K/N nemultipli de 4, C nealiniat, bloc/partajată/barieră greșite refuzate | `cargo test -p nvidia-ga104 --test sim_matmul vec4_step_keeps` | eroare, 0 scrieri în C | conform | PASS (SIMULARE) |
| T54 | 8 greșeli plantate (6 în model, 2 în test) | `cargo test -p nvidia-ga104 --test sim_matmul vec4` (cu modelul modificat) | fiecare pică cel puțin un test | 8/8 prinse | PASS (SIMULARE) |
| T55 | Treapta 6 pe placa reală (viteză + rezultat) | — | — | — | NOT TESTED (doar în fereastra cu Adrian, după I10 pe placă) |

## Rezultate — sesiunea 23 (30 oct 2026, cloud, fără placă) — I10d
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T50 | Zona de comenzi refolosită după semafor (I10d), cap-coadă prin tabelele de pagini | `cargo test -p nvidia-ga104 pushbuffer_area_is_reused_after_the_semaphore` | 12 lansări `add64` într-o zonă de 128 B (configurare + 2 lansări); a 3-a lansare refuzată cât zona e plină, fără nicio scriere; după semaforul 1, lansarea 3 la începutul zonei, lansarea 2 neatinsă; rezultat identic bit cu bit la fiecare lansare | conform; `PUT` = 13 | PASS (SIMULARE) |
| T51 | Aritmetica cercului (unde încape o porțiune) | `cargo test -p nvidia-ga104 placement_never_reaches_a_stretch_in_flight` | 10 cazuri: gol, la capăt, de la început, între capete după trecerea peste capăt, plin | conform | PASS |
| T52 | 7 greșeli plantate în `WorkQueue` (trecere peste capăt fără verificare, ramura „trecut peste capăt” fără limită, eliberare înainte de semafor, porțiunea de configurare neeliberată, aceeași valoare de semafor acceptată, semafor prea mare acceptat, capătul zonei cu 4 octeți prea lung) | testele `nvidia-ga104` rulate cu fiecare greșeală | fiecare greșeală prinsă de cel puțin un test | 7/7 prinse | PASS |

## Rezultate — sesiunea 22 (30 oct 2026, cloud, fără placă) — I10c
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T47 | Forma GR0 cerută de la GSP-RM (I10c) pe subdispozitivul clientului de lucru | `cargo test -p nvidia-ga104 gr_info_from_gsp_rm` | o comandă `0x20800a2a` acceptată; 6 GPC × 4 TPC × 2 SM, 48 warp-uri/SM, `sm_86` (valori SIM); memoria locală pentru 1 B/fir: 0x18000 pe TPC, 0x240000 în total | conform | PASS (SIMULARE) |
| T48 | Refuzurile GSP-RM (alt obiect decât subdispozitivul, altă mărime) | `cargo test -p nvidia-ga104 rm_refusals_for_gr_info` | codul RM (`INVALID_OBJECT_HANDLE`, `INVALID_PARAM_STRUCT`), nicio formă | conform | PASS (SIMULARE) |
| T49 | 8 greșeli plantate în răspuns (fiecare din cele 5 valori = 0, index greșit, lungime −8 / +1) + placă non-`sm_86` | `cargo test -p nvidia-ga104 planted_gr_info_mistakes_are_caught` | toate refuzate; datele lui GR1 ignorate; `index` = 0 acceptat | 8/8 prinse; `sm_87` refuzat la `check_sm86` | PASS (SIMULARE) |
| T50 | Lansarea `add64` (T41) cere întâi forma GR și verifică `sm_86` | `cargo test -p nvidia-ga104 first_launch_on_the_work_channel` | ca T41 | conform | PASS (SIMULARE) |
| T51 | Mărimea și indicii generați din antete (3712 B; 12, 13, 20, 23, 32; 58) | `cargo test -p nvidia-rm570 rm_api_layouts_570_144` | egale | conform | PASS |
| T52 | Răspunsul real al GSP-RM 570.144 pe RTX 3070 Ti | — | — | — | NOT TESTED (doar în fereastra cu Adrian, după I5–I8 pe placă) |

## Rezultate — sesiunea 21 (30 oct 2026, cloud, fără placă) — I10b
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T41 | Prima lansare pe canalul de lucru (I10b): după configurarea I10a, invalidarea cache-urilor + `SEND_PCAS_A`/`PCAS2_B` + semaforul de terminare (64 de biți, `RELEASE_WFI`), programul `add64` compilat cu NAK | `cargo test -p nvidia-ga104 first_launch_on_the_work_channel` | 2 intrări GPFIFO preluate; modelul citește programul, QMD-ul, argumentele **doar** prin tabelele de pagini; driverul vede `GPGet = 2`, semaforul `0x1_0000_0007` și rezultatul identic bit cu bit cu `reference::add_ftz` | conform | PASS (SIMULARE) — programul „rulează” pe procesor |
| T42 | Cuvintele lansării = cele din tinygrad | `cargo test -p nvidia-ga104 launch_words_match_tinygrad` | `0x1011`, `qmd >> 8`, `9`, `0x0110_0001` | conform | PASS (SIMULARE) |
| T43 | Refuzuri în driver (QMD nealiniat / în afara VA-ului nostru, semafor nealiniat / în afara VA-ului) | `cargo test -p nvidia-ga104 launch_driver_side_refusals` | eroare, nimic scris (`GPPut` neschimbat) | conform (5 cazuri) | PASS |
| T44 | 8 greșeli plantate în cuvintele lansării (bit necunoscut, adresa QMD, acțiunea PCAS, fără `SEND_PCAS_A`, jumătățile valorii, timestamp, semafor pe 32 de biți, adresa semaforului) | `cargo test -p nvidia-ga104 planted_launch_mistakes_are_caught` | fiecare prinsă de model sau de verificarea semaforului / rezultatului | 8/8 prinse | PASS (SIMULARE) |
| T45 | Citirea semaforului pe 64 de biți în două jumătăți (transport între citiri) | `cargo test -p nvidia-ga104 semaphore_reads_survive_a_carry` | recitire o dată, apoi eroare | conform | PASS |
| T46 | Lansarea pe placa reală | — | — | — | NOT TESTED (soneria nesunată; doar în fereastra cu Adrian) |

## Rezultate — sesiunea 20 (29 oct 2026, cloud, fără placă) — I10a
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T36 | Primul pushbuffer pe canalul de lucru (I10a): `SET_OBJECT` compute + ferestrele de memorie locală/partajată pe subcanalul 1 (ca tinygrad), intrarea GPFIFO în inel, `GPPut` în USERD | `cargo test -p nvidia-ga104 user_space work_queue` | după I9c–I9e, modelul `gpu-sim::cmdproc` citește inelul, USERD și pushbuffer-ul **doar** prin rădăcina primită de RM-ul fals, execută metodele, scrie `GPGet` înapoi; driverul îl citește și îl acceptă; inelul trece peste capăt (1023 de intrări), al 1024-lea refuzat fără nicio scriere | conform; cuvintele = exact ce dă `nvm(1, ...)` din tinygrad; **5 greșeli plantate în driver** (ferestre inversate, `GPPut` scris la `GPGet`, slot greșit, USERD neșters, adresa intrării) + **3 în octeți** (jumătăți inversate, subcanal greșit, metodă greșită, cuvânt lipsă), toate prinse | PASS (SIMULARE) |
| T37 | USERD șters înainte de alocarea canalului (`open_work_channel_cleared`) | idem | `GPGet = GPPut = 0` când GSP-RM programează canalul; fără ștergere modelul găsește „gunoi” și refuză | conform | PASS (SIMULARE) |
| T38 | Metodele de configurare în model (perechi `_A`/`_B`, biți în afara câmpului, `_B` fără `_A`, fără `SET_OBJECT`) + inelul cu trecere peste capăt | `cargo test -p gpu-sim cmdproc` | refuzate / înregistrate exact | conform (3 teste) | PASS (SIMULARE) |
| T39 | `AmpereAControlGPFifo` (USERD) generat din `clc56f.h` 570.144 | `tools/gen-rm570/generate.sh` + `cargo test -p nvidia-rm570` | 512 B, `GPGet` 0x88, `GPPut` 0x8c; restul fișierelor generate identice | conform | PASS |
| T40 | Soneria (scrierea jetonului la BAR0 + 0xbb0090) | — | — | — | NOT TESTED (doar descrisă: `work_queue::Doorbell`; nescrisă) |

## Rezultate — sesiunea 19 (29 oct 2026, cloud, fără placă) — I9e
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T33 | Canalul de lucru în grupul de canale (I9e): canal → compute → PROMOTE_CTX (fizic, apoi virtual) → jeton → programarea grupului | `cargo test -p nvidia-ga104 user_space` | prin cozile reale către `gpu-sim::fake_rm`: canalul sub TSG cu context share-ul, MAIN/PM/PATCH promovate pe buffere noi, jetonul = canal \| runlist << 16, grupul programat; inelul, USERD și bufferele găsite prin rădăcina primită de model | conform (6 teste noi + testul cap-coadă extins); 7 refuzuri RM întorc codul RM; greșeli plantate: 4 în codificator + 4 în secvență, toate prinse | PASS (SIMULARE) |
| T34 | Structurile noi generate din antetele 570.144 | `tools/gen-rm570/generate.sh` + `cargo test -p nvidia-rm570` | `regs.rs`, `structs.rs`, `sysinfo.rs` identice; `rmapi.rs` doar cu adăugări | conform | PASS |
| T35 | Soneria (scrierea jetonului în registru) | — | — | — | NOT TESTED (nescrisă; vine cu I10, doar cu `hw-write` și Adrian de față) |

## Rezultate — sesiunea 18 (29 oct 2026, cloud, fără placă) — C-39
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T30 | Antetele `booter_load`/`booter_unload`/`bootloader` cu forma fișierelor oficiale 570.144 | `cargo test -p nvidia-ga104 fw_hdr hs_load fw_check` | antetele citite; `bin_size` ≠ mărimea fișierului acceptat; 2 semnături de 384 B; Booter-ul se încarcă (`from_booter`, `booter_image`) | conform; greșeli plantate: verificarea veche `bin_size` = mărime (4 teste pică), numărul de semnături citit direct (9 teste pică) | PASS (SIMULARE: valorile de antet ale fișierelor oficiale, restul octeților umplutură) |
| T31 | OS-ul construit cu cyber-gpu local (`ef97110`) | copie de lucru a OS-ului (repo-ul OS neatins), dependențele `path` | build normal și `test-gpu-sim` | ambele trec | PASS (compilare); rulare în QEMU: NOT TESTED (QEMU lipsește în sesiune) |
| T32 | `gpu firmware` pe stick-ul real | în OS, la fereastră | `criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4)` | — | NOT TESTED (cere Release OS cu fixarea ≥ `ef97110`) |

## Rezultate — sesiunea 15 (28 oct 2026, cloud, fără placă) — C-36
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T27 | I4 pe stick, raport complet (`fw_check`) | `cargo test -p nvidia-ga104 fw_check` | toate 4 fișierele raportate (lipsă / altă mărime / alt conținut / oficial), antetul citit doar la cele oficiale, „0 accese la placa” | 7 teste trec; 3 greșeli plantate (fără SHA, fără mărime, antet necitit) prinse | PASS (SIMULARE, fișiere de probă `gpu-sim::fw_stick`, nu NVIDIA) |
| T28 | Rândurile I3b (`FwsecLocation::lines`) | `cargo test -p nvidia-ga104 fwsec` | 5 rânduri, `criteriul I3b: TRECE` pe lanțul de tip ASUS | conform | PASS (SIMULARE) |
| T29 | Schița pentru OS (`gpu firmware`, `gpu fwsec`) compilată în kernel | copie de lucru a OS-ului (repo-ul OS neatins), cyber-gpu local | build normal și `test-gpu-sim` | ambele trec | PASS (compilare); rulare în QEMU: NOT TESTED |
| T30 | Fereastra C-36 pe workstation | `docs/FEREASTRA_C-36.md` | — | — | NOT TESTED (posibil 29 oct, cu Adrian) |

## Rezultate — sesiunea 5 (26 oct 2026, cloud, fără placă)
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1–T4 | format, clippy, teste, build no_std | `sh tools/check.sh` (rustc 1.99.1) | toate trec | 134 trecute, 0 eșuate, 1 ignorat; build OK | PASS |
| T7 | CI pe GitHub | push pe `g0-g2-fundatie` | verde | rularea 14–15 roșie (lint clippy 1.99), reparat; 16–18 verzi | PASS |
| T23 | I4: firmware 570.144 (mărime + SHA-256) | `cargo test -p nvidia-ga104 firmware sha256` | 8 teste | conform | PASS (SIMULARE; sumele din fișierele oficiale) |
| T24 | I5–I8: WPR meta, LibOS, cozi, RPC, ELF + GSP simulat | `cargo test -p nvidia-ga104 gsp` | 20 de teste | conform | PASS (SIMULARE) |
| T25 | NAK: `add64.comp` → SM 8.6 | `sh tools/nak/build.sh add64` | binar determinist | 240 B, SHA-256 `a890b75c…98ce`, de 2 ori la fel | PASS (compilare); rulare NOT TESTED |
| T26 | Fereastra B0 pe workstation | partea B0 | — | — | NOT TESTED (planificată 27–28 oct) |

## Rezultate — sesiunea 3 (23 oct 2026, cloud, fără placă)
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1–T4 | format, clippy, teste, build no_std | `sh tools/check.sh` | toate trec | 102 trecute, 0 eșuate, 1 ignorat; build OK | PASS |
| T7 | CI pe GitHub | push pe `g0-g2-fundatie` | verde | rulările 7–8 verzi (repo public) | PASS |
| T19 | FWSEC_PROD găsit, ucode necitit | `cargo test -p nvidia-ga104 fwsec` | 11 teste; nicio citire în ucode | conform | PASS (SIMULARE) |
| T20 | regiunile GSP în 8 GiB | `cargo test -p nvidia-ga104 fb_layout` | 6 teste | conform | PASS (aritmetică) |
| T21 | radix3 | `cargo test -p nvidia-ga104 radix3` | 3 teste | conform | PASS (aritmetică) |
| T22 | tag-ul `stabil-2026-10-23` pe GitHub | `git push origin stabil-2026-10-23` | urcat | HTTP 403 (politica sesiunii cloud) | BLOCKED |

## Rezultate — sesiunea 2 (23 oct 2026, cloud, fără placă)
Hardware: container cloud Linux, Intel Xeon 2,1 GHz. **Fără placă NVIDIA.** Toolchain: rustc 1.97.0 stable, `x86_64-unknown-linux-gnu`.

| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1–T4 | format, clippy, teste, build no_std | `sh tools/check.sh` | toate trec | 82 trecute, 0 eșuate, 1 ignorat; build OK | PASS |
| T5 | logică hardware SIMULATĂ (nivel 0, BAR, I2, I3, configurația workstation) | inclus în T3 | zero scrieri; opriri cinstite | conform | PASS (SIMULARE) |
| T7 | CI pe GitHub | push pe `g0-g2-fundatie` | joburile `crates (stable/nightly)` verzi | verzi (rulările 2–4); rularea 1 roșie doar la `kernel-integration` (scos); rularea 5: joburile n-au primit calculator (0 pași), reluată o dată, la fel | PASS până la rularea 4; BLOCKED la 5 |
| T16 | matmul pe blocuri = referința | `cargo test -p gpu-compute` | bit cu bit | bit cu bit, 7 forme | PASS |
| T17 | bf16 | idem | 65 536/65 536 dus-întors; rotunjire = oracol pe 200 000 de valori | conform | PASS |
| T18 | viteză matmul 64×768×768 (orientativ) | `cargo test --release -p gpu-compute -- --ignored --nocapture` | — | referință 1,2 GFLOP/s, blocuri 14,2, bf16 14,5 (gazdă, un fir) | INFO |
| T8, T10–T15 | QEMU, placa fizică | — | — | — | NOT TESTED / BLOCKED |

## Rezultate — sesiunea 1 (23 oct 2026)
Hardware: laptop Lenovo, i5-1135G7, Iris Xe, Windows 11 Pro 26300. **Fără placă NVIDIA.**
Cod: ramura `g0-g2-fundatie`. Toolchain: rustc 1.99.1 `stable-x86_64-pc-windows-gnu`.

| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1 | format | `cargo fmt --all -- --check` | fără diferențe | fără diferențe, cod 0 | PASS |
| T2 | analiză statică | `cargo clippy --workspace --all-targets -- -D warnings` | 0 avertismente | 0, cod 0 | PASS |
| T3 | teste pe gazdă | `cargo test --workspace --release` | toate trec | 58 trecute, 0 eșuate | PASS |
| T4 | build `no_std` | `cargo build --workspace --target x86_64-unknown-none --release` | compilează | compilează, cod 0 | PASS |
| T5 | logică hardware SIMULATĂ | inclus în T3 (`gpu-sim`) | nivel 0 fără scrieri; dimensionare restaurată | conform | PASS (SIMULARE) |
| T6 | patch-ul de integrare se aplică | `git apply --check` pe `cyber-universe@2d038de` | se aplică | se aplică, cod 0 | PASS |
| T7 | kernel compilat cu patch-ul | job CI `kernel-integration` | compilează | nerulat local (modificarea clonei OS a fost refuzată de politica de permisiuni) | NOT TESTED |
| T8 | boot QEMU + `gpu` | QEMU | `gpu: nicio placa GA104` | QEMU indisponibil | NOT TESTED |
| T9 | inventar K0 (script) | `tools/k0-inventar-workstation.ps1` pe laptop | rulează, doar citire | rulează; BAR-urile Iris Xe 16 MiB / 256 MiB | PASS (nu e workstation) |
| T10 | identificare RTX 3070 Ti fizic | `gpu identifica` | BOOT_0 = 0x174xxxxx | — | BLOCKED (B1) |
| T11–T15 | memorie, transfer, prima comandă, verificare, repetare | — | — | — | BLOCKED (I4–I10) |

## Plan pentru seara de testare (în ordine; nu se sare peste o etapă eșuată)

### A. Fără workstation (oricând)
1. `sh tools/check.sh` → T1–T4 PASS.
2. Mută `ci/github-actions-ci.yml` în `.github/workflows/ci.yml` (cu un token care are scope-ul `workflow`), push, apoi CI: joburile `crates (stable)`, `crates (nightly)`, `kernel-integration` → T7.
3. Într-o clonă locală `cyber-universe` (de Adrian sau cu acordul lui): aplică patch-ul, `tools/build.sh --test`, scenariul
   `gpu` (mai jos) în QEMU → T8. Așteptat pe serial: `gpu: nicio placa GA104` (QEMU are VGA `1234:1111`).
   Scenariu propus pentru `tests/scenarios.txt` (necesită `test-shell`):
   ```
   gpu       | test-shell       | success | TASTATURA: astept taste;;cyber> gpu;;gpu: nicio placa GA104;;cyber> gpu plan;;I10;;cyber> test-gata;;TEST READY | | keys=g p u ret g p u spc p l a n ret t e s t minus g a t a ret
   ```

### B0. Fereastra de pe workstation, 27 sau 28 oct 2026 (~30 min) — lista exactă (actualizată 26 oct, sesiunea 5)

**Scop:** primele citiri reale ale plăcii (I0–I3, FWSEC) + verificarea fișierelor firmware de pe stick (I4).
**Nimic din fereastra asta nu scrie în placa video.** Pașii I5+ (Bus Master, falcon, GSP) **nu** se fac acum: codul care scrie în placă
nu e încă scris; când va fi, va avea o fereastră separată, cu Adrian de față și o listă nouă.

**Pregătire (acasă sau la birou, înainte; ~10 min, o singură dată):**
1. Release-ul OS de folosit: **`os-192`** (are `gpu gfw`, `gpu vbios` din C-19 și rândul „ecran:” în `hw` din C-26). Dacă OS-ul are și comanda `gpu firmware`
   (propunerea din `integration/cyber-universe/I4_gpu-firmware.md`), cu atât mai bine; dacă nu, pasul 8 de mai jos se sare.
2. Pe stick, în rădăcina lui, creează folderele `firmware\nvidia\ga102\gsp\` și copiază acolo, din pachetul oficial `linux-firmware`
   (https://gitlab.com/kernel-firmware/linux-firmware → folderul `nvidia/ga102/gsp/`, butonul de descărcare al fiecărui fișier):
   `booter_load-570.144.bin`, `booter_unload-570.144.bin`, `bootloader-570.144.bin`, `gsp-570.144.bin` (~61 MB) și, în `firmware\`,
   fișierul de licență `LICENCE.nvidia`. **Nu le pune pe GitHub, în hub sau în alt repo.**
3. Verifică pe Windows sumele (PowerShell, în folderul cu fișierele): `Get-FileHash *.bin -Algorithm SHA256`.
   Trebuie să iasă exact valorile din `docs/COMPATIBILITATE_FIRMWARE.md` (de ex. `gsp-570.144.bin` → `A8C3EBEE…7F7F`). Dacă una diferă: alt fișier, nu-l folosi.

**În fereastră (Kronos oprit și salvat; Secure Boot oprit doar pentru test):**
| # | Ce faci | Ce trebuie să vezi (PASS) | Dacă nu |
|---|---|---|---|
| 1 | Windows: `powershell -ExecutionPolicy Bypass -File tools\k0-inventar-workstation.ps1` | fișierul `rezultate\k0-*.txt` | se notează, se continuă |
| 2 | Repornire, F12 → stick → **Esc** → `cyber>` | promptul | oprire, poză |
| 3 | `lspci` | `10de:2482` la `b3:00.0` | oprire (placa nu e văzută) |
| 4 | `gpu identifica` + `log gpu 40` | RTX 3070 Ti, BOOT_0 `& 0x1ff00000 = 0x17400000`, „a refuzat 0 scrieri” | oprire, poză |
| 5 | `gpu plan` | oprirea la **I4** (Release fixat înainte de sesiunea 5) sau **I5** (după) | poză, se continuă |
| 6 | `gpu gfw` (I2) | `0x118128` bit 0 = 1, progres `0xff`, < 4 s, 0 scrieri | oprire (nu se trece la 7) |
| 7 | `gpu vbios` (I3, I3b) | `aa55` sau `NVGI` la `0x300000`; PCI-AT + EFI + FWSEC (`0xe0`); BIT cu `0x70`; FWSEC_PROD (`0x85`) găsit; 0 scrieri; ecranul viu | poză cu valorile; BLOCKED dacă nu e semnătură |
| 8 | `gpu firmware` (I4, dacă există în OS) | 4 rânduri „ok” (nume, mărime, SHA-256 = cele din tabel); 0 scrieri în placă | poză; „marimea”/„SHA-256” = alt fișier pe stick |
| 9 | `hw`, `log pci 40` | poza rândului „ecran: framebuffer la adresa fizica 0x…, mărime, LxI” (os-192) + BAR1 al plăcii (din `lspci`/`gpu identifica`): din ele `vram_window::screen_vram_range` calculează zona din VRAM pe care driverul n-o atinge niciodată | dacă adresa nu e în BAR1: se notează (zona se alege conservator) |
| 10 | `reboot` → Windows, Secure Boot pe **Enabled**, Kronos pornit, `nvidia-smi` OK | Kronos merge ca înainte | anunță sesiunea Kronos |

**Ce trimiți înapoi:** pozele de la pașii 3–8 (sau textul de pe ecran) → hub `rezultate/2026-10-2x-workstation-gpu/`. Durată: ~20–30 min.

### P. Proba de referință cu tinygrad (C-20 A) — PREGĂTITĂ, NERULATĂ; doar cu Adrian de față, după B0
**Ce e:** pornim workstation-ul de pe un stick cu Linux (nu Cyber Universe) și lăsăm driverul tinygrad (MIT) să pornească placa și să facă
o singură adunare de 64 de numere. Fiecare scriere în registre se notează (`NV_DEBUG=4`) → „jurnalul de aur” cu care se compară apoi
driverul nostru, întâi în simulare (`gpu-sim::golden_log`). **Asta scrie în placa reală** (prin tinygrad), deci intră doar în fereastra aprobată.
Ce NU prinde jurnalul: citirile, soneria (`BAR0+0xbb0000`), scrierile în VRAM prin BAR1 și în memoria calculatorului.

Pregătire (cu o zi înainte, pe laptop; ~20 min):
1. Un al doilea stick: Ubuntu 24.04 LTS „live” (de pe ubuntu.com, scris cu Rufus/balenaEtcher). La pornire, la meniul GRUB, `e` și se adaugă
   la linia `linux`: `modprobe.blacklist=nouveau,nvidia` (altfel nouveau ia placa).
2. Pe stickul cu date (cel de la B0) trebuie să existe deja `firmware/nvidia/ga102/gsp/*-570.144.bin` + `tools/proba-tinygrad/proba.sh`
   (copiat din acest repo).
3. Laptopul ThinkPad în aceeași rețea, cu `ssh` (comenzile se dau de acolo: ecranul workstation-ului se stinge când tinygrad preia placa).

În fereastră (~30 min, Kronos oprit și salvat):
| Min | Pas | Comanda (pe workstation, prin SSH) |
|---|---|---|
| 0–5 | pornire de pe stick-ul Ubuntu (cu blacklist); rețea prin cablu; `sudo apt install -y openssh-server zstd python3-venv git` | — |
| 5–8 | de pe laptop: `ssh ubuntu@<ip>`; găsirea stickului cu date: `lsblk` → ex. `/media/ubuntu/CYBER` | `lsblk` |
| 8–25 | proba | `sudo CONFIRM_ADRIAN=da bash /media/ubuntu/CYBER/tools/proba-tinygrad/proba.sh /media/ubuntu/CYBER` |
| 25–28 | repornire în Windows; `nvidia-smi` arată placa; Kronos pornește | — |
| 28–30 | folderul `proba-tinygrad/<data>/` de pe stick → hub `rezultate/<data>-proba-tinygrad/` (`jurnal.txt`, `consola.txt`, `SHA256SUMS`) | — |

Scriptul se oprește singur (fără nicio scriere) dacă: lipsește `CONFIRM_ADRIAN=da`, placa `10de:2482` nu e găsită, nouveau/nvidia e încărcat,
sumele SHA-256 ale firmware-ului nu se potrivesc. Fixat: tinygrad `9cf7829a6`, tinymesa `25.2.7.2` (NAK, compilatorul ales de Adrian).
Rezultat așteptat: o linie `PROBA: OK` + câteva mii de linii `wreg:`. Orice altceva = notat ca atare, fără reîncercări în aceeași fereastră.
DEDUS, de verificat atunci: tinygrad citește `/sys/bus/pci/...` și `/proc/self/pagemap` (root, VT-d oprit — ca la noi) și nu descarcă
nimic în afară de pachetele Python (firmware-ul îl ia din `/lib/firmware`, unde îl pune scriptul).
După: `gpu-sim::golden_log::parse` citește `jurnal.txt` (testat pe un jurnal scris de mână, 3 teste SIMULARE).

### B. Pe workstation — numai în fereastra aprobată de Adrian (Kronos oprit) — lista detaliată din sesiunea 2 (context)
Condiții: Kronos oprit și salvat; Adrian prezent; stick cu Release-ul care conține patch-ul; Secure Boot oprit temporar (decizia lui Adrian).
1. **Windows, înainte:** `powershell -ExecutionPolicy Bypass -File tools\k0-inventar-workstation.ps1` → fișierul `rezultate\k0-*.txt`.
   Notează manual din BIOS: IOMMU (AMD-Vi/VT-d), Above 4G Decoding, Resizable BAR, CSM.
2. Pornește de pe stick → `cyber>`.
3. `lspci` → rândul `10de:2482` (și `10de:228b` audio). Poză.
4. `gpu identifica` → poză + `log gpu 40`. PASS dacă: model RTX 3070 Ti, BAR0 mem32, D0, `NV_PMC_BOOT_0` cu `& 0x1ff00000 = 0x17400000`,
   `consecvent true`, **`0 scrieri in placa`**. Notează bitul 29 al BOOT_42.
5. `gpu plan` → poză (după sesiunea 2: oprirea trebuie să fie la **I4**, nu la I2).
5a. **I2 — GFW_BOOT (doar citire)** — necesită în OS o comandă care mapează UC primii **4 MiB** din BAR0 (vezi hub `integrare/cyber-gpu/`):
    `gpu gfw` → poză. PASS: `0x118128` cu bitul 0 = 1, `0x118234 & 0xff = 0xff`, în < 4 s (pe o placă pornită de mult timp: la prima citire),
    `0 scrieri`. FAIL: timeout (notează ultimele valori), `0xbadf…` (eroare de acces) sau `0xffffffff`. **Dacă FAIL: oprire, nu se trece la 5b.**
5b. **I3 — VBIOS (doar citire)**: `gpu vbios` → poză. PASS: primul cuvânt la BAR0+0x300000 = `…aa55` sau `0x4947564e` („NVGI”); lista de imagini
    conține PCI-AT, EFI și cel puțin o FWSEC (`0xe0`); tabela BIT are jetonul `0x70`; numărul de citiri < 300 000; `0 scrieri`; ecranul rămâne viu.
    Comparație: Windows raportează VBIOS 94.04.65.00.3e — dacă OS-ul poate afișa șirul de versiune din imagine, trebuie să coincidă.
    **28 oct, os-195: I0–I2 PASS pe placa reală; I3 FAIL la `0x24600` (imagine „NV”, `0x4e56`) → reparat în SIMULARE (C-30).** La fereastra
    următoare (Release cu C-31): `gpu vbios` + `gpu vbios --detalii` → poze cu toate rândurile (lista imaginilor și rezumatul cu „0 scrieri”).
    Dacă primul cuvânt nu e o semnătură: **BLOCKED** (umbrirea ROM ar cere o scriere) — se notează valoarea și se discută cu Adrian.
5c. **I3b — FWSEC (doar citire, aceeași fereastră PROM)**, numai dacă 5b a trecut și OS-ul afișează rezultatul lui `fwsec::locate`:
    PASS: tabela PMU conține aplicația `0x85`; descriptorul are versiunea 3; numărul de semnături = biții din `signature_versions`;
    ucode-ul încape în imaginile FWSEC; `0 scrieri`. Notează: numărul de semnături, `ucode_id`, `engine_id_mask`, mărimile IMEM/DMEM.
    FAIL la versiune/pointer = citirea noastră a formatului e greșită → se notează valorile, nu se ghicește.
6. Opțional, numai dacă Adrian acceptă riscul: `gpu dimensiuni da` → mărimi BAR; ecranul trebuie să rămână viu.
7. `reboot` în Windows; verifică că Kronos pornește normal (nvidia-smi OK).

### Ce se colectează pentru fiecare test
identificator, hardware/configurație, commit, comanda, așteptat, obținut, cod de ieșire/jurnal (`log gpu`), PASS/FAIL/BLOCKED/NOT TESTED
→ în acest tabel și în `Cyber-hub/rezultate/`.
