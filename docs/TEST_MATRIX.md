# Matricea de teste

## Rezultate — sesiunea 134 (21 nov 2026, ~12:10–13:10, cloud, fără placă) — C-155 punctul 3: argmax pe placă, SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T145a | proba: lista lipită-ln + dietă + copia logit-urilor + argmax în același simbol, 16 poziții, `EachLaunch` și `PerLayer`: cele 4 cuvinte = `argmax_result` al logit-urilor copiate, simbolul = argmax-ul procesorului, oracolul trece | `cargo test -p nvidia-ga104 --release --test sim_c155_argmax` | conform | conform | PASS (SIMULARE) |
| T145b | viteza maximă: doar argmax (fără copie), `Generation::run_top` = aceleași simboluri ca `run` cu copie + verificare; bufferul logit-urilor rămâne umplutura; o lansare mai puțin pe simbol | `… t145b` | conform | conform | PASS (SIMULARE) |
| T145c | algoritmul programului (256 fire, runde de 8, fluture xor 16…1, 8 alegeri) refăcut pe procesor = `argmax_result`, n de la 1 la 50 257, egalități, −0/+0, NaN, ±∞, subnormale, −FLT_MAX | `… t145c` | conform | conform | PASS |
| T145d | octeții = dump-ul NAK (SHA-256), `shfl` + `bar`, QMD 256 fire / 96 B / o barieră; `work_list` refuză argmax-ul | `… t145d` | conform | conform | PASS |
| T145e | NaN pus în logit-uri → cuvântul 0 = `0xffffffff`, 1 nefinit, oprire `LogitiNefiniti` | `… t145e` | conform | conform | PASS (SIMULARE) |
| T145f | program plantat care nu scrie → `Eroare` la poziția 0, niciun simbol; program „cu unu în plus” → prins de proba cu copie | `… t145f` | conform | conform | PASS (SIMULARE) |
| T145g | umplutura `0xEE`, alt n → eroare, nu simbol | `… t145g` | conform | conform | PASS |
| — | pe RTX 3070 Ti | — | — | — | NOT TESTED |

## Rezultate — sesiunea 126 (20 nov 2026, ~19:25–20:00, cloud, fără placă) — C-147 punctul 5: circuitele speciale (tensor cores), SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T142a | `mma16_f16`: adresele în bufferul constant la 0x700 (NVK 26.3.1), aceeași poziție în driver și în model | `cargo test -p nvidia-ga104 --release --test sim_c147_mma16` | conform | conform | PASS |
| T142b | D = A·B + C (16×16, A/B pe 16 biți, C/D f32), trimis ca de driver (pushbuffer, QMD, semafor), 8 seturi de numere: fiecare element în toleranța scrisă (`17 · 2⁻²³ · (|c| + Σ|a·b|)`) față de referința f64; modelul (ipoteza „blocuri de 8, trunchiere”) nu e identic peste tot | `… sim_c147_mma16 t142b` | conform | conform | PASS (SIMULARE) |
| T142c | program schimbat cu un octet, adrese puse la 0x28 (așezarea uneltei vechi), 64 de fire, 2 blocuri → refuzate, D rămâne NaN | `… t142c` | conform | conform | PASS (SIMULARE) |
| T142d | `gpu_compute::mma`: toate cele 65 536 de valori pe 16 biți dus-întors exact; rotunjirea la par; identitate·B = B; 5 greșeli plantate prinse (1 %, B transpus, C uitat, lungime greșită, NaN) | `cargo test -p gpu-compute --release mma` | conform | conform | PASS |
| T142e | programul compilat are 78 de instrucțiuni și suma SHA-256 scrisă; bufferul constant refuză adrese nealiniate sau 0 | `cargo test -p nvidia-ga104 --release --lib mma16` | conform | conform | PASS |
| — | pe RTX 3070 Ti | — | — | — | NOT TESTED (doar după „da”-ul separat al lui Adrian) |

## Rezultate — sesiunea 124 (20 nov 2026, ~13:25–14:20, cloud, fără placă) — C-151 punctul 1: tabloul de bord RUSD, SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T140a | pornire → citire → oprire față de modelul `SimRusd`: valorile (45,5 °C, 210/405 MHz, P8, 21 W), ordinea INIT, interval, mască / mască 0, INIT 0; cititorul `parse_kd` | `cargo test -p nvidia-ga104 --release --test sim_c151_rusd` | conform | conform | PASS (SIMULARE) |
| T140b | octeții celor 5 comenzi (adresa, `0 / 500 ms`, `0x1b / 0`, `0 / 0`, `0`); clientul și subdispozitivul nostru | `… sim_c151_rusd t140b` | conform | conform | PASS |
| T140c | `INIT` refuzat → o singură comandă; intervalul refuzat → pagina dată înapoi; fără model → `INVALID_COMMAND` ca rând | `… t140c` | conform | conform | PASS (SIMULARE) |
| T141a | dieta driverului (C-147 punctul 7): pushbuffer prin `ShadowedMemory` + înregistrări nezerorizate, 6 poziții, lista de 51 și cea lipită de 24: logit-uri, cache-uri și durate identice bit cu bit cu rularea fără dietă; cuvinte scrise pe simbol 623 → 55 și 299 → 28 | `cargo test -p nvidia-ga104 --release --test sim_c147_diet` | conform | conform | PASS (SIMULARE) |
| T141b | înregistrările rămase de la trimiterea de dinainte nu trec (`analyze_sequence`), `records_seen` corect după o oprire la jumătate | `… sim_c147_diet stale` | conform | conform | PASS |
| T141c | `driver_split`: ceasul rândurilor, două simboluri din jurnalul os-322 împărțite pe ferestre (fiecare µs numărat o dată), simbolurile lente lăsate deoparte; `ShadowedMemory` (doar cuvintele schimbate, eroarea uită umbra) | `cargo test -p nvidia-ga104 --release --lib driver_split kronos_fast` | conform | conform | PASS |
| T142a | P3 gata de legat (C-155 punctul 1): textul întreg (prompt + greedy până la context 16) prin `kronos_diag::send_token_list(Fused)` + copierea în RAM, cu dieta: logit-uri identice bit cu bit cu lista simplă, oracolul pe procesor trece la fiecare poziție (max 3,3e-7), `EachLaunch` și `PerLayer` | `cargo test -p nvidia-ga104 --release --test sim_c155_p3` | conform | conform | PASS (SIMULARE) |
| T142b | tăieturile `PerLayer` ale listei lipite (7 pe strat), 88 față de 196 lansări la #513, pushbuffer `EachLaunch` 4 256 B față de 9 440 B, programele listei | `… sim_c155_p3 per_layer` | conform | conform | PASS |
| T142c | un program lipit greșit (nu scrie nimic), plantat în modelul plăcii: oracolul îl prinde | `… sim_c155_p3 planted` | conform | conform | PASS |
| T140d | oprirea refuzată → nota „pagina rămâne rezervată” | `… t140d` | conform | conform | PASS (SIMULARE) |
| T140e | planul oprit: nimic trimis; planuri greșite refuzate înainte de trimitere | `… t140e` | conform | conform | PASS |
| T140f | citirea cu marcă de timp: scriere în curs, marcă schimbată, secțiune lipsă, ocupată după 10 încercări; placa fierbinte (motiv termic) | `… t140f` | conform | conform | PASS (SIMULARE) |
| T140g | `dashboard_at_end` într-un apel | `… t140g` | conform | conform | PASS (SIMULARE) |
| T140u | forma octeților, așezarea în 464 de octeți, `RUSD_SEQ_DATA_VALID` | `cargo test -p nvidia-ga104 --release --lib rusd` | conform | conform | PASS |
| T140-placă | GSP-RM acceptă `INIT` de la clientul nostru? pagina completată? valorile reale? | Release OS, modul ciclu, la sfârșitul lanțului | — | — | NOT TESTED |

`tools/check.sh` (Rust 1.100.0): format, clippy `-D warnings`, 597 teste (6 ignorate), build no_std — PASS.

## Rezultate — sesiunea 121 (20 nov 2026, ~04:25–05:30, cloud, fără placă) — C-147 punctul 3 (etapa 4): `q8_matvec32` + cursa `q8_race`, SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T137a | oracolul `row_gpu_order_w32`: în marjă față de procesor și de `q8_matvec` la 5 forme; diferit în ultimii biți la 64×768; 32 de fire ocupate de la 256 de intrări; un sfert calculat de mână | `cargo test -p gpu-compute q8` | conform | conform | PASS |
| T137b | `q8_matvec32`: programul = cel compilat (199 instr., SHA-256), bufferul constant = al lui `q8_matvec` + x aliniat la 16, QMD diferă doar prin registre; verdicte | `cargo test -p nvidia-ga104 q8_matvec32` | conform | conform | PASS |
| T137c | pe canalul de lucru: identic bit cu bit cu oracolul la 5 forme (până la 3072×768 și 768×3072), în marja I12; sub modelul „date vechi”: identic | `cargo test -p nvidia-ga104 --release --test sim_c147_w32` | conform | conform | PASS (SIMULARE) |
| T137d | 5 greșeli plantate (x nealiniat, x deplasat, grilă scurtă, număr schimbat, ordinea veche): niciuna nu trece | `… sim_c147_w32 planted` | conform | conform | PASS (SIMULARE) |
| T137e | cursa: A B B A, mărcile citite, y judecat, rândurile `kr` scrise și citite; regula câștigului (≥ 50 ‰, proba trecută; egalitate → rămâne) | `… sim_c147_w32 the_race` + `cargo test -p nvidia-ga104 q8_race` | conform | conform | PASS (SIMULARE) |
| T137f | unealta NAK reconstruită: `q8_matvec` (`d8c118cc…`) și `q8_fused` (`d86e53e2…`) identice; `q8_matvec32`: 2 × `ld.global.constant.b128`, 22 registre, fără `BAR.SYNC` | `sh tools/nak/build.sh q8_matvec32` + citirea listării | conform | conform | PASS (COMPILAT) |
| T137g | `tools/check.sh` (Rust 1.100.0): format, clippy `-D warnings`, 575 teste (6 ignorate), build no_std | `tools/check.sh` | verde | verde | PASS |
| T137h | pe RTX 3070 Ti: cursa (`kr`), `q8_matvec32` identic cu oracolul | (OS, după legare) | — | — | NOT TESTED |

## Rezultate — sesiunea 120 (20 nov 2026, ~01:25–02:00, cloud, fără placă) — C-147 punctul 2 (etapa 3a): programul lipit `q8_fused`, SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T136a | `q8_fused::cbuf`: așezarea în bufferul constant (72 octeți), refuzuri (QKV ≠ 3·E, rezidual = ieșire, nealiniat, cache lipsă) | `cargo test -p nvidia-ga104 q8_fused` | conform | conform | PASS |
| T136b | stratul lipit (7 lansări) față de cele 16: `x` + cache-urile identice bit cu bit, E 64 × 6 poziții și E 768 × 2; în toleranța I13 | `cargo test -p nvidia-ga104 --release --test sim_c147_fused fused_layer` | conform | conform | PASS (SIMULARE) |
| T136c | model mic (3 straturi), 4 poziții, trimis ca de OS: logit-uri și cache-uri identice bit cu bit; felurile `kp` lipite = lansările | `… sim_c147_fused fused_model` | conform | conform | PASS (SIMULARE) |
| T136d | numărători (#513: 195 → 87), `PROGRAMS_FUSED`, registrele vechi 19, `work_list` refuză programul lipit; 4 greșeli plantate neprinse = 0 | `… sim_c147_fused` | conform | conform | PASS (SIMULARE) |
| T136e | unealta NAK reconstruită: `q8_matvec` identic (SHA-256 `d8c118cc…`); `q8_fused` = produsul identic cu `q8_matvec`, GELU identic cu `gelu`, `fadd.ftz` pentru termen/rezidual, 19 registre, fără `BAR.SYNC` | `sh tools/nak/build.sh q8_fused` + citirea listării | conform | conform | PASS (COMPILAT) |
| T136f | tot pachetul | `sh tools/check.sh` (Rust 1.100.0) | conform | **561 PASS** (555 → +6), 6 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| T136g | pe RTX 3070 Ti: lista de 88, aceleași simboluri (și logit-uri) ca lista de 196 | — | — | — | NOT TESTED |

## Rezultate — sesiunea 119 (19 nov 2026, ~23:05–23:40, cloud, fără placă) — C-147 etapa 0 (măsurători), viteza maximă pregătită, limitele #651, toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T135a | felurile de lansare = lista trimisă de driver, unu la unu (cu/fără embedding, cu copierea în RAM; `EachLaunch` și `PerLayer`); suma pe feluri = timpul `kv` al acelorași simboluri; rândurile `kp` citite înapoi din textul jurnalului | `cargo test -p nvidia-ga104 --release --test sim_c147 kinds_are` | conform | conform | PASS (SIMULARE) |
| T135b | `KindProfile`: sume și mediane pe fel, refuză lungimi diferite / alt număr de lansări; rândurile `kp`, `kl`, `kc` citite înapoi cu `parse_journal` (64 MiB în 300 µs → 223 696 MB/s, 735 ‰) | `cargo test -p nvidia-ga104 kronos_perf` | conform | conform | PASS |
| T135c | copierea VRAM → VRAM cu `copy_b128`, de 3 ori într-o trimitere: destinația identică bit cu bit, cele 3 cuvinte verificate corecte, 3 timpi; o înregistrare lipsă = eroare | `… sim_c147 vram_copy` | conform | conform | PASS (SIMULARE) |
| T135d | starea de performanță: P0…P15 / mască ciudată / refuz; refuzul GSP-ului simulat = rând `ks … 8000xxxx`, nu eroare; viteza maximă `Blocked` implicit, durată finită (0, >3600, infinit refuzate), ștergerea în aceeași formă; cele două forme (A NVIDIA, B tinygrad) octet cu octet | `cargo test -p nvidia-ga104 gsp::perf` + `… sim_c147 pstate` | conform | conform | PASS (SIMULARE) |
| T135e | limitele #651: 388 lansări, 64 KiB, pas 389 (> 0x100 al OS-ului), VRAM ~250–290 MiB; `check` numește zona și pasul la setările lui #513 | `cargo test -p nvidia-ga104 k651_limits` | conform | conform | PASS |
| T135f | tot pachetul | `sh tools/check.sh` (Rust 1.100.0) | conform | **555 PASS** (545 → +10), 6 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| T135g | pe RTX 3070 Ti: rândurile `kp`/`kl`/`ks`/`kc` în `os-309` | — | — | — | NOT TESTED (după C-148) |

## Rezultate — sesiunea 116 (19 nov 2026, ~10:50, cloud, fără placă) — C-145: pauzele driverului (jurnal ținut, umbra lansărilor, timpul pe bucăți, verificarea după placă), toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T134a | jurnalul cu rutină: ține doar scrierile la adresele date și notele date, cel mult `max_pending`; pași, valori, citiri, alte scrieri merg imediat și iau cu ele ce era ținut; aceleași rânduri în aceeași ordine ca jurnalul din memorie; fără destinație nu schimbă nimic | `cargo test -p gpu-core routine_stretch` | conform | conform | PASS |
| T134b | `kronos_fast`: rutina = fereastra, cele 3 registre de invalidare, soneria (nimic altceva); umbra dă doar cuvintele schimbate, refuză nealiniat / în afară; `TimeSplit` cu cifrele `os-307` dă 3,39 / 0,24 simboluri/s și rândurile `kt` citite înapoi | `cargo test -p nvidia-ga104 kronos_fast` | conform | conform | PASS |
| T134c | un simbol ca în `os-307` (15 rânduri) × 18 poziții: stick-ul cu rânduri ținute = stick-ul rând cu rând, text identic; 18 goliri în loc de 270 | `cargo test -p nvidia-ga104 --release --test sim_c145 held_routine` | conform | conform | PASS (SIMULARE) |
| T134d | umbra sloturilor pe canalul simulat: aceleași simboluri, logit-uri identice bit cu bit, 18 640 B scriși față de 147 456 B (8 poziții × 36 lansări) | `… sim_c145 shadow` | conform | conform | PASS (SIMULARE) |
| T134e | verificarea după placă: aceleași simboluri și aceeași abatere ca verificarea între simboluri; la refuz (poziția 4 sau 1 din text) aceeași oprire, aceleași simboluri păstrate | `… sim_c145 check_after` | conform | conform | PASS (SIMULARE) |
| T134f | pe RTX 3070 Ti: `os-308` cu C-145 legat, rândurile `kt` | — | — | — | NOT TESTED (ciclul automat, după C-146) |

## Rezultate — sesiunea 113 (18 nov 2026, ~19:25–20:00, cloud, fără placă) — C-142: forma lui #513 în `gpu-sim`, limitele, contoarele de viteză, toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T133a | limitele lui #513 din forme: 196 lansări, 7 872 / 9 440 / 712 B de comenzi, zona 32 KiB, pas ≥ 197, ~106 MiB greutăți în 14 bucăți de 8 MiB, buget < 160 MiB; 24 de straturi → 64 KiB | `cargo test -p nvidia-ga104 k513_limits` | conform | conform | PASS (SIMULARE) |
| T133b | `ModelLimits::check` numește fiecare limită neîndeplinită (zona 16 KiB, pas 0x80, VRAM mic, bucată mică); bucată 0 / nemultiplu de 16 refuzată | `… check_names_every_limit` | conform | conform | PASS |
| T133c | contoarele: placa și procesorul separat, urcarea separat; mediane, simboluri/s, MB/s; ceasul plăcii înapoi = eroare; ceas lipsă = „?”, nu ghicit | `… speed_counts`, `… a_missing_host_clock` | conform | conform | PASS |
| T133d | rândurile `kv` în jurnal (5 + 4/poziție + 9, sau 14 fără pozițiile) și citite înapoi; o rulare mai veche din același text e ignorată | `… journal_rows_read_back` | conform | conform | PASS |
| T133e | **forma lui #513** (12 straturi, E 768, 12 capete, vocabular 16 384, context 512; greutăți generate) pe canalul simulat: zona 32 KiB, urcare prin arenă în 14 bucăți ≤ 8 MiB (recitite din VRAM = identice), 4 poziții × 1 sonerie de 196 lansări `EachLaunch`, logit-urile din RAM = procesorul (≤ 1e-3), greedy, rândurile `kv` citite înapoi | `cargo test -p nvidia-ga104 --release --test sim_k513` | conform | conform (16 s pe gazdă) | PASS (SIMULARE) |
| T133f | verificarea OS-ului refuză zona de 16 KiB și VRAM-ul mic înainte de orice urcare | `… the_os_check_refuses` | conform | conform | PASS |
| T133g | `gpu-sim`: copierea memoriei pe pagini + traducerea ținută pe pagină — aceleași rezultate în toate testele, pachetul de teste mai rapid | `sh tools/check.sh` | toate trec | **538 PASS**, 6 ignorate (531 → +7); format, clippy `-D warnings`, build no_std; 81 s (înainte 92 s fără testul nou) | PASS |
| T133h | pe RTX 3070 Ti: `kronos --gpu` cu #513, contoarele `kv` | — | — | — | NOT TESTED (după C-141, cu Adrian) |
| T133i | pe RTX 3070 Ti: FLR coboară WPR2 (propunerea din `docs/C-142_K513_VITEZA_FLR.md` §4) | — | — | — | NOT TESTED (cere „da” + comandă OS) |

## Rezultate — sesiunea 112 (18 nov 2026, ~12:30–14:45, cloud, fără placă) — C-139: I14 după `os-305` (alt argmax pe placă): reparația (invalidare înaintea fiecărei lansări), scara și diagnosticul din jurnal (`kronos_diag`), toate SIMULARE

Ipoteza „date vechi” e pusă în modelul de GPU cu `CommandProcessor::model_stale_reads` (oprit implicit): e o **ipoteză**, nu o măsurătoare.

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T132a | secvența reparată = cea veche + o invalidare înaintea fiecărei lansări, cuvânt cu cuvânt; `sequence_words_fresh(n)` = 8 + 12·n | `cargo test -p nvidia-ga104 fresh_sequence` | conform | conform | PASS (SIMULARE) |
| T132b | tabloul plăcii pe ipoteză (linii de 4 și 128 B): un strat pe trimitere, 4 poziții, x scris de driver (I13) trece fără nicio citire veche; simbolul întreg într-o trimitere (I14) cade cu toate logit-urile finite; ambele reparații trec; cu memorie coerentă cele 3 moduri dau aceiași biți | `cargo test -p nvidia-ga104 --test sim_i14_diag the_board_picture` | conform | conform | PASS (SIMULARE) |
| T132c | secvențele I11 (copiere dus-întors) și I12 (copiere, W·x, copiere) trec și cu modelul „date vechi” pornit, ca pe placă | `cargo test -p nvidia-ga104 under_the_stale_read_model` | conform | conform | PASS (SIMULARE) |
| T132d | scara pe ipoteză: proba de control = DATE VECHI la normalizarea 1 a stratului 1 (valoarea lui x scrisă de embedding), reparația trece și e aleasă, 2 trimiteri; jurnalul citit înapoi spune același lucru; niciun rând nu arată ca un pas sau ca o scriere în registru; pe modelul sănătos tot reparația e aleasă | `… the_ladder_confirms` | conform | conform | PASS (SIMULARE) |
| T132e | 3 straturi: stratul ascuns arată citirea veche în rândurile lui K/V (sunt ce dă `qkvb` al stratului 0) | `… a_hidden_layer` | conform | conform | PASS (SIMULARE) |
| T132f | invalidarea dintre lansări fără efect → „o trimitere pe strat” e încercată și aleasă (trimiteri de 35, 35, 17, 16, 2 lansări) | `… the_ladder_falls_back` | conform | conform | PASS (SIMULARE) |
| T132g | o lansare greșită, fără date vechi: în ultimul strat e numită direct de simbolul întreg (ieșirea nu urmează din intrările din memorie); doar în primul strat → „moștenit”, iar prefixul stratului o numește; scara nu alege niciun fel | `… a_wrong_launch` | conform | conform | PASS (SIMULARE) |
| T132h | `embed` greșit: embedding-ul singur = lansare greșită; simbolul trece cu embedding-ul scris de driver (lansarea 0 netrimisă); un text trece așa; `embed` + GELU greșite → prefixele (cu embedding-ul driverului) numesc GELU | `… a_wrong_embedding` | conform | conform | PASS (SIMULARE) |
| T132i | o trimitere care nu se termină oprește scara pe loc; nimic nu mai e trimis; verdictul 5 în jurnal | `… a_submission_that_does_not_finish` | conform | conform | PASS (SIMULARE) |
| T132j | explicația unui simbol deja rulat: (1) date vechi — nimic trimis, RAM = VRAM, primele 8 logit-uri și abaterea în jurnal; (2) logit-urile neajunse în RAM — etapele bune, 512 valori rămase `eeeeeeee`, `vram_pass` | `… explaining_a_token` | conform | conform | PASS (SIMULARE) |
| T132k | un text întreg în felul ales (linii de 4 și 128 B): toate pozițiile trec față de procesor; pe straturi la fel, aceleași simboluri; trimis ca la `os-305`: oprit la poziția 0 | `… a_text_in_the_chosen_mode` | conform | conform | PASS (SIMULARE) |
| T132l | **pasul întreg, cum îl descrie nota pentru OS**: sănătos și pe ipoteză → ok cu reparația, nimic explicat; copierea în RAM nu ajunge → textele trec cu logit-urile din VRAM, pasul = ESEC cu motivul; `embed` greșit → textele trec cu embedding-ul driverului, pasul = ESEC cu motivul; lansare greșită la orice poziție → niciun text trimis; greșită doar de la poziția 1, în ultimul strat → numită de explicație; într-un strat ascuns → „moștenit”, apoi `locate` o numește | `… the_step_as_the_note` | conform | conform | PASS (SIMULARE) |
| T132m | o poziție ulterioară pe modelul sănătos: simbol întreg și prefixe curate în toate felurile (și cu embedding-ul driverului), bit cu bit la poziția 0 | `… a_later_position` | conform | conform | PASS (SIMULARE) |
| T132n | memoria citită ca gunoi (NaN, infinit, zero, biți la întâmplare, umplutură, valori uriașe), 2 și 3 straturi, toate lungimile de prefix: niciodată panică, niciodată „trece”, niciun fel ales; citiri scurte = eroare a citirii; rulat și în build de depanare (verificarea depășirilor) | `… garbage_in_memory` | conform | conform | PASS (SIMULARE) |
| T132o | urma programelor (`layer::trace_gpu`, `model::trace_gpu`) urmează motorul de pe procesor | `cargo test -p gpu-compute the_trace_follows` | conform | conform | PASS (SIMULARE) |
| T132p | unități `kronos_diag`: trimiterile pe fiecare mod (și mărimea zonei de comenzi: 1 760 B modelul de control, 9 440 B #513); `Way` și embedding-ul driverului; raportul logit-urilor; jurnalul dus-întors; numele etapelor încap într-un rând | `cargo test -p nvidia-ga104 --lib kronos_diag` | conform | conform | PASS (SIMULARE) |
| T132q | modelul de control real (`mic_q8_0.kgr`, vectorii din hub pregătiți local): pe ipoteză alt argmax decât cel cerut de vectori, cu toate logit-urile finite; scara alege reparația (148 de rânduri de jurnal); cele 3 texte dau exact vectorii lui Kronos cu reparația; „pe straturi” și cu embedding-ul driverului la fel | `KGR1_VECTORI=$PWD/target/vectori-kgr1 cargo test -p nvidia-ga104 --test sim_i14_diag kgr1_control -- --ignored` | conform | conform | PASS (SIMULARE, local; ignorat în CI — vectorii sunt în hubul privat) |
| T132r | marjele celor 29 de poziții de control pe motorul de pe procesor: cea mai mică 0,028 (de 28 de ori toleranța), poziția căzută la `os-305` 0,28 → nu e rotunjire | `… kgr1_margins -- --ignored` | conform | conform | PASS (local; ignorat în CI) |
| T132s | cele două lansări de după straturi, greșite singure: capul de ieșire („logit-uri = Emb * h”) și norma finală sunt numite direct de simbolul întreg (intrările lor sunt în memorie); toate prefixele sunt curate; niciun fel ales | `… a_wrong_final_launch` | conform | conform | PASS (SIMULARE) |
| T132t | o lansare doar **pe jumătate** făcută din valori vechi (rândurile impare ale lui `qkv` din stratul 1 cu h-ul stratului 0 — cum ar arăta cache-uri separate pe fiecare procesor al plăcii): verdictul rămâne „date vechi”, marcat „o parte din ieșire”; aceeași lansare cu rânduri greșite pe care nicio valoare veche nu le explică = „lansare greșită” | `… a_partly_stale` | conform | conform | PASS (SIMULARE) |
| T132u | tot pachetul | `bash tools/check.sh` | 531 PASS | **531 PASS** (507 → +24), 6 ignorate (4 → +2); format, clippy `-D warnings`, build no_std | PASS |
| — | I14 cu scara și reparația pe RTX 3070 Ti | — | — | — | NOT TESTED (fereastra cu Adrian, după `os-306`) |

## Rezultate — sesiunea 108 (17 nov 2026, ~16:20, cloud, fără placă) — C-135: I9 pasul 5 după `os-304` (jetonul refuzat cu `0x40`): `open_work_channel_c135` — forma B direct, canal fără notificator de erori (încercarea N), apoi ChID ales de noi + jeton calculat (F), toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T131a | modelul care joacă ipotezele „globalele sunt necesare” + „notificatorul trebuie să fie memorie video” (`fake_rm::{PromoteRule, TokenRule}`) dă pentru `--i9 c133` exact ce a dat placa la `os-304`: A `0x57` la `PROMOTE_CTX`, B trece de promovare și compute, `0x40` la `GET_WORK_SUBMIT_TOKEN`, programarea netrimisă; aceleași handle-uri ca în jurnalul plăcii | `cargo test -p nvidia-ga104 c135_the_hypothesis` | conform | conform | PASS (SIMULARE) |
| T131b | încercarea N (pe modelul de mai sus și pe cel permisiv): un singur grup; canalul trimite `hObjectError` = 0, `internalFlags` = `0x14`, `errorNotifierMem` zero, `flags` = 0; forma B (4 intrări) înaintea obiectului; jetonul vine de la RM; grupul programat; fără a doua încercare; fără pagină de notificator, fără PM | `cargo test -p nvidia-ga104 c135_try_n` | conform | conform | PASS (SIMULARE) |
| T131c | jetonul refuzat din alt motiv (câmpul neatins): F în **grup nou**, `flags` = `0x00202000` (ChID 16, primit întocmai de la model), **niciun** control de jeton pentru F, jeton = runlist `<< 16` \| 16, grupul lui N neprogramat, memorie și handle-uri separate | `cargo test -p nvidia-ga104 c135_try_f_names` | conform | conform | PASS (SIMULARE) |
| T131d | ce a purtat un jeton refuzat se **notează și nu se folosește**: (1) refuz într-un răspuns întreg care are deja un id — id-ul e în raport, canalul reușit e al lui F cu id-ul lui; (2) refuz în antetul RPC — „fără răspuns” | `cargo test -p nvidia-ga104 c135_a_refused_token` | conform | conform | PASS (SIMULARE) |
| T131e | doar jetonul refuzat al lui N duce la F; orice alt refuz (grup, context share, canal, `PROMOTE_CTX`, compute, programare) și un timeout la jeton opresc imediat, cu pasul în raport; în F orice refuz oprește | `cargo test -p nvidia-ga104 c135_other_failures` | conform | conform | PASS (SIMULARE) |
| T131f | ChID-ul lui F e deja ocupat (15 canale făcute înainte): refuzul RM (`NO_FREE_FIFOS` în model) e rezultatul; nu se ghicește alt id | `cargo test -p nvidia-ga104 c135_try_f_in_use` | conform | conform | PASS (SIMULARE) |
| T131g | greșeli de partea noastră (rădăcină, fără runlist GR0, runlist prea mare pentru sonerie, fără MAIN/PATCH/FECS_EVENT/PRIV_ACCESS_MAP, global fără VA) nu trimit nimic; fără bufferul PM în golden merge | `cargo test -p nvidia-ga104 c135_bad_setup` | conform | conform | PASS (SIMULARE) |
| T131h | fără opțiune nimic nu se schimbă: drumul vechi și `--i9 c133` își descriu notificatorul ca înainte (`{0, 0xecc, 0}`, tipuri necunoscute) și trimit aceleași mesaje; canalul golden primește ChID 3 din `flags` | `cargo test -p nvidia-ga104 c135_off` | conform | conform | PASS (SIMULARE) |
| T131i | raportul ajunge în jurnal pe toate drumurile: N reușit (29 de rânduri), N refuzat + F reușit (59 de rânduri), refuz în antetul RPC (notă), încercare oprită înaintea jetonului | `cargo test -p nvidia-ga104 c135_report` | conform | conform | PASS (SIMULARE) |
| T131j | `c135` prin cozile reale (transportul de pe placă, cu GSP simulat): N reușește pe modelul care reproduce `os-304`; refuzul jetonului vine întreg prin cozi și duce la F | `cargo test -p nvidia-ga104 c135_through` | conform | conform | PASS (SIMULARE) |
| T131k | parametrii canalului: constructorul vechi = `(AsTinygrad, None)` octet cu octet; „fără notificator” schimbă doar `hObjectError`, `internalFlags`, mărimea din `errorNotifierMem`; ChID-ul ales schimbă doar `flags`; `flags` ↔ ChID (canalul golden = ChID 3); răspunsul unui control refuzat își păstrează parametrii | `cargo test -p nvidia-ga104 -- c135_channel_params c135_fixed c135_refused_control` | conform | conform | PASS |
| T131l | modelul: ChID din `flags` sau dat de model (regulă SIM), 0 e al GSP, id ocupat / în afara plajei refuzat | `cargo test -p gpu-sim channel_ids` | conform | conform | PASS (SIMULARE) |
| T131m | tot pachetul | `sh tools/check.sh` | 507 PASS | **507 PASS** (493 → +14), 4 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| — | `--i9 c135` pe RTX 3070 Ti | — | — | — | NOT TESTED (fereastra cu Adrian) |

## Rezultate — sesiunea 107 (17 nov 2026, ~14:20, cloud, fără placă) — C-133: I9 cu promovarea într-un singur mesaj PA + VA înaintea obiectului compute (`open_work_channel_c133`, încercările A și B), după `os-303` (I9 tot `0x57`), toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T130a | modelul care joacă ipotezele (a), (c) și „niciuna” (`fake_rm::PromoteRule`) reproduce `os-303` pe drumul vechi: fizicul trece, mesajul doar-VA primește `0x57` (și cu `--i9 c125`) | `cargo test -p nvidia-ga104 c133_the_hypothesis` | conform | conform | PASS (SIMULARE) |
| T130b | încercarea A (model permisiv și ipoteza (a)): TSG, context share, canal, **un singur** `PROMOTE_CTX` (MAIN, PM, PATCH cu PA + VA, `bInitialize`, fără `bNonmapped`) **înaintea** obiectului compute, apoi token și programare; VA-urile rezolvă PA-urile în tabelele noastre (walker independent); fără a doua încercare | `cargo test -p nvidia-ga104 c133_try_a` | conform | conform | PASS (SIMULARE) |
| T130c | ipoteza (c): A refuzată cu `0x57` la `PROMOTE_CTX` → B în **grup nou**: MAIN + PATCH (PA + VA) + FECS_EVENT și PRIV_ACCESS_MAP doar cu VA-ul din golden (rezolvat din rădăcina noastră); grupul lui A neprogramat, fără obiect; memorie și handle-uri separate | `cargo test -p nvidia-ga104 c133_try_b` | conform | conform | PASS (SIMULARE) |
| T130d | ambele refuzate: eroarea e `0x57` de la RM; raportul are ambele încercări (ce s-a trimis, unde s-a oprit) și ajunge în jurnal (51 de rânduri, numele `NV_ERR_OBJECT_NOT_FOUND`); un raport reușit spune varianta | `cargo test -p nvidia-ga104 c133_both_refused` | conform | conform | PASS (SIMULARE) |
| T130e | alt refuz decât al promovării (grup, canal, compute, programare) sau timeout la `PROMOTE_CTX`: oprire imediată, **fără** B, cu pasul notat | `cargo test -p nvidia-ga104 c133_other_failures` | conform | conform | PASS (SIMULARE) |
| T130f | greșeli de partea noastră (rădăcină greșită, golden fără runlist GR0, fără un buffer per canal, fără VA la un buffer global) nu trimit niciun mesaj | `cargo test -p nvidia-ga104 c133_bad_setup` | conform | conform | PASS (SIMULARE) |
| T130g | fără opțiune nimic nu se schimbă: drumul vechi trimite tot cele 2 `PROMOTE_CTX` după obiect; regula implicită a modelului e cea de dinainte | `cargo test -p nvidia-ga104 c133_off` (+ `c125_off_sends`) | conform | conform | PASS (SIMULARE) |
| T130h | canalul reușit (aici B): USERD zero, token din răspunsul RM + runlist-ul GR0 | `cargo test -p nvidia-ga104 c133_userd` | conform | conform | PASS (SIMULARE) |
| T130i | tot pachetul | `sh tools/check.sh` | 493 PASS | **493 PASS** (485 → +8), 4 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| — | `--i9 c133` pe RTX 3070 Ti | — | — | — | NOT TESTED (fereastra cu Adrian) |

## Rezultate — sesiunea 103 (16 nov 2026, ~19:30, cloud, fără placă) — C-125: I9 V2 (PTE privilegiat) + V3 (lista bufferelor de la RM), după `os-302` (I9 = `0x57`), toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| **PE PLACĂ** (OBSERVAT, `os-302`, 16 nov ~14:15, cu Adrian; hub `rezultate/2026-11-16-workstation-os302/`) | `gpu lant` varianta 1 | — | — | I0–I8 + golden TREC; I9 ESEC la `PROMOTE_CTX` virtual: `0x57` `NV_ERR_OBJECT_NOT_FOUND` | ESEC (motiv cunoscut) |
| T129a | opțiunile oprite (`I9Options::OS_302`, implicit) → **exact aceleași mesaje RM și același canal** ca `open_work_channel_cleared` (os-302); PTE-urile bufferelor de context fără PRIVILEGE | `cargo test -p nvidia-ga104 c125_off_sends` | conform | conform | PASS (SIMULARE) |
| T129b | V2: MAIN/PM/PATCH mapate cu PRIVILEGE = 1 pe fiecare pagină (citit cu walker-ul independent `gpu_sim::vram::translate`); inelul/USERD nu; mesajele RM identice cu os-302; PTE-ul diferă doar în bitul 5 | `cargo test -p nvidia-ga104 c125_v2` | conform | conform | PASS (SIMULARE) |
| T129c | V3: `GR_GET_CTX_BUFFER_INFO` trimis o dată, înainte de cele 2 `PROMOTE_CTX`; lista modelului (valori SIM) citită întreagă; jurnalul: opțiunile = 3, starea 0, 3 buffere, ale noastre „în lista RM”, „privilegiat”, MAIN marcat „mărime diferită” (+64 pagini, ca tinygrad) | `cargo test -p nvidia-ga104 c125_v3_lists` | conform | conform | PASS (SIMULARE) |
| T129d | V3 refuzat de RM (`NOT_SUPPORTED`): refuzul notat (cod + nume `NV_ERR_NOT_SUPPORTED`), canalul se creează și se programează mai departe | `cargo test -p nvidia-ga104 c125_v3_refusal` | conform | conform | PASS (SIMULARE) |
| T129e | cazul os-302 (al doilea `PROMOTE_CTX` refuzat cu `0x57`): eroarea oprește lanțul ca înainte, **raportul rămâne** (lista RM + bufferele noastre) și se scrie în jurnal | `cargo test -p nvidia-ga104 c125_report_survives` | conform | conform | PASS (SIMULARE) |
| T129f | greșeli plantate în cererea V3 (client, canal, alt obiect) prinse de modelul cu offset-uri scrise de mână; răspuns scurt / >64 intrări refuzat de parser; 0 intrări = listă goală | `cargo test -p nvidia-ga104 c125_planted` | conform | conform | PASS (SIMULARE) |
| T129g | antetele NVIDIA regenerate (bindgen 0.71.1, 570.144): în `nvidia-rm570/src/rmapi.rs` doar adăugări (`NV2080_CTRL_GR_GET_CTX_BUFFER_INFO_PARAMS` 5136 B, intrarea 80 B, comanda `0x20801219`) | `sh tools/gen-rm570/generate.sh …` + `git diff` | doar adăugări | doar adăugări | PASS |
| T129h | tot pachetul | `sh tools/check.sh` | 485 PASS | **485 PASS** (479 → +6), 4 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| — | V2/V3 pe RTX 3070 Ti | — | — | — | NOT TESTED (fereastra cu Adrian) |

## Rezultate — sesiunea 98 (13 nov 2026, ~13:30, cloud, fără placă) — PRIMUL JURNAL DE PE LAPTOP (GA106, `os-301`, `gpu tur --trezire`) citit și pus în model: C-116 pasul 3, partea de citire

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| **PE LAPTOP** (OBSERVAT; rulat de Adrian 13 nov ~07:35, `os-301` = cyber-gpu `835f24d`; hub `rezultate/2026-11-13-laptop-asus-os301/`: `gpu-jurnal-003.txt` + poza `gpu-tur-trezire.jpg`) | `gpu tur --trezire` pe ASUS TUF F15 (RTX 3060 Laptop, GA106) | — | BOOT_0 `0x176…`, GFW gata, VBIOS valid prin PROM, FWSEC găsit, VRAM ~6 GiB, WPR2 coborât, 2 scrieri în Command, 0 în registre | **TRECE, 45 de fapte:** BOOT_0 `0xb76000a1`, BOOT_42 `0x176a1000` (GA106, consecvent); GFW `0xff` la prima rundă; VBIOS fără IFR, **3 imagini** (PCI-AT 10de:2520 65024 o; FWSEC 10de:2200 22016 o la 0x0fe00; FWSEC 10de:2200 415744 o la 0x15400 — **fără imagine EFI**, spre deosebire de workstation), BIT la 0x1b0 cu **aceleași 17 jetoane** ca pe workstation; **`vbios.versiune = 94.06.1e.00.23` = exact șirul din Windows** (ordinea octeților jetonului 'i' confirmată pe placă); FWSEC: PMU 0x7aafe, descriptor v3 la 0x2c834, IMEM **0xe100** (workstation 0xe200), DMEM 0x800, 3 semnături, mască 0x0007, ucode_id 9; siguranța 0x8241e0 = 3 (v2/s2, ca GA104); **VRAM 6144 MiB**; WPR2 coborât → `i6.trece = da`; **scrieri: 2 în Command (pornit, repus), 0 în registrele plăcii** | PASS (pe placă, doar citire) |
| T128a | `gpu_sim::observed_laptop`: ROM-ul sintetic are cele 3 imagini la offset-urile/mărimile/ID-urile observate, BIT la 0x1b0, descriptorul FWSEC cu antetul 0x4ac = 44 + 3·384, tabela PMU în ultima imagine; BAR0 are cele 7 registre observate, iar cuvântul PROM `0xeb7faa55` **nu** e reprodus; configurația: Command `0x00100000` (Memory Space oprit), BAR5 `0x4001`, 0 scrieri | `cargo test -p gpu-sim --release observed_laptop` | conform | conform | PASS (SIMULARE) |
| T128b | **jurnalul real 003 (24 de rânduri) + cele 39 de fapte transcrise din poză** față de `run_tour_awake` pe modelul `observed_laptop`: `board_compare` → **36 LA FEL, 0 DIFERIT, 1 diferit intenționat (cuvântul PROM), 0 doar pe placă**; cele 8 fapte doar în simulare sunt exact rândurile tăiate din poză (ids, clasă, Command, ReBAR) + `chip.familie`/`chip.scrieri`; aceleași 8 citiri BAR0 jurnalizate, în aceeași ordine (fereastra PROM pliată în prima citire); 2 scrieri în configurație, 0 în registre; rândurile de jurnal de pe stick (`gpu: bar0 r …`) devin fapte | `cargo test -p nvidia-ga104 --release laptop_journal_003_agrees` | conform | conform | PASS (SIMULARE, față de un jurnal OBSERVAT) |
| T128c | același jurnal față de modelul **workstation-ului** → diferențe reale la `vbios.imagini`, `vbios.imagine1`, `fwsec.descriptor`, `fwsec.pmu`, `i6.vram`, `chip.nume` (modelul laptopului nu e workstation-ul cu alte ID-uri) | `cargo test -p nvidia-ga104 --release laptop_journal_is_not_explained` | conform | conform | PASS (SIMULARE) |
| T128d | tot pachetul | `sh tools/check.sh` | 479 PASS | **479 PASS** (475 → +4), 4 ignorate; format, clippy `-D warnings`, build no_std | PASS |

## Rezultate — sesiunea 96 (13 nov 2026, ~01:30, cloud, fără placă) — versiunea VBIOS din jetonul BIT 'i' + faptele UHL ale laptopului în model (C-116 / C-117), toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| **DIN WINDOWS** (citit din hub, `rezultate/2026-11-11-laptop-asus-tuf/REZULTAT.md`, 12 nov 22:14 + 23:03) | inventarul + raportul UHL ale lui Adrian pe ASUS TUF (doar citire, fără serii) | — | — | RTX 3060 Laptop `10de:2520` **rev A1, subsistem 1043:16F2**, **VBIOS 94.06.1E.00.23**, GA106M sm_86, **30 SM**, L2 3 MiB, 6 GiB/192 biți, BAR1 8 GiB @0x6000000000, PCIe Gen4 x8 activ (x16 max), D3cold suportat, ACPI `\_SB.PC00.PEG1.PEGP` | OBSERVAT (raport Windows, nu de driverul nostru) |
| T127a | `vbios::read_layout`: fără jetonul 'i' (0x69) → `version = None`; cu jetonul 'i' de **4** octeți → `None` (fără octetul „patch” nu ghicim); 0 scrieri | `cargo test -p nvidia-ga104 --release version_is_none_when_token_i` | conform | conform | PASS (SIMULARE) |
| T127b | cu jetonul 'i' de 5 octeți (octeți de test în forma versiunii raportate de Windows pentru workstation, `00 65 04 94 3e`): `BiosVersion {94, 04, 65, 00, 3e}`, afișat `94.04.65.00.3e`; `matches_report` acceptă `" 94.04.65.00.3E "`, refuză `94.04.65.00.3f`, `94.4.65.0.3e` și `""`; versiunea laptopului (`94.06.1E.00.23`) ≠ a workstation-ului | `cargo test -p nvidia-ga104 --release version_is_read_from_token_i` | conform | conform | PASS (SIMULARE) |
| T127c | turul C-54 pe modelul observat al workstation-ului (jetonul 'i' SIM cu versiunea din Windows) dă faptul nou **`vbios.versiune = 94.04.65.00.3e`**; toate celelalte fapte neschimbate; 0 scrieri | `cargo test -p nvidia-ga104 --release tour_on_observed_model` | conform | conform | PASS (SIMULARE) |
| T127d | `run_tour_awake` pe **modelul laptopului din faptele UHL** (`SimConfig::ga106_laptop_asus` + `ga106_bar0` + ROM-ul observat cu versiunea laptopului): `pci.subsistem = 1043:16f2`, `pci.bar1 = 0x6000000000 mem 64b prefetch`, `pci.rebar.bar1 = 8192 MiB`, `pci.legatura = gen4 x8 (max gen4 x16)`, `chip.familie = GA106`, **`vbios.versiune = 94.06.1e.00.23`**; 2 scrieri în configurație (pornit + repus), 0 în registre GPU, Command rămâne 0 | `cargo test -p nvidia-ga104 --release awake_tour_on_laptop_facts` | conform | conform | PASS (SIMULARE) |
| T127e | tot pachetul | `sh tools/check.sh` | 475 PASS | **475 PASS** (472 → +3), 4 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| pe laptopul ASUS | `gpu tur --trezire` (os-298+, după DA) → în jurnal `vbios.versiune` | — | **`94.06.1e.00.23`** (= raportul Windows); altceva = citirea noastră a ROM-ului e greșită sau PROM-ul arată altă imagine | — | NOT TESTED (OS-ul trebuie să lege `835f24d`+ ca să apară rândul) |
| pe workstation | `gpu lant` / `gpu tur` → `vbios.versiune` | — | **`94.04.65.00.3e`** (K0, `HARDWARE_WORKSTATION.md`) | — | NOT TESTED |

## Rezultate — sesiunea 95 (12 nov 2026, noaptea, cloud, fără placă) — C-116: laptopul ASUS TUF F15 (GA106) — recunoaștere doar-citire + „trezirea minimă” (Memory Space), toate SIMULARE

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| **PE LAPTOP** (citit din hub, `rezultate/2026-11-11-laptop-asus-tuf/`) | `gpu identifica` pe os-295, 12 nov 20:20 | — | — | NVIDIA **10de:2520** la 01:00.0, D0, **Command 0000** (Memory Space oprit de firmware), BAR0 16 MiB @0x85000000, BAR1 8 GiB (ReBAR), BAR3 32 MiB; driverul (`49e6685`) a refuzat corect „dispozitiv nesuportat”; **zero scrieri** | OBSERVAT (pozele + REZULTAT.md ale OS-ului) |
| T126a | `ids`: tabelul GA106 separat de GA104; `lookup` (GA104) nu știe `0x2520`; `lookup_ampere` dă `Ga106` pentru `0x2520` și `Ga104` pentru `0x2482`; GA107 (`0x25a0`) refuzat; `Family::Ga106.writes_allowed() == false` | `cargo test -p nvidia-ga104 --release ga106_table` | conform | conform | PASS |
| T126b | `probe::level0_ampere` pe modelul laptopului **așa cum îl lasă firmware-ul** (`SimConfig::ga106_laptop_asus`, Command 0): refuz `Blocked(MEMORY_SPACE_OFF)` **fără nicio citire din BAR0**, 0 scrieri; `probe_config_ampere` singur identifică `GeForce RTX 3060 Laptop`, BAR0 `0x85000000`, `memory_space_enabled() == false`; `level0` (GA104) refuză laptopul după ID | `cargo test -p nvidia-ga104 --release ga106_laptop_as_firmware` | conform | conform | PASS (SIMULARE) |
| T126c | Cu Memory Space pornit (model): `level0_ampere` → `GA106`, consecvent, doar 2 citiri BAR0 (`0x0`, `0xa00`), 0 scrieri; `level0_ampere` pe workstation = `level0` + familia; ID GA106 cu BOOT_0 de GA104 → refuz; D3 → refuz fără citire BAR0 | `cargo test -p nvidia-ga104 --release ga106_laptop_with ampere_level0 ga106_pci_id` | conform | conform | PASS (SIMULARE) |
| T126d | `read_tour::run_tour_ampere`: pe modelul laptopului cu ID schimbat → `chip.familie = GA106`, `chip.scrieri = interzise …`, 0 scrieri; `run_tour` refuză laptopul; pe laptopul cu Memory Space oprit se oprește înainte de prima citire BAR0; pe workstation = `run_tour` + `chip.familie = GA104` | `cargo test -p nvidia-ga104 --release ampere_tour` | conform | conform | PASS (SIMULARE) |
| T126e | `gpu_pci::mem_space::enable_memory_space` pe laptopul model (Command 0): **o singură scriere** `cfg w 0x004: 0x00100000 <- 0x00000002 => 0x00100002`, Bus Master rămâne oprit, pas `C-116 trezire minima: Memory Space pornit`; `restore_memory_space` → Command înapoi la 0, a doua scriere, pas `… Command repus`; pe workstation (deja pornit) **0 scrieri** | `cargo test -p gpu-pci --release laptop_one_write workstation_already_on` | conform | conform | PASS (SIMULARE) |
| T126f | refuzuri fără nicio scriere: placa absentă (`NotResponding`), D3, BAR0 nealocat (adresă 0), BAR1 numit de ReBAR fără adresă, antet care nu e tip 0; restaurarea refuză dacă s-au schimbat alți biți (Bus Master pornit între timp); bit 1 blocat la 0 (defect injectat) → `VerifyFailed`, valoarea veche repusă, 2 scrieri în jurnal | `cargo test -p gpu-pci --release refused_absent restore_refuses read_back_mismatch` | conform | conform | PASS (SIMULARE) |
| T126g | `read_tour::run_tour_awake` pe laptopul model: `Wake::EnabledAndRestored`, turul dă `GA106`, **2 scrieri în configurație** (pornit + repus), **0 scrieri în registre GPU**, Command rămâne 0 la final, `pci.trezire = Memory Space pornit de noi …`; pe workstation `Wake::NotNeeded`, 0 scrieri; când turul pică după trezire (BOOT_0 = 0xffffffff) Command e **tot** repus (2 scrieri); dispozitiv necunoscut → refuz înainte de orice scriere | `cargo test -p nvidia-ga104 --release awake_tour` | conform | conform | PASS (SIMULARE) |
| T126h | tot pachetul | `sh tools/check.sh` | 472 PASS | **472 PASS**, 4 ignorate; format, clippy `-D warnings`, build no_std | PASS |
| pe laptopul ASUS | `gpu tur` cu trezirea minimă (după DA) | — | jurnalul de citire: BOOT_0 = `0x176…`, GFW, VBIOS prin PROM | — | NOT TESTED (OS-ul trebuie să lege `ccec8f9`+ și să adauge comanda) |

## Rezultate — sesiunea 94 (12 nov 2026, seara, cloud, fără placă) — jurnalele 011 (I7 ESEC la repetare) și 012 (Booter Unload 0x88): starea plăcii în jurnal după un I7 / Unload eșuat

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| **PE PLACĂ** os-295, jurnalul 011 | `gpu lant` (varianta 1), repetarea de la ~18:00, stick la ~1 s/rând (C-113) | `diff` fără marcaje de timp față de `gpu-jurnal-010.txt` (primele 1500 de rânduri) | aceleași scrieri | **12 rânduri diferite**: 4 adrese RAM (LibOS, imaginea Booter, `GspFwWprMeta`) + rezultatul; Booter a terminat cu `MAILBOX0 = 0` (`FALCON_OS` scris după), **`GSP RISCV_CPUCTL = 0x10`** după așteptare → `pas i7-booter-gsp: ESEC` (timeout). Cauza nu se poate stabili din jurnal (`docs/os-295_ANALIZA_011_012.md` §1) | OBSERVAT (citire de jurnal real); I7 pe placă: **FAIL o dată din 2** azi |
| **PE PLACĂ** os-295, jurnalul 012 | `gpu recupereaza` (C-103), ~18:13, cel mai probabil aceeași pornire ca 011 | — | WPR2 coborât | Booter Unload (40 192 B) pornit → **`MAILBOX0 = 0x88`**, WPR2 rămâne ridicat; codul nu e publicat; diferența față de RM: fără FWSEC-SB înainte (§2) | OBSERVAT; C-103 pe placă: **FAIL** |
| T125a | `booter_load` cu `SimFault::RiscvStaysOff`: eroarea rămâne `Timeout`; după ea `nota: I7: Booter a terminat cu MAILBOX0 = 0, dar RISC-V-ul GSP nu a devenit activ; starea placii (doar citiri):` + **9 rânduri** `valoare …` în ordinea `RESUME_STATE_READS` + `I7_FAIL_EXTRA_READS` (`GSP RISCV_CPUCTL` fără bitul 7, `SEC2 MAILBOX0 = 0`, `GSP RISCV_BCR_CTRL` prezent), nicio scriere după notă; un I7 care trece n-are „starea placii” | `cargo test -p nvidia-ga104 --release i7_with_riscv_off` | cum e descris | la fel | PASS (SIMULARE) |
| T125b | `booter_load` cu `GspFwWprMeta` fără magic (SIM: Booter refuză cu `0xbad2`): `HardwareFault 0xbad2`, `nota: I7: Booter a oprit SEC2 cu MAILBOX0 != 0 …` + 9 rânduri cu `SEC2 MAILBOX0 = 0x0000bad2`; cealaltă notă lipsește | `cargo test -p nvidia-ga104 --release i7_with_booter_code` | cum e descris | la fel | PASS (SIMULARE) |
| T125c | `booter_unload` cu un program greșit (SIM: `0xdead`): după `MAILBOX0 (Booter a dat eroare) = 0x0000dead` urmează `nota: C-103 Booter Unload: MAILBOX0 != 0; starea placii (doar citiri):` + 9 rânduri, nicio scriere | `cargo test -p nvidia-ga104 --release booter_unload_error_journals` | cum e descris | la fel | PASS (SIMULARE) |
| T125d | `RISCV_ACTIVE_US` = 5 000 000 (nova); toate testele care așteaptă RISC-V activ trec neschimbate (`every_failure_is_reported_not_skipped`: `RiscvStaysOff` → `Timeout`) | `sh tools/check.sh` | 458 PASS | 458 PASS, 4 ignorate; clippy, format, build no_std | PASS |

## Rezultate — sesiunea 93 (12 nov 2026, cloud, fără placă) — după os-295 (I8 TRECE pe placă, I9 ESEC fără motiv în jurnal): motivul pe stick

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| **PE PLACĂ** os-295 | `gpu lant --varianta 1` pe RTX 3070 Ti (Adrian, 12 nov ~17:48; stick os-295, cyber-gpu `49e6685`), jurnalul `rezultate/os-automat/2026-11-12/gpu-jurnal-010.txt` (2425 rânduri) | `compara_variante 0x9c66000 gpu-jurnal-009.txt gpu-jurnal-010.txt` | I8 trece dacă RPC-urile sunt înaintea lui Booter (T122) | **`GSP_INIT_DONE rpc_result = 0`** după 11,17 s, `pas i8-gsp-init: ok`, 2 NOCAT (nu 844), **`pas i8e-golden: ok`** (11 mesaje RM acceptate), **`pas i9-canal-lucru: ESEC`** după 11 sonerii (al 11-lea mesaj = `PROMOTE_CTX` virtual), fără motiv în jurnal; „OPRIT CURAT la i9-canal-lucru”; ecranul înghețat de la pornirea GSP-RM, jurnalul complet pe stick | **I8 PASS, I8e PASS, I9 FAIL (motiv necunoscut)** — `docs/os-295_ANALIZA_I9.md` |
| T124 | `Journal::push_text`: text scurt = 1 rând `text motiv = "…"`; 61 de octeți = 2 rânduri (`motiv` / `motiv (urmare)`), lipite la loc dau textul; textul gol = 1 rând gol | `cargo test -p gpu-core long_text_is_journaled_in_pieces` | conform | conform | PASS |
| T124b | `chain::run_chain_ready`: după `pas i6-fwsec-frts: ESEC` urmează `text motivul esecului = "eroare hardware 0x57"`; un motiv de 61 de octeți = 2 rânduri; pasul oprit de poartă (I7 blocat în plan) are și el motivul („blocat in plan (I7)…”); pașii trecuți n-au motiv | `cargo test -p nvidia-ga104 a_failed_step_leaves_its_reason` | conform | conform | PASS (SIMULARE) |
| T124c | `QueueTransportBellKeep` prin cozile reale către `SimGsp` + `FakeRm`: în timpul primului apel GSP-RM trimite 1 NOCAT ENGINE (`0x27`, „Display Subsystem”), 1 `UCODE_LIBOS_PRINT`, 1 eveniment `0x1fff` cu `rpc_result 0x33`; răspunsul (acceptat) și refuzul comenzii a doua (`HardwareFault`) ca înainte; 2 sonerii; `events` = cele 3 numere, `skipped` = cele 3 mesaje întregi; `journal_skipped` → `total = 3`, 3 × „la pasul = "I9"”, NOCAT decodat (`NV_ERR_INVALID_DMA_SPECIFIER`, sursa), LibOS print decodat, `0x1fff` + `rpc_result 0x33`; nimic sărit = nimic scris | `cargo test -p nvidia-ga104 skipped_messages_are_kept_whole` | conform | conform | PASS (SIMULARE) |
| T124d | `variant_log`: jurnal în forma os-295 (fără motiv) → `fail_reasons` gol, tabelul spune „fara motiv in jurnal (driver dinainte de 12 nov 2026; era doar pe ecran)”; cu rândurile motivului în 2 bucăți → lipite: „argument invalid: RM_CONTROL: raspunsul e pentru alta comanda”, arătat în tabel; rândurile motivului nu sunt pași; un rând de motiv fără pas eșuat înainte = ignorat | `cargo test -p nvidia-ga104 the_reason_of_a_failed_step` | conform | conform | PASS |
| T124e | `compara_variante 0x9c66000 gpu-jurnal-009.txt gpu-jurnal-010.txt` pe jurnalele reale | — | tabel + „au trecut de I8: varianta 1” | 009: „I8 refuzat 0x59”, 010: „I8 TRECUT”, „motivul la i9-canal-lucru: fara motiv in jurnal …”, 51/40 rânduri diferite, verdicte „OPRIT CURAT la i8-gsp-init” / „… la i9-canal-lucru”, cod de ieșire 0 | PASS (citire de jurnal real) |
| Suita | `sh tools/check.sh` (Rust 1.97.0): format, clippy `-D warnings`, teste, build no_std | — | tot verde | **455 PASS** (4 ignorate) | PASS |

## Rezultate — sesiunea 89 (11 nov 2026, cloud, fără placă) — pregătirea pentru jurnalele de joi (C-108): `variant_log` + `compara_variante`

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T123 | `variant_log::read`: numărul și numele variantei, commit-ul din antet, rândurile `valoare varianta: …`, pașii, rezultatul (`PassedI8` doar cu `pas i8-gsp-init: ok`; `RefusedAtI8(0x59, nume)`; `StoppedBefore`; `Unknown`); `strip_prefix` scoate `[ timp] gpu: `; zgomot = antet, marcaje de timp NOCAT, `asteptat …`, rândurile variantei | `cargo test -p nvidia-ga104 --release --lib variant_log` | toate citite exact | exact | PASS |
| T123b | `variant_log::compare`: aceeași rulare pe altă pornire rece (alt commit, alte marcaje de timp, alt prefix) = IDENTIC, 21 rânduri cu sens; o adresă de assert + o cifră schimbate = 2 rânduri doar în referință, 2 doar aici; varianta 1 (rânduri în plus la I7, I8 trecut) = doar adăugările/lipsurile reale, soneria (mutată de la I8 la I7) nu e diferență; un rând repetat → surplusul `(xN)` | idem | cum e descris | la fel | PASS |
| T123c | `variant_log::Table`: ordonare referință → varianta 0, 1, 2, 3; „IDENTIC”, „I8 TRECUT”, „I8 refuzat: rpc_result 0x59 (NV_ERR_OPERATING_SYSTEM)”, „oprit inainte de I8, la i6-fwsec-frts”, NOCAT 844/1/1, setările pe scurt, `… inca N randuri`; concluzia numește varianta trecută sau „nicio varianta nu a trecut de I8” | idem | textul exact | exact | PASS |
| T123d | Jurnalul ferestrei complete SIMULATE (varianta 1, rândurile variantei la început ca în OS P6-78), cu prefixul OS-ului: `read` → varianta 1 `rpc-inainte`, 7 setări (ordinea = 1), `PassedI8`, fără NOCAT; identic cu el însuși; față de aceeași rulare cu I8 refuzat (forma os-283): doar în referință exact cele 3 rânduri ale refuzului, doar aici `pas i8-gsp-init: ok` + `i8e-golden`; tabelul: „au trecut de I8: varianta 1” | `cargo test -p nvidia-ga104 --release --lib variant_journal_of_the_simulated` | cum e descris | la fel | PASS (SIMULARE) |
| T123e | Unealta `compara_variante 0x9c66000 gpu-jurnal-009.txt gpu-jurnal-008.txt gpu-jurnal-007.txt` pe jurnalele reale din hub (os-283, os-280, os-279): toate „driver dinainte de C-108”, „I8 refuzat 0x59”, 844 NOCAT / 16 adrese; 008 față de 009: **2 rânduri** (`SEC2 CPUCTL` 0x10 vs 0x20, `GSP MAILBOX0` 0xcb vs 0 — citiri ale stării plăcii, nescoase de diff-ul de mână din sesiunea 88); 007 (format vechi): 48/4 rânduri (`0x1020` ×843); verdictele lanțului ca `verifica_jurnal_lant`; cod de ieșire 3 | `cargo run --release -p nvidia-ga104 --example compara_variante -- …` | cifrele din sesiunile 85–88 | regăsite | PASS (OBSERVAT: jurnale reale, citite pe gazdă) |
| Suita | `sh tools/check.sh` (Rust 1.97.0): format, clippy `-D warnings`, teste, build no_std | — | tot verde | **451 PASS** (4 ignorate) | PASS |

## Rezultate — sesiunea 88 (11 nov 2026, cloud, fără placă) — după os-283: ordinea RPC-urilor de pornire + variantele C-108

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T121 | `gsp::variants`: 10 variante numerotate 1..=10, nume ASCII ≤ 32, propoziție + sursă la fiecare, `get(0)`/`get(11)` = `None`, `settings(None)` = `OS_283`, varianta 1 = `OS_283` + doar ordinea, 1–8 pun RPC-urile înainte de Booter, 9–10 nu, oricare două variante diferă | `cargo test -p nvidia-ga104 --release --lib t121` | tot adevărat | tot adevărat | PASS (SIMULARE) |
| T121b | Fiecare variantă schimbă în `GspSystemInfo` **doar** `bIsPassthru`, `bIsPrimary`, `hostPageSize` (față de os-283); registrul are 2 sau 3 chei (a 3-a `RMDevidCheckIgnore`); `bDmemStack` după variantă; seturile tinygrad/nova/nouveau = tabelul C-107 | idem | octeții din afara celor trei câmpuri identici cu `sysinfo::encode` | identici | PASS (SIMULARE) |
| T121c | `Settings::journal`: rândurile `valoare varianta I8 (C-108): numarul`, `text varianta I8: numele`, apoi câte un rând `valoare varianta: …` pentru ordine, bIsPassthru, bIsPrimary (2 = după VGA), hostPageSize, chei, flags, bDmemStack | idem | rândurile exacte pentru varianta 3 și pentru os-283 | exacte | PASS |
| T121d | `init_rpcs_ready_for_i8`: coada goală → varianta 1 refuză, os-283 trimite; după `queue_init_rpcs` (secvențele 0, 1) → varianta 1 merge, os-283 refuză | idem | `Err`/`Ok(false)`/`Ok(true)`/`Err` | la fel | PASS |
| T122 | Modelul pornirii timpurii (`SimGsp::start_early_boot`: consumă ce e în coadă când GSP-RM pornește; `init_done_result` = 0x59 dacă lipsesc `SET_SYSTEM_INFO` + `SET_REGISTRY`) cu `wait_init_done` real: ordinea os-283 (RPC-urile la I8) → `GSP_INIT_DONE 0x59` ca pe placă; varianta 1 (RPC-urile + soneria la I7, înainte de Booter) → `rpc_result 0`; variantele 1–8 trec, 9–10 nu | `cargo test -p nvidia-ga104 --release --lib t122` | `HardwareFault 0x59` / `Ok` | la fel | PASS (SIMULARE — modelul e scris după sursele RM, nu după placă) |
| T78/T79/T86 (actualizate) | Lanțul complet de test rulează acum în ordinea variantei 1: soneria + 5 scrieri GSP înainte de Booter (+1 după), 2 sonerii înainte de RM (I7, I8); `check_i7_libos` acceptă soneria ca prima scriere | `cargo test -p nvidia-ga104 --release --lib user_space` | toți pașii trec, verdictul jurnalului `Passed` | la fel | PASS (SIMULARE) |
| Suita | `sh tools/check.sh` (Rust 1.97.0): format, clippy `-D warnings`, teste, build no_std | — | tot verde | **445 PASS** (4 ignorate) | PASS |

## Rezultate — sesiunea 86 (11 nov 2026, cloud, fără placă) — pregătirea analizei jurnalului os-280

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T120 | Jurnalul pe care îl va scrie rularea următoare, produs de driver în SIMULARE (6 asserturi într-un loc + 1 în alt loc **înainte** de cererea secvențiatorului, 2 înregistrări ENGINE `NV_ERR_INVALID_STATE` **după**, 1 `UCODE_LIBOS_PRINT`, `GSP_INIT_DONE` cu `0x59`), tipărit rând cu rând cu prefixul OS-ului (`[ timp] gpu: `) și citit înapoi de `nocat_log::digest`: 9 primite / 3 feluri / 1 LibOS / 10 sărite; jurnalizate în întregime nr. 1–4, 7, 8 (5 înainte, 1 după secvențiator); assert: adresă `0x12340`, linia 321, stiva `[0x10, 0x20]`, `count 3` la al doilea; ENGINE: `0x40` = `NV_ERR_INVALID_STATE`, subsystem 5; felurile ordonate după frecvență (primul acoperă 4); `rpc_result 0x59 (NV_ERR_OPERATING_SYSTEM)`, „eroarea vine din GSP_INIT_DONE”; textul rezumatului conține exact aceste cifre | `cargo test -p nvidia-ga104 --lib the_stick_journal_rows` | conform | conform | PASS (SIMULARE) |
| T120b | Formatul vechi (os-279): 843 rânduri `sarit (functia) = 0x1020` + 2 × `0x100c` → numărate ca „numite, dar nedecodate”, `rpc_result 0x59` fără nume (jurnalul vechi nu-l scria), total sărite 845; un jurnal oprit înainte de I8 → „niciun rand despre mesajele GSP-RM” | `cargo test -p nvidia-ga104 --lib nocat_log` | conform | conform | PASS |

Verificare pe jurnalul real: `verifica_jurnal_lant gpu-jurnal-007.txt 0x9c66000` → același verdict ca în sesiunea 85 („OPRIT CURAT la i8-gsp-init”), plus rezumatul nou:
„NOCAT (format vechi, ca os-279): 843 mesaje 0x1020 … si 2 mesaje 0x100c … numite, dar nedecodate / GSP_INIT_DONE: rpc_result 0x59 / I8: eroarea vine din GSP_INIT_DONE”.

Toate: `sh tools/check.sh` → format, clippy `-D warnings`, **440 teste PASS** (4 ignorate), build no_std PASS. **Nimic pe RTX 3070 Ti.**

## Rezultate — sesiunea 85 (11 nov 2026, cloud, fără placă) — după os-279: mesajele GSP-RM de dinaintea erorii 0x59, decodate

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T116 | Ce a trimis placa pe 11 nov (os-279): multe `GSP_POST_NOCAT_RECORD`, două `UCODE_LIBOS_PRINT`, apoi `GSP_INIT_DONE` cu `rpc_result = 0x59`. Primele 4 înregistrări și fiecare fel nou (recType, errorCode, subsystem) sunt decodate în jurnal (`text NOCAT: source`, numele `NV_STATUS`…), restul numărate; o înregistrare prea scurtă e notată; `UCODE_LIBOS_PRINT`: descriptorul motorului și mărimea; rezumatul (total, NOCAT, feluri, LibOS) înainte de cele 7 registre; `text GSP_INIT_DONE: rpc_result (nume NV_STATUS) = "NV_ERR_OPERATING_SYSTEM"`; eroarea rămâne `HardwareFault 0x59`; 40 de feluri → 16 păstrate, 24 numărate ca „peste limită”, `GSP_INIT_DONE` bun → Ok; nicio sonerie, nicio scriere | `cargo test -p nvidia-ga104 --lib nocat_records_and_libos_prints` | conform | conform | PASS (SIMULARE) |
| T117 | Decodarea `NV2080CtrlNocatJournalInsertRecord` (offset-urile = cele calculate de compilatorul C pentru `ctrl2080nvd.h`: 0, 8, 16, 20, 24, 92, 96, 104, 172, 176, 180; 1208 B); înregistrare ENGINE cu `NV_STATUS` numit; ASSERT ca în RM `journal.c` (adresă + linie, stiva de apeluri 5 adrese, `count`), fără nume `NV_STATUS` pentru o adresă; `UCODE_LIBOS_PRINT` cu buffer trunchiat; `Event::Text` tăiat la 32 de octeți și la NUL, caracterele neafișabile → `.`; tabelele generate (`nvidia-rm570::names`): `0x59` → `NV_ERR_OPERATING_SYSTEM`, `0x1020` → `GSP_POST_NOCAT_RECORD`, `0x100c` → `UCODE_LIBOS_PRINT`, 65 → `GET_GSP_STATIC_INFO`, sortate, fără dubluri | `cargo test -p nvidia-ga104 --lib gsp::nocat` · `cargo test -p gpu-core text_events` · `cargo test -p nvidia-rm570 names_from_the_headers` | conform | conform | PASS |
| T118 | C-103: Booter Unload în SIMULARE — după I6 + I7 (WPR2 ridicat, GSP-RM „activ”): reset GSP, reset SEC2, programul `BooterUnload` cu mailbox (0xFF, 0xFF) → WPR2 citește 0, RISC-V inactiv, `pas C-103 Booter Unload: WPR2 coborat: ok`, registrele WPR2 nescrise de noi; **apoi I6 (FWSEC-FRTS) rulează din nou în aceeași „pornire”**; WPR2 jos → `Wpr2NuEraRidicat`, 0 scrieri, niciun program încărcat; program greșit → `HardwareFault 0xdead` numit; `SimFault::Wpr2Stuck` → `InvalidArgument("… inca ridicat …")` numit | `cargo test -p nvidia-ga104 --lib tests_c103` | conform | conform | PASS (SIMULARE) |
| T119 | C-103: `gpu_pci::caps::reset_capabilities` citește FLR (DevCap bit 28), DevCtl bit 15, `No_Soft_Reset` (PMCSR bit 3), D-state, PME D3cold; 0 scrieri în configurație; `summary()` în română | `cargo test -p gpu-pci reset_capabilities` | conform | conform | PASS |
| T113 (actualizat) | ca înainte, plus rândul cu numele mesajului sărit (`OS_ERROR_LOG`), numele codului (`NV_ERR_OPERATING_SYSTEM`) și rezumatul (total = 1) între notă și registre | `cargo test -p nvidia-ga104 --lib init_done_with_an_error_code` | conform | conform | PASS (SIMULARE) |

Verificare pe jurnalul real: `verifica_jurnal_lant gpu-jurnal-007.txt 0x9c66000` → „OPRIT CURAT la i8-gsp-init (Bus Master pus la loc)”, I6 720 / I7 723 scrieri conforme,
I8 326 scrieri + 1 sonerie; cu rândurile noi (`text …`, rezumatul) adăugate artificial: același verdict (parserul le ignoră).

## Rezultate — sesiunea 78 (10 nov 2026, cloud, fără placă) — C-96: I8 „eroare hardware 0xbadf5620”

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T108 | Resetul GSP-ului care rulează GSP-RM (modelul răspunde `0xbadf5620` la primele citiri `HWCFG2` după eliberare, ca placa pe 10 nov): `reset` + `switch_to_falcon` trec, nucleul ajunge „falcon”, RISC-V oprit; cu `PrivErrorStuck` (eroarea nu se mai curăță) eroarea e `HardwareFault 0xbadf5620` după toată limita RM (4 s), nu timeout mut | `cargo test -p nvidia-ga104 --lib reset_of_a_running_gsp_rm` | trece; BCR final = falcon; greșeala persistentă raportată cu valoarea plăcii | conform | PASS (SIMULARE) |
| T109 | Cererea plăcii din 10 nov (`WRITE 0x110040 0`, `CORE_RESET`, `CORE_START`, `CORE_WAIT_FOR_HALT`, `CORE_RESUME`) până la `GSP_INIT_DONE`; toate comenzile în jurnal (`secv #i …`) **înainte** de prima scriere; `CORE_RESUME` resetează GSP-ul în RISC-V (BCR `0x0` la CORE_RESET, apoi `0x111`) și GSP-RM e activ la final | `cargo test -p nvidia-ga104 --lib board_request_of_10_nov` | 1 rulare, 5 comenzi; ordinea evenimentelor; `secv #0 WRITE 0x110040 0x0` | conform | PASS (SIMULARE) |
| T110 | Comanda care se oprește e numită în jurnal după `ESEC` (`secv #1 CORE_RESET`); eroarea poartă valoarea citită | `cargo test -p nvidia-ga104 --lib the_failing_command_is_named` | conform | conform | PASS (SIMULARE) |
| T113 | C-102: `GSP_INIT_DONE` cu `rpc_result = 0x59` → eroare `HardwareFault 0x59`, jurnalul numește mesajul sărit, `rpc_result`, `rpc_result_private`, „eroarea vine din GSP_INIT_DONE” + 7 registre citite | `cargo test -p nvidia-ga104 --lib init_done_with_an_error_code` | conform | conform | PASS (SIMULARE) |
| T114 | C-102: pointerul de scriere al cozii de stare = 89 (0x59) → aceeași eroare, jurnalul spune „eroarea vine din coada de stare” + registrele | `cargo test -p nvidia-ga104 --lib a_queue_pointer_out_of_range` | conform | conform | PASS (SIMULARE) |
| T115 | C-102: `CORE_RESUME` scrie în jurnal starea (7 valori, doar citiri) înainte de `0x840130 ← 2` și după predarea etapei 3; nicio scriere nouă (SCRATCH_14 neatins) | `cargo test -p nvidia-ga104 --lib core_resume_journals_the_state` | conform | conform | PASS (SIMULARE) |
| — | Driverul **vechi** (`24dc19c`) pe modelul nou: `boot_to_init_done_with_a_sequencer_request_simulated` pică cu `HardwareFault { code: 0xbadf5620 }` — exact eroarea plăcii | rulat o dată, apoi codul nou | — | reprodus | greșeală plantată prinsă |

Toate: `sh tools/check.sh` → format, clippy `-D warnings`, **420 teste PASS** (4 ignorate), build no_std PASS. **Nimic pe RTX 3070 Ti.**

## Rezultate — sesiunea 68 (8 nov 2026, cloud, fără placă) — C-83: verificarea jurnalului pentru fereastra de luni

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T107 | Jurnalul unei ferestre C-83 (I6 → I14, 17 pași; construit din fereastra simulată + accesele pe care le fac I11–I14: fereastra VRAM mutată și pusă la loc, soneria de lucru) citit cu `check_chain_journal_for(WINDOW_CHAIN_C83)`; lanțul ales singur din jurnal (`chain_for_journal`); greșeli: o scriere în `0x110040` la I12, I14 rulat după ce I13 a eșuat, oprire curată la I13 | `cargo test -p nvidia-ga104 --lib c83_window_journal` | TRECE pe jurnalul corect; verdictul vechi (până la I10) nu-l acceptă; scrierea străină numită cu pasul ei; „a rulat după oprirea”; „OPRIT CURAT la i13-strat” | conform | PASS (SIMULARE — text, nu placă) |

## Rezultate — sesiunea 58 (6 nov 2026, cloud, fără placă) — C-83: I13 (un strat Kronos întreg) și I14 (un simbol întreg)

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T100 | Modelele programelor (`layernorm_gpu`, `gelu_gpu`, `vadd_gpu`) față de motorul procesorului; toleranța stratului | `cargo test -p gpu-compute layer` | în toleranța 1e-4 + 1e-4·\|x\| | conform | PASS |
| T101 | I13 cap-coadă: 16 lansări pe poziție (6 programe NAK) pe canalul de lucru, cache-urile cheie/valoare cresc în VRAM; strat mic 64/2 pe 6 poziții, strat Kronos 768/12 pe 2 poziții | `cargo test --release -p nvidia-ga104 --test sim_i13_layer` | x după strat în toleranță la fiecare poziție (768: abatere < 1e-5) | conform | PASS (SIMULARE) |
| T102 | I13 greșeli plantate: rezidual lipsă, atenția cu o poziție în minus, GELU sărit | idem, `planted` | niciodată „trece” | conform | PASS (SIMULARE) |
| T104 | I14: simboluri întregi (embed + straturi + LayerNorm final + logit-uri, `16·straturi + 3` lansări), model sintetic 64/2/2 straturi/256 simboluri: prompt + 5 simboluri greedy; și lățimea Kronos 768/12 cu 1 strat, 512 simboluri | `cargo test --release -p nvidia-ga104 --test sim_i14_token` | simbolurile identice cu procesorul, logit-urile în 1e-3 + 1e-3·\|x\|; zona de pushbuffer refolosită (`submit_fenced`) | conform | PASS (SIMULARE) |
| T105 | I14: rândul de încorporare greșit, simbol în afara vocabularului, semafor care nu crește | idem, `wrong_token_row` | niciodată „trece” / refuz înainte de scriere | conform | PASS (SIMULARE) |
| T103 | Lanțul C-83 cu `i13-strat` după `i12-q8-matvec`, înainte de curățare | `cargo test -p nvidia-ga104 --lib c83_chain` | ordinea și oprirea corecte | conform | PASS (SIMULARE) |
| T106 | Vectorii de control KGR1 ai lui Kronos (hub `integrare/kronos/vectori`, S39/C-24; model minuscul cu greutăți la întâmplare, 2 straturi, 4 capete, n_emb 32, 16 384 simboluri): 3 texte; oracolul nostru pe procesor și, în SIMULARE, canalul de lucru (35 de lansări pe simbol); o greutate schimbată cu 1 % e prinsă. Vectorii NU sunt în repo (hub privat): se pregătesc local cu `tools/vectori-kgr1/pregateste.py`; fără ei testele sunt `ignored` (nu trec în CI), cu `--ignored` și fără fișiere PICĂ | `KGR1_VECTORI=… cargo test --release -p nvidia-ga104 --test sim_i14_token kgr1 -- --ignored` | argmax pe fiecare poziție și 5 simboluri greedy identice cu motorul de referință al lui Kronos; logit-urile ultimei poziții în 1e-3 (măsurat: procesor ≤ 4,6e-6, canal SIMULAT 2,4e-6) | conform | PASS local (procesor) / PASS local (SIMULARE); în CI: neexecutat |

## Rezultate — sesiunea 57 (6 nov 2026, cloud, fără placă) — C-83: I11 (copiere mare) și I12 (matrice × vector Q8_0)

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T93 | PTE pentru RAM (`mmu::encode_pte_sys`, `alloc_sys_mapped`) citite de un al doilea cititor scris de mână (`gpu_sim::vram::translate_any`); otrava refuzată | `cargo test -p gpu-sim --lib host_ram` | pagina RAM găsită doar de `translate_any`, la adresa dată de OS | conform | PASS (SIMULARE) |
| T94 | I11 cap-coadă: golden → canal de lucru → copiere RAM → VRAM → RAM (`copy_b128`, NAK) într-o intrare GPFIFO cu 3 mărci de timp; 16 B, 0x3230 B, 256 KiB | `cargo test -p nvidia-ga104 --test sim_i11_copy` | al doilea buffer RAM = modelul, bit cu bit; cuvintele VRAM la fel; paginile = cele date de OS | conform | PASS (SIMULARE) |
| T95 | I11 greșeli plantate: grilă cu un bloc prea mică, pagină RAM dublată, destinație doar-citire, înregistrări lipsă | `cargo test -p nvidia-ga104 --test sim_i11_copy planted` | fiecare prinsă (deplasarea exactă / refuz / „înregistrarea 0”) | conform | PASS (SIMULARE) |
| T96 | Oracolul Q8_0: ordinea NAK (`matvec_gpu_order`) față de motorul OS (`matvec_cpu`) în marja scrisă dinainte; marja prinde o greșeală de 1e-3 și o coloană inversată | `cargo test -p gpu-compute q8` | toate formele în marjă; greșelile prinse | conform | PASS |
| T97 | I12 cap-coadă: W și x în RAM → copiere sus → `q8_matvec` (NAK) → copiere jos, 4 mărci de timp; 1×32 … 3072×768 | `cargo test -p nvidia-ga104 --test sim_i12_q8` | verdict **identic** bit cu bit cu oracolul programului | conform | PASS (SIMULARE) |
| T98 | I12 greșeli plantate: x mutat cu un bloc, grilă cu un rând mai mică, QMD fără memorie partajată, un octet de greutate stricat | idem, `planted` | niciodată „trece”: rândul numit (63, 5) sau lansare refuzată | conform | PASS (SIMULARE) |
| T99 | Lanțul C-83: `WINDOW_CHAIN_C83` = lanțul complet + I11 + I12 înainte de curățare; eșec la I11 → I12 sărit, Bus Master pus la loc | `cargo test -p nvidia-ga104 --lib c83_chain` | ordinea și oprirea corecte | conform | PASS (SIMULARE) |

## Rezultate — sesiunea 54 (6 nov 2026, cloud, fără placă) — C-71: Booter-ul simulat lărgește WPR2 ca placa

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T92 | `gpu-sim`: Booter-ul simulat citește `GspFwWprMeta` din memoria de sistem simulată (`FakeGsp::wpr_meta_mem`) și coboară WPR2 LO la `gspFwWprStart`, HI rămâne cel scris de FWSEC (ca placa la 5 nov: după I6 `0x1ffe00000..0x1ffee0000`, după I7 `0x1f4100000..0x1ffee0000`) | `cargo test -p nvidia-ga104 --lib booter_widens` | după I7 WPR2 = valorile plăcii; `free_after_i7` = `free_below()`; meta cu magic greșit → Booter refuză (cod SIM `0xbad2`), WPR2 rămâne doar FRTS, zona refuzată | conform | PASS (SIMULARE) |

## Rezultate — sesiunea 53 (5 nov 2026, cloud, fără placă) — C-71: zona VRAM după I7

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T91 | Zona VRAM liberă după I7, față de WPR2 citit pe placă la 5 nov (`LO 0x01f41000 / HI 0x01ffee00` → `0x1f4100000..0x1ffee0000`); `booter_load` 24 684 B (real), `.fwimage` 60 MiB (DEDUS din WPR2: orice mărime în (59,97; 60,97] MiB dă același WPR2) | `cargo test -p nvidia-ga104 free_after_i7` | același început de WPR2 ca `GspFwWprMeta`; zona liberă = `free_below()`, capăt `0x1f4000000`; refuzat: WPR2 în altă parte (alt ELF), WPR2 doar FRTS, capăt greșit | conform | PASS (valori reale, aritmetică) |

## Rezultate — sesiunea 50 (5 nov 2026, cloud, fără placă) — C-71: fereastra VRAM în jurnalul lanțului

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T85 | Jurnalul real al ferestrei VRAM (`VramWindow` pe modelul PRAMIN din `gpu-sim`: tabelele de pagini, invalidarea, punerea la loc) pus în pasul I9 al jurnalului T82 | `cargo test -p nvidia-ga104 full_window_journal_checks` | lanțul tot TRECE; fereastra lăsată mutată → „fereastra VRAM (0x1700) lasata la …”; o mutare recitită diferit → „nu s-a mutat” | conform | PASS (SIMULARE) |
| T86 | I7 în ordinea veche (ca `os-246` pe 5 nov: fără resetarea GSP după FWSEC) pe placa simulată, apoi în ordinea nouă `booter_load_dma_libos` | `cargo test -p nvidia-ga104 old_i7_order` | vechi: I7 TRECE, I8 oprit la cozile GSP (ca pe placă), Bus Master la loc; nou: lanțul tot TRECE, jurnalul arată adresa LibOS păstrată; 2 greșeli plantate prinse (FALCON_OS greșit, cutia GSP necitită egal) | conform | PASS (SIMULARE) |
| T87 | După FWSEC, cutiile GSP nu păstrează scrierea (modelul plăcii, 5 nov); `write_libos_mailboxes` oprește înainte de orice scriere SEC2; după `gsp_reset_into_riscv` (BCR = 0x111) adresa rămâne | `cargo test -p nvidia-ga104 libos_mailboxes_need` | eroare cu valoarea citită (0), nicio scriere SEC2; apoi păstrată | conform | PASS (SIMULARE) |
| T88 | Jurnalul REAL `os246-gpu-jurnal-002.txt` (rândurile I7) citit cu `check_i7_libos` | `cargo test -p nvidia-ga104 i7_libos_rule` | raportat la prima scriere: „cutia GSP nu a pastrat adresa LibOS: GSP nu a fost resetat in RISC-V dupa FWSEC” | conform (și `verifica_jurnal_lant` pe tot jurnalul dă același verdict) | PASS (jurnal real, regula — nu placa) |
| T89 | Etapa WPR2 în turul de citire (faptul nou `i6.wpr2.etapa`) cu valorile brute de pe placă din 5 nov | `cargo test -p nvidia-ga104 wpr2_stage` | 0/0 → „coborat”; `0x01ffe000/0x01ffee00` → „FRTS (dupa I6)”; `0x01f41000/0x01ffee00` → „largit pentru GSP-RM (dupa I7)” | conform | PASS (valori reale, regula) |
| T90 | Soneria canalului de lucru: valoarea dedusă `0xbb0090` = antetul MIT 570.144 (`NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET` + `NV_VIRTUAL_FUNCTION_DOORBELL`) | `cargo test -p nvidia-ga104 --test rm570_crosscheck work_doorbell` | `0xb80000 + 0x30090 = 0xbb0090` = `push::DOORBELL` = `DOORBELL_INFERRED` = `chain_log::WORK_DOORBELL` (și verificat la compilare) | conform | PASS (aritmetică, fără placă) |

## Rezultate — sesiunea 49 (5 nov 2026, cloud, fără placă) — C-71: verificarea jurnalului pentru tot lanțul

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T82 | Jurnalul lanțului complet (cum l-ar salva OS-ul pe stick), citit pas cu pas cu `chain_log::check_chain_journal`: I6/I7 cu încărcarea DMA reală, I8 (prima scriere = soneria GSP, doar ferestrele GSP/SEC2), golden + I9 (doar sonerii + registrele ferestrei VRAM, o sonerie pe apel RM), I10, Bus Master pus la loc | `cargo test -p nvidia-ga104 full_window_journal_checks` | TRECE; 25 sonerii la golden + I9 = 25 apeluri RM; 7 greșeli plantate prinse (o sonerie lipsă, o sonerie în minus față de apeluri, o scriere străină la golden, Command pus greșit, pas doar-citire care scrie, I8 în afara ferestrelor, pași inversați); lanțul fără sonerii → „OPRIT CURAT la i8e-golden” | conform | PASS (SIMULARE) |
| T83 | T78 cu încărcarea DMA reală pentru I6/I7 (înainte: încărcare scurtată a simulatorului) | `cargo test -p nvidia-ga104 full_window_chain_passes` | toți cei 13 pași TREC | conform | PASS (SIMULARE) |
| T84 | Unealta `verifica_jurnal_lant` pe un jurnal fără pași | `cargo run --release -p nvidia-ga104 --example verifica_jurnal_lant -- gol.txt 0x123400000` | „DE ANALIZAT: lipsesc pasi: 0 din 13”, ieșire 2 | conform | PASS |

## Rezultate — sesiunea 48 (4 nov 2026, cloud, fără placă) — C-71: lanțul complet al ferestrei

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T78 | `WINDOW_CHAIN_FULL` rulat cap-coadă pe placa simulată (13 pași, cod real, GSP simulat care răspunde doar după sonerie) | `cargo test -p nvidia-ga104 full_window_chain_passes` | toți TREC; o sonerie pe mesaj RM; scrieri doar GSP/SEC2; `add64` bit cu bit; mediana 4000 ns (ceasul simulat); Bus Master la loc | conform | PASS (SIMULARE) |
| T79 | Același lanț cu transportul vechi (fără sonerie) | `cargo test -p nvidia-ga104 without_a_doorbell_per_rpc` | oprire la `i8e-golden` (timeout), pașii de după săriți, Bus Master la loc | conform | PASS (SIMULARE) |
| T80 | Porțile lanțului complet: doar-citire → oprire înainte de Bus Master; lista veche → oprire la I7; `READY_FOR_WINDOW` → merge mai departe | `cargo test -p nvidia-ga104 full_chain_gates` | conform | conform | PASS |
| T81 | `gpu lant` complet pe placa reală | — | TRECE sau oprire curată la primul eșec | — | NOT TESTED (fereastra cu Adrian) |

## Rezultate — sesiunea 47 (4 nov 2026, cloud, fără placă) — C-69: I7 verificat ca I6 + costul unei lansări

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T72 | I7 scrie exact cele 18 registre SEC2 (66 de scrieri pe Booter-ul SIM de 8 + 8 blocuri), cutiile = adresa `GspFwWprMeta` | `cargo test -p nvidia-ga104 i7_writes_exactly` | lista fixă, nimic în afara SEC2 | conform | PASS (SIMULARE) |
| T73 | Jurnalul I6 + I7 citit dintr-o trecere (`check_i6_i7_journal`) + 8 greșeli plantate în partea I7 | `cargo test -p nvidia-ga104 i6_i7_journal` | I6 78 scrieri, WPR2 trece, I7 66 scrieri, RISC-V activ; fiecare greșeală cu motivul ei | conform | PASS (SIMULARE) |
| T74 | `read_after_i7` după un Booter refuzat: cutia 0 = codul ROM-ului, RISC-V inactiv, 3 citiri, 0 scrieri | `cargo test -p nvidia-ga104 read_after_i7` | conform | conform | PASS (SIMULARE) |
| T75 | Costul unei lansări: marcaj + 16 lansări `add64` cu eliberare cu marcă de timp; o lansare refuzată → nicio cifră | `cargo test -p nvidia-ga104 --test sim_launch_cost` + `cargo test -p nvidia-ga104 launch_cost` | mediana = costul simulat ales de test (4070 ns); eroare la înregistrare lipsă/timp înapoi | conform | PASS (SIMULARE) — ceasul e al modelului |
| T76 | I7 pe placa reală (`verifica_jurnal_i7`) | — | TRECE | — | NOT TESTED (fereastra 1, după I6 TRECE) |
| T77 | Costul unei lansări pe placa reală | — | 17 înregistrări în ordine, timp crescător | — | NOT TESTED (fereastra 4, după I10) |

## Rezultate — sesiunea 46 (4 nov 2026, cloud, fără placă) — C-66: capătul WPR2 de 128 KiB + citirea de după I6

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T68 | Jurnalul real `gpu-jurnal-002.txt` (3 nov, placă) prin `verifica_jurnal_i6` | `cargo run --release -p nvidia-ga104 --example verifica_jurnal_i6 -- gpu-jurnal-002.txt` | ordinea I6 trece; HI citit după FWSEC | ordinea I6 **TRECE** (720 scrieri, 2 scrieri config); HI lipsește din jurnal (codul de atunci nu jurnaliza citirile WPR2) → HI confirmat din ecranul de pe 4 nov: `0x1ffee0` | PARȚIAL (date de pe placă, verificare pe gazdă) |
| T70 | `compare_frts_full`: valorile de pe placă din 4 nov (LO `0x1ffe00`, HI `0x1ffee0`) → TRECE, forma „ultimul bloc de 128 KiB”; formele vechi trec; 6 capete greșite prinse la HI; zonă prea mică/nealiniată → fără forma de bloc; regula veche dă exact mesajul din 4 nov | `cargo test -p nvidia-ga104 frts_last_128k` | conform | conform | PASS (SIMULARE) |
| T71 | I6 pe `gpu-sim` cu forma plăcii (implicită acum) + `read_frts_after_i6` fără permis: 0 scrieri, 3 citiri în jurnal, registru GSP refuzat, placă rece raportată | `cargo test -p nvidia-ga104 read_back_without_permit` | conform | conform | PASS (SIMULARE) |

## Rezultate — sesiunea 40 (3 nov 2026, cloud, fără placă) — C-58: verificarea capătului WPR2

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T65 | `compare_frts`: start ≠ cerut → eroare la LO `0x1fa824`; capăt = sfârșit exclusiv sau ultima pagină de 4 KiB → trece; orice alt capăt → eroare la HI `0x1fa828` (așteptat/citit) | `cargo test -p nvidia-ga104 frts_end_exclusive` | conform, cu valorile de pe placă (FRTS `0x1ffe00000..0x1fff00000`) | conform; 5 capete greșite prinse | PASS (SIMULARE) |
| T66 | I6 întreg pe `gpu-sim` cu ambele forme ale HI + un capăt greșit (mesajul numește HI) | `cargo test -p nvidia-ga104 hi_forms` | trece / trece / „verificare esuata la 0x1fa828…” | conform | PASS (SIMULARE) |
| T67 | `i6_log::wpr2_after_i6`: LO/HI citite din textul jurnalului (cu prefixul OS) și verificate cu aceeași regulă | `cargo test -p nvidia-ga104 wpr2_read_back` | trece pe ultima pagină; HI străin = eroare; doar citirea rece = nimic de verificat | conform | PASS (SIMULARE) |
| T68 | HI real citit după FWSEC pe placă (`gpu-jurnal-002.txt`, 736 rânduri) | `wpr2_after_i6` pe jurnal | ultima pagină `0x1ffeff000` (cel mai probabil) | — (jurnalul e pe stick) | NOT TESTED |
| T69 | `i6_log::check_stick_journal` + unealta `verifica_jurnal_i6`: jurnalul întreg de pe stick (Bus Master pornit, I6, citirea WPR2, Bus Master la loc) citit dintr-o trecere | `cargo test -p nvidia-ga104 whole_stick_journal` | 2 scrieri config; ordinea I6 trece; WPR2 ultima pagină trece; HI greșit → eroare la HI; jurnal doar-citire → nimic de verificat | conform | PASS (SIMULARE) |

## Rezultate — sesiunea 30 + 30b (1 nov 2026, cloud, fără placă) — C-49/C-51: pregătirea I6

| ID | Test | Comandă | Așteptat | Obținut | Stare |
|---|---|---|---|---|---|
| T56 | Siguranța „ucode version”: adrese (GSP 0x8241C0, SEC2 0x824140, +4·(id−1)), decodare (bitul cel mai înalt + 1), refuzuri (biți 31:16, 0xbadf…, 0xffffffff, fereastră < 16 MiB, id 0/17) | `cargo test -p nvidia-ga104 fuse` | 6 teste; o citire, 0 scrieri | conform; 6 greșeli plantate prinse | PASS (SIMULARE) |
| T57 | Alegerea semnăturii FWSEC cu forma de pe placă (ucode_id 9, 3 semnături, masca 0x0007): versiunea 0/1/2 → indexul 0/1/2; versiunea 3 → `Blocked` | `cargo test -p nvidia-ga104 fuse::tests::signature_choice` | conform | conform | PASS (SIMULARE: valorile siguranței sunt ale noastre) |
| T58 | Verificarea I6 doar citire (`i6_check`): 3 citiri pe placa rece, FRTS sub zona VGA, WPR2 ridicat = nu trece, 0xbadf…/0xffffffff/VRAM implauzibil = eroare | `cargo test -p nvidia-ga104 i6_check` | 3 teste, 0 scrieri | conform; 4 greșeli plantate prinse | PASS (SIMULARE) |
| T59 | I6 scrie exact cele 18 registre GSP din `docs/FEREASTRA_C-49.md` (78 de scrieri pe imaginea SIM de 20 de blocuri) | `cargo test -p nvidia-ga104 i6_writes_exactly` | lista fixă, nimic în afara GSP | conform | PASS (SIMULARE) |
| T61 | Jurnal pe loc (`Journal::with_sink`) | `cargo test -p gpu-core sink` | fiecare eveniment, și cele scoase din coadă | conform | PASS (SIMULARE) |
| T62 | Jurnalul I6 citit înapoi și verificat (`i6_log`) + 8 greșeli plantate în jurnal | `cargo test -p nvidia-ga104 i6_journal` | trece pe jurnalul simulat; fiecare greșeală cu motivul ei | conform | PASS (SIMULARE) |
| T63 | Bus Master (I5g): 1 scriere recitită, refuzuri, restaurare | `cargo test -p gpu-pci bus_master` | conform | conform; 6 greșeli plantate prinse | PASS (SIMULARE) |
| T64 | FWSEC copiat din PROM în RAM, neatins, peste granița imaginilor | `cargo test -p nvidia-ga104 ucode_and_signature` | octeți identici, 0 scrieri | conform | PASS (SIMULARE) |
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

### C-54 (1 nov, sesiunea 33) — bucla automată și lanțul (toate SIMULARE)
| Test | Ce verifică | Rezultat |
|---|---|---|
| `gpu_sim::observed` (2) | imaginile VBIOS și registrele modelului = valorile de pe placă | PASS (SIMULARE) |
| `read_tour` (4) | turul pe modelul real dă valorile de pe placă; fereastră mică → „sarit”; placa nu răspunde → eroare; lista onestă | PASS (SIMULARE) |
| `board_compare` (5) | rapoartele vechi → fapte; 0 diferențe față de model; un model greșit e prins; `fapt …` dus-întors | PASS (SIMULARE) |
| `chain` (6) | oprire la I7 blocat; eșec I6 → Bus Master pus la loc; modul automat nu rulează pași care scriu; pas doar-citire care scrie = eșec | PASS (SIMULARE) |
| `gpu tur` pe placă | rulat de bucla automată; raportul comparat cu `compara_raport` | NOT TESTED (comanda OS nu există încă) |
| `gpu lant` pe placă | doar cu Adrian de față | NOT TESTED |

| T138 `flr_reset` (4), C-151 punctul 3 | WPR2 ridicat → FLR → GFW → WPR2 = 0 → I6 + I7 a doua oară; WPR2 jos → nicio scriere; WPR2 rămas → oprire fără a doua încercare; FLR neanunțat / placa mută / BAR care refuză / GFW neterminat / tranzacții blocate | PASS (SIMULARE) |
| FLR pe RTX 3070 Ti | proba C-152 punctul 1 (OS, ciclul automat) | NOT TESTED |

| T139 viteza maximă (7) + `PerfGain`, C-147 punctul 4 | `boost_start`/`boost_finish` față de modelul de performanță din `gpu-sim`: acceptată (P8 → P0 → ștergere → P8), fără efect, refuzată, planul oprit, fără model, forma publică; octeții pe fir; eroarea de transport; cititorul `parse_ks`; comparația cu `os-313` | PASS (SIMULARE) |
| Viteza maximă pe RTX 3070 Ti | Release OS separat, modul ciclu (C-148 completarea 1) | NOT TESTED |

| T147 dieta 2 (2 + 4 unitare + 1 în `gpu-compute`), C-157 punctul 3 | un text de 16 poziții, lista lipită + dieta 1: timpul din două înregistrări = suma duratelor = `end − start` = `busy` la fiecare poziție; înregistrare lipsă / ceas înapoi prinse și cu două; `argmax_bits` = `argmax` (pe text și pe 4 000 de vectori cu −0/+0, subnormale, NaN, infinit); `RecordPlan` (1 664 → 416 cuvinte; #513: 356 → 8); `poll_planned` (somn, apoi citiri la 5 µs; timeout și Esc rămân erori); rândurile `kt` și cititorul lor | PASS (SIMULARE) |
| T148 #651 pe placă (4 teste), C-157 punctul 4 | formele = fișierul (206 665 728 B de tensori, 98 Q8_0 + 194 F32 = 292); lista simplă refuzată de constantele OS (pasul `0x100` < `0x185`, 32 KiB < 37 312 B), listele lipite (± argmax) încap; `ModelLimits::of_lists`; bugetul VRAM 270,3 MiB (cache-uri 72), blob continuu, `with_programs`; un model cu forma lui #651 (greutăți generate) urcat în 25 de bucăți, lista lipită, 4 poziții, oracolul trece (3,7e-6) | PASS (SIMULARE) |
| T149 poziția 0 rece (1), C-157 punctul 2 (os-329) | lista lipită #513: poziția 0 cu umbra goală = 11 264 cuvinte în sloturi + 1 064 pushbuffer; după încălzire 0 + 89 (+3); poziția 1 = 130 (= placa); 0,66 µs/cuvânt explică și pushbuffer-ul (690 µs) și poziția 0 rece din os-324 (±25 %) | PASS (aritmetică) |
| T150 proba P8 (mma16 pe placă), C-160 punctul 1 | 6 seturi fixe prin procesorul de comenzi simulat, judecate cu toleranța scrisă; cele 10 modele de adunare deosebite de seturi; rândurile `kt mma16` citite înapoi; greșeli plantate (D nescris, B transpus, alt set, set lipsă) prinse | PASS (SIMULARE) |
| T151 oracolul pasului de antrenare (3 + 4 unitare în `gpu-compute::train`), C-160 punctul 3 | procesor față de PyTorch, pasul de aur mic (`tests/date/pas_de_aur_mic/`), toleranțele din `PAS_DE_AUR.md`: **sgd** eroarea pe 10 pași ≤ 1,1e-7 rel., gradientul pasului 1 ≤ 3,6e-7 abs. (‖Δ‖/‖g‖ ≤ 6,3e-7), greutăți după 1 / 10 pași ≤ 1,2e-7 / 2,4e-7 (tol. 1e-6); **adamw_513** eroarea ≤ 8,8e-8, greutăți ≤ 1,9e-6 / 1,05e-5 (tol. 1e-4); gradientul verificat și independent cu diferențe finite; greșeli plantate (un gradient +0,2 %, rețeta greșită, ținte mutate) prinse | PASS (procesor, nu placă) |
| T152 `mm_train` (antrenare, programul 1), C-160 punctul 4 | binarul NAK (SHA-256, `ffma.ftz` + `fadd.ftz`, barieră); prin procesorul de comenzi simulat, cu modelul pe plăci al programului: **bit cu bit** egal cu `matmul_t_fma_ftz` pentru 4 forme × 6 combinații de opțiuni (plăci parțiale, intrări subnormale, −0,0 păstrat); față de oracolul de antrenare (`train::linear` / `linear_back`, sume f64) pe greutățile reale `in_proj` ale stratului 0 din modelul mic: Y = X·Wᵀ + b max \|d\| 9,5e-7, dX 1,9e-8, dW 1,2e-7 — toate sub marginea scrisă (cel mult 4,6 % din ea); refuzuri (opțiuni necunoscute, fără memorie partajată / barieră, 0 scrieri); greșeli plantate (modelul fără opțiuni, dW fără `TA`) prinse | PASS (SIMULARE) |
| T153 `xent` (antrenare, programul 7: softmax + entropia încrucișată + gradientul), C-160 punctul 4 | binarul NAK (SHA-256, `MUFU.EX2` / `LG2` / `RCP`, fluturi în warp, barieră); prin procesorul de comenzi simulat, cu modelul în ordinea sumelor programului: față de oracolul f64 `train::softmax_xent` în marginea scrisă (+ termenul FTZ adăugat după prima rulare) pentru V = 1, 255, 256, 257, 1 000, 16 384, răspândiri până la 200 și logit-urile reale ale modelului mic (gradient ≤ 75 % din margine, doar FTZ; eroarea ≤ 13 %); media erorii 44,193805307 față de `loss_and_grad` (\|Δ\| 3,9e-7); ținta greșită = NaN vizibil; refuzuri (bloc, memorie partajată, barieră: 0 scrieri); greșeli plantate prinse | PASS (SIMULARE) |
| Dieta 2 pe RTX 3070 Ti | Release OS, opțiune oprită implicit (nota `C-157_P3_dieta-doi.md`) | NOT TESTED |
| T154 `ln_back` (antrenare, programul 4: LayerNorm înapoi, dx), C-160 punctul 4 | binarul NAK (SHA-256, fără MUFU, fluturi în warp, barieră); prin procesorul de comenzi simulat, cu modelul în ordinea codului NAK (inclusiv fma-ul `dy·γ − m1` făcut de NAK): față de oracolul f64 `train::layernorm_back_rows` în marginea scrisă pentru C = 1, 64, 255, 256, 257, 768, 1 000, 2 048, media 100 și γ-urile reale ale modelului mic după 10 pași (≤ 21 % din margine); Σ dx ≈ 0 pe rând; două rulări = aceiași biți; refuzuri (bloc, memorie partajată, barieră: 0 scrieri); greșeli plantate (modelul altui program, μ/r ale altui rând, fără termenul x̂·m2, fără r) prinse | PASS (SIMULARE) |
| T155 `colsum` (antrenare, programul 2 + dγ/dβ ale programului 4), C-160 punctul 4 | binarul NAK (SHA-256, fără MUFU, fără barieră); un fir pe coloană, rândurile în ordine: dγ, dβ și suma deplasării față de `train::layernorm_back_rows` în marginea scrisă pentru 1…2 048 rânduri, C = 1…768 (≤ 50 % din margine); suma deplasării = dβ bit cu bit; grila scurtă lasă coloanele neatinse vizibile; greșeli plantate (modelul altui program, dβ în loc de dγ, un rând lipsă) prinse | PASS (SIMULARE) |
| T156 comparația după os-335 (P8 pe placă), C-157 punctul 2 | cifrele MĂSURATE ale plăcii (exacte + ‰ pe cele 6 seturi) față de modele calculate pe procesor: `GA104_OS335` (8 produse + acumulatorul pe pas, exponentul nominal, 1 bit în plus, tăiere spre zero) le reproduce pe toate 12; cele 10 modele vechi și 19 vecini (bloc 1…16 × 0…3 biți) nu; modelul în toleranță pe 20 de intrări aleatoare, refuzuri (bloc, biți, inf); rândurile `kt mma16 set: iesire (fnv)` / `model ga104` scrise și citite înapoi | PASS (procesor față de cifrele plăcii; potrivire pe amprentă, nu încă bit cu bit) |
| T157 `ln_fwd` (antrenare, programul 3: LayerNorm înainte care scrie μ și r), C-160 punctul 4 | binarul NAK (SHA-256, un singur `MUFU.RSQ`, fluturi în warp, barieră); prin procesorul de comenzi simulat, cu modelul în ordinea codului NAK (RSQ corect rotunjit): y, μ, r față de oracolul f64 `train::layernorm_rows` + `layernorm_stats` în marginea scrisă (cu 2⁻²² loc pentru RSQ-ul plăcii) pentru C = 1, 64, 255, 256, 257, 768, 1 000, 1 024, 2 048, 3 072, media 100, varianța ≈ ε, γ/β reale ale modelului mic (≤ 17 % din margine); lanțul `ln_fwd` → `ln_back` față de `train::layernorm_back_rows` (≤ 4,1e-7 relativ, toleranța 2⁻¹⁶); aceiași biți la două rulări; grila scurtă; refuzuri; 5 greșeli plantate prinse | PASS (SIMULARE) |
| T158 `gelu_tr` (antrenare, programul 5: GELU înainte / înapoi), C-160 punctul 4 | binarul NAK (SHA-256, 57 de instrucțiuni, un `MUFU.RCP` + un `MUFU.EX2`, modul la 0x44); prin procesorul de comenzi simulat, cu modelul în ordinea codului NAK (RCP / EX2 corect rotunjite): y = u·Φ(u) și du = g·(Φ + u·φ) față de oracolul f64 `train::gelu_rows` / `gelu_back_rows` în marginea scrisă (A&S 1,5e-7 + rotunjiri + 2⁻²² loc pentru fiecare unitate MUFU a plăcii) pe u ∈ [−12, 12] (10 007 și 3 001 puncte), valori speciale (0, ±subnormale, ±14, ±50, ±1e4), 4 096 de valori ca ieșirea fc (≤ 81 % din margine; modelul încape și fără locul plăcii); lanțul înainte → înapoi în același procesor de comenzi; aceiași biți la două rulări; modul 0 nu citește g; grila scurtă; refuzuri; 7 greșeli plantate + modelul altui program prinse | PASS (SIMULARE) |
| T159 `attn_back` (antrenare, programul 6, jumătatea înapoi: atenția cauzală înapoi), C-160 punctul 4 | binarul NAK (SHA-256, 319 instrucțiuni, 25 de registre, 5 `ffma` + 2 `fmul` + 2 `fadd`, fără MUFU / barieră / shfl / atomic, modul la 0x60, scala la 0x64); două lansări prin procesorul de comenzi simulat (modul 0 rândurile lui dS, modul 1 dQ \| dK \| dV), cu modelul = codul NAK: față de oracolul f64 `train::attention_back_rows` cu aceleași intrări, în marginea scrisă (lanțuri `ffma` + FTZ + rotunjirea oracolului), pe 7 forme (modelul mic B2 T32 C64 H4; T 70; d 40; d 1; H 1; softmax ascuțit; gradienți 1e-4): dS ≤ 19,4 %, dqkv ≤ 65,8 % din margine; față de P-ul în f64 al pasului (exact `loss_and_grad`) ≤ 2⁻¹⁶ relativ; masca nu citește deasupra diagonalei (NaN acolo: aceiași biți); lansarea = modelul bit cu bit; aceiași biți la două rulări; grilă scurtă; refuzuri; 10 greșeli plantate + modelul altui program prinse | PASS (SIMULARE) |
| T160 `attn_fwd` (antrenare, programul 6, jumătatea înainte: atenția cauzală pe T poziții, scrie P), C-160 punctul 4 | binarul NAK (SHA-256, 221 de instrucțiuni, 25 de registre, 2 `ffma`, un `mufu.exp2` după `fmul` cu log₂e `0x3fb8aa3b`, un `mufu.rcp` fără Newton, fără barieră / shfl / atomic, scala la 0x50); o lansare prin procesorul de comenzi simulat, modelul operație cu operație (EX2 / RCP rotunjite corect): față de oracolul f64 `train::attention_rows` în marginea scrisă (lanțuri + FTZ + 2⁻²² pentru fiecare MUFU), pe 8 forme (modelul mic; T 70; d 40; H 1; d 1; softmax ascuțit; foarte ascuțit până în FTZ; T 130): P ≤ 13,6 %, att ≤ 6,5 % din margine; P = +0 deasupra diagonalei; lansarea = modelul bit cu bit; aceiași biți la două rulări; rândurile lui P însumează 1; lanțul înainte → `attn_back` față de P-ul în f64 ≤ 2⁻¹⁴; două programe pe același procesor; grilă scurtă; refuzuri; 8 greșeli plantate prinse | PASS (SIMULARE) |
| T161 `emb_fwd` + `emb_back` (antrenare, programul 8: încorporările înainte / gradientul tabelelor pe segmente, fără `atomic`), C-160 punctul 4 | binarele NAK (SHA-256, 42 / 80 de instrucțiuni, 19 registre; `emb_fwd`: un singur `fadd.ftz`, fără împărțire / MUFU; `emb_back`: doar `fadd.ftz`, fără MUFU / barieră / shfl / atomic); lansări prin procesorul de comenzi simulat: „înainte” bit cu bit egal cu oracolul `train::embed_rows` (pasul de aur mic + 8 forme); „înapoi” față de oracolul f64 `train::embed_back_rows` în marginea scrisă (lanț f32 + FTZ + rotunjirile), pe 8 forme: ≤ 48,8 % din margine; pe datele pasului de aur mic față de gradientul PyTorch al pasului 1: max \|Δ\| 1,5e-8 (tol. 1e-6); lansarea = modelul bit cu bit; aceiași biți la două rulări; simbolurile nefolosite păstrează capul, pozițiile ≥ T rămân 0; grilă scurtă; două programe pe un procesor; refuzuri; 8 greșeli plantate prinse | PASS (SIMULARE) |
| T162 lista `lipita-ln-32` (`TokenList::FusedLn32`: LayerNorm nou + înmulțirile pe sferturi), C-157 punctul 2 după cursa de pe placă (22 nov 05:06) | textul modelului mic (3 straturi, 16 poziții, `EachLaunch` și `PerLayer`, dieta driverului) prin procesorul de comenzi simulat: oracolul pe procesor `Trece` la fiecare poziție, simbolurile = ale `lipita-ln` și `lipita-32`, abaterea față de fiecare < 1e-4 și biți diferiți față de fiecare (ambele schimbări au rulat); lista = `narrow(lipita-ln)`, aceleași cbuf-uri și grile ca `lipita`, 88 de lansări la #513, programele `PROGRAMS_FUSED_LN32`; lățimea 768; T148 (constantele OS-ului, #651) rulat și cu lista nouă | PASS (SIMULARE): față de procesor 3,6e-7, față de `lipita-ln` 3,0e-7, față de `lipita-32` 2,8e-7 | nimic pe placă încă (programele au rulat toate, separat) |
| T163 `gnorm` (antrenare, programul 9: norma totală a gradientului + factorul de tăiere, fără `atomic`), C-160 punctul 4 | binarul NAK (SHA-256, 135 de instrucțiuni, 19 registre, un `mufu.sqrt` + un `mufu.rcp`, fără `atom`); lansări prin procesorul de comenzi simulat (o lansare sau două: sume parțiale, apoi terminarea) față de oracolul f64 `train::grad_norm` în marginea scrisă, pe 10 forme (n 1 … 300 000, 1 172 de blocuri, scări 1e-18 … 1e18): ≤ 7,4 % din margine; pe gradientul PyTorch al pasului 1 (`pas_de_aur_mic/adamw_513`): N egal cu norma PyTorch (abatere 0); c = 1 / < 1 / max_norm ≠ 1 / zerouri / NaN / ∞; aceiași biți la două rulări; refuzuri; 8 greșeli plantate prinse | PASS (SIMULARE) |
| T164 `opt_step` (antrenare, programul 10: pasul SGD / AdamW pe toți parametrii, cu factorul de tăiere), C-160 punctul 4 | binarul NAK (SHA-256, 61 de instrucțiuni, 19 registre, `mufu.sqrt` + 2 `mufu.rcp`, niciun `ffma` — totul `precise`); SGD bit cu bit egal cu `train::sgd_step`; AdamW: m, v bit cu bit, w în marginea scrisă față de `train::AdamW::step` (6 cazuri, pașii 1 … 10, cu / fără tăiere): ≤ 71,2 %; cu unitățile mutate ±2 ulp ≤ 65,2 %; 8 greșeli plantate prinse; pasul de aur mic cu norma + tăierea + optimizatorul pe placa simulată (gradienții din oracol), 10 pași față de PyTorch: `sgd` 1,19e-7 / 2,38e-7 (tol. 1e-6, bit cu bit cu oracolul), `adamw_513` 1,94e-6 / 1,09e-5 (tol. 1e-4), normele ≤ 1,8e-7 relativ | PASS (SIMULARE) |
| T165 pasul de aur mic **cap-coadă** pe placa simulată (toate cele 10 programe de antrenare + `vadd`), C-160 punctul 5 | 75 de lansări pe pas (`adamw_513`; 73 la `sgd`), toate prin procesorul de comenzi, tensoarele intermediare în memoria plăcii; 10 pași față de PyTorch: eroarea ≤ 1,1e-7 rel., gradientul pasului 1 1,5e-7 abs. / 5,4e-7 rel., greutățile `sgd` 6,0e-8 / 2,4e-7 (tol. 1e-6), `adamw_513` 2,4e-6 / 1,28e-5 (tol. 1e-4), normele ≤ 8,9e-8; față de oracolul pe procesor ≤ 3,6e-7 (gradient); 4 greșeli plantate în lista driverului prinse (tăierea uitată doar după 10 pași); lățimile lui #513 cu greutăți inventate, 1 strat, un pas `adamw_513` față de oracol (gradient 5,2e-7, greutăți 6,1e-6); toate 12 straturile (`#[ignore]`, de mână): 385 de lansări, ~1 545 MiB, gradient 5,1e-7, greutăți 6,9e-6 | PASS (SIMULARE) |
