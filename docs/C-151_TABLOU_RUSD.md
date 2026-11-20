# C-151 punctul 1 — tabloul de bord al plăcii („RM User Shared Data”, RUSD) — SIMULARE

Sesiunea 124, 20 nov 2026. Comanda: C-151 punctul 1 (Adrian, 19 nov 23:37: „da” la treapta 1; pe placă **aprobat**: pornirea
zonei și citirea ei **la sfârșitul lanțului**, după ce rezultatele sunt pe stick; un refuz = rând în jurnal).
Nota pentru OS: `integration/cyber-universe/C-151_tablou-rusd.md`. Codul: `nvidia_ga104::gsp::rusd`; modelul: `gpu_sim::fake_rm::SimRusd`.

## 1. Ce e
O pagină de memorie de sistem (464 de octeți folosiți) pe care **programul NVIDIA din placă (GSP-RM) o completează singur**, la un
interval cerut (500 ms la plăcile GeForce): temperatura, ceasurile nucleului și memoriei (MHz), starea de performanță, încărcarea,
motivele încetinirii, puterea medie și limita de putere. Driverul o pornește cu **două comenzi** și apoi doar **citește memorie** —
nu citește niciun registru al plăcii și nu scrie nimic în placă în afară de cele două comenzi (și oprirea lor).

## 2. Regula „trei surse” — ÎNDEPLINITĂ pentru formă, memorie și citire; o diferență scrisă (mânerele)
| Ce | Sursa 1 | Sursa 2 | Sursa 3 | Verdict |
|---|---|---|---|---|
| forma mesajelor | NVIDIA 570.144 `ctrl2080internal.h`: `INIT_USER_SHARED_DATA` 0x20800afe `{NvU64 physAddr}` (0 = oprire); `SET_DATA_POLL` 0x20800aff `{NvU64 polledDataMask; NvU32 pollFrequencyMs}` (16 octeți, calculat de compilator) | NVIDIA 570.144 `gpu_user_shared_data.c`: ordinea INIT → intervalul (masca 0, 500 ms) → masca (interval 0); oprirea = masca 0 | nouveau, seria „GSP telemetry via RUSD” (Mohamed Ahmed, 14 iul 2026, doar r570): aceleași două comenzi, aceeași ordine, „0 = păstrează intervalul” | ÎNDEPLINITĂ |
| rutarea / dreptul | `g_subdevice_nvoc.c`: steagurile `0xc0` = `INTERNAL \| ROUTE_TO_PHYSICAL`, fără `NON_PRIVILEGED` | `control.h`: `0xc0` = „kernel privileged” | `INTERNAL_STATIC_KGR_GET_INFO` are aceeași clasă de drepturi (`0x1c0c0`) și **a fost acceptată de GSP-RM pe RTX 3070 Ti de la clientul nostru** (I10c) | ÎNDEPLINITĂ |
| de unde vine memoria | `gpuCreateRusdMemory`: `sizeof(NV00DE_SHARED_DATA)`, **memorie de sistem, contiguă fizic**, umplută cu zero, adresa văzută de placă (`AT_GPU`) | nouveau: „zero-filled, physically contiguous, coherent sysmem buffer, hand its bus address to the firmware” | `cl00de.h`: structura are doar câmpuri aliniate la 8; nicio versiune sau semnătură — așezarea e fixată de antetul cu care e construit firmware-ul (570.144, al nostru) | ÎNDEPLINITĂ |
| cum se citește | `cl00de.h` `RUSD_READ_DATA`: marca de timp → copia → marca din nou; egale = valid; până la 10 încercări; 0 = nedisponibil; `u64::MAX` = scriere în curs | comentariul din `NV00DE_SHARED_DATA` (aceeași regulă) | nouveau `r570_rusd_read_section`: 10 încercări, `-ENODATA` la 0 | ÎNDEPLINITĂ |
| GA104 | `_rusdPollingSupported`: adevărat pentru orice placă cu GSP (`IS_GSP_CLIENT`) | nouveau: tabelul r570 (toate plăcile cu GSP pe firmware 570) | — (nicio dovadă pe placă) | DEDUS, probabil |

**Diferența (DEDUS, ~70 %):** NVIDIA și nouveau trimit cele două comenzi cu clientul **intern** al GSP-RM (`hInternalClient` /
`hInternalSubdevice`, din răspunsul `GET_GSP_STATIC_INFO`, pe care nu-l citim). Noi le trimitem cu **clientul și subdispozitivul
nostru**, ca la viteza maximă. Steagurile o permit (aceeași clasă ca `KGR_GET_INFO`, acceptată); dacă GSP-RM leagă zona de clientul
intern, va refuza `INIT` → un rând `8000xxxx`, nimic altceva trimis. Dacă se întâmplă, pasul următor (cu „da”-ul lui Adrian) ar fi
citirea `GET_GSP_STATIC_INFO` pentru mânerele interne.

**Ce NU spun sursele:** ce unități are încărcarea (`gpuPercentBusy`) — o scriem brută; dacă `GSP_RM_CONTROL` cu `physAddr` de memorie
de sistem cere ceva în plus de la OS (nouveau nu face nimic în plus). Firmware-ul NVIDIA nu a fost deschis sau analizat.

## 3. Așezarea (570.144, x86-64, tipărită cu `offsetof` de compilatorul C din antetul NVIDIA; 464 de octeți)
| Secțiune | Poziție | Ce folosim |
|---|---|---|
| `clkPublicDomainInfos` | 72 | marca + MHz: nucleu (80), memorie (84) |
| `clkThrottleReason` | 96 | marca + masca motivelor (104); 0x20/0x40 = încetinire termică |
| `perfDevUtil` | 112 | marca + încărcarea nucleului (120), a memoriei (124) |
| `perfCurrentPstate` | 264 | marca + masca stării (272), ca `PERF_GET_CURRENT_PSTATE` |
| `powerLimitGpu` | 280 | marca + limita impusă (292, mW) |
| `temperatures[GPU]` | 296 | marca + `NvTemp` (304, grade C × 256, cu semn) |
| `avgPowerUsage` | 368 | marca + puterea medie a plăcii (376, mW) |

## 4. Secvența (`gsp::rusd`)
1. `rusd_start` (oprit implicit): `INIT(adresa paginii)` → `SET_DATA_POLL(0, 500 ms)` → `SET_DATA_POLL(0x1b, 0)` (ceasuri, perf,
   putere, temperatură). Fiecare răspuns = un rând `kd`; un refuz oprește pornirea (fără eroare).
2. așteptare `3 × 500 ms` (GSP-RM a completat pagina de cel puțin două ori).
3. `read_dashboard`: fiecare secțiune citită după regula de mai sus; rânduri `kd` pentru valori + masca secțiunilor citite + masca
   celor ocupate după 10 încercări.
4. `rusd_stop`: `SET_DATA_POLL(0, 0)` → `INIT(0)`. Dacă `INIT(0)` e refuzat, nota `kd zona comuna NU s-a oprit` — pagina rămâne
   rezervată până la repornire (GSP-RM ar putea scrie în ea).
`dashboard_at_end` le face pe toate patru într-un apel. Verificările (adresa 0 sau nealiniată, masca necunoscută, interval în afara
100–1000 ms) se fac **înainte** de primul octet trimis.

## 5. Modelul din `gpu-sim` (SIMULARE)
`FakeRm::rusd = Some(SimRusd::new(regula))`; regulile (ipoteze): `Accept`, `RefuseInit`, `RefusePoll`, `RefuseDeinit`. `render(t)`
dă pagina după o citire a GSP-RM (valori SIM: 45,5 °C, 210/405 MHz, P8, 21 W, limita 290 W). Fără model (implicit) comenzile sunt
refuzate cu `INVALID_COMMAND`, ca înainte.

## 6. Teste — T140 (`tests/sim_c151_rusd.rs`, 7) + 3 în `gsp::rusd`
| Test | Ce verifică | Rezultat |
|---|---|---|
| T140a | pornire → citire → oprire: toate valorile, ordinea pe fir, modelul oprit la final, cititorul `parse_kd` | PASS (SIMULARE) |
| T140b | octeții celor 5 comenzi; clientul și subdispozitivul nostru | PASS |
| T140c | `INIT` refuzat (o comandă, nimic altceva), interval refuzat (pagina dată înapoi fără ștergerea măștii), fără model | PASS (SIMULARE) |
| T140d | oprirea refuzată: nota „pagina rămâne rezervată” | PASS (SIMULARE) |
| T140e | planul oprit: nimic trimis, nicio așteptare; planuri greșite refuzate înainte de trimitere | PASS |
| T140f | citirea: scriere în curs + marcă schimbată → reîncercare; secțiune lipsă ≠ ocupată; ocupată după 10; placa fierbinte (87 °C, motiv termic) | PASS (SIMULARE) |
| T140g | `dashboard_at_end` într-un apel | PASS (SIMULARE) |
| unități | forma octeților, așezarea în 464 de octeți, validitatea mărcilor (`RUSD_SEQ_DATA_VALID`) | PASS |

`tools/check.sh` (Rust 1.100.0): format, clippy `-D warnings`, **597 teste** (587 → +10; 6 ignorate), build no_std.

## 7. Ce NU știm (NETESTAT pe RTX 3070 Ti)
- dacă GSP-RM acceptă `INIT` de la clientul nostru (vezi diferența de la §2);
- dacă pagina se completează la 500 ms și ce valori dă (temperatura reală, ceasurile în P8 / după viteza maximă);
- unitățile încărcării.
Punctul 2 (citirea înainte și după viteza maximă, pragul de temperatură) vine după ce tabloul merge pe placă.
