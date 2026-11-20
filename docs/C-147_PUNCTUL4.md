# C-147 punctul 4 — viteza maximă (`PERF_BOOST`), gata de legat — SIMULARE

Sesiunea 123, 20 nov 2026. Comanda: C-147 punctul 4 + nota centrului de comandă din 20 nov 09:20 (`os-313`: `ks` = P8, `kc` = 3 % din
608 GB/s → condiția (a) îndeplinită; P5 trece înaintea lui P3). Pregătirea (formele, sursele) e din sesiunea 119:
`docs/C-147_ETAPA0.md` §3. Aici: secvența completă gata de legat, modelul în `gpu-sim`, cititorul și comparația.

## 1. Sursele, verificate încă o dată (NVIDIA open-gpu-kernel-modules 570.144, MIT, citite ca specificație)
| Ce | Unde | Ce spune |
|---|---|---|
| forma mesajului | `ctrl2080internal.h` | `NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X` = 0x20800a9a, `{NvBool flags; NvU32 duration}` |
| cine trimite și cu ce mânere | `kern_perf_boost.c`, `kperfBoostSet_IMPL` | driverul NVIDIA transformă `PERF_BOOST` public în `..._SET_2X` și îl trimite prin RM-ul fizic (GSP) cu **mânerele clientului însuși** (`RES_GET_CLIENT_HANDLE(pSubdevice)`, `RES_GET_HANDLE(pSubdevice)`) — exact ce facem noi |
| rutarea | `g_subdevice_nvoc.c` | steagurile `0xc8` = `INTERNAL` \| `ROUTE_TO_PHYSICAL` \| `NON_PRIVILEGED` |
| plicul RPC | `vgpu/rpc.c`, `rpcRmApiControl_GSP` | `GSP_RM_CONTROL` cu `flags = RMAPI_RPC_FLAGS_NONE` — la fel ca plicul nostru |
| semnificația | `ctrl2080perf.h` | `CMD` 1:0 (0 ștergere, 2 la maxim), `CUDA` 4:4, `CUDA_PRIORITY` 6:6; durata ≤ 3600 s; „alți factori pot limita starea” |

**Dovadă de pe placă pentru plic:** o comandă tot `INTERNAL` (`INTERNAL_STATIC_KGR_GET_INFO`, I10c), cu același client și subdevice și
același plic, a fost **acceptată de GSP-RM pe RTX 3070 Ti** (I10 a trecut). Deci steagul `INTERNAL` nu respinge clientul nostru.
Regula „trei surse” pentru forma mesajului: antetul + codul care îl trimite + tabelul de rutare (toate NVIDIA) — **îndeplinită**;
tinygrad trimite altă formă (publică) și nu verifică răspunsul, deci nu e a patra confirmare, doar o alternativă.

## 2. Secvența (`nvidia_ga104::gsp::perf`)
1. `boost_start` — o singură dată, după deschiderea canalului de lucru: `ks inainte` → cererea (0x52, 600 s) → `ks cerere` →
   `ks dupa cerere` (doar dacă e acceptată) → `ks … verdict` (0 oprită / 1 urcată / 2 acceptată fără efect / 3 refuzată).
2. Lanțul neschimbat (`i14-simbol`, `kronos --gpu`, `kp`, `kc`).
3. `boost_finish` — în locul citirii `ks la final` de azi: dacă cererea a fost acceptată, `ks inainte de stergere` → ștergerea în
   aceeași formă (0x10, durata 0) → `ks stergerea`; apoi mereu `ks la final`.

Refuzurile plăcii sunt rânduri; doar transportul sau un răspuns pentru altă comandă sunt erori. Planul e oprit implicit
(`BoostPlan::default().enabled == false`): atunci se citește doar starea.

## 3. Modelul din `gpu-sim` (SIMULARE)
`FakeRm::perf = Some(SimPerf::new(regula))`: repaus P8, cu cererea P0 (valori SIM). Regulile: `Accept`, `AcceptNoEffect`,
`Refuse(status)`. Forma publică primește `NOT_SUPPORTED` — **ipoteza modelului** (steagurile ei `0x318` nu au `ROUTE_TO_PHYSICAL`),
nu răspunsul cunoscut al GSP-RM. Fără `perf` (implicit) modelul refuză totul, ca înainte (T135d neschimbat).

## 4. Teste — T139 (`tests/sim_c147_boost.rs`, 7) + `kronos_perf::gain_compares_two_runs`
| Test | Ce verifică | Rezultat |
|---|---|---|
| T139a | acceptată: P8 → P0 → (înainte de ștergere P0) → ștergere → P8; ordinea pe fir: citire, cerere, citire / citire, ștergere, citire; 7 rânduri; cititorul | PASS (SIMULARE) |
| T139b | octeții: comanda 0x20800a9a, `52 00 00 00 58 02 00 00`, ștergerea `10 00 …`; duratele 0, 3601, „infinit” refuzate înainte de trimitere | PASS |
| T139c | refuzată (0x56): rând `80000056`, verdict 3, nicio ștergere, lanțul continuă | PASS (SIMULARE) |
| T139d | acceptată fără efect: verdict 2, ștergerea se trimite oricum | PASS (SIMULARE) |
| T139e | planul oprit: doar cele două citiri, nota „oprita”, verdict 0 | PASS (SIMULARE) |
| T139f | GSP fără model de performanță: citiri și cerere refuzate = rânduri, nu erori | PASS (SIMULARE) |
| T139g | forma publică refuzată de model; eroare de transport și răspuns pentru altă comandă = `Err` | PASS (SIMULARE) |
| `PerfGain` | comparația a două jurnale (`kp toate`, `kc` MB/s, `ln1`) în ‰ | PASS |

`tools/check.sh` (Rust 1.100.0): format, clippy `-D warnings`, **587 teste** (579 → +8; 6 ignorate), build no_std.

## 5. Ce NU știm (NETESTAT pe RTX 3070 Ti)
- dacă GSP-RM acceptă cererea de la clientul nostru (plicul e dovedit, comanda nu);
- cât de repede urcă placa și dacă rămâne sus între lansări (de aceea citirea înainte de ștergere);
- cât câștigăm: estimare grosieră, nu măsurătoare — dacă ceasurile urcă din P8 la P0, normalizările și `kc` ar trebui să se
  îmbunătățească de multe ori; înmulțirile mari mai puțin (sunt limitate și de program). Cifrele vin din testul cu Release-ul OS.
