# Cum folosește OS-ul crate-urile cyber-gpu (pentru sesiunea cyber-universe)

Actualizat: 26 octombrie 2026 (sesiunea 6). Nimic de aici nu e aplicat în `cyber-universe`; OS-ul decide singur.
**Nimic nu e testat pe RTX 3070 Ti.**

## Starea de acum (C-16, făcut de sesiunea OS)
OS-ul folosește crate-urile fixate pe commit-ul `41e44ba` (= tag-ul `stabil-2026-10-23`, care există doar local: urcarea tag-urilor
din cloud e refuzată, 403). Așa cum e scris în `kernel/Cargo.toml` și `user/Cargo.toml` ale OS-ului:

```toml
gpu-core     = { git = "https://github.com/Adriancu91/cyber-gpu", rev = "41e44ba" }
gpu-pci      = { git = "https://github.com/Adriancu91/cyber-gpu", rev = "41e44ba" }
nvidia-ga104 = { git = "https://github.com/Adriancu91/cyber-gpu", rev = "41e44ba" }
gpu-sim      = { git = "https://github.com/Adriancu91/cyber-gpu", rev = "41e44ba", optional = true } # doar teste: SIMULARE
gpu-compute  = { git = "https://github.com/Adriancu91/cyber-gpu", rev = "41e44ba" }               # programe ring 3
```

## Pentru C-19 (mutarea fixării) — 26 oct 2026
- **Nu există ramura `main`** în cyber-gpu: tot lucrul e pe `g0-g2-fundatie` (nu o creăm noi, regula sesiunii). Fixarea se face pe un
  **commit** de pe `g0-g2-fundatie` cu CI verde — recomandat: ultimul commit „Stare: sesiunea 5 …” (vezi `docs/IMPLEMENTATION_STATUS.md`).
- `gfw::wait_gfw_boot`, `vbios::read_layout`, `fwsec::locate` sunt neschimbate față de sesiunea 4.
- Nou (doar adăugări): `firmware::{GSP_570_144, load_gsp_set, verify, FirmwareSource, STICK_DIR}`, `sha256`, `gsp::*` (structurile GSP-RM
  570.144, doar SIMULARE), `Ga104Backend::{set_firmware, has_firmware}`. Propunere separată: `I4_gpu-firmware.md` (comanda `gpu firmware`).
- Schimbări de valoare: `firmware::GSP_VERSION` = `"570.144"`; `initialize()` fără firmware se oprește la I4 cu mesajul
  „I4: firmware-ul GSP 570.144 nu a fost dat …” (înainte: blocaj de licență la I4); `gpu plan` va arăta I4 ca implementat și oprirea la I5.
  Dacă un scenariu CI al OS-ului verifică textul vechi de la I4, trebuie actualizat.

## Nou în sesiunea 6 (26 oct seara) — doar adăugări, opțional pentru OS
- Crate nou **`nvidia-rm570`** (no_std, fără dependențe, doar date generate din antetele MIT NVIDIA 570.144 + open-gpu-doc, cu notificarea lor
  în `crates/nvidia-rm570/LICENSE-NVIDIA`). **`nvidia-ga104` depinde acum de el** (același repo; cargo îl ia automat la `git = …, rev = …`).
  Licența rămâne MIT.
- `nvidia-ga104::qmd` (descrierea unei lansări, QMD V03_00), `gsp::registry` (tabelul SET_REGISTRY), `gsp::rpc::{encode_split, join_split}`.
- `gpu-sim::cmdproc` (procesorul de comenzi simulat: rulează `add64` cap-coadă pe procesor) și `gpu-sim::golden_log` (parserul jurnalului
  probei tinygrad) — **doar SIMULARE / teste**.
- Nimic nu scrie în placă. Interfața veche (inclusiv tot ce folosește os-191 de la `dac2e32`) e neschimbată.

## Nou în sesiunea 7 (27 oct noaptea) — doar adăugări, opțional pentru OS
- `gpu-core::hal::Mmio` are o **metodă nouă cu valoare implicită**: `fn is_simulation(&self) -> bool { false }`. Implementările OS-ului
  se construiesc neschimbate; **nu o suprascrieți** pentru BAR-ul real (doar modelele `gpu-sim` întorc `true`).
- `nvidia-ga104::gsp::falcon` (I6–I7 pe registre) și `gsp::sequencer` (secvențiatorul GSP + așteptarea `GSP_INIT_DONE`): pe un BAR real cer
  `HardwareWritePermit` (opțiunea `hw-write`, oprită implicit) → **OS-ul nu trebuie să le cheme** în afara ferestrei cu Adrian.
- `nvidia-ga104::fw_hdr` (antetele fișierelor `bootloader` / `booter_load`, doar citire din bytes).
- `gpu-compute`: `math::fma_f32`, `math::ftz`, `reference::add_ftz`, `reference::matmul_fma`, `reference::matmul_fma_ftz` (oracolele programelor NAK).
- `gpu-sim`: `fake_gsp` (GSP fals pe registre), `cmdproc::matmul_naive_on_cpu` / `matmul_coalesced_on_cpu`.

## Promisiunea de compatibilitate
Față de `41e44ba`, interfața publică a `gpu-core`, `gpu-pci`, `nvidia-ga104` (probe) și `gpu-compute` se schimbă **doar prin adăugare**.
Verificare: `git diff 41e44ba HEAD -- crates/gpu-core crates/gpu-pci crates/nvidia-ga104 crates/gpu-compute | grep -E '^-.*\bpub\b'`
nu întoarce nimic (24 oct). O schimbare incompatibilă, dacă devine inevitabilă, se anunță în `docs/IMPLEMENTATION_STATUS.md` și în hub.

## Ce a apărut după 41e44ba (opțional; mutarea fixării e decizia OS-ului / a lui Adrian)
| Crate / modul | Ce face | Scrie în placă? | Testat |
|---|---|---|---|
| `nvidia-ga104::gfw` | I2: așteaptă GFW_BOOT (2 registre pe lista permisă, limită 4 s) | nu | SIMULARE |
| `nvidia-ga104::vbios` | I3: citește VBIOS prin fereastra PROM, lanț PCIR/NPDE, tabela BIT | nu | SIMULARE (imagini sintetice) |
| `nvidia-ga104::fwsec` | localizează descriptorul FWSEC în VBIOS (ucode-ul nu se citește) | nu | SIMULARE |
| `nvidia-ga104::fb_layout`, `radix3` | aritmetică pentru zonele GSP din VRAM și lista de pagini | nu (calcul pur) | gazdă |
| `nvidia-ga104::firmware` (I4), `sha256` | citește și verifică fișierele GSP 570.144 de pe stick (mărime + SHA-256) | nu | SIMULARE (sumele: din fișierele oficiale) |
| `nvidia-ga104::gsp` | WPR meta, argumentele LibOS, cozile de mesaje, RPC (570.144) | nu (nimic nu atinge placa) | SIMULARE |
| `gpu-compute::rows` | softmax cauzal, atenție, cross-entropy pe felii de rânduri (mai multe nuclee) | — (procesor) | gazdă, bit cu bit |

## Exemplu minimal: identificarea plăcii (doar citire)
Tiparul folosit deja de OS în `kernel/src/drivers/gpu.rs`: un adaptor `ConfigSpace` peste ECAM și un `Mmio` peste prima pagină a BAR0
(mapată UC), ambele **refuză orice scriere** și o numără. Apoi `nvidia_ga104::probe::{probe_config, probe_mmio}` → raportul nivel 0. În teste, același cod rulează
pe `gpu-sim` cu `SimConfig::workstation_rtx3070ti()` (10de:2482, BAR0 16 MiB, BAR1 256 MiB, BAR3 32 MiB) — ieșirea trebuie să spună
SIMULARE. Pe QEMU fără placă: „nicio placa NVIDIA”.

## Exemplu minimal: o operație pe mai multe nuclee, cu referința și toleranța declarate
```rust
use gpu_compute::{reference, rows};
// oracolul, pe un fir:
reference::causal_attention(&q, &k, &v, &mut want, t, heads, d)?;
// rapid: fiecare fir p din n își face felia (fără stare comună; ieșiri disjuncte)
let (r0, r1) = rows::split_rows(t, n, p)?;
rows::causal_attention_rows(&q, &k, &v, &mut got[r0 * w..r1 * w], t, heads, d, r0)?;
// toleranța declarată: 0 (identic bit cu bit), pentru orice n
assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
```
Cu SIMD propriu (FMA) rotunjirea diferă de oracol; toleranța folosită de `kronos-calcul` în OS (|x − ref| ≤ 1e-5·(1 + |ref|)) e
potrivită pentru matmul de mărimea aceea; pentru k mult mai mare, limita teoretică crește cu k (vezi testele din `gpu-compute::kernels`).
