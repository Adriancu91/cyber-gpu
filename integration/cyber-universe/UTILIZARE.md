# Cum folosește OS-ul crate-urile cyber-gpu (pentru sesiunea cyber-universe)

Actualizat: 24 octombrie 2026. Nimic de aici nu e aplicat în `cyber-universe`; OS-ul decide singur.
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
