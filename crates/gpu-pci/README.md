# gpu-pci

Descoperire PCI/PCIe: antet, decodarea BAR (doar citire), dimensionarea BAR (cu permis, restaurare, verificare), capabilități, ReBAR, raport nivel 0.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-pci` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
