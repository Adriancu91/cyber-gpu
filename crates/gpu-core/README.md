# gpu-core

Vocabularul comun: erori structurate (`GpuError`), contractul I1 (`ComputeDevice`), stări, capabilități, trăsăturile hardware (`ConfigSpace`, `Mmio`, `Clock`), jurnalul limitat.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-core` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
