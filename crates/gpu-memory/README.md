# gpu-memory

Invarianții M1–M6: tipuri de adrese, poarta IOMMU, allocator de intervale, politica DMA, tabela de buffere cu generații.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-memory` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
