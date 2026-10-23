# gpu-sim

**SIMULARE.** Modele software (configurație PCI, BAR0, ceas) pentru teste pe gazdă. Nu sunt dovadă de funcționare hardware.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-sim` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
