# gpu-sim

**SIMULARE.** Modele software (configurație PCI, BAR0, ceas) pentru teste pe gazdă. Nu sunt dovadă de funcționare hardware.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-sim` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.

- `observed` (C-54): modelul plăcii de pe workstation construit din ce a citit OS-ul pe placă (rapoartele din hub `rezultate/`);
  ce încă ghicește e în `observed::STILL_SIM`. Comparația cu un raport nou: `cargo run --release -p nvidia-ga104 --example compara_raport -- raport.txt`.
