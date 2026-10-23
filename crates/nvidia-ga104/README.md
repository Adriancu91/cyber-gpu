# nvidia-ga104

Backend GA104: ID-uri, BOOT_0/BOOT_42, registre permise (fără scriere BAR0), sonda nivel 0, planul I0–I10, inventar firmware, codificare pushbuffer/GPFIFO.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p nvidia-ga104` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
