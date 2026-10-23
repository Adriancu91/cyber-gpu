# gpu-command

Primitive de trimitere neutre: inel PUT/GET cu validarea GET, fence-uri monotone, așteptări limitate.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-command` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
