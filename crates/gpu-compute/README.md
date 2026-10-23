# gpu-compute

Runtime-ul tensorial: `TensorDesc` (I2), math `no_std`, operatorii de referință f32 (oracolul), backend-ul CPU prin I1.

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-compute` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
