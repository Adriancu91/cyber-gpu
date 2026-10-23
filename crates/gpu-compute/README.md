# gpu-compute

Runtime-ul tensorial: `TensorDesc` (I2), math `no_std`, operatorii de referință f32 (oracolul), `kernels` (matmul pe blocuri și matmul cu greutăți bf16 — identice bit cu bit cu referința), backend-ul CPU prin I1 (operația nouă `MATMUL_BF16W` = 9).

Măsurătoare orientativă (gazdă cloud, Xeon 2,1 GHz, un fir, `cargo test --release -- --ignored`): matmul 64×768×768 —
referința 1,2 GFLOP/s, pe blocuri 14,2 GFLOP/s, cu greutăți bf16 14,5 GFLOP/s. **Atenție:** ținta `x86_64-unknown-none` compilează
fără SSE (virgulă mobilă în software), deci în kernel viteza va fi mult mai mică până când OS-ul compilează cu SSE/AVX (K6-01 le-a pornit).

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-compute` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
