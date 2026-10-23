# gpu-compute

Runtime-ul tensorial: `TensorDesc` (I2), math `no_std`, operatorii de referință f32 (oracolul), `kernels` (matmul pe blocuri și matmul cu greutăți bf16 — identice bit cu bit cu referința), backend-ul CPU prin I1 (operația nouă `MATMUL_BF16W` = 9).

Măsurătoare orientativă (gazdă cloud, Xeon 2,1 GHz, un fir, `cargo test --release -- --ignored`): matmul 64×768×768 —
referința 1,2 GFLOP/s, pe blocuri 14,2 GFLOP/s, cu greutăți bf16 14,5 GFLOP/s. **Atenție:** ținta `x86_64-unknown-none` compilează
fără SSE (virgulă mobilă în software), deci în kernel viteza va fi mult mai mică până când OS-ul compilează cu SSE/AVX (K6-01 le-a pornit).

## Pe mai multe nuclee (`rows`)
Biblioteca nu pornește fire (nu știe nimic de planificatorul OS-ului); oferă funcții care lucrează pe **felii de rânduri**, fără stare
globală, ca fiecare nucleu să-și calculeze felia lui în propriul buffer de ieșire:
- se împart simplu, prin felii: `matmul`, `kernels::matmul_blocked`, `kernels::matmul_bf16w` (rândurile lui `a` și `out`),
  `layernorm`, `bias_add`, `gelu_*`, `embedding` (o felie din `ids`);
- au nevoie de `rows`: `softmax_rows_range` (cauzal: rândul știe al câtelea e), `causal_attention_rows` (rândul i citește k, v pe 0..=i),
  `cross_entropy_terms` + `cross_entropy_mean` (fiecare felie dă termenii ei; un singur fir îi adună în ordine);
- `split_rows(randuri, parti, parte)` dă împărțirea (deterministă, echilibrată).

Promisiunea: oricum ar fi împărțite rândurile, rezultatul unit e **identic bit cu bit** cu `reference` (toleranță 0); testele verifică
asta pentru 1, 2, 3, 4, 7, 16 și 40 de părți. Doar pe gazdă (SIMULARE nu e cazul: e calcul pe procesor); pe nucleele OS-ului: NOT TESTED.

```rust
// pe fiecare nucleu p din n (fire pornite de OS):
let (r0, r1) = gpu_compute::rows::split_rows(t, n, p)?;
gpu_compute::rows::causal_attention_rows(&q, &k, &v, &mut out[r0 * w..r1 * w], t, heads, d, r0)?;
```

- Depinde de: vezi `Cargo.toml` (fără dependențe externe).
- Teste: `cargo test -p gpu-compute` (pe gazdă).
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`.
