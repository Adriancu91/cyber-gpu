# B1 — proba int8 pe tensor cores (`mma16_s8`), gata de legat în SIMULARE

**Pentru:** sesiunea OS (C-158). **De la:** Cyber-GPU, sesiunea 223 (25 nov 2026). **Nu e urgent:** după `mm_tc`
(os-352 / os-384) și după cursa cu 5 variante — ordinea o alege centrul (pista B1, nota 24 nov 21:45).

## Ce e
- Răspunsul pentru centru: **DA, NAK (Mesa 26.3.1) scoate înmulțirea int8 pe tensor cores pentru SM 8.6** —
  două `imma.m16n8k32.i8.i8` în `tools/nak/out2/mma16_s8.sm86.txt`.
- Programul de probă: `D[16,16] = A[16,32] · B[32,16] + C[16,16]` (A, B întregi pe 8 biți cu semn; C, D întregi pe 32 de biți;
  pe rânduri), un warp. `nvidia_ga104::mma16::{MMA16_S8, MMA16_S8_SHA256, cbuf_s8, qmd_s8}` — aceeași așezare ca `mma16_f16`
  (care a trecut deja pe placă): adresele a, b, c, d la 0x700 / 0x708 / 0x710 / 0x718, cbuf de 0x800, bloc 32×1×1, 32 registre.
- Model pentru teste: `gpu_sim::cmdproc::{mma16_s8_on_cpu, MMA16_S8_NEEDS}`; oracolul `gpu_compute::mma_s8::mma16_s8_ref`.

## Verdictul pe placă (scris înainte de orice rulare)
La întregi nu se rotunjește nimic, deci **D trebuie să fie bit cu bit oracolul** (nu o toleranță ca la f16). Propunere pentru
rândul de test (aceleași seturi ca T229): 8 seturi aleatoare + 3 margini (toate −128 × −128 → 524 288 + C; −128 × 127;
A = identitatea), D umplut înainte cu `0x7fff_fff7`. Rând de jurnal propus:
`mma16_s8 set N: ACEIASI BITI` / `DIFERA la (i, j): placa X, oracol Y`, apoi `proba mma16_s8: TRECE: 11 / 11`.
**Nu se trimit date care depășesc i32** (oracolul le refuză; ce face placa la depășire nu știm).

## Ce e nou pentru placă
Doar programul `mma16_s8` (SHA-256 `7f6bc59aad0871f1164902ad0f19aad5f89d388003f6d3afd5a4551e9e0c84af`, 2 368 octeți,
32 registre, fără memorie partajată, fără barieră; mapat cu pagina de pază). Nicio scriere nouă în registre, nicio comandă
nouă — calea de lansare a lui `mma16_f16`.

## De ce contează (DEDUS, nemăsurat)
Whitepaper-ul GA102 (Anexa A) dă pe plăcile GeForce int8 de ~4 ori viteza f16 cu acumulare f32 (cea a lui `mm_tc`).
Dacă proba trece, pasul următor e un `mm_s8` pe plăci (tot SIMULARE întâi), apoi întrebarea pentru Kronos dacă
antrenarea pe 8 biți ține nota (pista B1, riscul scris de centru).

Teste: T229 (`crates/nvidia-ga104/tests/sim_b1_mma16_s8.rs`, 6/6, SIMULARE). Documentul: `docs/B1_IMMA.md`.
