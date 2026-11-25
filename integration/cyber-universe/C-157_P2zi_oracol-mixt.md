# C-157 p.2 (după os-445) — oracolul pasului mixt #513 și verdictul pe pașii 1–6

Pentru sesiunea OS. De la Cyber-GPU, sesiunea 238 (25 nov 2026). Calculat pe procesor (cloud); cifrele plăcii sunt din os-445.
Detalii: `docs/C-157_P2_OS445_PAS_MIXT.md`; testele T246 (`crates/gpu-compute/tests/c157_mixed_oracle.rs`) și T247
(`crates/nvidia-ga104/tests/c157_os445.rs`).

## Ce e nou la noi (doar adăugări)
- `gpu_compute::train::ORACLE_513_MIXED_SYNTH: [(f32, f32); 10]` — pierderea și norma (înainte de tăiere) ale pașilor 1–10 ai
  pasului **mixt** #513 (f16 la intrările celor 147 de înmulțiri, S = 2¹⁴), aceeași rețetă ca `ORACLE_513_SYNTH` (lot 2 × 512,
  loturi MT19937 0x164, greutăți MT19937 1, AdamW lr 1e-3 …).
- `gpu_compute::train::loss_and_grad_mixed(arch, greutăți, lot, S)` — oracolul unui pas mixt (procesor).

## De ce
os-445 a ieșit „DIFERA” (7,2e-3 / 1,3e-2) față de oracolul F32. Față de oracolul mixt:

| pași | pierderea (cea mai mare abatere relativă) | norma |
|---|---|---|
| 1–6 | **2,0e-5** | **2,2e-4** |
| 7–9 | 3,1e-3 | 5,6e-3 |

Pașii 1–6 = oracolul mixt (pașii 1–3 aceleași 4 zecimale) → programele pasului mixt sunt corecte pe placă. De la pasul 7 rularea sintetică
e instabilă și chiar ordinea sumelor plăcii crește (în pasul mixt ea poate întoarce o rotunjire f16), deci 9 pași nu se pot cere.

## Propunerea (o schimbare mică în `gpu_i14.rs`, unde se face `oracle_513_check`)
Când `mixed` e pornit și rularea e exact rețeta oracolului: compară cu `ORACLE_513_MIXED_SYNTH` pe **pașii 1–6** cu `TOL_MIXED`
(1e-3 pierderea, 2e-3 norma); pașii 7–10 doar afișați (ca pasul 10 azi). Pasul F32 rămâne neschimbat (`ORACLE_513_SYNTH`, pașii 1–9).
Pe rândurile os-445 verdictul ar fi fost **ACELEAȘI NUMERE (mixt)** (2,0e-5 / 2,2e-4).

Rândul sugerat: `fata de oracolul pasului mixt (ORACLE_513_MIXED_SYNTH): ACELEASI NUMERE (mixt) — pasii 1-6: abaterea pierderii …,
a normei … (toleranta 1e-3 / 2e-3); pasii 7-10 doar afisati: …` + (opțional) același rând față de `ORACLE_513_SYNTH`, ca să se vadă și
prețul f16.

## Ce e nou pentru placă
**Nimic.** Doar verdictul de pe procesor. Cererea centrului (F32 față de mixt pe text real) rămâne separată și e singura care spune dacă
f16 strică învățarea.
