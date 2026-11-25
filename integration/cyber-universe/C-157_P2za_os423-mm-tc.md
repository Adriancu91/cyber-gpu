# C-157 p.2 — os-423 (`test-placa --mm-tc`): ce a ieșit față de predicție, ce urmează

Sesiunea Cyber-GPU 228, 25 nov 2026, în cloud, fără placă. Comparația întreagă: `docs/C-157_P2_COMPARATII.md` §„os-423”.

## Pe scurt
- **Corect, mai bine decât speram:** 6 / 6 seturi în margine și **132 780 / 132 780 de elemente identice bit cu bit cu modelul nostru**
  (`gpu_compute::mm_tc::model`). Predicția avea doar 50 % încredere. Sumatorul circuitelor speciale e dovedit pe placă.
- **Lent:** 1,03 TFLOP/s în loc de 8–17. Cauza (DEDUS, din listare + timpi): fiecare fir citește cele 32 de valori ale feliei **una câte
  una, așteptând-o pe fiecare** (~700 de cicluri fiecare) → ~12 µs pe felie, cât s-a măsurat. Circuitele speciale stau aproape tot timpul.
- **Ocuparea** (citirea din REZULTAT) e înmulțitorul, nu cauza: la treapta noastră cea mai mare (64 KiB) `mm_tc` are 2 blocuri pe unitate
  → cel mult ×2 (~2 TFLOP/s).

## Ce cerem OS-ului
**Nimic nou.** Nicio schimbare în cod, nimic nou pentru placă din comparația asta. Proba ocupării (gata, `C-165_proba-ocuparii.md`)
rămâne utilă — pentru `mm_tc` prezicem **×1,7–2,0** la 64 KiB.

## Predicții scrise acum (DEDUS), pentru testele care vin
| Test | Prezis |
|---|---|
| proba ocupării, `mm_tc` la 64 KiB | ×1,7–2,0 față de os-423 |
| `mm_tc128` așa cum e, qkv-513 | ~3–4,5 TFLOP/s |
| `mm_tc` cu citirile în avans (programul următor de la noi, încă nescris) | ~5–10 TFLOP/s |

## Ce urmează la Cyber-GPU
Programul `mm_tc` cu citirile în avans (cum face deja atenția gen 3, trecută pe placă: indice strâns cu `min()`, 32 de citiri în zbor,
apoi rotunjirea și scrierea). Verificarea listării: `tools/nak/citiri_seriale.py` / testul T234 trebuie să arate ≥ 16 citiri în zbor.
Va fi predat separat, cu „ce e nou pentru placă: exact ce”.
