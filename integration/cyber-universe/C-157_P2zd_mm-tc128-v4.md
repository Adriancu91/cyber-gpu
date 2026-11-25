# C-157 p.2 — `mm_tc128_v4`: `mm_tc128` cu citirile în avans de 128 de biți (gata pentru placă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 231, 25 nov 2026, în cloud, fără placă. Detalii: `docs/C-157_P2_MM_TC128_V4.md`.

## Ce e
Același program ca `mm_tc128` (aceeași lansare: 256 de fire, plăci 128 × 128, grila ⌈n/128⌉ × ⌈m/128⌉, 24 KiB memorie comună, același cbuf,
aceiași biți așteptați ca `mm_tc`), cu leacul lui `mm_tc_v4`: felia următoare citită cât lucrează circuitele speciale, câte **4 numere
vecine pe citire** (16 citiri de 128 de biți în loc de 128 de câte un număr). **168 de registre** → 1 bloc (8 grupuri de fire) pe unitate.

| | |
|---|---|
| Binarul | `tools/nak/out2/mm_tc128_v4.sm86.sass`, SHA-256 `8bd6361b65a2975832135fc81d2db402b76fbb3736dce45c60ae817e93352d93` |
| Rust | `nvidia_ga104::mm_tc128_v4::{MM_TC128_V4, MM_TC128_V4_SHA256, REGISTERS (168), BLOCK, SHARED, grid, fits}` — `Args` / `cbuf` = ale lui `mm_tc`; `grid` / `BLOCK` / `SHARED` = ale lui `mm_tc128` |
| Simulare | `gpu_sim::cmdproc::{mm_tc128_v4_on_cpu, MM_TC128_V4_NEEDS}` (refuză produsele nealiniate) |
| QMD | ca `mm_tc128`, **dar registrele = 168** (din `mm_tc128_v4::REGISTERS`) |

## Atenție: doar produse aliniate
**Înainte de lansare: `mm_tc128_v4::fits(&args)`** (aceeași regulă ca `mm_tc_v4::fits`). Dacă zice nu → `mm_tc128` (sau `mm_tc`).
O citire de 128 de biți nealiniată pe placă ar fi o eroare de memorie. Formele #513 sunt aliniate dacă bufferele sunt.

## Ce cerem OS-ului (când are placa)
Încă un rând în proba `--mm-tc`, **după `mm_tc_v4`**: aceleași seturi (cele care trec `fits`; celelalte „nealiniat”) + qkv-513.
- **Așteptat:** aceiași biți ca `mm_tc` pe toate elementele, gărzile neatinse.
- **Viteza prezisă (DEDUS):** qkv-513 **~4–7 TFLOP/s**, probabil cam cât `mm_tc_v4` (mai puține citiri, dar ocupare pe jumătate).
  Rândul acesta decide care placă merge mai departe în pasul mixt #513. Dacă nu e măcar ×1,2 față de `mm_tc_v4`, rămâne `mm_tc_v4`.

## Ce e nou pentru placă
Doar programul `mm_tc128_v4` (SHA de mai sus). Nicio scriere în registre, niciun firmware, nicio schimbare la `mm_tc` / `mm_tc128` /
`mm_tc_v4`. Nicio schimbare incompatibilă (doar adăugări).
