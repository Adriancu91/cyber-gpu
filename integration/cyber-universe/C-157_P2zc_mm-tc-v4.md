# C-157 p.2 — `mm_tc_v4`: `mm_tc_pf` cu citiri de 128 de biți (gata pentru placă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 230, 25 nov 2026, în cloud, fără placă. Detalii: `docs/C-157_P2_CITIRI_128.md`.

## Ce e
Același program ca `mm_tc` / `mm_tc_pf` (aceeași lansare, același cbuf, 128 de fire, plăci 64 × 64, 24 KiB memorie comună, aceiași biți
așteptați), dar fiecare citire aduce **4 numere vecine** deodată: 8 citiri pe felie în loc de 32. Și folosește **128 de registre** (nu 248).

| | |
|---|---|
| Binarul | `tools/nak/out2/mm_tc_v4.sm86.sass`, SHA-256 `488ca341fd5bf400cbaf41eebe65998c5f7ae167b8c5443828ef451e1967acf1` |
| Rust | `nvidia_ga104::mm_tc_v4::{MM_TC_V4, MM_TC_V4_SHA256, REGISTERS (128), BLOCK, SHARED, grid, fits}` — `Args` / `cbuf` / `grid` = ale lui `mm_tc` |
| Simulare | `gpu_sim::cmdproc::{mm_tc_v4_on_cpu, MM_TC_V4_NEEDS}` (refuză produsele nealiniate) |
| QMD | ca `mm_tc`, **dar registrele = 128** (din `mm_tc_v4::REGISTERS`) |

## Atenție: doar produse aliniate
**Înainte de lansare: `mm_tc_v4::fits(&args)`.** Cere: lungimea rândului lui A (k, sau m cu TA) și a lui B (n, sau k cu TB) multiplu de 4,
adresele lui A și B multiplu de 16 octeți. Dacă `fits` zice nu → lansați `mm_tc_pf` (sau `mm_tc`). O citire de 128 de biți nealiniată
pe placă ar fi o eroare de memorie, nu un rezultat greșit. Formele #513 (qkv, MLP, înapoi) sunt toate aliniate dacă bufferele sunt.

## Ce cerem OS-ului (când are placa)
Încă un rând în proba `--mm-tc`, după `mm_tc_pf`: aceleași seturi (cele care trec `fits`; celelalte se sar și se notează „nealiniat”) +
qkv-513, cu `mm_tc_v4`.
- **Așteptat:** aceiași biți ca `mm_tc` pe toate elementele, gărzile neatinse.
- **Viteza prezisă (DEDUS):** qkv-513 **~5–9 TFLOP/s** (os-423: 1,03; `mm_tc_pf` prezis 3–5). Dacă nu e măcar ×1,5 față de `mm_tc_pf`,
  frâna e altundeva (scrierea în memoria comună / barierele).

## Ce e nou pentru placă
Doar programul `mm_tc_v4` (SHA de mai sus). Nicio scriere în registre, niciun firmware, nicio schimbare la `mm_tc` / `mm_tc_pf`. Nicio
schimbare incompatibilă (doar adăugări).
