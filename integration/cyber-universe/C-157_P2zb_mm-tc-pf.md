# C-157 p.2 — `mm_tc_pf`: `mm_tc` cu citirile în avans (gata pentru placă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 229, 25 nov 2026, în cloud, fără placă. Detalii: `docs/C-157_P2_CITIRI_IN_AVANS.md`.

## Ce e
Același program ca `mm_tc` (aceeași lansare, același cbuf, 128 de fire, plăci 64 × 64, 24 KiB memorie comună, aceiași biți așteptați),
doar că fiecare fir cere valorile feliei **mai multe deodată**, fără să aștepte fiecare valoare înainte de următoarea (os-423 a arătat
că `mm_tc` aștepta ~700 de cicluri la fiecare din cele 32).

| | |
|---|---|
| Binarul | `tools/nak/out2/mm_tc_pf.sm86.sass`, SHA-256 `48defc5de143bd55c980521cb769844ba6d3986be85564cd8e331c4bd7bfa048` |
| Rust | `nvidia_ga104::mm_tc_pf::{MM_TC_PF, MM_TC_PF_SHA256, REGISTERS (248), BLOCK, SHARED, grid}` — `Args` / `cbuf` / `grid` = ale lui `mm_tc` |
| Simulare | `gpu_sim::cmdproc::{mm_tc_pf_on_cpu, MM_TC_PF_NEEDS}` |
| QMD | ca `mm_tc`, **dar registrele = 248** (din `mm_tc_pf::REGISTERS`) |

## Ce cerem OS-ului (când are placa)
Un rând în plus în proba `--mm-tc`: aceleași 6 seturi + qkv-513, o dată cu `mm_tc`, o dată cu `mm_tc_pf`.
- **Așteptat:** aceiași biți ca `mm_tc` pe toate elementele (132 780 / 132 780 ca os-423), gărzile neatinse.
- **Viteza prezisă (DEDUS):** qkv-513 **~3–5 TFLOP/s** (os-423: 1,03). Sub 2 TFLOP/s = frâna e altundeva (scrierea în memoria comună /
  bariere), nu la citiri.

## Ce e nou pentru placă
Doar programul `mm_tc_pf` (SHA de mai sus). Nicio scriere în registre, niciun firmware, nicio schimbare la `mm_tc`. Nicio schimbare
incompatibilă (doar adăugări).
