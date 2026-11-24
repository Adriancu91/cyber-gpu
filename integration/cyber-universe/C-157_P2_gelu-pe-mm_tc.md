# Pentru sesiunea OS — `mm_tc_gelu`: lipirea GELU' merge acum și în pasul MIXT (`--mixt --lipit`)

**De la:** Cyber-GPU, sesiunea 199 (24 nov 2026), commit `338dbc3` pe `g0-g2-fundatie`. **Stare:** SIMULARE, gata de legat. Nimic rulat
pe placă. Planul și rezultatul: `docs/C-160_P7_LIPIRE_GELU_TC.md`.

## Ce e

Până azi `step_mixed` refuza `with_fused_gelu` (la voi: `SimBoard::new` refuză `mixed && fused`, `antrenare_sim.rs` rândul ~121).
Acum „proj-dX” + derivata GELU merg într-o singură lansare pe circuitele speciale: programul nou **`mm_tc_gelu`** (`mm_tc` neschimbat
până la scrierea finală, apoi exact operațiile lui `gelu_tr` mod 1). T207 (`antrenare mini`, 5 pași AdamW, S = 1 024): **aceiași biți**
ca pasul mixt nelipit, singur și cu `ln_back_res`; 76 → 74 lansări pe pas (#513: 12 mai puține, ~1,8 GB trafic mai puțin pe bucată, DEDUS).

## Cum se leagă (doar adăugări în bibliotecă — codul vostru de azi compilează neschimbat)

```rust
let plan = Plan::new_tiles3(arch, b, t, &inputs, base)?.with_fused_back().with_fused_gelu();
// programe: plan.programs() + plan.fused_programs() + mm_tc (ca azi) + mm_tc_gelu (nou)
let list = plan.step_mixed(Recipe::AdamW513, &adam, t, ts::LOSS_SCALE)?;      // conține MixedLaunch::TcGelu
let (zone, qmds) = ts::pack_mixed_gelu(&list, prog_va, TC, TCG, ZONE)?;        // TCG = unde e încărcat mm_tc_gelu
```

- Cod: `nvidia_ga104::mm_tc_gelu::MM_TC_GELU`, SHA-256 `03de80cfffdce1805e7dd343ae8fafa4853c32c6dcf635697efc9ba1e62557d8`,
  **64 registre**, bloc 128×1×1, 24 KiB partajați + o barieră (ca `mm_tc`), 24 688 octeți (încape într-un loc de 64 KiB).
- Simulare: `cp.register_program_with_needs(MM_TC_GELU, c::mm_tc_gelu_on_cpu, c::MM_TC_GELU_NEEDS)` și codul scris la `TCG`
  (în testul nostru: slotul de după `mm_tc`, `PROGS + 0x1_0000 · 24`).
- `MixedLaunch::TcGelu { args, tag, layer }` e o variantă nouă; voi folosiți doar `matches!(…, Tc { .. })` și `slot()` — merg neschimbate.
  `name()` = `"mm_tc_gelu"`, `slot()` = `SLOT_TC`. Dacă numărați lansările pe circuitele speciale, adunați și `TcGelu`.
- `pack_mixed` (cel de azi) **refuză** o listă cu `TcGelu` (mesaj: „foloseste pack_mixed_gelu”), deci o legare pe jumătate nu trimite
  nimic greșit. `pack_mixed_gelu(…, tc_gelu_va = 0, …)` pe o listă fără `TcGelu` = `pack_mixed`.
- Propunere: în `antrenare … --placa-sim --mixt --lipit` scoateți refuzul și comparați lipit / nelipit ca la P6-170 (așteptat: aceiași biți).

## Ce e nou pentru placă

**Un singur program, `mm_tc_gelu`.** Nimic în registre, nimic prin GSP. Ordinea pe placă: după ce `--antrenare-mini --mixt` (os-384) a
trecut cu `mm_tc` și după `mm_train_gelu` pe pasul F32 (`--lipit`). Nu e o cerere de test pe placă acum.
