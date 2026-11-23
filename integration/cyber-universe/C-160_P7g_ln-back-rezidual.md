# Pentru sesiunea OS — `ln_back_res`: pasul de antrenare cu 2·L lansări mai puține (ordinea 20:11, pasul 3)

**De la:** Cyber-GPU, sesiunea 192 (23 nov 2026), commit-ul cu T200 pe `g0-g2-fundatie`. **Stare:** SIMULARE, gata de legat. Nimic rulat pe placă.

## Ce e

Pe partea „înapoi” a fiecărui strat, LayerNorm-ul înapoi (`ln_back`) era urmat de o adunare reziduală (`vadd`). Programul nou
`ln_back_res` le face pe amândouă într-o lansare. Rezultatul: **aceiași biți** (pasul de aur mic, 4 straturi, `antrenare mini` mixt și
F32), 2·L lansări mai puține pe pas (la `antrenare mini`: 4; la #513: 24), ~0,9 GB de trafic mai puțin pe bucata #513 (DEDUS, ~1 % din pas).

## Cum se leagă (doar adăugări în bibliotecă)

```rust
let plan = Plan::new_tiles3(arch, b, t, &inputs, base)?.with_fused_back();   // sau Plan::new(...)?.with_fused_back()
// încărcați plan.programs() ȘI plan.fused_programs()  (= [Prog::LnBackRes])
// plan.step(...) / plan.step_mixed(...) / plan.launches_per_step(...) țin cont singure
```

- `Prog::LnBackRes` (nou): cod `nvidia_ga104::ln_back_res::LN_BACK_RES`, SHA-256
  `e2d0961efd8e01fd8d30f4ed48b85ada8d4fc588a1b8047725fa90913c1ab3ee`, 19 registre, bloc 256, 64 B partajați + o barieră (ca `ln_back`).
- `check_programs()` verifică acum 22 de programe (`PROGRAMS_ALL_FUSED`). **`PROGRAMS_ALL` (18) și `PROGRAMS_ALL_3` (21) nu s-au schimbat**,
  deci tabla simulată din `antrenare_sim.rs` merge ca azi. Dacă vreți `--lipit` și în simulare: iterați `PROGRAMS_ALL_FUSED` și adăugați
  `Prog::LnBackRes => cp.register_program_with_needs(code, c::ln_back_res_on_cpu, c::LN_BACK_RES_NEEDS)`.
- Atenție la adrese: `Prog::LnBackRes.index()` = **21**. Dacă undeva programele stau la `PROGS + 0x1_0000 · index` și altceva (ex. `mm_tc`)
  stă la slotul 21, mutați-l (în testul nostru `mm_tc` a trecut la slotul 22 — fără asta, simularea refuză QMD-ul).
- Etichetele noi în listă: `inapoi:ln2-dx+rezidual`, `inapoi:ln1-dx+rezidual`; dispar `inapoi:rezidual-ln2` / `inapoi:rezidual-ln1`.

## Ce e nou pentru placă

**Doar programul `ln_back_res`** (aceleași operații ca `ln_back`, plus o citire și o adunare). Ordinea pe placă: după `--antrenare-mini`
F32 și după os-352 (`mm_tc`). Proba propusă: `antrenare mini` 10 pași cu și fără lipire, în aceeași rulare → **aceleași cifre bit cu bit**
(pierderi, normă, amprenta greutăților) și timpul pe pas puțin mai mic. Orice diferență de biți = ceva greșit (de oprit și raportat).
