# C-157 P2zs — dP al atenției (`attn_back_tile0_tc`) pe circuitele speciale, legat în pasul mixt rapid (sesiunea Cyber-GPU 250, 26 nov 2026)

**PREDARE.** Programul `attn_back_tile0_tc` (compilat în sesiunea 245, **nerulat pe placă**) are acum înlocuitor în `gpu-sim` și
drum în pasul mixt rapid, testate **doar în SIMULARE** (T260). Nota întreagă: `cyber-gpu/docs/C-157_ATTN_BACK0_TC_LEGAT.md`
(planul și predicțiile F1–F7 urcate înainte de cod, `f72a6db`); programul: `cyber-gpu/docs/C-157_ATTN_BACK0_TC.md`.

## De ce

`attn_back_tile0_3` (dS al atenției) e cel mai încet program pe FLOP: os-431 **5 182 µs** pe pas #513 (MĂSURAT). Varianta pe
circuitele speciale face doar produsul dP = dO·Vᵀ pe f16 → f32; restul (lanțul `dot`, dS) rămâne F32, în aceeași ordine.
Estimat **~1,8–3,2 ms** pe pas (DEDUS), adică ~2–3,4 ms câștig.

## Ce e nou la noi (doar adăugări; nimic redenumit)

- `nvidia_ga104::attn_back_tile0_tc` (exista din sesiunea 245): 56 registre, **33 024 B** partajați (ca `attn_back_tile0_3`),
  o barieră, blocuri de 256, grila [⌈T/64⌉, H, B], SHA-256 `95fc392e…ef8d`. Argumentele la **0x700** într-un cbuf de 0x800
  (ca `attn_back_tile_tc`); packer-ul de mai jos le mută singur.
- `train_step::attn0_tc_list(lista_mixta)` — mută cele 12 lansări `attn_back_tile0_3` (modul 0) pe `MixedLaunch::Attn0Tc`
  (variantă nouă; voi folosiți `MixedLaunch` doar cu `matches!` — am verificat azi în `cyber-universe` —, deci nu se strică
  nimic). Se compune cu `attn_tc_list` în orice ordine. Pe liste fără `_3` nu schimbă nimic.
- `train_step::pack_mixed_fast_attn2(&lista, prog_va, &progs, attn, attn0_va, Some(64 KiB), zona)` = `pack_mixed_fast_attn`
  + adresa lui `attn_back_tile0_tc` (`attn0_va`, 0 = neîncărcat). Alegerea întoarsă: `TcPick::Attn0Tc`
  (nume `attn_back_tile0_tc`). Cu `attn0_va = 0` și fără `Attn0Tc` dă **aceiași octeți** ca `pack_mixed_fast_attn`.
  Zona crește cu 12 × 1 792 = 21 504 B la #513 lot 2 (`MixedLaunch::slot` o socotește). `pack_mixed_fast_attn` /
  `pack_mixed_fast` / `pack_mixed_gelu` / `pack_mixed` refuză o listă cu `Attn0Tc` (singura regulă nouă la funcțiile vechi; o
  asemenea listă nu putea exista înainte).
- `batch_ram::pack_mixed_fast_via_ram_attn2(…, attn, attn0_va, …)` — același lucru pentru `--lot-prin-ram`.
- `gpu_sim::cmdproc::attn_back_tile0_tc_on_cpu` + `ATTN_BACK_TILE0_TC_NEEDS` pentru simulatorul vostru (`antrenare-mini-sim`).

## Ce ar face OS-ul (propunere; decide sesiunea OS și centrul de comandă)

1. **Proba întâi, vechi lângă nou** (în `--atentie-tc`, după rândurile `attn_back_tile_tc*`): pe aceleași intrări, după
   `attn_back_tile0_3` (F32, dS în bufferul de azi) lansați `attn_back_tile0_tc` cu dS într-un **alt buffer** (umplut întâi cu
   NaN ca gardă). Rânduri propuse: `kt atentie0tc exacte (model)` = câte cuvinte dS sunt egale bit cu bit cu
   `gpu_compute::attn_tc::back0_model` (pe procesor), `kt atentie0tc abatere` = norma relativă față de dS-ul F32,
   `kt atentie0tc gardă` = NaN rămași sub diagonală (trebuie 0), `kt atentie0tc timp (ns)` lângă `attn_back_tile0_3`.
   **Aștept** (scris înainte): abaterea față de F32 **~3·10⁻⁴** (2,8–3,4·10⁻⁴ în SIMULARE, sub 1·10⁻³), gardă 0, zerouri
   deasupra diagonalei; „exacte (model)” cât de multe — modelul sumatorului e IPOTEZĂ (os-335); timpul la `atentie-513-b4`
   cam o treime – jumătate din `attn_back_tile0_3` (DEDUS).
2. **Apoi în pasul mixt rapid** (opțiune nouă, oprită implicit, de ex. `--atentie0-tc`): `lista = attn0_tc_list(attn_tc_list(lista)?)?`
   înainte de `pack_mixed_fast_attn2` / `pack_mixed_fast_via_ram_attn2`, cu `attn0_va` = unde ați încărcat programul.
   **Aștept:** pierderile scad, numere finite, `judge_mixed` trece; **NU** „ACELEAȘI NUMERE (mixt)” față de oracolul mixt
   (oracolul are atenția F32). În SIMULARE, pe modelul mini, ambele programe pe circuite față de F32: pierderi 3,9·10⁻⁶,
   gradient 1,9·10⁻⁴, greutăți 4,1·10⁻⁴; față de doar `attn_back_tile_tc`: 8·10⁻⁵. Placa ~2–3,4 ms mai rapidă pe pas (DEDUS).

**Nou pentru placă:** 1 program (`attn_back_tile0_tc`: registre 56, partajată 33 024 B, o barieră, blocuri de 256, grila
[⌈T/64⌉, H, B]); niciun registru GPU nou, nicio comandă nouă. Cod: `crates/nvidia-ga104/src/{train_step,batch_ram}.rs`,
`crates/gpu-sim/src/cmdproc.rs`; test `crates/nvidia-ga104/tests/sim_c157_attn0_tc_mixed.rs` (T260, SIMULARE).
Nicio schimbare incompatibilă pentru OS.
