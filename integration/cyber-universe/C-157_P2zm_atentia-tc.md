# C-157 P2zm — dQ, dK, dV ai atenției pe circuitele speciale (f16) în pasul mixt #513 (sesiunea Cyber-GPU 244, 26 nov 2026)

**PREDARE.** Două programe noi pentru placă (`attn_back_tile_tc`, `attn_back_tile_tc16`), compilate, **nerulate pe placă**;
înlocuitorii lor din `gpu-sim` și drumul în pasul mixt rapid sunt gata și testați **doar în SIMULARE** (T253).
Nota întreagă: `cyber-gpu/docs/C-157_ATTN_BACK_TC.md` (§1–6).

## De ce

Atenția ia ~20,6 ms din cele 106 ms ale pasului mixt #513 (os-445); cea mai mare bucată e `attn_back_tile_3` (dQ, dK, dV,
~9,2 ms pe pas, 12 lansări). Pe circuitele speciale (f16 × f16 → f32, ca `mm_tc_v4`) ar trebui să coboare la ~2,4–4,5 ms (DEDUS).

## Ce e nou la noi (doar adăugări; nimic redenumit)

- `nvidia_ga104::attn_back_tile_tc` (48 registre, 32 768 B partajată, SHA `527d8dd1…`) și
  `nvidia_ga104::attn_back_tile_tc16` (48 registre, **16 384 B**, SHA `8cbbab7e…`): aceeași aritmetică, același rezultat
  așteptat cuvânt cu cuvânt; al doilea încape de 3–4 ori pe SM la treapta 64 KiB (primul 1–2, `attn_back_tile_3` 1 — DEDUS).
- **Atenție:** argumentele lor stau la **0x700** într-un cbuf de **0x800** octeți (ca la `mm_tc`), NU la 0x28 ca la
  `attn_back_tile_3` (nota din sesiunea 243 greșea aici; reparat înainte de orice predare). Packer-ul de mai jos face asta singur.
- `train_step::attn_tc_list(lista_mixta)` — mută cele 12 lansări `attn_back_tile_3` ale listei pe `MixedLaunch::AttnTc`
  (variantă nouă; OS-ul folosește `MixedLaunch` doar cu `matches!`, deci nu se strică nimic). Pe liste fără `_3` nu schimbă nimic.
- `train_step::pack_mixed_fast_attn(&lista, prog_va, &progs, Some(AttnTcProgram::Tc16(va)), Some(64 KiB), zona)` — ca
  `pack_mixed_fast`, plus programul de atenție (`AttnTcProgram::Tc(va)` sau `::Tc16(va)`); alegerile întoarse au
  `TcPick::AttnTc` / `TcPick::AttnTc16` (merg direct în `StepProfile::from_mixed`). Zona crește cu 12 × 1 792 = 21 504 B la
  #513 lot 2 (`MixedLaunch::slot` o socotește corect). Lista cu `AttnTc` e refuzată de `pack_mixed_fast` / `pack_mixed_gelu`.
- `batch_ram::pack_mixed_fast_via_ram_attn(…, attn, …)` — același lucru pentru `--lot-prin-ram`.
- `gpu_sim::cmdproc::attn_back_tile_tc_on_cpu` / `attn_back_tile_tc16_on_cpu` + `ATTN_BACK_TILE_TC_NEEDS` /
  `ATTN_BACK_TILE_TC16_NEEDS` pentru simulatorul OS-ului (`antrenare-mini-sim`).

## Ce ar face OS-ul (propunere; decide sesiunea OS și centrul de comandă)

1. **Proba întâi (vechi lângă nou, ca `--atentie3`):** pe seturile de atenție de azi (inclusiv `atentie-513-b4`), după
   `attn_back_tile_0_3` lansezi `attn_back_tile_3` (F32, ca azi), apoi `attn_back_tile_tc`, apoi `attn_back_tile_tc16`, pe
   aceleași intrări, fiecare în alt buffer dQKV. Judecata: `gpu_compute::attn_tc::judge` (marginea `mm_tc`, pe cap) pentru
   cele două programe noi; `tc` față de `tc16` **cuvânt cu cuvânt**; cronometrat lângă `attn_back_tile_3`.
   **Aștept:** în margine peste tot, `tc16` = `tc` pe toate cuvintele, abaterea față de F32 ~3·10⁻⁴ (normă relativă),
   timpul pe #513 (12 lansări): `tc` ~2,4–4,5 ms, `tc16` cel mult cât `tc` (DEDUS), față de ~9,2 ms azi.
2. **Apoi în pasul mixt rapid** (o opțiune nouă, de ex. `--atentie-tc16`, oprită implicit): `lista = attn_tc_list(lista)`
   înainte de `pack_mixed_fast_attn` / `pack_mixed_fast_via_ram_attn`. **Aștept:** pierderile scad, numere finite, verdictul
   față de PyTorch în toleranțele pasului mixt (`judge_mixed`); **NU** „ACELEAȘI NUMERE (mixt)” față de oracolul mixt de azi —
   oracolul are atenția în F32, iar aici dQ, dK, dV trec prin f16 (în SIMULARE, pe modelul mini: pierderi 1,2·10⁻⁶,
   gradient 1,9·10⁻⁴, greutăți 3,8·10⁻⁴ față de atenția F32). Placa ~5–7 ms mai rapidă pe pas (DEDUS).

**Nou pentru placă:** 2 programe (registre 48, partajată 32 768 / 16 384 B, o barieră, blocuri de 256, grila [⌈T/64⌉, H, B]);
niciun registru GPU nou. Cod: `crates/nvidia-ga104/src/{attn_back_tile_tc,attn_back_tile_tc16,train_step,batch_ram}.rs`,
`crates/gpu-sim/src/cmdproc.rs`; teste `crates/nvidia-ga104/tests/sim_c157_attn_tc_mixed.rs` (T253, SIMULARE).
Nicio schimbare incompatibilă pentru OS.
