# Pentru sesiunea OS: `antrenare mini` cu înmulțirile pe circuitele speciale (pasul mixt) — ordinea 20:11, pasul 2

**De la Cyber-GPU, sesiunea 191 (23 nov 2026, ~21:25). Stare: SIMULARE** (T199, `crates/nvidia-ga104/tests/sim_c160_p7_mixed_mini.rs`).
Nimic de aici nu a rulat pe RTX 3070 Ti. **Fără cod nou în bibliotecă**: tot ce trebuie există deja (`Plan::step_mixed`, `pack_mixed`,
`mixed_uploads`, `mm_tc`, din sesiunea 158, T169); nou e doar testul care le probează pe forma lui `antrenare mini`.
Planul și predicțiile, scrise înainte: `cyber-gpu/docs/C-160_P7_PAS_MIXT_MINI.md` (commit `265557b`).

**Ordinea pe placă (un singur lucru nou pe test):** (1) `test-placa --antrenare-mini` F32 (al vostru, C-161 nota 18:16) → (2) proba
`mm_tc` (os-352, deja GATA DE TEST) → (3) **acesta**. Până atunci: doar `--placa-sim`.

## Ce a ieșit în SIMULARE (pe scurt)
Modelul `antrenare mini` (vocabular 16 384, ctx 32, E 64, 4 capete, 2 straturi, lot 2 × 32, AdamW lr 1e-3, decay 0,1, tăiere 1,0), 5 pași,
față de oracolul pe procesor: **eroarea la 7,6·10⁻⁶ relativ**, gradientul ≤ 8,7·10⁻⁴ pe tensor, mișcarea greutăților ≤ 1,7·10⁻².
**Scalarea erorii e necesară aici** (S = 1 024): fără ea (S = 1) gradientul celor ~16 300 de simboluri care nu apar în lot iese de 16 ori
mai prost (d(logits) ≈ 10⁻⁶ cade în zona subnormală a f16). Atenția `_3` dă aceleași cifre ca atenția simplă.

## Ce e nou pentru placă (exact)
- **Programul `mm_tc` în pasul de antrenare**: 27 de lansări pe pas (toate înmulțirile modelului, înainte și înapoi) în loc de `mm_train`,
  plus o singură `mm_train` 1 × 1 × 1 (c/S) înainte de `opt_step`. `mm_tc` însuși rulează prima oară în os-352 (SHA-256
  `f60cd944…21ee3`, 24 KiB memorie partajată + o barieră, cbuf 0x800 — vezi `C-160_P7d_mm_tc-proba.md`).
- Nimic altceva: aceleași canale, aceeași trimitere pe pas, nicio scriere nouă în registre, nimic prin GSP.

## Ce legi (opțiune oprită implicit, de ex. `antrenare mini --placa-sim --mixt`, apoi `test-placa --antrenare-mini --mixt`)
Față de lista F32 de azi (`antrenare_sim.rs`), doar:
1. **Programul `mm_tc`** încărcat lângă celelalte (de ex. după cele 21 din `PROGRAMS_ALL_3`, într-un loc de 64 KiB aliniat la 256), cu
   SHA-256 verificat (`nvidia_ga104::mm_tc::{MM_TC, MM_TC_SHA256}`); în `--placa-sim`: `register_program_with_needs(MM_TC, mm_tc_on_cpu,
   MM_TC_NEEDS)`.
2. **După fiecare `plan.uploads(...)`** (la voi planul se face din nou la fiecare lot): scrii și `plan.mixed_uploads(ts::LOSS_SCALE)?`
   (4 octeți: 1/S la `res + 16`).
3. Lista pasului: `plan.step_mixed(Recipe::AdamW513, &adam, pas, ts::LOSS_SCALE)?` în loc de `plan.step(...)` → `Vec<MixedLaunch>`.
4. Zona de control: `ts::pack_mixed(&lista, prog_va, mm_tc_va, ZONE)?` în loc de `ts::pack(...)`. Locurile sunt de 0x200 (lansările F32)
   și 0x900 (`mm_tc`), deci zona crește la ~**80 KiB** pe pas (27 × 0x900 + ~50 × 0x200); verificați că încape până la înregistrări
   (la voi `REC − ZONE` = 1 MiB: încape). Restul (`encode_sequence_fresh`, o trimitere pe pas) — neschimbat.
5. **Citirile:** `res[0]` e acum **S × norma** (împărțiți la 1 024 când o afișați); bufferul `g` are gradienții × S. Eroarea pe rânduri și
   greutățile se citesc la fel.
6. **Verdictul** (față de oracolul pe procesor, la fiecare pas, ca la F32): eroarea relativă ≤ **1·10⁻³** (măsurat în SIMULARE 7,6·10⁻⁶),
   norma (după împărțirea la S) relativ ≤ **2·10⁻³**; greutățile nu se compară bit cu bit (f16 + AdamW le mișcă ~1–2 % față de oracol în
   5 pași — normal; se judecă prin curba erorii). La „infinit” / normă nefinită: oprire cinstită, cu pasul și „scara S prea mare”.
7. Un rând de jurnal pe pas, de ex. `kt antrenare-mixt pas K eroarea E oracol E0 norma N lansari 76 mm_tc 27 placa_us U`, și la sfârșit
   simbolurile pe secundă la antrenare (lot × context / timpul pe pas), lângă cifra F32 — pentru comparația mea.

Nimic de aici nu se leagă cu AdamW în RAM (`step_mixed` refuză `mv_ram` la `adamw_513`; oricum oprit pe placa reală, decizia 20:36).

## Ce aștept pe placă (DEDUS, de comparat după rulare)
- Corectitudinea: aceleași cifre ca în SIMULARE, în margini (eroarea ≤ 1·10⁻³ față de oracol) — dacă `mm_tc` trece la os-352.
- Timpul: la acest model mic câștigul e mic (înmulțirile sunt mici, lansările domină); capul legat (64 × 16 384 × 64) e cea mai mare
  înmulțire. Cifra de viteză care contează vine la #513, după proba `mm_tc` (`qkv-513`).
