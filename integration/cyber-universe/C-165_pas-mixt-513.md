# Pentru OS: pasul mixt (`mm_tc`) pe #513 — scara erorii S = 2¹⁴ (C-165, prioritatea 2 a centrului)

Sesiunea Cyber-GPU 217, 25 nov 2026. Totul **SIMULARE**; nimic nu a rulat pe placă. Detalii și cifre: `docs/C-165_PAS_MIXT_513.md`, T223.

## Ce e nou (doar adăugări, nicio schimbare incompatibilă)
- `nvidia_ga104::train_step::LOSS_SCALE_513: f32 = 16384.0` — scara pentru lotul #513 (2 × 512 simboluri, vocabular 16 384).
- `nvidia_ga104::train_step::loss_scale_for(vocab, tokens) -> f32` — regula generală (1 024 pentru `antrenare mini`, 16 384 pentru #513).
- `LOSS_SCALE` (1 024) rămâne cum e; `antrenare mini --mixt` nu se schimbă.

## Ce înseamnă pentru `--antrenare-513 --mixt` (când îl faceți)
- Același cod ca la `--antrenare-mini --mixt` (`plan.mixed_uploads(S)` o dată, apoi `plan.step_mixed(Recipe::AdamW513, &adam, t, S)`
  pe fiecare pas, `ts::pack_mixed_gelu` pentru zona de control), cu **S = `ts::LOSS_SCALE_513`** în loc de `ts::LOSS_SCALE`.
- Planul: același ca azi (`new_tiles3` …); `with_mm_tile` / `with_mm_choice` nu contează la pasul mixt (toate înmulțirile modelului merg
  pe `mm_tc`); `with_colsum_split` și varianta lipită (`with_fused_back().with_fused_gelu()`) merg.
- Lista unui pas (T223, SIMULARE): `placi3 + colsum`: **484 de lansări, 147 pe `mm_tc`**, zona de control **499 KiB**;
  varianta lipită: **448, 135 `mm_tc` + 12 `mm_tc_gelu`**, **481 KiB**. Programele noi față de pasul F32: `mm_tc` (și `mm_tc_gelu` la lipit).
- m, v în RAM (`new_*_mv_ram`) nu merg cu pasul mixt (refuzat cu eroare, ca până acum).

## Ce a ieșit în SIMULARE (modelul mini cu lotul 32 × 32 = același d(logits) ca #513)
- S = 2¹⁴: aceleași cifre ca oracolul în toleranțele pasului mixt (gradientul ≤ 8,8·10⁻⁴), cea mai mare valoare d(logits)·S = 16, nicio
  valoare subnormală.
- S = 1 024 (scara de azi): 53 % din d(logits) subnormale în f16, dar aceeași eroare — deci și el ar merge la pornire. Recomandăm totuși
  2¹⁴ (rezervă când softmax-ul se ascute; DEDUS).

## Ordinea pe placă (neschimbată de nota asta)
1. os-352: `test-placa --mm-tc` — proba `mm_tc` singură (**încă nerulată pe placă**).
2. os-384: `test-placa --antrenare-mini --mixt` (încă nerulat).
3. Abia apoi `--antrenare-513 --mixt` cu `LOSS_SCALE_513`. Cifre așteptate: DEDUS, dacă `mm_tc` atinge 5 TFLOP/s, înmulțirile pasului
   scad de la ~440 ms la ~60–120 ms; cifra reală o dă proba `mm_tc` (timpul la `qkv-513`).

**Ce e nou pentru placă din această notă:** nimic (doar o constantă pe procesor).
