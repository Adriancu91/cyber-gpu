# C-165, pasul următor: `mm_train_tile_gelu` — GELU' lipit pe plăcile de 64 × 64

**Sesiunea 214, 24 noiembrie 2026, ~21:12–21:48 (ora României), în cloud, FĂRĂ placă.** Totul de aici e verificat **doar în
SIMULARE** (pe procesor). Nimic nu a rulat pe RTX 3070 Ti.

## De ce
Cursa C-165 (os-411/412) încă n-a rulat pe placă (e programată la noapte). Următorul punct din coadă, scris de sesiunea 213:
„variantele pentru `mm_train_gelu`”. Azi, într-un plan cu **ambele** opțiuni — `with_mm_tile` (înmulțirile pe plăci de 64 × 64,
os-410: ×1,68) și `with_fused_gelu` (GELU' lipit în „proj-dX”, −L lansări) — cele L înmulțiri „proj-dX+gelu” rămâneau pe
`mm_train_gelu`, adică pe plăcile vechi de 16 × 16 (viteza lui `mm_train`). Deci lipirea și plăcile se încurcau una pe alta.

## Ce e
`tools/nak/mm_train_tile_gelu.comp` = `mm_train_tile.comp` neschimbat (încărcările, plăcile 64 × 64, 4 × 4 pe fir, ordinea
`ffma`-urilor) + finalul lui `mm_train_gelu.comp` caracter cu caracter (GELU' cu `fma(u, pdf, cdf)` scris explicit; u citit la
`[rând, coloană]` din locul bias-ului; fără BIAS). Același bufer constant ca `mm_train_gelu` (`mm_train_gelu::Args::cbuf()`),
altă lansare: blocuri de 256, o bucată de 64 × 64 din C pe bloc, 8 KiB partajați.

| | |
|---|---|
| Compilator | Mesa NAK 25.0.7 (reconstruit în sesiune; martorul `mm_train_tile` a ieșit **identic**, SHA-256 `f8acce21…`) |
| SHA-256 | `0c4f96bad3e3dbce87d5c309ba7e2c3d426003ff2097e86dc782ce472ccb11af` (27 040 octeți) |
| Registre | 34 (cât `mm_train_tile`), 0 memorie locală |
| `ffma.ftz` | 368 = 256 (bucla lui `mm_train_tile`, același număr) + 16 × 7 (finalurile) |
| Finalul | cele 16 finaluri = finalul lui `mm_train_gelu` instrucțiune cu instrucțiune (25 de instrucțiuni, forma fluxului de date; `tools/nak/compara_epilog.py`, care prinde și o greșeală plantată `ffma` → `fadd`) |

## Predicții (scrise înainte de teste) și ce a ieșit
1. Programul singur = oracolul (`gelu_tr_gpu_back` peste `matmul_t_fma_ftz`) = `mm_train_gelu`, bit cu bit, pe toate cele 4
   combinații de opțiuni permise, cu plăci parțiale în M, N, K → **CONFIRMAT** (24 de cazuri + formele „proj-dX” ale pasului
   mic și #513 pe 64 de rânduri).
2. Greșelile se prind: finalul uitat, grila cu un rând de blocuri lipsă; grila lui `mm_train_gelu` (16 × 16, mult mai mare) e
   inofensivă (blocurile în plus nu scriu nimic); BIAS și C = u refuzate → **CONFIRMAT**.
3. Pasul #513 (`new_tiles3`, rezident, ambele lipiri, `with_mm_tile`) cu `with_mm_tile_gelu`: aceeași listă, doar cele 12 lansări
   „proj-dX+gelu” mutate, aceleași cbuf-uri, aceeași arenă → **CONFIRMAT** (349 de lansări, 12 mutate).
   Am scris în test „de 4 ori mai puține blocuri” — **greșit**: sunt de **16 ori** mai puține (147 456 → 9 216), pentru că
   blocurile vechi fac 16 × 16 rezultate, cele noi 64 × 64, cu același număr de fire (256). Corectat în test.
4. Pasul de aur mic, 10 pași, sgd și adamw_513: cu opțiunea = fără, **bit cu bit** (pe `_3` + ambele lipiri, pe `Plan::new` +
   GELU' lipit, cu alegerea cursei, și singură fără GELU' lipit = `with_mm_tile`), verdictul față de PyTorch TRECE, același număr
   de lansări, nicio lansare `mm_train_gelu` rămasă → **CONFIRMAT**.

## Cât ar câștiga (DEDUS, NETESTAT pe placă)
„proj-dX” la #513: 12 × 2 · 1 024 · 3 072 · 768 = ~58 GFLOP pe pas (~10 % din cele 599). Pe plăcile vechi (~0,6 TFLOP/s, os-395)
~97 ms; pe `mm_train_tile` (×1,68 la os-410) ~58 ms. Deci:
- față de „lipit + `--mm-placi`” cum era până acum: **~−40 ms pe pas**;
- față de „nelipit + `--mm-placi`” (rândul de azi al OS-ului): doar ce aduce lipirea însăși — 3 · 12 = 36 de lansări mai puține
  și două treceri prin `[1 024, 3 072]` mai puține pe strat, **~2–4 ms pe pas**.
Câștigul mare rămâne la cursă (ceasul / plăcile); acesta doar scoate încurcătura dintre lipire și plăci.

## Interfața (doar adăugări)
- `nvidia_ga104::mm_train_tile_gelu` (`MM_TRAIN_TILE_GELU`, `_SHA256`, `REGISTERS`, `BLOCK`, `SHARED`, `CBUF_BYTES`, `grid`).
- `Prog::MmTrainTileGelu` (indicele 29), `PROGRAMS_ALL_TILE_GELU` (30), `PROGRAMS_MM_TILE_GELU_EXTRA`; `check_programs()`
  verifică 30. Indicii 0–28 neschimbați.
- `Plan::with_mm_tile_gelu()` (= `with_mm_tile()` + mutarea), `Plan::mm_tile_gelu()`, `Plan::mm_tile_gelu_programs()`.
  Se combină cu `with_mm_choice`, `resident`, `with_recompute`, `with_colsum_split`, `step_until_opt`.
- `gpu_sim::cmdproc::{mm_train_tile_gelu_on_cpu, MM_TRAIN_TILE_GELU_NEEDS}`.

## Teste
T220: `crates/nvidia-ga104/tests/sim_mm_train_tile_gelu.rs` (4 teste) + `t220_*` în `sim_c160_p5_golden.rs` (2 teste).

## Surse
Ideea plăcilor: siboehm SGEMM_CUDA kernel 5 (MIT, doar ideea), deja notată pentru `mm_train_tile`. Finalul GELU' e al nostru
(`gelu_tr.comp`, sesiunea 147). Nimic copiat; nimic GPL.
