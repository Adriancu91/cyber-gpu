# C-160 punctul 4 — programul 3 al antrenării: LayerNorm înainte, care păstrează μ și r (`ln_fwd`)

Sesiunea 146, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Programul „înainte” prin LayerNorm al pasului de antrenare. Față de `layernorm256` (generarea: un singur vector de cel mult 1 024 de numere,
ținut în registre) acesta:
- ia **multe rânduri** (un bloc de 256 de fire pe rând, grila = numărul de rânduri) și **orice lățime C** (două treceri prin rând, din memorie);
- **scrie μ și r = 1/√(var + ε) ale fiecărui rând**, pe care le citesc programele „înapoi” (`ln_back`, `colsum`) — asta leagă cele două
  jumătăți ale LayerNorm-ului;
- μ e rotunjit **o singură dată** (`precise` în GLSL): μ-ul scris pentru drumul înapoi e exact cel scăzut din x (altfel NAK îl lipea în câte
  un `ffma` pe fiecare element, ca în `layernorm256`).

`tools/nak/ln_fwd.comp` → Mesa NAK 25.0.7: **3 456 octeți**, 216 instrucțiuni, 19 registre, 32 de octeți partajați, o barieră;
SHA-256 `895fc7988e387cd8ff279695996f5a78e3f326874fa8013aac5efea31e9f2204`. Bufferul constant 0: x / γ / β / y / μ / r la 0x28…0x50,
C la 0x58, 1/C la 0x5c, ε la 0x60 (0x64 octeți).

Ordinea (citită din `tools/nak/out/ln_fwd.sm86.txt`, modelul `gpu_compute::layer::ln_fwd_gpu256` o urmează): s = Σ x (`fadd.ftz`, de la 0),
fluturi în warp + cele 8 sume în ordine; μ = `fmul.ftz`(S, 1/C); d = `fadd.ftz`(x, −μ); q = `ffma.ftz`(d, d, q); aceeași reducere;
r = `MUFU.RSQ`(`ffma.ftz`(Q, 1/C, ε)); y = `ffma.ftz`(`fmul.ftz`(d, r), γ, β). Un singur `MUFU.RSQ` pe rând: unitatea plăcii e aproximativă,
deci modelul o ia corect rotunjită și **placa nu se compară bit cu bit**, ci în margine. Oracolul: `gpu_compute::train::layernorm_rows`
(adăugat; exact `layernorm` din `loss_and_grad`, care a dat numerele PyTorch la pasul de aur mic, T151) + `layernorm_stats` pentru μ și r.

## Marginea, scrisă înainte de măsurare (T157)
u = 2⁻²⁴, P = ⌈C/256⌉ + 12 rotunjiri pe drumul unei sume. Δμ = (P+1)·u·Σ|x|/C + 2u·|μ|; varianța: (P+3)·u·Σd² + C·(Δμ + u·max|d|)², plus
1/C și fma-ul; **r: jumătate din eroarea relativă a varianței + 2⁻²² (loc pentru `MUFU.RSQ` al plăcii, ~2 ulp)**; y: |γ|·(Δd·r + |d|·Δr + u·|x̂|)
+ 2u·|y| + FTZ. μ scris: Δμ + u·|μ|; r scris: Δr + u·r.

## Ce a ieșit (SIMULARE, modelul față de oracol)
| Caz (rânduri × C) | y: max \|Δ\| (cât din margine) | μ / r (cât din margine) |
|---|---|---|
| 3 × 1 (varianța 0: r = 1/√ε) | 0 | 0 / 28 % |
| 4 × 64 / 3 × 255 / 3 × 256 / 3 × 257 | 2,4e-7 (9–12 %) | ≤ 6 % / ≤ 12 % |
| 4 × 768 (#513) / 2 × 1 000 / 2 × 1 024 | 2,4e-7 (11–12 %) | ≤ 1 % / ≤ 8 % |
| 2 × 2 048 / 2 × 3 072 (peste ce ține `layernorm256`) | 2,4e-7 / 3,6e-7 (≤ 10 %) | ≤ 0,03 % / ≤ 12 % |
| 3 × 768, media 100 (μ rotunjit contează) | 1,6e-5 (7 %) | 7 % / 8 % |
| 2 × 768, varianța ≈ ε (ε contează) | 1,2e-7 (13 %) | 0,5 % / 0 |
| γ, β reale ale modelului mic (cele 5 LayerNorm-uri, după 10 pași `sgd`), 64 × 64 | ≤ 4,8e-7 (≤ 17 %) | ≤ 5 % / ≤ 14 % |

**Lanțul înainte → înapoi** (două lansări în același procesor de comenzi: `ln_fwd` scrie μ și r, `ln_back` le citește), față de
`train::layernorm_back_rows`: abaterea maximă a lui dx / cel mai mare |dx| = 9,6e-8 … 4,1e-7 (C = 64, 768, 768 cu media 50, 2 048), sub
toleranța scrisă dinainte 2⁻¹⁶ ≈ 1,5e-5.

Și: modelul lansat = modelul pe rând, bit cu bit; două rulări = aceiași biți; grila scurtă lasă rândurile neatinse; refuzuri (bloc greșit,
fără memorie partajată / barieră: nimic scris); greșeli plantate prinse (modelul altui program; μ / r ale altui rând; varianța ÷ (C − 1);
ε uitat; β uitat).

**Ce NU e dovedit:** cât de exact e `MUFU.RSQ` pe placa noastră (marginea lasă 2⁻²²; dacă placa iese din ea, se vede la primul test);
viteza (două citiri ale rândului din memorie + una pentru y: la C = 768 placa citește de 3 ori 3 KiB pe rând — neglijabil la pașii de aur).

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Un program nou, `ln_fwd` (același fel de QMD ca `layernorm256` / `ln_back`: 256 de fire, 32 de octeți partajați, o barieră; bufferul
constant 0x64 octeți), un `MUFU.RSQ` pe rând. Nepredat: punctul 6 cere întâi un singur program „înapoi” pe placă.

## Ce urmează (punctul 4, mai departe)
Programul 5 (GELU înainte / înapoi), apoi atenția înapoi, încorporările, norma + tăierea, AdamW.
