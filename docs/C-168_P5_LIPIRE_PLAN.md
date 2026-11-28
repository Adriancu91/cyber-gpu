# C-168 punctul 5 — lipirea lansărilor mărunte (F4): inventar, plan, predicții — scrise ÎNAINTE de cod

**Sesiunea 290, 28 nov 2026, cloud, fără placă.** Tot ce e aici e CALCUL (din cifrele măsurate pe placă în os-503) sau DEDUS.
Nimic nu a rulat pe RTX 3070 Ti în sesiunea asta.

Cererea (C-168 p.5): „GELU + bias în înmulțire, Adam + norma într-o trecere, sumele pe coloane lipite. Țintă: ~21 → ≤ 11 ms.”

## 1. Inventarul: ce costă azi lansările mărunte (os-503, lot 6, pasul 140,6 ms, 491 lansări)

n = 6 × 512 = 3 072 rânduri, C = 768, F = 3 072; un tensor n×C f32 = 9,44 MB, n×F = 37,7 MB. Octeții sunt cei pe care programul
trebuie să-i citească și să-i scrie (CALCUL din argumente); viteza = octeți / timpul măsurat. Vârful plăcii (fișa) ≈ 608 GB/s.

| Program | Lansări | Timp (măsurat) | Octeți (CALCUL) | GB/s | Ce înseamnă |
|---|---|---|---|---|---|
| `opt_step` (AdamW) | 1 | 4 786 µs | 28 B × 98,0 M = 2,75 GB | **573** | 94 % din vârf: nu mai e loc de viteză, doar de octeți mai puțini |
| `gelu_tr` (12 înainte + 12 înapoi) | 24 | 4 247 µs | 2,27 GB | 533 | la memorie; câștigul vine doar din a nu mai citi/scrie |
| `colsum_part` (sume pe coloane, pasul 1) | 73 | 3 418 µs | ~1,49 GB | 436 | la memorie |
| `vadd` (reziduurile: 24 înainte, 24 înapoi) | 48 | 2 483 µs | 1,36 GB | 547 | la memorie |
| `ln_back` | 25 | 1 629 µs | ≥ 0,71 GB | 434 | |
| `xent` | 1 | 1 454 µs | 0,40 GB | 277 | sub memorie (loc ~0,7 ms) |
| `ln_fwd` | 25 | 1 423 µs | 0,47 GB | 332 | sub memorie (loc ~0,5 ms) |
| `colsum` (pasul 2, sumele sumelor) | 98 | 1 347 µs | mic | — | ~14 µs pe lansare = aproape doar costul lansării |
| `gnorm` (norma) | 2 | 733 µs | 0,39 GB | 535 | la memorie |
| **Total mărunte** | **297** | **21 520 µs** | | | |

**Concluzia inventarului (CALCUL):** aproape toate programele mărunte merg deja la 70–94 % din viteza memoriei. Un program
„mai bun” nu le mai grăbește; se câștigă doar dacă **datele intermediare nu mai trec prin memorie** (lipire) sau dacă
lansările foarte mici (cele 98 `colsum`) se adună.

## 2. Podeaua (CALCUL) și ce înseamnă pentru ținta ≤ 11 ms

- AdamW cu `m`, `v`, `w`, `g` în F32 citește/scrie 28 B pe parametru: **4,5 ms la vârful plăcii**, oricum am lipi. Lipirea normei
  în Adam nu scade asta (norma trebuie știută înainte de primul `w` scris); poate doar scoate citirea separată a lui `g`
  (0,73 ms) dacă sumele de pătrate se fac acolo unde gradienții se scriu (alți biți: altă ordine a sumei).
- `xent` citește și scrie logit-urile (0,40 GB): ≥ 0,66 ms.
- Ce rămâne după toate lipirile din §3: **~12–15 ms (DEDUS)**. **Ținta ≤ 11 ms nu se atinge fără o decizie a lui Adrian**
  despre starea lui Adam (de ex. `m`, `v` în bf16 = −8 B pe parametru ≈ −1,4 ms, dar alt antrenament: alt oracol, altă rețetă).
  Nu fac asta fără „da”-ul lui; o notez ca întrebare în hub.

## 3. Lipirile, în ordinea în care le fac (cea mai ieftină și mai sigură întâi)

| Pas | Ce | Program | Biți | Câștig prezis lot 6 (DEDUS) |
|---|---|---|---|---|
| **P5a** | reziduurile înapoi lipite în `ln_back` (`ln_back_res`) **și în pasul mixt rapid** | existent (`ln_back_res`, **dovedit pe placă** în os-415: oracolul trecut) | **aceiași** (demonstrat în SIMULARE în sesiunea 192; aici din nou pe pasul rapid) | **0,5–1,1 ms** (−24 `vadd`) |
| P5b | GELU' lipit în coada lui „proj-dX” pe v8n / v7 (cum e `mm_tc_gelu` pe `mm_tc`-ul vechi) | nou (`mm_tc_v8n_gelu` / `mm_tc_v7_gelu`) | aceiași ca v8n + `gelu_tr` (aceeași coadă, aceleași operații) | 1,3–2,0 ms (−12 `gelu_tr`, −75 MB pe strat) |
| P5c | GELU înainte în coada lui „fc” (scrie și `u`, și `act`) | nou | aceiași | 0,6–1,0 ms (−12 `gelu_tr`) |
| P5d | reziduurile înainte în coada lui „out” / „proj” (`c = a·b + bias + x`) | nou (fanion în coada v8n / v7) | aceiași (o adunare f32 în plus, aceeași ordine ca `vadd`) | 0,6–1,0 ms (−24 `vadd`) |
| P5e | sumele de bias ale lui `fc-db` în coada lui P5b (sume parțiale pe blocuri de rânduri) | în P5b | **alți biți** (altă ordine a sumei; judecat față de oracol) | 0,5–0,9 ms |
| P5f | cele 98 `colsum` finale adunate într-o lansare la capătul pasului (biasurile nu se folosesc până la Adam) | nou (`colsum_multi`) | aceiași (aceeași sumă pe fiecare coloană) | 0,6–1,0 ms |
| P5g | norma: sumele de pătrate din `colsum`/înmulțirile dW, `gnorm` doar finalul | nou | alți biți | 0,4–0,7 ms |

Total prezis: **4,5–7,7 ms** → mărunte ~21,5 → **~14–17 ms**, pasul lot 6 ~134 → ~128 ms (cu P3e + P4b deja socotite separat).
Istoric care cere modestie: os-328 (lipirea la scris) a dat cât s-a prezis, os-415 (lipirea pe F32) a dat ~0 (sub zgomot, când
înmulțirile erau 93 % din pas), os-498 ~jumătate din prezis. De aceea fiecare pas are fereastra lui și se măsoară separat.

## 4. P5a în detaliu — predicțiile (scrise înainte de cod)

Ce se schimbă: planul OS-ului (`new_tiles3 … .with_mm_tile().with_colsum_split()`) primește și `.with_fused_back()`.
`step_mixed` dă atunci `LnBackRes` (o lansare Std) în loc de `LnBack` + `Vadd` la „ln2” și „ln1” pe fiecare strat; toți
packerii rapizi (inclusiv `pack_mixed_fast_attn9`) împachetează lansările Std după `prog_va`, deci **niciun cod nou în
bibliotecă**: OS-ul încarcă în plus `ln_back_res` (`Plan::fused_programs()`), ca la os-415.

- **I1 (SIMULARE):** modelul mic aliniat, 5 pași, pasul rapid (v7 / v8 / v8n pe formă, atenția pe circuite înainte și înapoi,
  `attn9`), cu și fără `with_colsum_split`: pierderile, gradientul pasului 1 și greutățile de la sfârșit **bit cu bit** ca
  fără `with_fused_back`.
- **I2 (SIMULARE):** #513 lot 2 și lot 6: lista are **exact 24 de lansări mai puțin** (491 → 467 la lot 6), 24 `vadd` dispar,
  24 `ln_back` devin `ln_back_res`, nimic altceva; alegerile înmulțirilor (v7 / v8 / v8n, garda) și `attn_fwd_tc` neschimbate.
- **I3 (placa, DEDUS):** `vadd` 2,48 → ~1,24 ms; `ln_back_res` ≈ `ln_back` + ~18 µs (citește încă un n×C); câștig net
  **0,5–1,1 ms** la lot 6 (~0,2–0,4 ms la lot 2). Sub 0,3 ms la lot 6 = RATAT.
- **I4 (placa):** oracolul lot 2 dă aceleași numere ca rândul fără lipire.

## 5. Reguli

Măsurătoare înainte (os-503 / rândul P4b) → schimbare → aceeași măsurătoare după; fiecare program nou trece întâi prin
modelul pe procesor și emularea literală (ca T299); cod care scrie în placa reală doar sub opțiunea OS-ului.

## 6. Rezultatul P5a (SIMULARE) — se completează după cod

T306 (`crates/nvidia-ga104/tests/sim_c168_p5a_ln_back_res_mixed.rs`), SIMULARE:

| Predicția | Rezultat | Stare |
|---|---|---|
| I1: model mic, 5 pași rapizi, aceiași biți | pierderile, gradientul pasului 1, greutățile = aceiași biți, cu și fără colsum împărțit (76 → 72, 94 → 90 lansări) | **CONFIRMAT** (SIMULARE) |
| I2: #513 exact −24 lansări, restul neschimbat | lot 2 și 6: 484 → 460 (pe placă 491 → 467 cu copierile lotului); 24 `vadd` și 24 `ln_back` → 24 `ln_back_res`; restul listei identic, în ordine; alegerile (lot 6 13 / 0 / 134, 12 `attn_fwd_tc`) identice | **CONFIRMAT** (SIMULARE) |
| I3: câștig 0,5–1,1 ms la lot 6 | — | NOT TESTED (placa) |
| I4: oracolul lot 2 aceleași numere | — | NOT TESTED (placa) |

Aflat: packerul nu verifică dacă un program `Std` e încărcat (`prog_va` greșit nu e refuzat) — verificarea rămâne a OS-ului (ca la
os-415). Nota pentru OS: `integration/cyber-universe/C-168_P5a_rezidual-lipit-in-pas-rapid.md`.

Următorul pas (P5b): planul + predicțiile pentru GELU' în coada lui v8n / v7 („proj-dX”), apoi codul.
