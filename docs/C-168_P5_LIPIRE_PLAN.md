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

## 7. P5b — GELU' în coada înmulțirilor rapide: planul și predicțiile (sesiunea 291, 28 nov, scrise ÎNAINTE de cod)

**Ce e azi (os-503 / planul P5a):** pe fiecare strat, înapoi, „proj-dX” (`dY · Wproj`, m = n·T = 3 072, n = F = 3 072,
k = C = 768 la lot 6) merge pe v8n / v8 / v7 (alegerea pe valuri), apoi `gelu_tr` mode 1 citește `u` și rezultatul și scrie
`dwide2` (3 × 37,7 MB pe strat). Lipirea există doar pe `mm_tc`-ul vechi (`mm_tc_gelu`, 64×64, sesiunea 199) — de 3–4 ori mai
lent decât v8n, deci planul rapid nu o folosește.

**Ce fac:**
1. Aceeași sursă cu un fanion nou `-DGELU`: `mm_tc_v8.comp` → `mm_tc_v8_gelu` și `mm_tc_v8n_gelu` (cu `-DV8N`),
   `mm_tc_v7.comp` → `mm_tc_v7_gelu`. Fără `-DGELU` sursa dă **exact aceiași octeți** ca azi (verific recompilând v7 / v8 /
   v8n și comparând SHA-256). Coada cu `-DGELU` = a lui `mm_tc_gelu.comp`, literă cu literă (u din locul biasului, 0x718).
2. `nvidia_ga104::mm_tc_fast_gelu` (programele, registrele, `Args` = `mm_tc_gelu::Args`, refuzurile), doar adăugări.
3. În simulator: înlocuitorii `mm_tc_v7_gelu_on_cpu` / `mm_tc_v8_gelu_on_cpu` / `mm_tc_v8n_gelu_on_cpu` = aritmetica lui
   `mm_tc_gelu_on_cpu` cu blocul / grila / alinierea programului.
4. `train_step::pack_mixed_fast_attn10` = `attn9` + `GeluFast { v7, v8, v8n }`: un `TcGelu` aliniat ia varianta GELU a
   programului pe care l-ar alege `pick_long_k` pentru același produs fără GELU; dacă acea variantă nu e încărcată → refuz
   înainte de orice trimis (nu cad tăcut pe `mm_tc_gelu`). Cu `None` = `attn9` octet cu octet. `TcPick` primește
   `V7Gelu` / `V8Gelu` / `V8nGelu` (doar adăugări).
5. OS-ul: plan rapid + `.with_fused_gelu()`, încarcă cele trei programe.

**Predicțiile (J1–J7):**
- **J1 (CALCUL, NAK):** fără `-DGELU` cele trei programe vechi ies cu aceleași SHA-256. Cu `-DGELU`, coada GELU' din fiecare
  listare NAK = coada din `mm_tc_gelu.sm86.txt`, instrucțiune cu instrucțiune (`compara_epilog.py`).
- **J2 (CALCUL, NAK):** registrele: v8n_gelu ≤ 168 (păstrează 3 blocuri pe SM), v8_gelu ≤ 128 (păstrează 2 blocuri pe SM —
  **riscul cel mai mare**: peste 128 → 1 bloc pe SM), v7_gelu ≤ 128 (4 blocuri). Dacă v8_gelu trece de 128, `pick` nu-l va
  alege oricum pe #513 (lot 6 0 produse pe v8), îl notez și merg mai departe.
- **J3 (SIMULARE):** model mic aliniat, 5 pași rapizi cu `with_fused_back` + `with_fused_gelu`: pierderile, gradientul
  pasului 1 și greutățile de la sfârșit **bit cu bit** ca același plan cu `mm_tc_gelu` (aceeași aritmetică) și ca planul P5a
  fără `with_fused_gelu` (aritmetica lui `mm_tc_gelu` = `mm_tc` + `gelu_tr` mode 1, sesiunea 199).
- **J4 (SIMULARE):** #513 lot 6: lista are **exact L = 12 lansări mai puțin** (467 → 455 pe placă), 12 `gelu_tr` dispar, cele
  12 „proj-dX” devin `TcGelu` pe varianta GELU a aceluiași program ales azi; restul listei identic, în ordine.
- **J5 (SIMULARE):** refuzuri fără nimic trimis: variantă neîncărcată, adresă nealiniată, `c == u`, BIAS.
- **J6 (placa, DEDUS):** `gelu_tr` înapoi ~2,5 ms la lot 6 (12 × ~210 µs: 113 MB pe strat la ~533 GB/s) dispare; coada citește
  în plus `u` (37,7 MB pe strat, ~70–110 µs) → câștig net **1,0–1,8 ms** la lot 6. Sub 0,5 ms = RATAT.
- **J7 (placa):** oracolul lot 2: aceleași numere ca rândul P5a.

## 8. Rezultatul P5b (CALCUL + SIMULARE, sesiunea 291) — se citește cu §7

Unealta NAK a doua (Mesa 26.3.1) reconstruită în cloud în sesiunea asta; T307 = `crates/nvidia-ga104/tests/sim_c168_p5b_gelu_fast.rs`.

| Predicția | Rezultat | Stare |
|---|---|---|
| J1a: fără `-DGELU` aceiași octeți | `mm_tc_v7` 953ed6b8…, `mm_tc_v8` 1b910a53…, `mm_tc_v8n` 9a0702bc…, `mm_tc_v8r` 6e28d531…, `mm_tc_gelu` 03de80cf… — recompilate, **identice** (git nu vede nicio schimbare în `out2/`) | **CONFIRMAT** (CALCUL) |
| J1b: coada GELU' = a lui `mm_tc_gelu` | `compara_epilog.py --float` (nou: doar operațiile în virgulă mobilă, comparate ca arbore al valorii scrise): v7_gelu 32 / 32, v8_gelu 64 / 64, v8n_gelu 64 / 64 cozi **identice** cu a lui `mm_tc_gelu` (21 operații). Diferența găsită de forma veche a scriptului era doar ordinea: NAK mută testul de semn (`fsetp`) un loc mai sus / jos, iar adresa scrierii între ultimele două înmulțiri — aceleași operații pe aceleași valori | **CONFIRMAT** (CALCUL) |
| J2: registrele | v7_gelu **128**, v8_gelu **128**, v8n_gelu **144** — exact ca programele fără GELU (4 / 2 / 3 blocuri pe SM), 0 „spills” | **CONFIRMAT** (CALCUL) — riscul pe v8 nu s-a produs |
| J3: model mic, 5 pași rapizi, aceiași biți | P5a fără lipire = `mm_tc_gelu` = variantele rapide, bit cu bit (pierderi, gradientul pasului 1, greutăți); 90 → 88 lansări pe pas; plus S4: același produs forțat pe fiecare din cele 3 programe = biții lui `mm_tc_gelu` | **CONFIRMAT** (SIMULARE) |
| J4: #513 exact −12 lansări | lot 2 și lot 6: 460 → 448 (pe placă 467 → 455); 12 `gelu_tr` înapoi dispar; „proj-dX” pe varianta GELU a aceluiași program: **lot 6 12 × v8n_gelu, lot 2 12 × v8_gelu**; restul listei și alegerile identice, în ordine; QMD-urile cu programul, registrele, blocul, grila lor | **CONFIRMAT** (SIMULARE) |
| J5: refuzuri | variantă neîncărcată → refuz înainte de trimis (fără cădere tăcută pe `mm_tc_gelu`); adresă nealiniată; BIAS; `c == u`; `None` = `attn9` octet cu octet; prin RAM (`via_ram_attn10`) aceleași alegeri și aceeași listă | **CONFIRMAT** (SIMULARE) |
| J6: câștig 1,0–1,8 ms la lot 6 | — | NOT TESTED (placa) |
| J7: oracolul lot 2 aceleași numere | — | NOT TESTED (placa) |

Limita comparației J1b (cinstit): ea arată că cele 21 de operații GELU' sunt aceleași pe aceleași intrări; nu verifică registrul care
ține constanta 1,0 (o intrare încărcată înainte de coadă) și nu înlocuiește proba pe placă. Bucla K a fiecărui program e cea
neschimbată (aceleași octeți ai sursei până la scriere), deci suma `v` e a programului fără GELU (dovedită pe placă: v7 os-490,
v8 / v8n os-500).

Următorul pas: P5c (GELU înainte în coada lui „fc”, scrie `u` și `act`) cu plan + predicții întâi; pe placă: rândul P5b (nota
`integration/cyber-universe/C-168_P5b_gelu-in-coada-rapida.md`).

## 9. P5c — GELU înainte în coada lui „fc”: planul și predicțiile (sesiunea 292, 28 nov, scrise ÎNAINTE de cod)

**Ce e azi (planul P5a / P5b):** pe fiecare strat, înainte, „fc” (`h2 · Wfcᵀ + bias`, m = n·T = 3 072, n = F = 3 072, k = C = 768
la lot 6, fanionul TB + BIAS) merge pe v8n / v8 / v7 și scrie `u` (F32, 37,7 MB la lot 6); apoi `gelu_tr` mode 0 citește `u` și
scrie `act` = GELU(u) (F32, 37,7 MB). `u` trebuie păstrat (îl citește înapoi GELU', P5b), deci lipirea scrie **amândouă**.
Profilul os-507 lot 6: `gelu_tr` 4,25 ms pe 24 de lansări; cele 12 înainte mută 75 MB pe strat → ~141 µs fiecare la ~533 GB/s.

**Ce fac:**
1. Aceleași surse cu un fanion nou `-DGELU_FWD` (`mm_tc_v8.comp` → `mm_tc_v8_geluf`, cu `-DV8N` → `mm_tc_v8n_geluf`,
   `mm_tc_v7.comp` → `mm_tc_v7_geluf`): scrierea rămâne cea de azi (`c = v + bias`, adică `u`), plus `act[rând, col]` = corpul
   lui `gelu_tr.comp` mode 0, literă cu literă, pe aceeași valoare. Adresa lui `act` vine la 0x730 (argumentele cresc de la 0x30 la
   0x38 octeți; cbuf-ul rămâne 0x800). Fără fanion sursa dă **exact aceiași octeți** ca azi.
2. `nvidia_ga104::mm_tc_fast_gelu_fwd` (programele, SHA-256, registrele, `Args` = `mm_tc::Args` + `act`, refuzurile: `act == c`,
   `act` nealiniat, fără BIAS ok), doar adăugări.
3. În simulator: înlocuitorii `mm_tc_v{7,8,8n}_geluf_on_cpu` = aritmetica programului fără fanion + `gelu_tr_gpu(u, None)`.
4. `train_step`: `MixedLaunch::TcGeluFwd` (variantă nouă, doar adăugare), `Plan::with_fused_gelu_fwd()` (fc + gelu → o lansare
   cu eticheta `inainte:fc+gelu`), `pack_mixed_fast_attn12` = `attn11` + `GeluFwdFast { v7, v8, v8n }`: un `TcGeluFwd` aliniat ia
   varianta `geluf` a programului ales de `pick_long_k` pentru același produs (v8 → `v8_geluf`, nu există `v8r_geluf`); varianta
   neîncărcată → refuz înainte de trimis; nealiniat → refuz (nu există `mm_tc_geluf` lent; planul OS-ului nu cere lipirea atunci).
   `None` = `attn11` octet cu octet. `TcPick::V7GeluF` / `V8GeluF` / `V8nGeluF`.
5. Unealta NAK a doua (Mesa 26.3.1) trebuie reconstruită în cloud (~35 min) — de aceea codul vine în sesiunea următoare.

**Predicțiile (L1–L7):**
- **L1 (CALCUL, NAK):** fără `-DGELU_FWD` toate programele vechi (v7 / v8 / v8n / v8r și variantele GELU') ies cu aceleași SHA-256.
- **L2 (CALCUL, NAK):** registrele: v8n_geluf ≤ 168 (3 blocuri pe SM), v7_geluf ≤ 128 (4), v8_geluf ≤ 128 (2) — **riscul cel mai
  mare**: GELU înainte ține mai multe valori vii decât GELU' (u și cdf odată cu suma); dacă v8_geluf trece de 128 îl notez, iar
  `pick` oricum nu-l alege la lot 6 (0 produse pe v8).
- **L3 (CALCUL):** coada `act` din fiecare listare = corpul lui `gelu_tr` mode 0 (`compara_epilog.py --float` față de
  `gelu_tr.sm86.txt`, aceleași operații ftz pe aceeași valoare), iar coada `u` = cea a programului fără fanion.
- **L4 (SIMULARE):** model mic aliniat, 5 pași rapizi cu `with_fused_gelu_fwd` (+ P5a + P5b): pierderile, gradientul pasului 1,
  greutățile **bit cu bit** ca fără `with_fused_gelu_fwd`.
- **L5 (SIMULARE):** #513 lot 2 și lot 6: exact **−12 lansări** (12 `gelu_tr` înainte), „fc” pe varianta `geluf` a programului
  ales azi (lot 6: v8n_geluf), restul listei și alegerile identice.
- **L6 (placa, DEDUS):** lot 6: −12 × ~141 µs (`gelu_tr` înainte) + 12 × ~71 µs (scrierea lui `act` în coadă) → câștig net
  **0,6–1,1 ms**. Sub 0,4 ms = RATAT.
- **L7 (placa):** oracolul lot 2: aceleași numere ca rândul fără P5c.

## 10. P5a comparat cu os-508 (sesiunea 293, 28 nov 14:20) — placa

Sursa: hub `rezultate/2026-11-28-workstation-os508-rezidual-lipit/REZULTAT.md` (nucleul os-508 `c4ebc1e`, `--pas-rezidual-lipit`,
două rânduri într-o pornire, `ceas 60`, max 62 °C). Martorul = os-507 (aceeași placă, aceleași programe, fără lipire).

| Predicția | Măsurat | Stare |
|---|---|---|
| I3: câștig 0,5–1,1 ms la lot 6 (sub 0,3 = RATAT) | 119,8 → **118,8 ms** = **−1,0 ms** (25 635 → 25 865 simb/s, +0,9 %) | **CONFIRMAT** (placa) |
| I3: ~0,2–0,4 ms la lot 2 | 60,8 → 59,9 ms = −0,9 ms | **RATAT ÎN BINE** (placa; o singură măsurătoare pe rând, zgomotul între porniri nu e cunoscut la lot 2 — de privit la rândul următor) |
| I4: oracolul lot 2 aceleași numere | „ACELEAȘI NUMERE” (pierderea 6,3·10⁻⁶, norma 6,4·10⁻⁵) | **CONFIRMAT** (placa) |

Aflat: la lot 6 câștigul e la capătul de sus al benzii, deci cele 24 `vadd` costau cât am socotit (~50 µs fiecare) și `ln_back_res`
aproape nimic în plus. La lot 2 câștigul pare cât la lot 6, deși mută de 3 ori mai puțini octeți: probabil la lot 2 costul fiecărei
lansări mărunte e mai mult așteptare / pornire decât memorie (DEDUS) — de aceea P5c (−12 lansări) ar putea da și la lot 2 mai mult
decât banda de memorie.

## 11. Rezultatul P5c (CALCUL + SIMULARE, sesiunea 293) — se citește cu §9

Unealta NAK a doua (Mesa 26.3.1) reconstruită în cloud (~35 min); T310 = `crates/nvidia-ga104/tests/sim_c168_p5c_gelu_fwd.rs`;
comparatorul nou `tools/nak/compara_gelu_fwd.py`.

| Predicția | Rezultat | Stare |
|---|---|---|
| L1: fără `-DGELU_FWD` aceiași octeți | `mm_tc_v7` 953ed6b8…, `mm_tc_v8` 1b910a53…, `mm_tc_v8n` 9a0702bc…, `mm_tc_v8r` 6e28d531…, `mm_tc_v7_gelu` ed808204…, `mm_tc_v8_gelu` d0d061ce…, `mm_tc_v8n_gelu` 81e6e4ab…, `mm_tc_gelu` 03de80cf… — recompilate, **identice** (git nu vede nicio schimbare în `out2/`) | **CONFIRMAT** (CALCUL) |
| L2: registrele | `mm_tc_v7_geluf` **128**, `mm_tc_v8_geluf` **128**, `mm_tc_v8n_geluf` **144** — exact ca programele simple (4 / 2 / 3 blocuri pe SM); 0 „spills” în memorie (v7: aceleași 14 copii în registre ca `mm_tc_v7`) | **CONFIRMAT** (CALCUL) — riscul pe v8 nu s-a produs |
| L3: coada `act` = `gelu_tr` mod 0; coada `u` = a programului simplu | `compara_gelu_fwd.py`: v7 32 / 32, v8 64 / 64, v8n 64 / 64 cozi = arborele lui `gelu_tr` mod 0 (58 de operații) din `out/gelu_tr.sm86.txt` — **programul de pe placă, făcut de Mesa 25.0.7**; fiecare coadă pornește din registrul scris în C (adresa din 0x710) și scrie la adresa din 0x730; restul operațiilor (virgulă mobilă + `hmma`), fără cozi, = programul simplu, în aceeași ordine (503 / 586 / 413). 3 greșeli plantate (o constantă, o adunare, adresa `act`) prinse | **CONFIRMAT** (CALCUL) |
| L4: model mic, 5 pași rapizi, aceiași biți | P5a + P5b ± P5c: pierderile, gradientul pasului 1, greutățile **bit cu bit**, cu vitezele os-500 (fc pe v7_geluf) și cu vitezele forțate pe v8 + v8r încărcat (fc pe v8_geluf, restul pe v8r); 88 → 86 lansări; plus S4: produsul 256×512×128 (TB + BIAS) pe fiecare din cele 3 programe = C și act ale perechii fc + `gelu_tr`, bit cu bit | **CONFIRMAT** (SIMULARE) |
| L5: #513 exact −12 lansări | lot 2 și lot 6: 448 → 436 (lista P5b, fără copierile lotului); 12 `gelu_tr` înainte dispar; „fc” pe varianta `geluf` a programului de azi: **lot 6 12 × v8n_geluf, lot 2 12 × v8_geluf** (unde azi e v8r: v8r nu are variantă `geluf`, deci v8_geluf — aceiași biți); celelalte alegeri și toate celelalte cbuf-uri identice, în ordine; QMD-urile cu adresa, registrele, blocul, grila variantei; cbuf-ul = al lui „fc” + `act` la 0x730 | **CONFIRMAT** (SIMULARE) |
| refuzuri | `gelu_fwd = None` = `attn11` octet cu octet; o listă cu fc+gelu refuzată de `attn11`, de `pack_mixed_gelu` și de `attn12` fără programele `geluf`; variantă neîncărcată, adresă nealiniată, fără v7 / v8, produs nealiniat, `act` suprapus peste A / B / C / bias — refuzate înainte de trimis; prin RAM (`via_ram_attn12`) aceleași alegeri și aceeași listă | **CONFIRMAT** (SIMULARE) |
| L6: câștig 0,6–1,1 ms la lot 6 | — | NOT TESTED (placa) |
| L7: oracolul lot 2 aceleași numere | — | NOT TESTED (placa) |

Limitele, cinstit: (1) comparația L3 rescrie cele două listări cu identități **exacte** (|x| și −x ca modificatori, −0 + x = x pe o
valoare deja FTZ, semnul scos din înmulțire, `mov C` = C, `mufu.*.f32` = `mufu.*`); orice altă diferență ar rămâne. Ea arată că pe
aceeași intrare se fac aceleași operații, nu că `MUFU.RCP` / `MUFU.EX2` dau aceiași biți pe placă (sunt aceleași instrucțiuni, deci
ar trebui). (2) Pe lot 2, unde azi „fc” merge pe v8r, P5c îl mută pe v8_geluf (v8 + coadă): cu biții identici, dar o înmulțire cu
~9 % mai lentă (os-505: v8r 0,981 față de v8 1,063 ms la qkv) — câștigul la lot 2 poate fi mai mic; la lot 6 (v8n) nu se schimbă
nimic din asta. (3) Programele `v8_geluf` / `v8n_geluf` au ~70 KiB (peste 64 KiB); OS-ul le alocă după mărime (`program_map_bytes`),
deci nu e o problemă acolo, dar orice loc fix de 64 KiB pe program le-ar suprapune (testul nostru avea așa ceva — reparat).

Nota pentru OS: `integration/cyber-universe/C-168_P5c_gelu-inainte-in-coada-fc.md`. Următorul pas: P5d (reziduurile înainte în coada
lui „out” / „proj”) cu plan + predicții întâi; pe placă: rândurile P5b și P5c.

## 12. P5d — reziduurile înainte în coada lui „out” și „proj”: planul și predicțiile (sesiunea 294, 28 nov, scrise ÎNAINTE de cod)

**Ce e azi (planul P5a + P5b + P5c):** pe fiecare strat, înainte, „out” (`att · Woutᵀ + bias`, m = n·T, n = C, k = C) scrie `tmp`,
apoi `vadd` scrie `xmid = xin + tmp`; la fel „proj” (`act · Wprojᵀ + bias`, k = F) scrie `tmp`, apoi `vadd` scrie `xout = xmid + tmp`.
Fiecare `vadd` citește 2 × n×C și scrie n×C (lot 6: 28,3 MB, ~52 µs măsurat: os-503 `vadd` 2 483 µs / 48). `tmp` nu mai e citit de
nimeni după `vadd` (stratul următor îl rescrie înainte să-l citească), deci lipirea poate scrie direct `xmid` / `xout`.

**Ce fac:**
1. Aceleași surse cu un fanion nou `-DRES`: `mm_tc_v8.comp` → `mm_tc_v8_res` (și `-DV8N` → `mm_tc_v8n_res`), `mm_tc_v7.comp` →
   `mm_tc_v7_res`, **și `mm_tc_v8r.comp` → `mm_tc_v8r_res`** (spre deosebire de P5c: „out” / „proj” au n = C = 768, iar la lot 2
   valurile dau v8, deci azi merg pe v8r; fără `v8r_res` lipirea le-ar muta pe v8, ~9 % mai lent). Scrierea devine
   `c = (v + bias) + res[rând, col]`: aceeași valoare `v + bias` ca azi, apoi **o adunare f32** cu reziduul — exact ce face `vadd`
   (`a + b`, `fadd.ftz`; adunarea e comutativă bit cu bit). Adresa lui `res` la 0x730 (ca `act` la P5c; fanioanele `-DRES` și
   `-DGELU_FWD` nu se folosesc împreună). Fără fanion sursele dau **exact aceiași octeți** ca azi.
2. `nvidia_ga104::mm_tc_fast_res` (programele, SHA-256, registrele, `Args` = `mm_tc::Args` + `res`; refuzuri: `res` 0, nealiniat la 4,
   suprapus peste A / B / C / bias), doar adăugări.
3. În simulator: `mm_tc_v{7,8,8n,8r}_res_on_cpu` = aritmetica programului fără fanion + `add_f32_ftz` cu reziduul.
4. `train_step`: `MixedLaunch::TcRes` (variantă nouă), `Plan::with_fused_res_fwd()` (în `step_mixed`: „out” + `rezidual1` →
   `inainte:out+rez` / `recalcul:out+rez`, „proj” + `rezidual2` → `inainte:proj+rez`; C = ieșirea lui `vadd`, `res` = cealaltă
   intrare a lui `vadd`), `pack_mixed_fast_attn13` = `attn12` + `ResFast { v7, v8, v8n, v8r }`: un `TcRes` ia varianta `_res` a
   programului pe care `attn12` l-ar da aceluiași produs (inclusiv v8 → v8r când `v8r_va ≠ 0`); variantă neîncărcată / nealiniat →
   refuz înainte de trimis. `None` = `attn12` octet cu octet. `TcPick::V7Res` / `V8Res` / `V8nRes` / `V8rRes`.
5. OS-ul: plan rapid + `.with_fused_res_fwd()`, încarcă cele patru programe.

**Predicțiile (M1–M7):**
- **M1 (CALCUL, NAK):** fără `-DRES` toate programele vechi (v7 / v8 / v8n / v8r și variantele GELU' / `geluf`) ies cu aceleași SHA-256.
- **M2 (CALCUL, NAK):** registrele = ale programelor simple (v7_res 128, v8_res 128, v8n_res 144, v8r_res 120), 0 „spills”: coada ține
  o singură valoare în plus (reziduul citit), mai puțin decât GELU înainte (care a încăput).
- **M3 (CALCUL):** în fiecare listare, coada = a programului simplu + **exact o `fadd` în plus pe element**, al cărei operand e o
  citire globală de la adresa din 0x730 + același deplasament ca scrierea lui C; restul operațiilor în virgulă mobilă (cu `hmma`)
  identice, în aceeași ordine.
- **M4 (SIMULARE):** model mic aliniat, 5 pași rapizi (P5a + P5b + P5c ± P5d): pierderile, gradientul pasului 1 și greutățile
  **bit cu bit** ca fără `with_fused_res_fwd`; plus același produs forțat pe fiecare din cele 4 programe = „mm + `vadd`”, bit cu bit.
- **M5 (SIMULARE):** #513 lot 2 și lot 6: exact **−24 lansări** (436 → 412 față de lista P5c), cele 24 `vadd` înainte dispar, „out” și
  „proj” pe varianta `_res` a programului de azi; celelalte alegeri și cbuf-uri identice, în ordine.
- **M6 (placa, DEDUS):** lot 6: −24 × ~52 µs (`vadd`) + 24 × ~18 µs (citirea lui `res` în coadă, 9,4 MB la ~533 GB/s) → câștig net
  **0,5–1,1 ms**; lot 2: **0,3–0,9 ms** (la os-508 lansările mărunte la lot 2 au costat cât la lot 6). Sub 0,3 ms la lot 6 = RATAT.
- **M7 (placa):** oracolul lot 2: aceleași numere ca rândul fără P5d.

## 13. Rezultatul P5d (CALCUL + SIMULARE, sesiunea 294) — se citește cu §12

Unealta NAK a doua (Mesa 26.3.1) reconstruită în cloud; T311 = `crates/nvidia-ga104/tests/sim_c168_p5d_res_fwd.rs`; comparatorul nou
`tools/nak/compara_res.py`.

| Predicția | Rezultat | Stare |
|---|---|---|
| M1: fără `-DRES` aceiași octeți | `mm_tc_v7` / `v8` / `v8n` / `v8r`, `mm_tc_v7_gelu` / `v8_gelu` / `v8n_gelu`, `mm_tc_v7_geluf` / `v8_geluf` / `v8n_geluf`, `mm_tc_gelu` — recompilate, **identice** (git nu vede nicio schimbare în `out2/`) | **CONFIRMAT** (CALCUL) |
| M2: registrele | `mm_tc_v7_res` **128**, `mm_tc_v8_res` **128**, `mm_tc_v8n_res` **144**, `mm_tc_v8r_res` **120** — exact ca programele simple (4 / 2 / 3 / 2 blocuri pe SM); 0 „spills” în memorie (v7: aceleași 14 copii în registre ca `mm_tc_v7`); mărimi 33 760 / 50 576 / 49 488 / 49 344 B, toate sub 64 KiB | **CONFIRMAT** (CALCUL) |
| M3: coada = a programului simplu + o `fadd` cu `res` | `compara_res.py`: v7 32 / 32, v8 64 / 64, v8n 64 / 64, v8r 64 / 64 cozi: valoarea scrisă vine dintr-o `fadd.ftz` chiar înainte de scriere, cu o citire globală de la adresa din 0x730 cu **același indice** ca adresa lui C; fără aceste 4 instrucțiuni, restul (232 / 348 / 370 / 348 operații în virgulă mobilă, `hmma`, citiri și scrieri) = programul simplu, în ordine. 3 greșeli plantate (`fadd` → `fmul`, altă adresă pentru `res`, alt indice) prinse | **CONFIRMAT** (CALCUL) |
| M4: model mic, 5 pași rapizi, aceiași biți | P5a + P5b + P5c ± P5d: pierderile, gradientul pasului 1, greutățile **bit cu bit**, pe 4 vitezele: os-500 (out / proj pe v7_res), v8 (v8_res), v8 + v8r încărcat (v8r_res), v8n (v8n_res); 86 → 82 lansări; plus S4: produsul 256×512×128 (TB + BIAS) pe fiecare din cele 4 programe = produs + `vadd`, bit cu bit | **CONFIRMAT** (SIMULARE) |
| M5: #513 exact −24 lansări | lot 2 și lot 6: 436 → 412 (lista P5c, fără copierile lotului); 24 `vadd` înainte dispar; „out” / „proj” pe varianta `_res` a programului de azi: **lot 6 24 × v8n_res, lot 2 24 × v7_res**; celelalte alegeri și cbuf-uri identice, în ordine; QMD-urile cu adresa, registrele, blocul, grila variantei; cbuf-ul = al produsului cu C = ieșirea lui `vadd` și `res` la 0x730 | **CONFIRMAT** (SIMULARE) |
| refuzuri | `res_fwd = None` = `attn12` octet cu octet; o listă cu „+rez” refuzată de `attn12`, de `pack_mixed_gelu` și de `attn13` fără programele `_res`; variantă neîncărcată, adresă nealiniată, fără v7 / v8, produs nealiniat, `res` suprapus peste A / B / C / bias — refuzate înainte de trimis; prin RAM (`via_ram_attn13`) aceleași alegeri și aceeași listă | **CONFIRMAT** (SIMULARE) |
| M6: câștig 0,5–1,1 ms la lot 6, 0,3–0,9 la lot 2 | — | NOT TESTED (placa) |
| M7: oracolul lot 2 aceleași numere | — | NOT TESTED (placa) |

Aflat, cinstit: (1) planul presupunea că la lot 2 „out” / „proj” merg pe v8r; în SIMULARE, cu vitezele os-500 și garda os-503, ele merg
pe **v7** (n = C = 768 dă puține blocuri de 128×128). `mm_tc_v8r_res` rămâne pentru cazul în care valurile aleg v8 (dovedit în S1 / S4),
dar #513 nu-l folosește azi. (2) Comparația M3 arată că pe aceeași valoare se face aceeași adunare cu reziduul, nu că `FADD.FTZ` de pe
placă dă aceiași biți ca `vadd` — e aceeași instrucțiune, deci ar trebui; o arată oracolul (M7).

Nota pentru OS: `integration/cyber-universe/C-168_P5d_rezidual-inainte-in-coada.md`. Următorul pas: P5f (cele 98 `colsum` finale într-o
lansare, aceiași biți) cu plan + predicții întâi; P5e și P5g dau alți biți și vin după; pe placă: rândurile P5b, P5c, P5d.

## 14. P5b, P3g și P5c comparate cu placa: os-512 și os-514 (sesiunea 295, 28 nov 18:40)

Surse (hub, `rezultate/os-automat/2026-11-28/`): os-508 = `ciclu-auto-20261128-1259.txt` (lot 2) / `-1304` (lot 6), martorul;
os-512 (`--pas-gelu-lipit --pas-v8r`) = `-1554` (lot 2) / `-1559` (lot 6) / `-1613` (lot 6 pe datele reale); os-514
(`--pas-gelu-inainte-lipit` peste os-512) = `-1739` (lot 2) / `-1744` (lot 6). Toate „placa, median pe 10 pași”, nucleul cu
`cyber-gpu 408c69c`, aceeași placă, `ceas 60`.

| Predicția | Măsurat pe placă | Stare |
|---|---|---|
| J4: 467 → 455 lansări la lot 6 | 455 (os-512 lot 6), 448 la lot 2 | **CONFIRMAT** (placa) |
| J6: câștig net 1,0–1,8 ms la lot 6 | placa 119,03 → **118,82 ms** = −0,2 ms (pasul întreg 118,77 → 118,76) | **RATAT** (sub 0,5) |
| J7: oracolul lot 2 aceleași numere | ACELEAȘI NUMERE (pierderea 6,3e-6, norma 6,4e-5); lot 6 pe datele reale = S141 (4,2e-5 / 5,6e-4) | **CONFIRMAT** (placa) |
| K5: v8r ~217–225 µs pe lansare la lot 2 (5,2–5,4 ms pe 24) | „înainte:fc” pe v8r: 12 lansări **2 594 µs** = 216 µs (os-508 pe v8: 2 836 µs = 236 µs; ×1,09) | **CONFIRMAT** (placa; la marginea bună) |
| K6: lot 6 neschimbat (±0,5 ms) | niciun v8 / v8r în pasul lot 6; 118,8 ms | **CONFIRMAT** (placa) |
| L5: −12 lansări | 455 → 443 (lot 6), 448 → 436 (lot 2) | **CONFIRMAT** (placa) |
| L6: câștig 0,6–1,1 ms la lot 6 (sub 0,4 = RATAT) | placa 118,82 → **117,89 ms** = **−0,93 ms**; pasul întreg 117,77 ms = **26 085 simb/s** (cel mai bun pe loturi sintetice) | **CONFIRMAT** (placa) |
| L7: oracolul lot 2 aceleași numere | ACELEAȘI NUMERE (6,3e-6 / 6,4e-5) | **CONFIRMAT** (placa) |

**De ce J6 a ratat (CALCUL, din profil + listarea NAK):** `gelu_tr` înapoi a dispărut exact cât am socotit (lot 6: 24 → 12 lansări,
4 258 → 1 700 µs = **−2 558 µs**; prezis ~−2,5 ms), dar „proj-dX” cu coada GELU' a crescut cu **+2 286 µs** (7 438 → 9 724 µs,
12 lansări = +190 µs pe lansare), nu cu +70–110 µs. La lot 2 la fel: `gelu_tr` −876 µs, „proj-dX” pe v8 → v8_gelu +608 µs (+51 µs pe
lansare). În listarea `tools/nak/out2/mm_tc_v8n_gelu.sm86.txt` coada e scrisă element cu element: fiecare din cele 64 de valori ale unui
fir stă în propriul bloc păzit (`if (gr < m && gc < n)`), cu o citire `ld.global.b32` a lui `u`, lanțul GELU' (~20 de operații
dependente) și scrierea — iar citirea următoare pornește abia după. Așteptarea memoriei (~600 de cicluri) se plătește deci de 64 de ori
pe rând în fiecare val de blocuri: ~64 × 0,35 µs ≈ 20–25 µs pe val; lot 6 are 8 valuri (1 152 blocuri, 144 pe val) → ~190 µs pe
lansare = exact ce s-a măsurat. Programul simplu citește la fel `bias`, dar acela e un vector mic (din memoria rapidă); `u` e citit o
singură dată, din memoria plăcii. Socoteala din §7 a presupus că citirea lui `u` merge cu viteza memoriei (533 GB/s); merge de fapt cu
~200 GB/s.

**De ce P5c a ținut:** coada `geluf` doar **scrie** în plus (`act`), nu citește; scrierile nu așteaptă. „înainte:fc” 7 702 → 8 753 µs
(+88 µs pe lansare, prezis +71), `gelu_tr` înainte −1 700 µs → −0,93 ms. La lot 2 câștigul e mic (−0,13 ms): „fc” mergea pe v8r
(2 594 µs), iar v8r nu are variantă `geluf`, deci trece pe `mm_tc_v8_geluf` (3 087 µs).

**Ce înseamnă pentru P5d (M6, scris înainte de rândul pe placă — îl corectez ACUM, înainte să vină):** coada `_res` are aceeași
formă (`tools/nak/out2/mm_tc_v8n_res.sm86.txt`: 64 de citiri `res` în plus, câte una pe bloc păzit). Lot 6, „out” / „proj” pe v8n_res:
288 blocuri = 2 valuri × ~20 µs ≈ +40 µs pe lansare × 24 ≈ +0,96 ms, față de `vadd` −1,28 ms (os-514: 24 × 53 µs) → **M6 revizuit:
−0,1…−0,6 ms la lot 6**; lot 2 (v7_res, 32 de citiri pe fir, 1 val) ≈ +10 µs × 24 față de `vadd` −0,45 ms → **0…−0,4 ms**. Predicția
veche (−0,5…−1,1 / −0,3…−0,9) rămâne scrisă; dacă placa o atinge, socoteala de aici e greșită.

**Unde suntem:** lot 6 sintetic 117,8 ms = 26 085 simb/s (os-514); pe datele reale 118,0 ms = 26 031 (os-512). Leacul pentru coadă
(citirile pornite înainte, aceiași biți) = P5e de mai jos.

## 15. P5e — coada cu citirile pornite înainte (aceiași biți): planul și predicțiile (sesiunea 295, scrise ÎNAINTE de cod)

**Ce fac:** aceleași surse (`mm_tc_v7.comp`, `mm_tc_v8.comp`, `mm_tc_v8r.comp`) cu un fanion nou `-DPF`, doar împreună cu `-DGELU` sau
`-DRES`. În coadă, valorile citite (`u` pentru GELU', `res` pentru reziduu) ale bucății următoare (i, j) se citesc **înainte** să se
lucreze bucata curentă, cu indicele strâns în margini (`min(gr, m−1)`, `min(gc, n−1)`, ca citirea să fie mereu la o adresă validă și
fără ramificație); scrierea rămâne păzită ca azi. Aritmetica pe fiecare element e literă cu literă aceeași, pe aceeași valoare → aceiași
biți. Programe noi: `mm_tc_v7_gelu_pf`, `mm_tc_v8_gelu_pf`, `mm_tc_v8n_gelu_pf`, `mm_tc_v7_res_pf`, `mm_tc_v8_res_pf`,
`mm_tc_v8n_res_pf`, `mm_tc_v8r_res_pf` (`geluf` nu citește nimic, nu are nevoie). În bibliotecă doar adăugări: constantele (octeți,
SHA-256, registre) în `mm_tc_fast_gelu` / `mm_tc_fast_res`; dacă registrele sunt aceleași ca la programele de azi, OS-ul pune adresele
`_pf` în `GeluFast` / `ResFast` și nimic altceva nu se schimbă (aceleași `Args`, bloc, grilă, memorie partajată).

**Predicțiile (N1–N7):**
- **N1 (CALCUL):** fără `-DPF` cele 15 programe de azi (v7 / v8 / v8n / v8r, 3 `_gelu`, 3 `_geluf`, 4 `_res`, `mm_tc_gelu`) ies cu
  aceleași SHA-256.
- **N2 (CALCUL):** registrele `_pf` = ale variantelor de azi (128 / 128 / 144; 128 / 128 / 144 / 120). Riscul: +8 registre vii în
  coadă; la v8n ≤ 168 tot 3 blocuri pe SM, la v8 / v8r / v7 peste 128 = blocuri pierdute → atunci varianta nu e folosită.
- **N3 (CALCUL):** în listare, citirile lui `u` / `res` nu mai stau fiecare în blocul ei păzit: cele 4 citiri ale bucății următoare
  pornesc împreună, înaintea lanțului bucății curente; operațiile în virgulă mobilă ale fiecărui element = ale variantei de azi, în
  ordine (unealta `compara_pf.py`, cu greșeli plantate).
- **N4 (SIMULARE):** modelul mic, 5 pași rapizi cu adresele `_pf` în `GeluFast` / `ResFast`: aceiași biți ca azi; lista #513 identică
  în afara adreselor programelor.
- **N5 (placa, DEDUS):** lot 6 „proj-dX+gelu” 9,75 → **7,6–8,2 ms** (−1,5…−2,1 ms; podeaua = citirea lui `u` la viteza memoriei,
  ~70 µs pe lansare); lot 2 `mm_tc_v8_gelu` 3,39 → 2,85–3,05 ms. Sub −0,8 ms la lot 6 = RATAT (atunci așteptarea nu era cauza).
- **N6 (placa, DEDUS):** cu P5d legat, „out” / „proj” pe `_res_pf`: încă −0,5…−0,9 ms la lot 6 față de `_res`.
- **N7 (placa):** oracolul lot 2 aceleași numere.
