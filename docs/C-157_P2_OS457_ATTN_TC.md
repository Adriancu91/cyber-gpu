# C-157 p.2 — comparația os-457: atenția înapoi (dQ, dK, dV) pe circuitele f16

Sesiunea 247, 26 nov 2026, în cloud, **fără placă**. Cifrele plăcii sunt MĂSURATE de OS (os-457, 26 nov 09:43,
`Cyber-hub/rezultate/2026-11-26-workstation-os457-atentia-tc/REZULTAT.md`, jurnalul brut
`rezultate/os-automat/2026-11-26/ciclu-auto-20261126-0943.txt`, rândurile 103–113 și 193–195). Restul e DEDUS sau SIMULARE.

## 1. Ce a ieșit pe placă (MĂSURAT)

| Set | tc (ns) | tc16 (ns) | `_3` F32 (ns) | în margine (`judge`) | tc16 = tc | abaterea față de F32 |
|---|---|---|---|---|---|---|
| mic-b2-t16-d16 | 29 632 | 24 928 | 109 856 | 86 ‰ / 86 ‰ | DA | 2,77·10⁻⁴ |
| t130-d64 | 121 888 | 93 600 | 277 920 | 46 ‰ / 46 ‰ | DA | 2,87·10⁻⁴ |
| t64-d40 | 60 096 | 42 432 | 91 776 | 45 ‰ / 45 ‰ | DA | 2,76·10⁻⁴ |
| t200-h3 | 217 024 | 197 312 | 378 528 | 40 ‰ / 40 ‰ | DA | 2,76·10⁻⁴ |
| ascutit | 79 680 | 57 696 | 165 344 | NU (elementul 394: 3,0900992·10⁻⁷, marginea 3,65·10⁻¹²) | DA | 3,02·10⁻⁴ |
| atentie-513-b4 (mediana a 5) | **1,16 ms** | **0,87 ms** | **1,59 ms** | (doar placa: gărzi neatinse) | DA (4 608 cuvinte) | 2,97·10⁻⁴ |

Corectitudinea a ieșit **exact ca în SIMULARE** (4 / 5 cu marginea veche, `ascutit` iese din cauza subnormalelor f16 — explicat în
sesiunea 246, `docs/C-157_ATTN_TC_SUBNORMALE.md`); viteza a ieșit **sub predicție**.

## 2. Predicția noastră pentru timp — și o greșeală de formă în ea

Rândul trimis OS-ului (nota P2zm) spunea: „la atentie-513-b4 tc ~0,2–0,37 ms pe lansare (2,4–4,5 ms pe 12), față de ~0,77 la `_3`”.
**Cele 0,77 ms sunt ale lotului 2** (profilul os-456: 9,2 ms / 12), dar setul `atentie-513-b4` are **lotul 4** (384 de blocuri, nu 192):
în aceeași rulare `attn_back_tile_3` a luat **1,60 ms** pe lansare. Deci predicția corectă pentru setul măsurat era **~0,4–0,75 ms**
(același raport 2,0–3,8× față de `_3`). Greșeala e a noastră (am pus cifra lotului 2 lângă numele setului lotului 4).

| | prezis (corectat la lotul 4) | MĂSURAT | |
|---|---|---|---|
| tc16 pe lansare | 0,40–0,75 ms | 0,87 ms | **×1,16–2,2 peste** |
| tc pe lansare | 0,40–0,75 ms | 1,16 ms | ×1,5–2,9 peste |
| câștig față de `_3` | ×2,0–3,8 | tc16 ×1,83, tc ×1,38 | sub bandă |
| tc16 față de tc | „cel mult cât tc” | ×1,33 mai repede | CONFIRMATĂ |

**La pasul mixt #513 (lotul 2), DEDUS:** 384 de blocuri la lotul 4 față de 192 la lotul 2; cu 3 blocuri tc16 pe SM (treapta 64 KiB,
cu rezerva de 1 KiB) intră 144 deodată → 3 valuri la lotul 4, 2 la lotul 2. Lotul 2 ar lua deci între jumătate și două treimi din
0,87 ms → **~0,44–0,58 ms pe lansare, ~5,2–7,0 ms pe pas (× 12) față de 9,2 ms azi → câștig ~2,2–4,0 ms pe pas** (OS-ul a scris ~5 ms,
la capătul bun al benzii).

## 3. De ce e mai lent decât am prezis (DEDUS — întrebarea 1 a OS-ului)

1. **Lucrul circuitelor e mic, așteptarea citirilor e mare.** Un pas de placă (64 × 64 × 64) înseamnă, pe fiecare subgrup, 16 `hmma`
   (~0,3 µs). Programul citește plăcile f32 din memorie în **8 runde a câte 4 citiri scalare** pe fir (`[[dont_unroll]]`, bucla `hh`),
   iar NAK 26.3.1 ține cel mult ~5 citiri în zbor (aflat în sesiunea 229 la `mm_tc_pf`) → **~8 așteptări de memorie pe pas de placă**.
   Pe seturile mici fiecare bloc stă singur pe un SM (2–12 blocuri), deci timpul = blocul cel mai lung: t64-d40 (3 pași) 42 µs,
   ascutit (5 pași) 58 µs, t130-d64 (7 pași) 94 µs → **~8–14 µs pe pas de placă** (cu ~10–20 µs fix pe lansare). Adică **> 90 % din
   timpul unui bloc e așteptare de citiri**, nu circuite. Exact boala lui `mm_tc` înainte de `mm_tc_pf` / `mm_tc_v4`.
2. **Câte blocuri pe SM (treapta 64 KiB, `occupancy`):** `_3` 1, tc 1–2, tc16 3–4. Dacă timpul ar fi doar valuri × blocul, tc16 ar fi
   ×2–2,7 mai rapid decât tc; e doar ×1,33. Deci cu mai multe blocuri pe SM fiecare bloc merge mai încet (citirile se calcă pe picioare:
   L1 / L2 / cozile de memorie). Cauza exactă: NECUNOSCUT (fără contoare pe placă); nu e lățimea memoriei (~170 MB cerute pe lansare,
   ~195 GB/s, sub copia VRAM de 282 GB/s măsurată și mult sub 563 GB/s văzuți la `kc`).
3. **Pe FLOP:** ~2,7 GFLOP pe lansare la lotul 4 (plăci întregi) → tc16 **~3,1 TFLOP/s**, `_3` ~1,7, `mm_tc_v4` 8,28 (os-443). Predicția
   (studiul 242, A3) presupunea eficiența `mm_tc_v4` „și de până la 2 ori mai încet”; programul a ieșit la ~38 % din ea, pentru că nu
   are nimic din ce a făcut `mm_tc_v4` rapid (citiri de 128 de biți, citirea plăcii următoare peste produse).

**Remediul propus (următorul program, nescris încă):** `attn_back_tile_tc16_v4` — aceeași aritmetică, aceleași valori f16 în aceleași
locuri ale memoriei partajate, deci **aceiași biți** ca tc16 (judecabil cuvânt cu cuvânt pe placă), dar: (a) citiri de 128 de biți
(4 + 4 `vec4` pe fir pe pas în loc de 32 de citiri scalare → ~2 așteptări în loc de ~8), doar când T și adresele sunt aliniate la 4
(altfel OS-ul rămâne pe tc16); (b) placa următoare citită în registre înainte de produsele plăcii curente. DEDUS: pas de placă
~8–14 → ~3–6 µs, adică **~0,35–0,55 ms pe lansare la lotul 4** (×1,6–2,5 față de tc16), **~2,5–4 ms pe pas** la lotul 2. Se vede doar pe placă.

## 4. Seturile mici: placa = modelul? (predicții R1–R3, scrise ÎNAINTE de testul T257)

Testul T257 (SIMULARE, pe procesor) recalculează cu modelul nostru (`attn_tc::model`, sumatorul os-335) cifrele pe care OS-ul le-a
scris de pe placă, cu aceeași formulă (`rel_ppm` din `gpu_i14`: norma relativă față de dQ/dK/dV F32, în milionimi).
Atenție: P-ul plăcii nu e bit cu bit P-ul modelului (atenția înainte: 69 097 / 178 162 de elemente exacte, restul în marginea MUFU),
deci dS și intrările diferă puțin; de aceea R1–R2 cer potrivire la cifrele tipărite, nu biți.

- **R1 — abaterea față de F32.** Pe cele 5 seturi mici modelul dă **aceleași cifre tipărite** (x,yy·10⁻⁴) ca placa pe cel puțin 4 din 5,
  iar pe toate cel mult ±0,02·10⁻⁴: 2,77 / 2,87 / 2,76 / 2,76 / 3,02.
- **R2 — cel mai rău ‰ (`attn_tc::judge`).** Aceleași cifre ca placa pe cel puțin 3 din cele 4 seturi în margine (86 / 46 / 45 / 40),
  restul ±3 ‰; `ascutit` iese din margine la **același element 394, cu aceeași valoare 3,0900992·10⁻⁷**.
- **R3 — marginea nouă (`judge_nominal`, sesiunea 246)** pe ieșirea modelului: 5 / 5 în margine. Fiindcă placa a dat aceleași cuvinte
  la tc și tc16 și aceleași cifre ca modelul, verdictul plăcii cu marginea nouă ar fi tot 5 / 5 (DEDUS; îl scrie OS-ul, nota P2zo).

### Rezultatul R1–R3 (T257, `crates/nvidia-ga104/tests/sim_c157_os457_attn_tc.rs`, SIMULARE)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| R1 | aceleași cifre ca placa pe ≥ 4 / 5, toate ±0,02·10⁻⁴ | **5 / 5 identice**: 2,77 / 2,87 / 2,76 / 2,76 / 3,02·10⁻⁴ | CONFIRMATĂ |
| R2 | același ‰ pe ≥ 3 / 4; `ascutit` la elementul 394 cu 3,0900992·10⁻⁷ | **4 / 4 identice** (86 / 46 / 45 / 40 ‰); `ascutit`: elementul 394, 3,0900992·10⁻⁷, referința 3,090158315899316·10⁻⁷ și marginea 3,646918216634182·10⁻¹² — **toate cifrele ca pe placă** | CONFIRMATĂ |
| R3 | marginea nouă 5 / 5 | **5 / 5** (`ascutit` 46 ‰, celelalte identic cu vechea) | CONFIRMATĂ |

**Ce înseamnă:** pe toate cifrele pe care OS-ul le-a tipărit, placa și modelul nostru de pe procesor se potrivesc complet; diferențele
de P din atenția înainte nu ajung în aceste cifre. Cu marginea nouă (P2zo) proba ar ieși **TRECE 5 / 5** (DEDUS: verdictul îl
scrie OS-ul pe placă). Partea de corectitudine a atenției înapoi pe f16 e închisă; rămâne viteza (§3).
