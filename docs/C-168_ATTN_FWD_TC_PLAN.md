# C-168 punctul 4 — atenția înainte pe circuitele speciale (`attn_fwd_tc`): planul și predicțiile (scrise ÎNAINTE de cod)

Sesiunea Cyber-GPU 281, 27–28 nov 2026, în cloud, FĂRĂ placă. Comanda: C-168 (Adrian, 27 nov 20:00), punctul 4: „atenția
înainte pe circuitele speciale, într-o singură trecere (fără matricea întreagă în memorie). Țintă: 16,1 → ≤ 6 ms pe pas.”
Punctul „înainte”: os-496, lot 6, pasul record 147,7 ms — `attn_fwd_tile3` **16 149 µs** pe 12 lansări (10,9 %).
Punctul 3 (v8) așteaptă proba pe placă `--mm-tc-v8` (nota P3a); acest plan nu depinde de el.
Socotelile: T298 (`crates/nvidia-ga104/tests/calc_c168_attn_fwd_tc_plan.rs`, CALCUL + DEDUS; nimic pe placă, nimic măsurat
pentru programul nou). Cifrele plăcii de mai jos sunt MĂSURATE de OS (os-496, `ciclu-auto-20261127-1954.txt`).

## 1. Ce spun cifrele de azi (T298a, T298c)

| Programul (lot 6, o lansare = un strat) | produse 64×64×64 | µs / lansare | TFLOP/s |
|---|---|---|---|
| `attn_fwd_tile3` (F32: S = Q·Kᵀ, softmax, O = P·V) | 2 | 1 346 | **2,02** |
| `attn_back_tile_tc16_v4` (f16: dQ, dK, dV) | 3 | 455 | **8,96** |
| `attn_back_tile0_tc_v4` (f16: dP, apoi dS) | 1 | 533 | 2,55 |

- Un produs pe toate capetele unui strat = 36 de perechi de plăci sub diagonală × 72 de capete = **1,36 GFLOP**.
- Programul înapoi pe circuite face **3 produse în o treime din timpul** celui înainte, care face 2. Deci frâna e F32-ul și
  trecerile prin memorie, nu forma atenției.
- Matricea S / P: azi atenția înainte o trece prin memoria plăcii de **5 ori** (S scris, S citit, P scris, P citit de două ori) =
  2 430 MiB pe pas ≈ **5,7–6,4 ms** din cei 16,1 doar memorie (DEDUS, 400–450 GB/s).

## 2. Alegerea: două treceri pe chei, P scris o singură dată (nu FlashAttention „cu o trecere” — încă)

Atenția înapoi de azi (`tile0_tc_v4` și `tc16_v4`) **citește P** din memorie. Sunt două drumuri:

| | A. două treceri (planul) | B. o trecere + P recalculat înapoi (FlashAttention-2, public) |
|---|---|---|
| înainte | trecerea 1: S, maximul și suma pe rând; trecerea 2: S din nou, P exact, O += P·V → **3 produse** | S, softmax „din mers”, O → **2 produse**; scrie doar L (log-suma) pe rând |
| înapoi | **neschimbat** (citește P ca azi) | `tile0` trebuie să recalculeze S din Q, K, L → **+1 produs** |
| produsele pasului | 3 + 4 = **7** | 2 + 5 = **7** (la fel, T298e) |
| P în memorie | scris o dată (486 MiB pe pas, ~1,2 ms), citit de 2 ori înapoi | deloc (−906 MiB din arenă, −~3 ms de citiri) |
| riscul | mic: înapoi nu se atinge, P are aceeași formă | mare: trei programe noi deodată; rescalarea lui O cere fie aranjarea acumulatorilor (nedefinită în GLSL), fie un drum prin memoria comună la fiecare placă |

**Planul: A acum, B după ce A e măsurat.** A păstrează interfața (P [B, H, T, T] F32, doar sub diagonală, aceeași arenă, aceleași
programe înapoi) — o lansare se înlocuiește cu alta în pas, doar adăugări în Rust. A respectă „fără matricea întreagă în memorie”
**pentru atenția înainte** (nu mai citește nici S, nici P; le ține în memoria comună, placă cu placă); P se scrie doar pentru că
înapoi îl cere. B rămâne pasul următor scris aici, ca să nu se piardă: tot câștigul ei e la memorie (P și dS), nu la FLOP.

De ce nu „o trecere” în A: P exact (împărțit la suma rândului) se poate scrie doar după ce suma întregului rând e cunoscută. Într-o
singură trecere ar trebui rescalați acumulatorii lui O cu `exp(m_vechi − m_nou)` pe fiecare rând, iar în `GL_KHR_cooperative_matrix`
**nu se știe ce element al acumulatorului e pe ce rând** (aranjarea e a implementării). Două treceri ocolesc problema cinstit:
P e final când se înmulțește cu V, deci O nu se rescalează niciodată.

## 3. Programul `attn_fwd_tc` (ce scriu în sesiunea următoare)

`tools/nak/attn_fwd_tc.comp`, compilat cu unealta a doua (Mesa NAK 26.3.1, `tools/nak/build2.sh`). Argumentele = cele ale lui
`attn_fwd_tile3` (`qkv`, `p`, `att`, T, C, H, d, scale; același cbuf, `crate::attn_fwd::Args`), ca să se schimbe doar programul.
Doar formele **aliniate** (d = 64, T multiplu de 64, adresele multiplu de 16 B — ca `attn_back_tile_tc16_v4::fits`); pe rest
rămâne `attn_fwd_tile3`.

- **Grila în perechi (T298d):** [⌈T/64⌉ / 2, H, B]; blocul x face rândurile-placă x și 7 − x una după alta → fiecare bloc are
  exact 9 plăci de chei. La lot 6: 288 de blocuri pe 96 de locuri = **3 valuri pline** (grila obișnuită: 32 de unități față de
  idealul 27 → ~16 % pierdut în coadă). Aritmetica fiecărui rând nu se schimbă.
- **256 de fire = 8 grupuri de 32**, fiecare grup are 16 rânduri × 32 de coloane din placa 64 × 64 (aranjarea din `tc16_v4`:
  `wr = (w >> 1) · 16`, `wc = (w & 1) · 32`), 2 acumulatori 16×16.
- **Q stă în registre:** cele 4 fragmente A (16 × 64 f16) ale grupului, încărcate o dată pe rând-placă → zona Q a memoriei comune
  se refolosește pentru P.
- **Memoria comună (T298b): 28 416 B** → 2 blocuri pe SM la treapta de 64 KiB (azi `tile3`: 37 888 B → 1 bloc pe SM):
  - R1 = 16 640 B: K (f16, rânduri de 144 B), apoi S (f32, rânduri de 65), apoi Vᵀ (f16, rânduri de 144 B);
  - R2 = 9 216 B: Q la încărcare, apoi P (f16);
  - 2 560 B: m, l pe rând + parțialele celor 4 fire ale fiecărui rând.
- **Trecerea 1 (pe fiecare placă de chei jt ≤ rândul-placă):** K: citiri de 128 de biți în registre (fără ramuri, ca v4–v8) → f16
  în R1 → barieră → S = Q·Kᵀ (4 pași de 16, K pe coloane: `ldsm`) → barieră → S f32 în R1 → barieră → fiecare rând are 4 fire ×
  16 valori: x = s·scale (cu masca diagonalei), maximul și suma „din mers” (`l = l·2^((m−m')·log2e) + Σ 2^((x−m')·log2e)`).
  La sfârșit cele 4 parțiale ale rândului se unesc într-o ordine fixă → m, l, `inv = 1/l`.
- **Trecerea 2 (aceleași plăci):** S din nou (același cod, aceiași biți ca în trecerea 1) → P = 2^((x − m)·log2e) · inv în F32:
  **scris în `p`** (citiri / scrieri de 128 de biți, doar sub diagonală; deasupra nu se scrie nimic, ca azi) și f16 în R2; Vᵀ (citit în
  registre în timpul produsului S, întors la scriere) în R1 → barieră → O += P·Vᵀ (4 pași de 16, `ldsm` pe ambele).
- **La sfârșit:** O (f32, nescalat — P e deja final) prin memoria comună → `att` (ca `tc16_v4`).
- Ce înseamnă „aceeași aritmetică” aici: Q, K, V, P rotunjite f16 (`round_f16`, cu subnormale), sumele f32 în pași de 16 **în ordine**
  (modelul `mm_tc`, trecut pe placă bit cu bit în os-423 și după), `exp` ca `MUFU.EX2` (aproximativ pe placă, ~2 ulp) → verdictul pe
  placă e **o margine față de oracol**, nu biți (ca la `tc16`). Modelul pe procesor nou: `gpu_compute::layer::attn_fwd_tc_gpu`.

## 4. Predicțiile (scrise înainte de cod; verdictul pe placă = marginea față de `attn_fwd_tile3` + timpul)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| H1 | compilarea (NAK 26.3.1) | **≤ 128 registre** (2 blocuri pe SM), 0 memorie locală, 28 416 B memorie comună | 55 % |
| H2 | listarea | toate citirile operanzilor din memoria comună sunt `ldsm` (K, P, Vᵀ, Q la încărcare); 32 `hmma` 16×8×16 pe grup și placă în trecerea 2 (16 în trecerea 1); `mufu.ex2` 16 pe fir și placă în fiecare trecere; **3 bariere** pe placă în trecerea 1, **≤ 5** în trecerea 2 | 60 % |
| H3 | SIMULARE (`gpu-sim`) | emularea programului = modelul pe procesor bit cu bit (cu `ex2_ftz`) pe margini (d < 64 refuzat, T = 64 / 128 / 512, B, H, rândurile perechi); față de `attn_fwd_tile3` (F32), pe date de mărimea #513: O eroare relativă (norma) **≤ 1·10⁻³**, P **≤ 2·10⁻³**; deasupra diagonalei lui P nimic scris | 85 % |
| H4 | pe placă, proba `--atentie-fwd-tc` (lângă `attn_fwd_tile3`, aceleași intrări) | O și P în marginea H3 pe toate seturile aliniate; nealiniatele sărite, cu motiv | 80 % |
| H5 | pe placă, timpul pe lansare, lot 6 (azi 1 346 µs) | **550–930 µs** (×1,45–2,45 mai rapid) | 45 % |
| H6 | în pas, lot 6 (cu v7; v8 nu schimbă atenția) | atenția înainte 16,1 → **6,6–11,1 ms**; placa 147,7 → **~138–143 ms** | 40 % |
| H7 | oracolul mixt, lot 2 | **ACELEAȘI NUMERE** (toleranțele 1·10⁻³ / 2·10⁻³; înapoi primește deja P rotunjit f16 la dV) | 75 % |

Socoteala H5–H6 (DEDUS, T298e): aceleași 3 produse ca `tc16_v4` (455 µs); presupun **×1,2–1,8** mai încet pe FLOP decât el (mai
multe bariere pe placă, S dus și adus prin memoria comună, `exp`-urile, 2 blocuri pe SM în loc de 3; perechile câștigă înapoi o parte),
plus scrierea lui P (40,5 MiB, 0–100 µs). **Ținta C-168 (≤ 6 ms) cere 8,2 TFLOP/s pe cele 3 produse — adică exact viteza lui
`tc16_v4`.** O scriu cinstit: cu A singur o ating doar dacă programul nou e la fel de bun ca cel mai bun program de atenție de azi
(~20 %). Restul până la ≤ 6 ms vine din B (fără P: −1,2 ms la scriere aici, −~2,3 ms la citirile înapoi).

**Prag scris:** dacă H1–H4 ies și pe placă `attn_fwd_tc` e peste 930 µs, frâna e în trecerea 2 (S prin memoria comună + 5 bariere),
nu în produse → pasul următor e S păstrat în registre cu aranjarea măsurată a acumulatorilor (o probă separată, întâi), nu B.

## 5. Tăiat acum (cu motiv)

- **B (FlashAttention-2 cu P recalculat înapoi):** trei programe deodată și o schimbare a arenei; după ce A e măsurat (§2).
- **Rescalarea lui O în registre:** aranjarea acumulatorilor nu e definită în GLSL; o folosim doar după o probă pe placă (§4 prag).
- **`cp.async`:** nu există în NAK 26.3.1 (`docs/C-168_V8_PLAN.md` §1); citirile plăcii următoare merg prin registre, ca la v7 / v8.
- **Formele nealiniate și d ≠ 64:** rămân pe `attn_fwd_tile3` (în #513 toate sunt aliniate).

## 6. Pașii (câte unul pe sesiune, fiecare testat)

1. GLSL-ul + compilarea (H1, H2) + `nvidia_ga104::attn_fwd_tc` (constantele, `fits`, grila în perechi, QMD-ul; doar adăugări).
2. Modelul pe procesor + emularea în `gpu-sim` (H3), test T299.
3. Nota pentru OS: proba `--atentie-fwd-tc` lângă `attn_fwd_tile3` (H4, H5) → PREDARE.
4. După probă: `TcPick` pentru atenția înainte în pas (`pack_mixed_fast_attn*`, doar adăugări) → PREDARE (H6, H7).

## 7. Surse

- Ideile publice: FlashAttention / FlashAttention-2 (Dao și alții, articolele arXiv 2205.14135 și 2307.08691): plăcile ținute în
  memoria comună, softmax-ul „din mers”, recalcularea lui S înapoi; perechile de rânduri pentru echilibrul cauzal (idee publică,
  folosită și de alte programe de atenție). Nicio linie de cod copiată; codul e al nostru.
- `docs/C-157_ATENTIE_TC.md` (precizia f16, A4), `tools/nak/attn_back_tile_tc16_v4.comp` (aranjarea grupurilor, a noastră),
  `docs/C-168_V8_PLAN.md` §1 (ce poate NAK 26.3.1: `ldsm` doar pe A pe rânduri / B pe coloane).

## 8. Pasul 1 făcut (sesiunea 282, 28 nov 2026, cloud, FĂRĂ placă): programul scris și compilat — H1 și H2 CONFIRMATE pe listare

`tools/nak/attn_fwd_tc.comp`, compilat cu unealta a doua (Mesa 26.3.1 reconstruit în cloud; `mm_tc_v7` a ieșit **identic**,
`953ed6b8…`). Binarul: `tools/nak/out2/attn_fwd_tc.sm86.sass`, 19 952 B, SHA-256 `ce2b2f22…7e929257`. În Rust (doar adăugări):
`nvidia_ga104::attn_fwd_tc` (constantele, `fits` / `fits_cbuf`, `cbuf` cu argumentele la 0x700, grila în perechi, `row_tiles`,
`qmd`) + 5 teste pe listare și pe reguli. **Nimic rulat pe placă; modelul pe procesor și SIMULAREA sunt pasul 2.**

| # | Prezis | Ieșit (listarea NAK) | Verdict |
|---|---|---|---|
| H1 | ≤ 128 registre, 0 memorie locală, 28 416 B | **120 registre**, 0 memorie locală, 0 copii pe memorie, **27 648 B** → 2 blocuri pe SM (registre 2 × 256 × 120 = 61 440) | CONFIRMATĂ |
| H2 | toți operanzii prin `ldsm`; 16 `hmma` în trecerea 1, 32 în trecerea 2; 16 `ex2` pe fir și placă; 3 / ≤ 5 bariere | `ldsm` 8 × x4 (Q, P) + 48 × x2 (K, Vᵀ), **0 `movm`**; singurele citiri simple din memoria comună = 12 × 128 de biți ale plăcii f32 S / O; `hmma` 16 + (16 + 16); bariere **3** pe placă în trecerea 1, **5** în trecerea 2 (13 în program); `ex2` **16 + 1** în trecerea 1 (+1 = reajustarea sumei), 16 în trecerea 2, 4 la unire; 1 `rcp`; 8 `shfl`; 0 `ffma`; toate citirile / scrierile globale de 128 de biți (20 / 8) | CONFIRMATĂ (cu „+1” la `ex2`, scris aici) |

Ce s-a schimbat față de §3 (și de ce), toate fără efect pe aritmetica rândului:

- **Rândurile plăcii f32 S / O: 68 de numere, nu 65** — lecția os-467 (cu rânduri de 65 NAK a scris acumulatorii cu scrieri de
  64 de biți la adrese 4 mod 8 și blocul nu s-a terminat pe placă). Memoria: trei zone de 64 × 144 B = 27 648 B (Z: Q la încărcare,
  apoi Vᵀ; X: K în trecerea 2; Y: K în trecerea 1, apoi P f16); S și O stau peste Z + X.
- **Unirea celor 4 bucăți ale unui rând prin `subgroupShuffle`** (cele 4 fire ale rândului sunt în același grup de 32), nu prin
  memoria comună: nicio barieră în plus, 2 560 B mai puțin. Ordinea e fixă: q = 0, 1, 2, 3, aceeași pentru toate cele 4 fire.
- **Trei încercări de compilare, scrise cinstit:** (1) V citit cu o placă înainte → **144 registre** (1 bloc pe SM); mutat: V
  pleacă după produsul S (zboară cât se scrie S și se calculează P), K-ul următor pleacă după scrierea lui Vᵀ (zboară cât lucrează
  O += P·V) → 120. (2) `live ? exp2(..) : 0` și `!diag || …` deveniseră **ramificații** (184 de blocuri în listare) → `mix(…, bool)`
  (o alegere) și o limită pe placă `lim = mix(63, r, diag)` → 40 de blocuri. (3) Bucla pașilor lui P (16 câte 16) desfăcută cu o
  ieșire uniformă → copiile de registre din jurul `hmma` de la O: 10 pe pas + ~65 în jurul buclei → 8 în pași + 16 la ieșire (în tot programul 216 → 168 `mov`).
- **NAK a refolosit citirea lui K(0)** din trecerea 1 în trecerea 2 (memoria e doar-citire, deci e corect): 20 de citiri globale în
  listare, nu 24.
- **Corectură la §3:** am scris „deasupra nu se scrie nimic, ca azi” — **greșit**: `attn_fwd_tile3` scrie 0 și în plăcile de deasupra
  diagonalei (faza C, ramura `jt > bx`). `attn_fwd_tc` scrie toate plăcile până la cea de pe diagonală inclusiv (cu 0 exact deasupra
  diagonalei în ea) și **nu scrie** plăcile de deasupra. Programele înapoi nu folosesc acele numere (alegeri, încercate cu NaN acolo:
  T188, T189, T258, T261). Pasul 2 verifică asta și pentru pasul întreg (P cu NaN în plăcile nescrise → aceiași biți înapoi); proba
  de pe placă (pasul 3) compară P doar până la diagonală.

**Pasul următor (2):** modelul pe procesor `gpu_compute::layer::attn_fwd_tc_gpu` + emularea în `gpu-sim` (H3, T299), cu `exp2`
ca `ex2_ftz` și rotunjirile f16 ale lui Q, K, V, P.

## 9. Pasul 2 făcut (sesiunea 284, 28 nov 2026, cloud, FĂRĂ placă): modelul pe procesor + SIMULAREA — H3 CONFIRMATĂ (SIMULARE)

Ce s-a scris (doar adăugări, nicio interfață schimbată):

- **Modelul:** `gpu_compute::attn_fwd_tc` (`fwd_tile` = o placă de rânduri a unui cap, `fwd_model` = toată lansarea, `rel_norm`),
  plus numele din plan `gpu_compute::layer::attn_fwd_tc_gpu` / `attn_fwd_tc_tile_gpu`. Ordinea e cea din listarea NAK 26.3.1 (toate
  operațiile f32 `fadd.ftz` / `fmul.ftz` / `fmnmx.ftz`, 0 `ffma`): Q, K, V, P rotunjite f16; produsele în pași de 16 cu sumatorul
  potrivit pe os-335 (ipoteza `mm_tc`); fiecare rând pe 4 sferturi (coloanele 4q + 16m + i) cu maximul și suma din mers; unirea
  sferturilor în ordinea 0, 1, 2, 3; `ex2` și `rcp` corect rotunjite (pe placă sunt aproximative → pe placă verdictul e o margine).
- **Înlocuitorul în simulator:** `gpu_sim::cmdproc::attn_fwd_tc_on_cpu` + `ATTN_FWD_TC_NEEDS` (27 648 B, o barieră): argumentele de la
  0x700, grila în perechi, scrie P doar până la placa diagonală inclusiv; refuză (fără nicio scriere) d ≠ 64, T nemultiplu de 64,
  adrese nealiniate la 16 B, grila peste perechi / H și o intrare care nu e f16 finit.
- **T299** (`crates/nvidia-ga104/tests/sim_c168_attn_fwd_tc.rs`): o **emulare literală a GLSL-ului** — bloc cu bloc, grup cu grup,
  fir cu fir, barierele ca faze; memoria comună = cele 6 912 cuvinte ale programului (perechi f16 ca `packFloat2x16`, plăci f32 pe
  rânduri de 68), fiecare `coopMatLoad` / `coopMatStore` cu adresa, pasul și aranjarea lui, fiecare vec4 global verificat în buffer,
  **fiecare cuvânt citit și scris între două bariere verificat pentru curse**.

| # | Prezis (§4) | Ieșit (SIMULARE) | Verdict |
|---|---|---|---|
| H3 a | emularea = modelul bit cu bit pe margini | **aceiași biți** (P unde e scris, O) pe T 64 / 128 / 192 (impar: blocul din mijloc are o placă) / 320 / 256, B 1–2, H 1–2, amplitudini 1, 4, 12, și pe capetele 0 și 11 ale formei #513 (T 512, toate cele 4 blocuri); **0 curse** între bariere | CONFIRMATĂ |
| H3 b | d < 64 refuzat | d = 32, T = 96, qkv la 8 mod 16, un K = 10⁶ (în afara f16): `fits` = fals și înlocuitorul refuză, **nimic scris** | CONFIRMATĂ |
| H3 c | față de `attn_fwd_tile3` (F32) la mărimea #513: O ≤ 1·10⁻³, P ≤ 2·10⁻³ | B 1, T 512, C 768, H 12: amplitudine 1 → **O 2,7·10⁻⁴, P 7,6·10⁻⁵** (cel mai rău P 5,5·10⁻⁵ absolut); amplitudine 2 → **O 4,0·10⁻⁴, P 2,8·10⁻⁴** | CONFIRMATĂ |
| H3 d | deasupra diagonalei lui P nimic scris | plăcile de deasupra plăcii diagonale rămân NaN (prin emulare și prin pushbuffer + QMD); în placa diagonală 0 exact deasupra diagonalei, ca `attn_fwd_tile3` | CONFIRMATĂ |

Verificări în plus:

- **Comparațiile nu sunt oarbe:** 6 greșeli plantate, toate prinse — 2 bariere scoase (→ curse găsite), Vᵀ neîntors, a doua placă
  a perechii uitată (în emulare); unirea sferturilor inversată, fără mască în trecerea 2, P trunchiat în loc de rotunjit (în model).
- **Înapoi nu citește plăcile nescrise** (cerința din §8): `attn_tc::back0_model` (dS sub diagonală) și `attn_tc::model` (modelul lui
  `tc16_v4`) dau **aceiași biți** cu NaN acolo (ce lasă `attn_fwd_tc`) ca și cu 0 (ce scria `attn_fwd_tile3`).
- O = P16·V16: față de o sumă f64 pe P și V rotunjite f16, cel mai rău 5,3·10⁻⁸ (verificare a modelului față de propria definiție).

**Ce NU arată pasul 2:** nimic despre placă. Sumatorul circuitelor e ipoteza potrivită pe os-335; `ex2` / `rcp` ale plăcii sunt
aproximative; că NAK a tradus GLSL-ul exact cum îl emulăm e verificat doar pe listare (H2). Pe placă se judecă **marginea** față de
`attn_fwd_tile3`, nu biții (H4), și timpul (H5).

**Pasul următor (3):** nota pentru OS — proba `--atentie-fwd-tc` lângă `attn_fwd_tile3` pe aceleași intrări (H4, H5) → PREDARE.
