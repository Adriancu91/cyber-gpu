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

## 10. Pasul 3 făcut (sesiunea 285, 28 nov 2026, cloud, FĂRĂ placă): proba `--atentie-fwd-tc` pentru OS — PREDARE

Ce s-a scris (doar adăugări): `nvidia_ga104::attn_fwd_tc_probe` — seturile, ce se umple cu NaN și ce se citește (`fwd_ranges`),
`launch_bytes` (cbuf 0x800 + QMD **la treapta 64 KiB**; la treapta proprie, 32 KiB, ar încăpea doar 1 bloc pe SM), `judge`
(O și P pe j ≤ i ca norme relative față de `attn_fwd_tile3`, 0 exact deasupra diagonalei în placa diagonală, plăcile de deasupra plăcii
diagonale nescrise = NaN, gărzile), `Probe::lines` și `parse`. Nota pentru OS: `integration/cyber-universe/C-168_P4a_atentie-fwd-tc-proba.md`.

**De ce seturi noi:** din cele 5 seturi mici ale probei atenției **niciunul nu e aliniat** (d 16 / 40, T 130 / 200 / 96); ar fi rămas
doar `atentie-513-b4`, citit pe benzi. Seturile noi (citite întregi): `tc-t64` (o placă), `tc-t192-b2` (3 plăci: blocul din mijloc face
una — cazul impar al perechilor; 2 texte), `tc-t256-h2` (2 capete, perechi pline), `tc-t320-ascutit` (5 plăci, qkv în [−4, 4)).

**T301 (SIMULARE):** ambele programe prin pushbuffer + QMD, ca în OS (`attn_fwd_tile3` cu `launch_bytes3`, `attn_fwd_tc` cu
`launch_bytes`), NaN înainte de fiecare, citirile judecate: 4 / 4 seturi mici trec, cuvintele lui `attn_fwd_tc` = modelul pe procesor
(toate), iar `atentie-513-b4` (benzile, din modele) trece. 8 greșeli plantate în citire, toate prinse (un cuvânt scris deasupra plăcii
diagonale, un ne-zero deasupra diagonalei, garda lui P, garda lui att, un rând de P cu +10 %, att decalat cu un rând, un NaN rămas sub
diagonală, lungime greșită / set nealiniat / buffer nealiniat refuzate). Rândurile de raport citite înapoi cu `parse`.

| Set | O față de `tile3` | P față de `tile3` | cel mai rău P (absolut) |
|---|---|---|---|
| `tc-t64` | 3,7·10⁻⁴ | 2,3·10⁻⁴ | 1,5·10⁻⁴ |
| `tc-t192-b2` | 3,9·10⁻⁴ | 2,7·10⁻⁴ | 2,5·10⁻⁴ |
| `tc-t256-h2` | 3,7·10⁻⁴ | 2,7·10⁻⁴ | 2,4·10⁻⁴ |
| `tc-t320-ascutit` | **6,1·10⁻⁴** | 5,4·10⁻⁴ | 1,0·10⁻³ |
| `atentie-513-b4` (benzi) | 1,9·10⁻⁴ | 5,2·10⁻⁵ | 4,9·10⁻⁵ |

**Predicțiile pentru placă (scrise acum, înainte de rulare):**

- **H4 (80 %, neschimbată):** 5 / 5 seturi în margine (O ≤ 1·10⁻³, P ≤ 2·10⁻³), zerourile, nimic scris deasupra plăcii diagonale,
  gărzile neatinse. Cel mai aproape de prag: `tc-t320-ascutit` (O ×1,6 sub prag). „Exacte față de model” pe placă **sub** total
  (`ex2` / `rcp` aproximative) — doar informativ.
- **H5 pe proba OS (45 %):** H5 (lot 6, 550–930 µs pe 3 valuri de 96 de blocuri) trecută la `atentie-513-b4` (192 de blocuri = 2 valuri):
  **0,37–0,62 ms** față de `attn_fwd_tile3` ~1,00 ms (os-364…375) → ×1,6–2,7. DEDUS.
- **Prag (neschimbat):** H4 trece și timpul peste 0,62 ms → frâna e în trecerea 2; pasul următor e proba aranjării acumulatorilor, nu
  legarea în pas.

**Ce NU arată pasul 3:** nimic despre placă — seturile, citirile și verdictul sunt verificate pe modele.

**Pasul următor (4):** după rezultatul probei pe placă: `TcPick` pentru atenția înainte în pas (`pack_mixed_fast_attn*`, doar
adăugări) → PREDARE (H6, H7).

## 11. os-504 comparat + pasul 4: planul legării în pas (sesiunea 289, 28 nov 2026, ~08:15, cloud, FĂRĂ placă — scris ÎNAINTE de cod)

**os-504 (MĂSURAT de OS, `rezultate/2026-11-28-workstation-os504-atentia-fwd-tc/` din hub):**

| # | Prezis | Pe placă | Verdict |
|---|---|---|---|
| H4 | 5 / 5 seturi în margine (O ≤ 1·10⁻³, P ≤ 2·10⁻³) | 5 / 5; la #513 O 1,9·10⁻⁴, P 5,2·10⁻⁵ (= SIMULAREA, T301) | **CONFIRMATĂ** |
| H5 (proba) | `atentie-513-b4` 0,37–0,62 ms (×1,6–2,7) | **0,184 ms** (tile3 1,003 → ×5,45); seturile mici ×1,9–2,4 | **RATATĂ în bine** (de 2–3,4 ori mai rapid decât intervalul) |

Ce înseamnă H5 ratată în bine (DEDUS, nu măsurat): pe 3 produse de 64×64×64 pe placă și pe toate capetele (`atentie-513-b4`:
4 × 12 × 36 plăci × 3 produse = 0,68 GFLOP… ×2 pentru treceri) programul nou face ~**11 TFLOP/s** pe produsele utile — peste
`tc16_v4` (8,96), nu de 1,2–1,8 ori sub el cum presupusesem. Presupunerea greșită: că S dus prin memoria comună și cele 5 bariere
costă cât produsele; pe placă costul s-a ascuns în cele 2 blocuri pe SM. Pragul din §4 / §10 („peste 0,62 ms → proba aranjării
acumulatorilor”) **nu s-a atins** → pasul 4 (legarea în pas) e următorul, cum scrie în plan.

**Pasul 4 — ce scriu (doar adăugări, ca la `attn_tc_list` / `pack_mixed_fast_attn2`):**

1. `MixedLaunch::AttnFwdTc(Launch)` — o lansare `attn_fwd_tile3` a listei, mutată pe `attn_fwd_tc` (aceleași argumente; cbuf-ul
   de 0x800 cu argumentele la 0x700, `attn_fwd_tc::cbuf_from_f32`; **grila în perechi** `attn_fwd_tc::grid`, nu grila lui tile3).
   Locul în zona de control: `SLOT_TC` (ca celelalte programe de pe circuite).
2. `train_step::attn_fwd_tc_list(list)` — mută doar lansările `attn_fwd_tile3` care trec `attn_fwd_tc::fits_cbuf`; restul rămân
   (formele nealiniate pe tile3, fără eroare: asta e regula din §3). Se compune cu `attn_tc_list` / `attn0_tc_list` în orice ordine.
3. `train_step::pack_mixed_fast_attn9(…, long_k, attn_fwd_va, attn, …)` = `pack_mixed_fast_attn8k` cu `attn_fwd_va` (adresa lui
   `attn_fwd_tc`, 0 = neîncărcat) pus după `long_k`; alegerea nouă `TcPick::AttnFwdTc`; QMD-ul: 120 registre, 27 648 B, 1 barieră,
   256 de fire, grila în perechi, treapta pasului. `attn_fwd_va = 0` și fără `AttnFwdTc` în listă = `attn8k` octet cu octet;
   o listă cu `AttnFwdTc` și `attn_fwd_va = 0` e refuzată înainte de orice trimitere. Plus `batch_ram::pack_mixed_fast_via_ram_attn9`.
   Packerii vechi refuză o listă cu `AttnFwdTc` (ca la `Attn0Tc`).

**Predicțiile pasului 4 (scrise acum; H6 și H7 din §4 reluate cu cifrele os-504):**

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| H6a | SIMULARE: legarea | `attn_fwd_va = 0` = `attn8k` octet cu octet (mini și #513); în #513 lot 6 **exact 12** lansări mutate (una pe strat), toate celelalte alegeri neschimbate (v7 12 / v8n 135 fără gardă; 13 / 134 cu garda P3e) | 95 % |
| H6b | SIMULARE: un model mic aliniat (d = 64, T = 64, 2 straturi, 5 pași) prin pushbuffer + QMD | pierderile / gradientul / greutățile **în marginile pasului mixt** față de același pas cu tile3 (`MIX_TOL_*`), nu aceiași biți (f16 în produse) | 85 % |
| H6c | pe placă, timpul atenției înainte, lot 6 | 12 × ~0,25 ms (1,346 / 5,45) = **2,6–4,0 ms** (azi 16,1) — în margine sus pentru că în pas L2-ul e împărțit | 60 % |
| H6d | pe placă, pasul lot 6 (os-503 / os-504: 140,5–140,7 ms) | **126–130 ms = 23 600–24 400 simb/s**; cu garda P3e împreună **121–126 ms = 24 400–25 400 simb/s** | 55 % |
| H7 | pe placă, oracolul mixt lot 2 | **ACELEAȘI NUMERE** (în toleranțe; os-504 a dat marginile H4 pe #513) | 75 % |

Ținta C-168 p.4 (≤ 6 ms pe pas) e atinsă dacă H6c iese (DEDUS până la rândul pe placă). FlashAttention-2 întreg (B din §2) nu mai e
necesar pentru țintă; rămâne scris, pentru memoria lui P (−906 MiB din arenă), nu pentru timp.
