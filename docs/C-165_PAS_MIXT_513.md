# C-165 / prioritatea 2 — pasul mixt (`mm_tc`) pe forma lui #513: scara erorii S = 2¹⁴ (SIMULARE)

Sesiunea 217, 25 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE** sau **DEDUS**: nimic nu a rulat pe RTX 3070 Ti.

## De ce acum
`mm_train_tile` = 93 % din pasul #513 (os-412); cursa de noapte (os-413) a găsit doar ×1,50 pe qkv (`mm_tv_128x128`). Pasul următor
din ordinea centrului (`idei/2026-11-24-centru-ordinea-prioritatilor.md`, locul 2) = **circuitele speciale + umbra pe 16 biți**:
toate înmulțirile pasului pe `mm_tc` (f16 × f16 → f32). `Plan::step_mixed` există (T169, T199), dar a fost probat doar pe 64 de
simboluri pe lot (`antrenare mini`, lot 2 × 32). #513 are **1 024 de simboluri pe lot** (2 × 512) → d(logits) pentru simbolurile care
nu sunt ținta ≈ S / (16 384 · 1 024) — la S = 1 024 (scara de azi, `ts::LOSS_SCALE`) cade **exact la cel mai mic număr normal f16**
(6,1·10⁻⁵), deci jumătate din valori ar fi subnormale. Nota `C-157_P2u_oracol-513.md` §4 a propus S = 2¹⁴; aici se verifică, pe
procesor, înainte să ajungă pe placă.

Pe placă, ordinea rămâne cea scrisă de OS (`stare/cyber-universe.md`): întâi proba `mm_tc` singură (os-352, `--mm-tc`, **nerulată
încă**), apoi `--antrenare-mini --mixt` (os-384, nerulat), abia apoi #513 mixt. Nimic din nota asta nu sare peste ele.

## Ce se face (test nou + o constantă; nimic schimbat în ce există)
1. `train_step::LOSS_SCALE_513: f32 = 16 384.0` și `train_step::loss_scale_for(vocab, tokens)` = cea mai mică putere a lui 2 care ține
   S / (V · N) ≥ 2⁻¹⁰ ≈ 10⁻³ (regula din `C-157_P2u` §4), între 1 și 2²⁴ — doar adăugări; `LOSS_SCALE` (1 024) rămâne neschimbat.
2. T223, `crates/nvidia-ga104/tests/sim_c165_mixed_513.rs`:
   - **(a) lista #513 adevărată** (M513: 12 straturi, E 768, vocabular 16 384, lot 2 × 512), cele două planuri pe care OS-ul le
     rulează azi (`new_tiles3 + with_mm_tile + with_colsum_split` și varianta „lipit pe plăci”
     `with_fused_back + with_fused_gelu + with_mm_tile_gelu`): `step_mixed(S = 2¹⁴)` se construiește, câte lansări, câte pe `mm_tc`
     / `mm_tc_gelu`, toate argumentele acceptate, mărimea zonei de control (`pack_mixed_gelu`). Fără rulare (#513 e prea mare pentru
     simulator).
   - **(b) rularea pe forma cu 1 024 de simboluri pe lot**: modelul `antrenare mini` (vocabular 16 384, E 64, 4 capete, 2 straturi,
     context 32) cu **lotul 32 × 32 = 1 024 de simboluri** — același d(logits) ca #513 (V · N identic), dar încape în simulator.
     2 pași pe același lot, AdamW ca `antrenare mini`; oracolul pe procesor; pasul mixt cu S = 1 024 și cu S = 2¹⁴, atenția `_3`.

Măsurile sunt cele din T199 (`C-160_P7_PAS_MIXT_MINI.md`): eroarea pe pas, gradientul pasului 1 pe tensor (‖Δ‖/‖g‖, după împărțirea la
S, fără `in_proj_bias`), **rândurile rare ale lui `emb_token`** (simbolurile care nu apar în lot: acolo se vede f16).

## Predicțiile, scrise ÎNAINTE de cod și de prima rulare (se raportează oricum ar ieși)
1. **Lista #513**: `step_mixed` = lansările lui `step` (fără plăci) + 1; pe `mm_tc` **147** lansări pe pas (12 pe strat × 12 + 3 la
   capăt: logits înainte, dX și dW ale capului legat); în varianta lipită 12 dintre ele sunt `mm_tc_gelu` (135 + 12). Toate argumentele
   acceptate (formele #513 sunt multipli de 64, K multiplu de 32). Zona de control a unui pas: ~**0,4–0,5 MiB** (147 × 0x900 + restul
   × 0x200).
2. **S = 2¹⁴, lot 1 024**: în toleranțele scrise ale pasului mixt — eroarea ≤ 1·10⁻³, gradientul pasului 1 ≤ 5·10⁻³ pe tensor,
   rândurile rare ≤ 1·10⁻³ (aceeași mărime a d(logits) ca la mini cu S = 1 024, ~10⁻³, deci aceleași cifre ca T199: rare ~5·10⁻⁴).
3. **S = 1 024, lot 1 024** (scara de azi pe forma #513): rândurile rare **mai rele decât la S = 2¹⁴, de 1,5–5 ori** (valorile stau pe
   granița normal / subnormal: pasul f16 de acolo, 6·10⁻⁸, e ~10⁻³ din valoare, deci nu catastrofal ca S = 1 la mini, dar vizibil);
   tot ≤ 5·10⁻³.
4. **Nicio depășire la S = 2¹⁴**: |S · d(logits)| ≤ S / N = 16 (ținta), mult sub 65 504; norma finită, toate greutățile finite.
5. `loss_scale_for(16 384, 1 024)` = 2¹⁴, `loss_scale_for(16 384, 64)` = 1 024 (scara pe care a trecut T199) — regula dă scara de azi pentru mini.

## Ce e nou pentru placă
Nimic. Propunerea pentru OS (după os-352 și os-384 pe placă): `--antrenare-513 --mixt` cu `LOSS_SCALE_513`.

## Rezultate (sesiunea 217, T223 `crates/nvidia-ga104/tests/sim_c165_mixed_513.rs`, SIMULARE)
| Predicție | Ce a ieșit | Verdict |
|---|---|---|
| 1. Lista #513 | `placi3 + mm-placi + colsum`: 483 lansări F32 → **484 mixt, 147 pe `mm_tc`**, zona 499 KiB; `lipit-placi`: 447 → **448, 135 `mm_tc` + 12 `mm_tc_gelu`**, zona 481 KiB; nicio înmulțire a modelului rămasă F32; toate argumentele acceptate | CONFIRMATĂ |
| 2. S = 2¹⁴, lot 1 024 | eroarea 9,0·10⁻⁷; gradientul pasului 1 cel mai rău 8,8·10⁻⁴ (`straturi.0.ln2.bias`); `emb_token` 4,7·10⁻⁴; rândurile rare **4,1·10⁻⁴** | CONFIRMATĂ |
| 3. S = 1 024 mai rău de 1,5–5 ori pe rândurile rare | **53 % din d(logits) sub 2⁻¹⁴ (subnormale în f16)**, dar rândurile rare 4,13·10⁻⁴ — **raport 1,00**; eroarea 2,0·10⁻⁷, gradientul 9,1·10⁻⁴ | **GREȘITĂ** |
| 4. Nicio depășire la 2¹⁴ | max \|S · d(logits)\| = **15,999** (= S / N), 0 % subnormale; norma, gradientul și greutățile finite | CONFIRMATĂ |
| 5. Regula scării | `loss_scale_for(16 384, 64)` = 1 024, `(16 384, 1 024)` = 16 384 | CONFIRMATĂ |

**De ce a greșit predicția 3 (DEDUS):** eroarea rândurilor rare nu vine din d(logits), ci din rotunjirea la f16 a **stărilor ascunse**
(cealaltă intrare a înmulțirii dW = d(logits)ᵀ · h), aceeași la ambele scări. Pasul f16 lângă 2⁻¹⁴ e 2⁻²⁴ ≈ 6·10⁻⁸, adică ~10⁻³ din
valoare — de 2–4 ori mai grosier decât f16 normal, dar erorile pe 1 024 de simboluri se compensează în sumă (f32). Deci **pe forma #513, la
pornire, S = 1 024 ar fi mers și el.**

**Ce recomandăm totuși: S = 2¹⁴ (`LOSS_SCALE_513`).** La 2¹⁴ nu mai e nicio valoare subnormală și cea mai mare e 16 (de 4 000 de ori sub
65 504), deci rezerva e mare în ambele direcții; pe măsură ce modelul învață, softmax-ul se ascute și cele mai multe probabilități scad
mult sub 1/V — atunci, la S = 1 024, tot mai multe valori ar coborî adânc în zona subnormală (ca S = 1 la mini, T199: de 10 ori mai rău).
Asta e DEDUS, nu măsurat; o probă cu greutăți antrenate ar lămuri-o (nu avem încă greutăți #513 antrenate în SIMULARE).

**Ce e nou pentru placă:** nimic. Ordinea pe placă rămâne: os-352 (`--mm-tc`, proba singură), os-384 (`--antrenare-mini --mixt`),
abia apoi `--antrenare-513 --mixt` cu `LOSS_SCALE_513` (sau `loss_scale_for(V, B·T)`).
