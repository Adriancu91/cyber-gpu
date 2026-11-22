# C-160 punctul 7 — proba atenției „pe plăci” pentru placa reală (`attn_tile_probe`)

Sesiunea 166, 22 nov 2026, în cloud, **fără placă**. Planul, marginile și predicțiile de mai jos sunt scrise și urcate **înainte** de
cod. Totul despre viteză e **DEDUS**; corectitudinea se verifică doar în **SIMULARE** până rulează pe placă.

## Ce probăm
Cele trei programe noi ale atenției de antrenare, în ordinea unui pas, **singure** (în afara pasului de antrenare și a generării):
`attn_fwd_tile` (înainte: P și att) → `attn_back_tile0` (înapoi, modul 0: rândurile lui dS) → `attn_back_tile` (înapoi, modul 1:
dQ | dK | dV). Pe placă e nou: cele trei binare (SHA-256 din module) și QMD-urile cu memorie partajată 37 / 32 / 32 KiB + barieră.
Nimic altceva nou în același test. Pe placă: după pasul de aur mic F32 (os-351) și proba `mm_tc` (os-352), câte una.

## Planul (scris ÎNAINTE de cod)
Modul nou `nvidia_ga104::attn_tile_probe` (doar adăugări), ca `mm_tc_probe`:
- **Seturi** (intrările = un hash din (sămânță, element), ca OS-ul să le scrie pe bucăți): qkv în [−A, A), dO în [−1, 1).
  1. `mic-b2-t16-d16` (B 2, T 16, C 32, H 2 — forma pasului de aur mic), A = 2;
  2. `t130-d64` (B 1, T 130, C 128, H 2) — T peste două plăci de 64, ultima parțială, A = 2;
  3. `t64-d40` (B 2, T 64, C 40, H 1) — o placă exactă, d care nu e putere a lui 2, A = 2;
  4. `t200-h3` (B 1, T 200, C 192, H 3) — rânduri lungi, A = 2;
  5. `ascutit` (B 1, T 96, C 64, H 1) — A = 6: softmax aproape „un singur 1” (exponenți mari negativi);
  6. **`atentie-513-b4`** (B 4, T 512, C 768, H 12 — o treime din bucata #513), A = 2, **5 repetări** din fiecare program → timpul median.
- **Un buffer pe set:** qkv, dO, apoi P, att, dS, dqkv, fiecare urmat de 256 de cuvinte-gardă; ieșirile + gărzile umplute cu NaN
  înainte (un element nescris nu poate trece, o scriere peste capăt se vede). Setul 6: ~151 MB (corectat după cod; în prima versiune scrisesem ~133).
- **Ce se citește înapoi:** seturile 1–5 tot (P, att, dS, dqkv + gărzile). Setul 6 pe benzi: pentru b ∈ {0, 3}, toate cele 12 capete,
  rândurile i ∈ {0, 300, 511} din P și dS, rândurile b·T + i din att, rândul b·T + 511 din dqkv, + cele 4 gărzi (~0,34 MB; în prima versiune scrisesem ~0,6).
- **Judecata, câte una pe program, astfel încât fiecare să fie judecat singur:**
  - `attn_fwd_tile` față de modelul pe procesor `layer::attn_fwd_tile_row_gpu` (pe qkv-ul scris): deasupra diagonalei P exact +0;
    altfel **marginea MUFU** (modelul are `EX2` / `RCP` rotunjite corect, placa le are aproximative):
    |ΔP_ij| ≤ 2⁻¹⁸·P_ij + (i + 3)·2⁻¹²⁶, |Δatt_ie| ≤ (2⁻¹⁸ + 2·(i + 1)·2⁻²⁴)·Σ_j |P_ij·V_je| + 2·(i + 1)·2⁻¹²⁶.
    (Socoteala: EX2 și RCP au eroarea relativă ≤ ~2⁻²² fiecare pe Ampere — ghidul public CUDA dă 2 ulp pentru `exp2f` rapid; x, suma
    și 1/sumă adună ≤ ~3·2⁻²², plus rotunjirile care pot cădea altfel pe intrări puțin diferite → 2⁻¹⁸ are ×~4 loc.)
  - `attn_back_tile0` față de `layer::attn_back_row_gpu` calculat **pe P-ul citit de pe placă** (nu pe al modelului): așa MUFU-ul
    plăcii nu mai contează, iar programul n-are MUFU → **predicția: bit cu bit**. Verdictul „în margine” cere
    |Δ| ≤ 2⁻²⁰·max|rând| + 2⁻¹²⁶ (margine de siguranță, nu ținta); câte sunt exacte se scrie separat.
  - `attn_back_tile` față de `layer::attn_back_col_gpu` pe P-ul **și** dS-ul citite de pe placă → **predicția: bit cu bit**; aceeași
    margine de siguranță pe rândul lui dqkv.
  - Pe setul 6 aceleași funcții, pe rândurile citite (dQ, dK, dV ale poziției 511 au nevoie doar de rândul 511 din P și dS).
- **Rânduri `kt atentie …`** (seturi, în margine, elemente exacte pe program, promile, gărzi, amprente FNV, timpul median pe program),
  rânduri de raport și cititorul lor din jurnal (`parse_attn_tile`), ca la `mm_tc`.
- **SIMULARE (T176):** placa simulată rulează lanțul cu `gpu_sim` (programele „pe procesor” ale celor trei binare), apoi judecata
  pe bytes citiți exact ca OS-ul; plus greșeli plantate.

## Predicții (scrise acum)
1. **Corectitudine pe placă:** 6 / 6 seturi în margine la toate trei programele; gărzile neatinse; nicio scriere deasupra diagonalei
   în afară de +0.
2. `attn_back_tile0` și `attn_back_tile`: **100 % bit cu bit** față de model pe intrările citite de pe placă (n-au MUFU, n-au
   `atomic`, ordinea sumelor e fixă) — orice abatere = o greșeală de-a noastră în model sau în program.
3. `attn_fwd_tile`: P și att **nu** toate bit cu bit (MUFU-ul plăcii): estimez **20–80 %** din elementele P exacte (cele mai multe
   diferențe de 1–2 unități în ultima poziție), cel mai rău **≤ 300 ‰** din marginea MUFU; în SIMULARE 100 % exacte (modelul e
   chiar programul simulat).
4. **Viteza pe setul 6 (B 4, o lansare = un strat, DEDUS din `tools/buget-antrenare/fir_limitat.py` cu B = 4):**
   `attn_fwd_tile` **~0,53 ms** (interval 0,27–1,1), `attn_back_tile0` **~0,49 ms** (0,25–1,0), `attn_back_tile` **~0,42–0,68 ms**
   (0,2–1,4). Toată atenția ~1,5 ms pe strat la B 4 → pe bucata #513 (×3 texte, ×12 straturi) **~52–61 ms**, față de ~257 ms pentru
   programele vechi de 32 de fire (DEDUS) și ~156 ms toată bucata PyTorch (MĂSURAT de Kronos).
5. SIMULARE: greșeli plantate prinse — un rând de P stricat cu 2⁻¹⁴ (iese din marginea MUFU), un dS stricat cu o unitate (nu mai e
   exact; în marginea de siguranță dacă e mic), o gardă scrisă (INVALID), un element nescris (NaN → INVALID), un set lipsă la probă.

## Ce NU face
Nu schimbă `attn_fwd_tile`, `attn_back_tile`, `attn_back_tile0` și nici programele vechi. Nu le leagă de pasul de antrenare. Pe placă
doar prin Release-ul OS, cu „ce e nou pentru placă” scris în nota pentru OS.

## Surse
Planul probei e cel din `mm_tc_probe` (sesiunea 160) și `colsum_probe` (sesiunea 158). Precizia `exp2f` / `__frcp_rn` aproximative:
ghidul public CUDA (anexa „Mathematical Functions”, „intrinsic functions”). Formulele atenției: `gpu_compute::train` (oracolul
pasului de aur, T151).

## Rezultatul (SIMULARE, T176; codul după planul `608f377`)
- Modulul `nvidia_ga104::attn_tile_probe` (doar adăugări) + testul `crates/nvidia-ga104/tests/sim_c160_p7_attn_tile_probe.rs`.
- **T176a:** 5 / 5 seturi mici prin procesorul de comenzi simulat (cbuf + QMD din `launch_bytes`, cele trei lansări în ordine, citirea
  pe `read_ranges`): toate trei programele în margine, **toate elementele exacte** (154 018 / 86 690 / 201 984) — așteptat, programele
  simulate sunt chiar modelele (egalitatea model = GLSL e T173–T175). Pe placă, exactitatea lui `attn_fwd_tile` e predicția 3 (20–80 %).
- **T176b, predicția 5: DA** — P stricat cu 2⁻¹⁴ după lansarea înainte: doar `attn_fwd_tile` NU TRECE, cele două „înapoi” rămân 100 %
  exacte (judecate pe P-ul plăcii — izolarea cerută); o unitate în dS: în marginea de siguranță, exacte − 1; gardă scrisă: INVALID doar
  la programul acela; `attn_back_tile` nelansat: INVALID; grila înainte cu un bloc lipsă: INVALID; un set lipsă: proba nu trece.
- **T176c:** un set mediu pe benzi rulat întreg: citirea pe benzi = exact ieșirea modelului pe benzi.
- **T176d:** setul de timp judecat pe ieșirea modelului (benzile), rândurile `kt atentie …` citite înapoi de `parse_attn_tile`.
- **Două socoteli greșite în plan, corectate:** bufferul setului de timp e **~151 MB** (144 MiB), nu ~133 (socoteala de mână greșită cu ~18 MB, cât un qkv); citirea
  înapoi e **~0,34 MB**, nu ~0,6.
- `tools/check.sh`: format, clippy (Rust 1.97), **835 de teste** (+7 ignorate), build no_std — PASS.
- Predicțiile 1–4 (pe placă) și viteza: **NOT TESTED** până la rularea pe placă. Nota pentru OS:
  `integration/cyber-universe/C-160_P7e_atentia-pe-placi-proba.md`.

## Rezultatul pe placa reală (os-358, 23 nov 2026, 00:36; comparat în sesiunea 171)
- Predicțiile 1–3 (corectitudinea): **DA** — 6/6 seturi, gărzile neatinse; „înapoi” 100 % exacte; `attn_fwd_tile` 38,8 % exacte, cel mai
  rău 89 ‰.
- Predicția 4 (viteza): **RATATĂ de 4–9 ori** — 2,49 / 1,81 / 3,99 ms față de ~0,53 / ~0,49 / ~0,42–0,68; pe bucata #513 ~297 ms.
  Cauzele (DEDUS din listările NAK, `tools/buget-antrenare/nak_bucle.py`) și ce tai mai departe: `docs/C-157_P2_COMPARATII.md`, secțiunea
  os-358. Greșeala estimării: a socotit doar FLOP și octeți, nu instrucțiunile și așteptările din codul compilat.
