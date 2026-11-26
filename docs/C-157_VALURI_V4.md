# C-157 — valurile `mm_tc_v4` pe pasul mixt #513: unde se duc cele 106 ms și dacă lansarea dublă dX + dW mai merită

Sesiunea 241, 26 nov 2026, în cloud, **fără placă**. Cifrele plăcii sunt MĂSURATE de OS (os-431, os-443, os-445); tot ce e
calculat aici e **DEDUS** (aritmetică pe procesor). Nimic nu a rulat pe RTX 3070 Ti în sesiunea asta.

## 1. De ce acum

Studiul programului persistent (sesiunea 236, `docs/C-157_PROGRAM_PERSISTENT.md` §5, punctul 2) a lăsat deschis: „lansarea dublă
dX + dW se va repeta pe `mm_tc_v4` (pasul mixt), unde înmulțirile devin mai scurte și cozile cântăresc mai mult — de măsurat”.
Pasul mixt rapid a rulat pe placă (os-445: placa **106,0 ms**, pasul întreg 116,9 ms), dar fără profil pe lansări, deci nu știm
cât din cele 106 ms sunt înmulțiri și cât restul. Niciun test nou pe placă de la os-445.

## 2. Ce avem (citit, nu presupus)

- `mm_tc_v4`: blocuri de 128 de fire, plăci 64 × 64, 128 de registre, 24 KiB memorie comună → la treapta 64 KiB **2 blocuri pe
  unitate** (96 de locuri pe 48 de unități), la 32 KiB 1 bloc (48 de locuri).
- os-443, qkv-513: **2 623 680 ns la 64 KiB**, **4 737 216 ns la 32 KiB**. *Corectat după commit-ul predicțiilor:* setul qkv-513 al
  probei e **6 144 × 2 304 × 768** (`mm_tc_probe.rs`: qkv-ul unei bucăți de 6 × 1 024 de rânduri) = 3 456 de blocuri → **36 de valuri**
  la 64 KiB (72 880 ns pe val) și 72 la 32 KiB (~65 795 ns pe val). Prima variantă a codului a luat forma pasului (1 024 de rânduri)
  și a ieșit de 6 ori prea mare (+499 %) — greșeala mea de citire, prinsă de testul V1; predicțiile V1–V4 nu depindeau de ea.
- os-431 (pasul F32 #513, aceeași listă de programe în afara înmulțirilor, aceeași treaptă 64 KiB): placa 255 896 µs, din care
  înmulțirile (`mm_tv_128x128`, 147 lansări) **223 234 µs** → restul **32 662 µs**.
- Lista pasului mixt (`Plan::new_tiles3(M513, 2, 512).with_mm_tile().with_colsum_split().step_mixed`): 484 de lansări, 147 `mm_tc`
  în 15 forme (cozile valurilor: 1024 × 768 = 192 de blocuri = exact 2 valuri; `out-dW` 768 × 768 = 144 = 1,5 valuri).

## 3. Predicțiile (scrise ÎNAINTE de cod și de calcule)

- **V1.** Înmulțirile pasului mixt pe placă (DEDUS: 106,0 ms − restul os-431) ≈ **73 ms**; modelul valurilor `mm_tc_v4` (un timp pe
  val, calibrat doar pe qkv-513, liniar în K) le reface **în ±10 %**.
- **V2.** Pe plăcile 64 × 64 cozile valurilor sunt mici: locurile goale din cele 147 de lansări **sub 5 %** din timpul înmulțirilor.
- **V3.** Lansarea dublă dX + dW pe `mm_tc_v4` (podeaua, blocuri împachetate perfect) câștigă **sub 2 %** din placa pasului mixt
  (pe F32 era 3,1 % peste alegerea pe formă) → nu merită un program nou.
- **V4.** Restul (~32,7 ms) devine **~31 %** din placă; cea mai mare bucată a lui e atenția (os-431: `attn_*` 20,6 ms ≈ 19 % din
  placa mixtă) → următoarea tăietură mare e atenția, nu cozile înmulțirilor.

## 4. Ce se face în sesiune

Cod (doar adăugare) în `nvidia_ga104::wave_model`: timpii pe val ai `mm_tc_v4` din os-443 și predicția unei liste de produse;
în `persistent_study`: podeaua lansării duble pentru `mm_tc_v4`. Test T250 pe procesor (rândurile os-443 / os-445 / os-431 citite
din `tests/rapoarte/`). Nimic nou pentru placă; la final, o cerere opțională către OS: profilul pasului mixt (aceeași comandă cu
profilul pornit), ca să înlocuiască DEDUS-ul de aici cu o măsurătoare.

## 5. Rezultatul (sesiunea 241, ~03:55; T250, 6/6 pe procesor; cifrele = DEDUS din timpii MĂSURAȚI de OS)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| V1 | înmulțirile ≈ 73 ms; modelul le reface în ±10 % | DEDUS din placă: 106 001 − 32 662 = **73 339 µs** (= 8,17 TFLOP/s în medie, aproape cât qkv-513 singur, 8,28); modelul valurilor `mm_tc_v4` (calibrat doar pe qkv-513): **73 260 µs, −0,1 %** | CONFIRMATĂ |
| V2 | locurile goale < 5 % | **1,3 %** din timpul înmulțirilor; doar 3 forme au val parțial: `out-dW` 768 × 768 (144 de blocuri, 21 % goale), `qkv-dW` 2 304 × 768 (432, 8 %), `logit` (4 096, 0,8 %) | CONFIRMATĂ |
| V3 | lansarea dublă dX + dW < 2 % din placă | podeaua (plafonul, împachetare perfectă): 49 137 → 48 197 µs = **939 µs = 0,9 % din placă** | CONFIRMATĂ |
| V4 | restul ~31 %, mai ales atenția (~19 %) | restul **30,8 %**; atenția (`attn_fwd_tile3`, `attn_back_tile_3`, `attn_back_tile0_3`) **20 581 µs = 19,4 % din placă, 63 % din rest** | CONFIRMATĂ |

Cod (doar adăugări, nimic redenumit): `wave_model::{TC_V4_TILE, OS443_TC_V4, TC_V4_CAL_K, tc_v4_blocks, tc_v4_wave_ns,
tc_v4_launch_ns, tc_v4_idle_permille}`, `persistent_study::{backward_pairs_mixed, tc_v4_pair_cost}` și
`train_profile::StepProfile::from_mixed(list, picks)` — profilul pasului mixt, cu fiecare `mm_tc` numit după programul ales de
`pack_mixed_fast` (`mm_tc_v4` / `mm_tc_pf` / `mm_tc_gelu`). Test `tests/c157_valuri_v4.rs` (T250a–f).

### Ce înseamnă (pe scurt)

1. **Placa pasului mixt (106 ms) = ~73 ms înmulțiri + ~33 ms restul** (DEDUS). Potrivirea de −0,1 % spune că (a) toate cele 15 forme
   merg pe placă cam la viteza lui qkv-513 și (b) restul s-a păstrat cât în os-431. Nu e o dovadă: două presupuneri se pot compensa.
   Dovada = profilul pe lansări al pasului mixt (mai jos).
2. **Cozile valurilor nu mai contează pe `mm_tc_v4`.** Plăcile de 64 × 64 fac multe blocuri, aproape toate formele umplu valuri întregi.
   Lansarea dublă dX + dW ar lua cel mult ~0,9 ms din 106 → **nu merită un program nou**; firul „program persistent la antrenare” se
   închide aici.
3. **Ce urmează la antrenare, în ordinea câștigului (DEDUS):** (a) înmulțirile însele (69 % din placă; 8,2 TFLOP/s e încă departe de
   ce pot circuitele speciale ale plăcii — următorul câștig e în program, partea „c”: scrierile în memoria comună); (b) **atenția pe
   circuitele speciale** (19 % din placă; azi F32) — singura bucată mare din rest; (c) lucrul procesorului în afara plăcii (~11 ms
   din 117, deja pregătit: `--lot-prin-ram`, os-447).

### Ce NU e dovedit

- Timpul înmulțirilor din pasul mixt nu e măsurat direct: e placa os-445 minus restul os-431 (alt pas, aceleași programe în afara
  înmulțirilor, aceeași treaptă 64 KiB). Formele transpuse (TA / TB) pot merge altfel decât qkv-513; modelul nu le deosebește.
- Podeaua lansării duble e un plafon (împachetare perfectă), nu o predicție.

### Cererea pentru OS (opțională, nimic nou pentru placă)

Profilul pe lansări al pasului mixt: OS-ul scrie deja câte un record după fiecare lansare și citește pașii întregi pentru pasul F32
(`StepProfile::from_launches`); pentru lista mixtă numele vin acum din `StepProfile::from_mixed(&lista, &alegeri)` (alegerile = al
treilea rezultat al lui `pack_mixed_fast`). O rulare `--mixt-513 --mixt-513-rapid` cu profilul pornit ar înlocui DEDUS-ul de aici cu
MĂSURAT. Nota: `integration/cyber-universe/C-157_P2zl_profil-pas-mixt.md`.

**Ce e nou pentru placă:** nimic (aceleași programe, aceeași listă; doar numele pentru profil). **Nicio schimbare incompatibilă.**
