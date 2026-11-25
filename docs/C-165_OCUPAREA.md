# C-165 / pista A — Ocuparea unităților de calcul (treapta memoriei comune din QMD)

Pornit de studiul centrului din 25 nov 07:00 (`Cyber-hub/idei/2026-11-25-centru-inmultirea-la-altii.md`). Sesiunea 224,
în cloud, **FĂRĂ placă**. Tot ce e aici e **DEDUS** sau **SIMULARE**; nimic nu e măsurat pe RTX 3070 Ti.

## 1. Părerea noastră despre citirea câmpurilor

**Da, în mare ați citit bine.** `qmd.rs` (`ComputeQmd::encode_with_shared`) pune azi:

| câmp QMD V03_00 | ce punem | pentru 8 KiB pe bloc |
|---|---|---|
| `SHARED_MEMORY_SIZE` | memoria unui bloc, rotunjită la 256 B | 8 192 |
| `MIN_SM_CONFIG_SHARED_MEM_SIZE` | treapta ≥ memoria **unui** bloc | 3 (= 8 KiB) |
| `TARGET_SM_CONFIG_SHARED_MEM_SIZE` | aceeași | 3 (= 8 KiB) |
| `MAX_SM_CONFIG_SHARED_MEM_SIZE` | treapta pentru 48 KiB | 17 (= 64 KiB) |

Codificarea (n/4 KiB + 1, trepte 8/16/32/64/96 KiB) e cea din Mesa NAK (`gv100_sm_config_smem_size`, MIT, rescrisă).
Câmpurile nu au documentație publică; înțelesul („cât din cei 128 KiB L1 ai unității devine memorie comună”) e **DEDUS**
din nume, din ghidul CUDA (capacitatea aleasă de driver ca să încapă mai multe blocuri) și din cifrele noastre.

Trei precizări:

1. **Registrele la `mm_tv_128x128`: pragul e ≤ 80, nu ≤ 85.** Registrele se dau pe grup de fire în bucăți de 256
   (= 8 pe fir), deci 85 se rotunjește la 88: 88 × 32 × 8 grupuri = 22 528 pe bloc → 65 536 / 22 528 = 2 blocuri.
   La 80: 20 480 pe bloc → 3 blocuri. (Fișa publică pentru capacitatea 8.6: 65 536 registre, 1 536 fire, 48 grupuri,
   16 blocuri pe unitate, cel mult 255 registre pe fir.)
2. **Rezerva de 1 KiB pe bloc.** Ghidul CUDA spune că pe capacitatea 8.x driverul CUDA rezervă 1 KiB de memorie comună
   pe bloc „pentru sistem”. Nu știm dacă e un lucru al plăcii sau doar al driverului CUDA. **Un indiciu:** dacă ar fi al
   plăcii și placa ar folosi strict `TARGET`, `mm_train_tile` (8 KiB + 1 = 9 KiB) n-ar încăpea în treapta de 8 KiB și n-ar
   porni deloc — dar pornește de o lună. Deci ori rezerva nu există la noi, ori placa urcă singură treapta între
   `MIN` și `MAX`. Socotim mai jos ambele variante (R0 = fără rezervă, R1 = cu 1 KiB).
3. **A doua explicație, pe care proba trebuie s-o excludă (ipoteza H2):** placa alege deja singură treapta în
   intervalul `MIN…MAX` (64 KiB), deci ocuparea e deja bună și pierderea e în altă parte. Atunci `64x64k8` ar fi mai
   rapidă din alt motiv (bucla cu 8 în loc de 16 pe felie). Proba de mai jos o deosebește de H1 (placa folosește `TARGET`)
   pentru că pune `MIN = TARGET` = treapta cerută.

Cifrele noastre se potrivesc mai bine cu H1 (`64x64k8` ×1,6 cu 2 blocuri; variantele de 12 KiB → treapta 16 KiB → 1 bloc,
fără câștig), dar asta e tot **DEDUS**.

## 2. Câte blocuri ar încăpea (DEDUS; `nvidia_ga104::occupancy`)

| program | fire | registre | memorie | 8 KiB | 16 KiB | 32 KiB | 64 KiB | 96 KiB | limita fără memorie |
|---|---|---|---|---|---|---|---|---|---|
| `mm_train_tile` | 256 | 34 (→40) | 8 KiB | **1** / 0 | 2 / 1 | 4 / 3 | 6 / 6 | 6 / 6 | 6 (fire, registre) |
| `mm_tv_64x64k8` | 256 | 34 (→40) | 4 KiB | 2 / 1 | 4 / 3 | 6 / 6 | 6 / 6 | 6 / 6 | 6 |
| `mm_tv_128x128` | 256 | 90 (→96) | 8 KiB | 1 / 0 | 2 / 1 | 2 / 2 | 2 / 2 | 2 / 2 | 2 (registre) |
| `mm_tc` | 128 | 64 | 24 KiB | 0 / 0 | 0 / 0 | 1 / 1 | 2 / 2 | 4 / 3 | 8 (registre) |

(R0 / R1. „0” = nu încape: placa ar refuza sau ar urca singură treapta.) `MAX` rămâne 64 KiB (Mesa NVK), deci
96 KiB nu intră în probă; plăcile GA10x au de fapt 100 KiB ca treaptă maximă, iar codificarea Mesa se oprește la 96 —
nefolosită de noi.

## 3. Proba ocupării — ce se rulează pe placă (prin OS)

Un singur lucru nou pentru placă: **valoarea câmpurilor `MIN`/`TARGET` din QMD** (pe care le scriem deja). Niciun registru,
nicio comandă nouă, aceleași programe, aceleași adrese, forma qkv a pasului #513.

`occupancy::PROBE` (10 rulări, în ordine): `mm_train_tile` la 8 → 16 → 32 → 64 KiB; `mm_tv_64x64k8` la 8 → 16 → 32;
`mm_tv_128x128` la 8 → 16 → 32 (32 în plus față de cererea centrului, pentru varianta R1).

### Predicții (scrise înainte de cod)

| | predicția | ce înseamnă dacă iese altfel |
|---|---|---|
| P1 | fiecare rulare dă **aceiași biți** ca martorul (`mm_train_tile` la 8 KiB) | STOP: treapta schimbă rezultatul → nu se leagă nimic |
| P2 | H1: `mm_train_tile` 64 KiB / 8 KiB ≤ 0,85 din timp (ținta ×1,5–3) | > 0,97 → H2: placa urcă deja singură; ocuparea nu e buba |
| P3 | H1: timpul scade monoton 8 → 16 → 32 KiB la `mm_train_tile`; 32 → 64 câștig mic (memoria devine frâna) | — |
| P4 | `mm_tv_64x64k8`: 8 → 16 KiB câștigă (R0: 2 → 4 blocuri) | nimic la 16, câștig la 32 → R1 |
| P5 | `mm_tv_128x128`: 8 → 16 câștigă sub R0; sub R1 abia la 32 | aceeași deosebire R0/R1 |
| P6 | puterea plăcii crește cu ocuparea (azi 123–133 W) | dacă timpul scade fără putere mai mare → altă cauză |

Regula verdictului: P1 întâi (biții). H1 confirmată = P2 adevărată. R0 / R1 = care dintre P4/P5 se potrivește.

## 4. Verificările T230 (SIMULARE; scrise înainte de cod)

- **T230a** `occupancy::blocks_per_sm` dă tabelul din §2 (R0 și R1), cu limita numită (fire / registre / memorie / blocuri).
- **T230b** `ComputeQmd::encode_with_sm_config` (doar adăugare) pune `MIN = TARGET` = treapta cerută, `MAX` ca azi, și
  **toate celelalte câmpuri identice** cu `encode_with_shared`; la treapta 8 KiB pentru 8 KiB = exact octeții de azi.
- **T230c** refuză: treaptă care nu e 8/16/32/64 KiB, treaptă mai mică decât memoria blocului, treaptă peste `MAX`.
- **T230d** `occupancy::best_step` = cea mai mică treaptă la care **fiecare** program din listă ajunge la limita lui fără
  memorie (cel mult 64 KiB): `[mm_train_tile]` → 64 KiB sub R0 și R1; `[mm_tv_128x128]` → 16 KiB (R0) / 32 KiB (R1).
- **T230e** în modelul procesorului de comenzi, cele 10 rulări din `PROBE` dau **aceiași biți** ca martorul (modelul nu
  are ocupare: verifică doar că un QMD cu altă treaptă e citit corect și nu schimbă rezultatul).
- **T230f** modelul refuză un QMD cu `TARGET` mai mic decât memoria programului sau `MIN > TARGET` / `TARGET > MAX`
  (lansarea nu pornește, C rămâne neatins), și ține lista treptelor văzute (`CommandProcessor::sm_configs`).

## 5. Ce urmează (după proba pe placă)

- Dacă H1: `Plan` cu o singură treaptă pe tot pasul (`best_step` peste toate programele pasului) — schimbarea treptei
  între lansări probabil cere unitatea goală, de evitat între cele ~480 de lansări; apoi dieta de registre la
  `128x128` (≤ 80) și `mm_tc` cu mai puțină memorie pe bloc.
- Dacă H2: căutăm pierderea în altă parte (citiri de 128 de biți, bucla K desfăcută — punctul 5 al studiului).

## Surse

- Ghidul de programare CUDA (capacitatea memoriei comune, ocuparea, tabelul capacităților 8.6) — doar ideile.
- NVIDIA Ampere GPU Architecture Tuning Guide (memoria comună pe 8.6, rezerva pe bloc) — doar ideile.
- Mesa NAK `qmd.rs` (MIT): codificarea treptei; deja notată în `docs/PROVENANCE.md`.
- Studiul centrului: `Cyber-hub/idei/2026-11-25-centru-inmultirea-la-altii.md`.
