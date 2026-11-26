# C-157: `token32` cu G mare — treapta SM aleasă după G + proba barierei largă (sesiunea 240, 26 nov 2026, cloud, FĂRĂ placă)

**Stare:** cod (doar adăugări) + SIMULARE (T249, 4/4). **NU RULAT pe placă.** Nimic nou trimis la placă; nicio comandă nouă pentru OS
încă (vezi §5).

## 1. De ce

Nota `C-157_TOKEN32.md` §5 a lăsat deschis: `token32_qmd` cere treapta cea mai mică a memoriei comune (8 KiB), unde încap doar 2–3
grupuri `token32` pe SM; câștigul de viteză cere G = 384–768, adică treapta 64 KiB. Fără treaptă potrivită, un G mare ar însemna
grupuri care nu pornesc deodată → bariera ar aștepta la nesfârșit (blocare pe placă). Pasul (b) al ordinii propuse (aceeași probă
a barierei la G = 384 și 768) avea nevoie și el de o funcție care să permită 8 și 16 grupuri pe SM (`groups_for` permite doar 1–2).

## 2. Ce s-a adăugat (doar adăugări, nimic redenumit)

- `nvidia_ga104::kronos_token`:
  - `TOKEN32_KERNEL` (32 de fire, 92 de registre, 2 176 B) pentru aritmetica ocupării (`occupancy::SM86`);
  - `token32_max_per_sm()` = 16 (limita de blocuri a SM-ului, la treapta 64 KiB);
  - `token32_step_for(G, SM-uri)`: cea mai mică treaptă 8/16/32/64 KiB la care **toate** cele G grupuri încap deodată, socotit
    cu rezerva de 1 KiB pe bloc (citirea mai strictă): 2 / 5 / 10 / 16 pe SM; refuză G = 0, 0 SM-uri și G > 16 × SM-uri;
  - `token32_groups_for(GrShape, pe_SM)`: 1..=16 pe SM, altfel refuz;
  - `token32_qmd_wide(prog, cbuf, G, SM-uri)`: QMD-ul lui `token32_qmd` cu `MIN = TARGET =` treapta aleasă
    (`ComputeQmd::to_bytes_with_sm_config`).
- `nvidia_ga104::grid_barrier`: `MAX_PER_SM_WIDE` = 16, `groups_for_wide(GrShape, 8 | 16)` (G = 384 / 768 pe 3070 Ti). Programul
  barierei (19 registre, fără memorie comună) nu are nevoie de altă treaptă: 16 pe SM încap la oricare.
- Testul de lucru `common::token_sim::TokenSim::token32_sms` (opțiune; implicit oprită → `token32_qmd` ca înainte).

## 3. Predicțiile (scrise înainte de rularea testului)

| | Predicție | Ieșit |
|---|---|---|
| P1 | pe 48 de SM-uri: G ≤ 96 → 8 KiB; 97–240 → 16 KiB; 241–480 → 32 KiB; 481–768 → 64 KiB; 769 → refuz. Deci G = 384 → **32 KiB**, G = 768 → **64 KiB** | **DA** (T249a) |
| P2 | `token32_qmd_wide` la G ≤ 96 = exact octeții lui `token32_qmd` | **DA** (T249b; la G = 384 octeții diferă, cum trebuie) |
| P3 | SIMULARE: un simbol din lista de azi + argmax cu `token32` la G = 384 și 768 (QMD lărgit) = cele 25 de lansări separate, aceiași biți ai logit-urilor și aceleași 4 cuvinte argmax; contorul barierei = (n − 1) · G; modelul vede treapta în QMD: TARGET = 9 (32 KiB) la G = 384 și 17 (64 KiB) la G = 768 | **DA** (T249c: 2 poziții × 2 G, modelul mic) |
| P4 | SIMULARE: proba barierei la G = 384 și 768 (R = 1 și 1 001) → TRECE; `groups_for_wide` refuză 0, 1, 2, 3, 17 pe SM | **DA** (T249d) |

## 4. Rezultate

Toate cele 4 predicții: **DA**, în SIMULARE (`crates/nvidia-ga104/tests/sim_c157_token32_wide.rs`, T249a–d, 4/4). Ce spune
asta: driverul alege corect treapta pentru G, QMD-ul o poartă (modelul o citește: MIN = TARGET = 9 la G = 384, 17 la G = 768), iar
programul pe simbol dă aceiași biți cu 384 / 768 de grupuri ca lansările separate. Ce NU spune: nimic despre placă (§5).

## 5. Ce NU e dovedit și ce urmează pe placă

- Că placa ține într-adevăr 10 / 16 blocuri `token32` pe SM la treptele 32 / 64 KiB e **DEDUS** (C-165: proba ocupării a arătat
  treapta 64 KiB folosită de `mm_tc*`, os-443, dar nu un program cu barieră pe toată placa).
- Ordinea rămâne: (a) os-451, proba barierei la G = 48 / 96; (b) doar dacă trece: aceeași probă cu `groups_for_wide` (8 și 16 pe SM),
  o opțiune nouă în OS (de ex. `--bariera-larga`), cerută prin hub când (a) are rezultat; (c) `token32` cu G din (b) și
  `token32_qmd_wide`, sub o opțiune oprită implicit, comparat cu lista de azi (aceleași 16 simboluri) și cu timpul de azi.
- Nicio PREDARE către OS în această sesiune: (b) nu are rost înainte ca (a) să treacă.
