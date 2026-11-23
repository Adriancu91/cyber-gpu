# C-157 punctul 2 (după os-375) — atenția `_3` legată în pas (`Plan::new_tiles3`, `Plan::new_p_only_tiles3`)

Sesiunea 187, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE**; costul e **DEDUS** (nemăsurat în pas).

## De ce
Pe placă (os-375, ciclul 15:57) cele trei programe `_3` (`attn_fwd_tile3`, `attn_back_tile0_3`, `attn_back_tile_3`) au dat
**6 / 6 seturi identice bit cu bit** cu programele vechi și timpii 1,00 / 0,83 / 1,60 ms față de 1,52 / 1,55 / 2,26 (`_2`).
Condiția scrisă în sesiunea 182 („legarea în pas după ce placa confirmă”) e îndeplinită. Azi pasul de antrenare pe plăci
(`Plan::new_tiles`, C-160 p.7) lansează încă `_2`. Pasul acesta adaugă **planuri noi** care lansează `_3`, lângă cele vechi.

## Ce se schimbă (doar adăugări)
- `train_step::Prog` primește 3 variante noi, **la sfârșit**: `AttnFwdTile3`, `AttnBackTile0_3`, `AttnBackTile3` (cod, SHA-256,
  registre, bloc, memorie partajată = ale modulelor `attn_fwd_tile3`, `attn_back_tile0_3`, `attn_back_tile_3`).
- **`PROGRAMS_ALL` rămâne cu 18 programe.** Motiv: în OS, `shell/antrenare_sim.rs` (`SimBoard::new`) trece prin `PROGRAMS_ALL` și
  întoarce eroare pentru un program fără model pe procesor; dacă cele 3 ar intra acolo, simularea OS-ului s-ar opri la
  următoarea urcare a versiunii. Nou: `PROGRAMS_ALL_3` (21 = cele 18 + cele 3, în această ordine), `PROGRAMS_TILES3` (14 = cele
  11 fără atenție + cele 3 `_3`) și `PROGRAMS_TILES3_P_ONLY` (15 = + `attn_p_tile`). `Prog::index()` se ia din `PROGRAMS_ALL_3`
  (indicii celor 18 vechi rămân aceiași); `check_programs` verifică toate cele 21 SHA-256.
- `Plan::new_tiles3(…)` = arena lui `Plan::new_tiles`; `Plan::new_p_only_tiles3(…)` = arena lui `new_p_only_tiles` (un singur P),
  plus variantele `_mv_ram` (m, v ale AdamW în RAM, ca `new_tiles_mv_ram`). În lista pasului se schimbă **doar** programul celor
  3 lansări de atenție pe strat: `inainte:atentia` → `attn_fwd_tile3`, `inapoi:atentia-dS` → `attn_back_tile0_3`,
  `inapoi:atentia-dqkv` → `attn_back_tile_3`; **același cbuf, aceeași grilă** [⌈T/64⌉, H, B]. Relansarea `inapoi:atentia-P`
  rămâne pe **`attn_p_tile`** (P-ul lui `attn_fwd_tile3` = al lui `attn_fwd_tile2` bit cu bit, T190; `attn_p_tile` = `_2`, T186).
  Cer d = C/H ≤ 64, ca `new_tiles`. Planurile vechi trimit exact ce trimiteau.
- QMD-ul poartă memoria partajată a programului `_3` (`attn_back_tile_3`: 33 024 octeți).

## Predicții (scrise acum)
1. **Lista:** `new_tiles3` are aceiași octeți de arenă ca `new_tiles`, `new_p_only_tiles3` ca `new_p_only_tiles`; același număr
   de lansări; fiecare lansare are aceeași etichetă, același strat, **același cbuf și aceeași grilă**; diferă doar programul
   celor 3·L lansări de atenție (relansările P neschimbate). `Plan::programs()` = `PROGRAMS_TILES3` / `PROGRAMS_TILES3_P_ONLY`.
2. **Pasul de aur mic, 10 pași, `sgd` și `adamw_513` (SIMULARE):** `new_tiles3` = `new_tiles` **bit cu bit**, iar verdictul față
   de PyTorch trece. Cinstit: în simulare `_3` rulează pe **același model pe procesor** ca `_2` (T188–T190 au arătat că
   emularea literală a GLSL-ului `_3` = modelul), deci egalitatea aici dovedește **legătura** (programul potrivit, cbuf, grilă,
   memorie partajată, ordinea în trimitere), nu calculul plăcii. Calculul `_3` = vechii pe placa reală e dovedit separat
   (os-375, 6 / 6 bit cu bit).
3. **P refăcut:** `new_p_only_tiles3` = `new_tiles3` bit cu bit (4 straturi, 3 pași AdamW, 3 relansări `attn_p_tile`).
4. **Amestecul:** `_3` înainte cu `_2` înapoi (și invers) dau aceiași biți — fiecare program e interschimbabil cu perechea lui.
5. **Greșeli plantate prinse:** modul 0 lansat pe `attn_back_tile_3` (iese fără să scrie dS → alți biți sau pas refuzat);
   QMD-ul lui `attn_back_tile_3` cu memoria partajată a lui `attn_back_tile_2` (mai mică cu 256 de octeți: dacă diferă) sau
   fără memorie partajată → refuzat de procesorul de comenzi; relansarea P scoasă → alți biți.
6. **Compatibilitate:** `PROGRAMS_ALL` neschimbat (18, aceiași indici), deci `SimBoard::new` din OS merge neschimbat.
7. **Costul (DEDUS, nemăsurat în pas):** pe pasul mic nu contează. Pe bucata #513 (T = 512, d = 64, 12 straturi) atenția pe pas
   ~191 ms (`_2`) → **~123 ms** (`_3`), din timpii os-375 × numărul de lansări (sesiunea 186).

## Test
`crates/nvidia-ga104/tests/sim_c160_p5_golden.rs`, testele `t195_*` (SIMULARE): lista, pasul mic 10 pași ambele rețete,
4 straturi + greșeli plantate, amestecul. Nimic nu rulează pe RTX 3070 Ti.

## Ce e nou pentru placă (când OS-ul îl leagă)
Doar cele 3 programe `_3` **în pas** (pe placă au rulat deja în proba `--atentie3`, os-375). Nimic altceva.

## Riscuri
- Pe placă pasul întreg cu `_3` n-a rulat; proba „pasul de aur mic pe placă” e o comandă separată (nu o cer aici).
- `attn_back_tile_3` are 80 de registre și 33 024 octeți partajați — 3 blocuri pe SM, ca `_2` (os-375 a rulat-o așa).

## Surse
Codul nostru (T186, T188–T191, os-375 `rezultate/2026-11-23-workstation-os375-atentie3/`); `docs/C-160_P7_PLACI_IN_PAS.md`;
`docs/C-157_P2_ATTN_TILE3.md`.

## Rezultat
(se completează după cod)
