# C-160 punctul 7 — programele „pe plăci” legate în pas (`Plan::new_tiles`, `Plan::new_p_only_tiles`)

Sesiunea 177, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE**; costul e **DEDUS** (nemăsurat).

## De ce
Pasul de antrenare (`train_step::Plan::step`) lansează azi atenția cu programele vechi (`attn_fwd`, `attn_back` cu blocuri de
32, un fir pe rând). La os-358 programele „pe plăci” au ieșit corecte pe placă, iar a doua generație (`attn_*_2`, același
calcul, bucle mai scurte) e în proba „vechi lângă nou” (os-364). Pentru pasul #513 (T = 512) contează doar programele pe
plăci; până acum nu exista niciun plan care să le lanseze în pas. Pasul acesta le leagă — **ca plan nou, lângă cele vechi**.

## Ce se schimbă (doar adăugări)
- `train_step::Prog` primește 4 variante noi, **la sfârșit**: `AttnFwdTile2`, `AttnBackTile0_2`, `AttnBackTile2`, `AttnPTile`
  (cod, SHA-256, registre, bloc, memorie partajată = ale modulelor `attn_fwd_tile2`, `attn_back_tile0_2`, `attn_back_tile_2`,
  `attn_p_tile`). OS-ul are deja brațul `_ => 0` în `gpu_i10.rs`, deci compilează neschimbat.
- `PROGRAMS` (13) și `PROGRAMS_P_ONLY` (14) rămân la fel. Nou: `PROGRAMS_ALL` (18 = cele 14 + cele 4, în această ordine, deci
  `Prog::index()` al celor vechi rămâne același), `PROGRAMS_TILES` (14 = cele 11 fără atenție + cele 3 pe plăci) și
  `PROGRAMS_TILES_P_ONLY` (15 = + `attn_p_tile`). `check_programs` verifică toate cele 18 SHA-256.
- `Plan::new_tiles(…)` = arena lui `Plan::new` (P păstrat pe strat); `Plan::new_p_only_tiles(…)` = arena lui `new_p_only`
  (un singur P). În lista pasului se schimbă **doar** lansările atenției: `inainte:atentia` → `attn_fwd_tile2`,
  `inapoi:atentia-dS` (modul 0) → `attn_back_tile0_2`, `inapoi:atentia-dqkv` (modul 1) → `attn_back_tile_2`, relansarea
  `inapoi:atentia-P` → `attn_p_tile`; **același cbuf**, grila [⌈T/64⌉, H, B] a programului pe plăci. Cer d = C/H ≤ 64 (altfel
  constructorul refuză). `Plan::new`, `new_p_recalc`, `new_p_only` trimit exact ce trimiteau. `Plan::programs()` dă lista
  planului.
- QMD-ul lansărilor pe plăci poartă memoria partajată a programului (`Prog::shared`).

## Predicții (scrise acum)
1. **Lista:** `new_tiles` are aceiași octeți de arenă ca `new`, `new_p_only_tiles` ca `new_p_only`; același număr de lansări
   (73 `sgd` / 75 `adamw_513` pe pasul mic; + 1 relansare la `p_only_tiles`); fiecare lansare are aceeași etichetă, același
   strat și **același cbuf** ca în planul vechi corespunzător; diferă doar programul și grila celor 3 (4) lansări de atenție.
2. **Pasul de aur mic, 10 pași, `sgd` și `adamw_513` (SIMULARE):** verdictul față de PyTorch **trece** (aceleași margini).
   **Nu** bit cu bit cu `Plan::new`: T = 32, iar rândurile 16…31 au suma în 16 parțiale (T173). Diferența față de `Plan::new`:
   erorile pe pas relativ ≤ 1·10⁻⁶, greutățile după 10 pași în normă relativă ≤ 1·10⁻⁵.
3. **„Înapoi” pe plăci = „înapoi” vechi, bit cu bit:** cu P-ul pe plăci, aceeași listă dar cu `attn_back` vechi la cele două
   lansări „înapoi” dă **exact** aceiași biți (T174 / T175 / T182 / T183 o spun program cu program; aici în pasul întreg).
4. **P refăcut:** `new_p_only_tiles` = `new_tiles` **bit cu bit** (10 pași, ambele rețete; 4 straturi, 3 pași AdamW, 3 relansări
   `attn_p_tile`).
5. **Greșeli plantate prinse:** relansarea scoasă; relansarea pe `qkv`-ul altui strat; modul 0 lansat pe `attn_back_tile_2`
   (programul iese fără să scrie, dS rămâne cel vechi); QMD-ul unei lansări pe plăci fără memoria partajată (refuzat).
6. **Costul (DEDUS, nemăsurat):** pe pasul mic nu contează. La #513 (T = 512, d = 64) atenția trece de la programele de 32 la cele
   pe plăci; cât câștigă se știe doar după os-364 (`attn_*_2` pe placă) — nu dau cifră nouă aici.

## Ce NU face
Nu schimbă niciun program, nicio interfață veche, proba `attn_tile_probe`, pasul mixt. **Nimic nou pentru placă:** cele 4
programe sunt deja predate (`attn_*_2`, os-364) sau vin după os-364 (`attn_p_tile`); planul nou se folosește pe placă doar după
ce os-364 arată biții lui `attn_*_2`, și doar printr-o comandă nouă.
