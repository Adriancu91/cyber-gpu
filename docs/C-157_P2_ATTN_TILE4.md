# C-157 punctul 2 — `attn_back_tile_4`: tăietura în buclele dK / dV ale lui `attn_back_tile_3`

## Planul și predicțiile (sesiunea 188, 23 nov 2026, scrise ÎNAINTE de cod)

**De ce aici.** Pe placă (ciclul 15:57) `attn_back_tile_3` e cel mai lung din lanțul atenției: 1,60 ms din 3,43 (47 %), iar
modelul din listare (`tools/buget-antrenare/nak_bucle.py`) îl dă **limitat de numărul de instrucțiuni** (emisie 0,43 ms). Pe bucle
(DEDUS, din listarea NAK a lui `_3`, 1,77 GHz, setul `atentie-513-b4`):

| Bucla (capul în listare) | Instrucțiuni / iterație | `ffma` | Emisie | Lanț |
|---|---|---|---|---|
| dQ, plăcile de sub diagonală (10) | 105 | 64 | 0,053 ms | 0,023 |
| dQ, placa diagonalei, cu gardă (17) | 261 | 64 | 0,038 | 0,020 |
| urcarea plăcilor dK / dV (128) | 79 | 0 | 0,051 | 0,100 |
| **dK / dV, sub diagonală (135)** | 107 | 64 | **0,108** | 0,049 |
| **dK / dV, placa diagonalei, cu gardă (142)** | **444** | 64 | **0,128** | 0,082 |

Bucla diagonalei dK / dV are de **4 ori** mai multe instrucțiuni decât cea de sub diagonală pentru același număr de `ffma`: garda
`if (i < T && t0 + r <= i)` pe fiecare din cele 4 rânduri, la fiecare din cele 4 pași desfăcuți → 16 ramuri pe iterație.

**Observația care face tăietura sigură.** Garda nu mai e necesară pe o placă **întreagă** a diagonalei: la urcare, `sa` primește deja
exact **+0** unde `t0 + r > i` (deasupra diagonalei) și `sb` primește +0 unde `i ≥ T`. Pe placa diagonalei (`it == bx`) toate rândurile
`i < T` dacă placa e întreagă (`it·64 + 64 ≤ T`). Atunci `fma(+0, b, acc)` lasă `acc` neschimbat **dacă `b` e finit și `acc` e +0 sau
nenul**. De ce `acc` e +0 acolo: placa diagonalei e **prima** din bucla `it` (care pornește de la `bx`), `acc` e pus la +0 înaintea
buclei, iar pentru coloana `r` pașii mascați sunt `ii < r` — toți **înaintea** primului pas valid (`ii = r`). Deci în pașii mascați
`acc` e încă +0, și `+0 + (±0) = +0` (rotunjire la cel mai apropiat, și cu FTZ). La dQ ordinea e inversă (placa diagonalei e **ultima**,
pașii mascați `j > t` vin **după** cei valizi, unde `acc` ar putea fi −0 dintr-o valoare subnormală negativă tăiată la zero) — de
aceea **dQ nu se atinge**.

### Schimbarea (una singură)
- **K1:** în trecerile dK / dV, condiția plăcilor fără gardă devine `it * 64 + 64 <= T` (înainte: `it > bx && it * 64 + 64 <= T`),
  adică și placa întreagă a diagonalei merge pe bucla fără gardă (aceeași ordine a sumelor: `ii0`, `u`, `a`, `k`, aceleași adrese
  `sa[ii·64 + r]`, `sb[ii·64 + tx + 16k]`). Placa parțială de la capăt (`T` nemultiplu de 64) rămâne pe bucla cu gardă.
- Restul = `attn_back_tile_3.comp`, neschimbat (argumentele, modul 1, grila, blocurile, memoria partajată, dQ, urcările, scrierile).
- Program nou **lângă** `_3` (neschimbat): `tools/nak/attn_back_tile_4.comp`, modulul `nvidia_ga104::attn_back_tile_4`, modelul
  pe procesor tot `attn_back_col_gpu`. Nelegat în pas și nepredat până nu e proba „vechi lângă nou” (pasul următor, separat).

### Predicții
- P1 — **aceiași biți** ca `attn_back_tile_3` (deci ca `attn_back_tile` și `attn_back` modul 1) pe cele 10 forme ale testelor, pentru
  intrări finite. Încredere ~90 % (riscul: un caz de semn al lui zero pe care nu l-am văzut).
- P2 — limita cinstită: cu Inf / NaN în Q sau dO, coloanele de deasupra diagonalei pot primi NaN în plus (0 · Inf). Nu schimbă nimic
  în antrenare (pasul se oprește oricum la normă nefinită); testul cu NaN deasupra diagonalei în **P / dS** trece în continuare
  (acolo nu se citește nimic).
- P3 — registre ≤ 80 (3 blocuri pe SM ca `_3`); încredere ~75 % (bucla diagonalei dispare, una dintre cele două rămâne).
- P4 — în listare: bucla cu gardă rămâne (pentru placa parțială), dar pe setul mare nu mai rulează; model emisie **0,43 → ~0,31–0,33 ms**
  (×1,3–1,4), lanțul atenției în model 3,43 → ~3,0–3,1 ms pe placă cu factorul măsurat ×3,75 → `attn_back_tile_4` **~1,1–1,3 ms**
  față de 1,60 (DEDUS; pe placă NOT TESTED).
- P5 — greșeli plantate prinse: (a) condiția fără `+ 64 <= T` (și placa parțială fără gardă) → diferă pe formele cu `T` nemultiplu
  de 64 sau trece (dacă și acolo `sa` / `sb` sunt 0 — atunci notez că și placa parțială poate merge fără gardă); (b) bucla `it`
  pornită de la `bx + 1` → dK / dV greșite.

## Rezultat în SIMULARE (sesiunea 188, după plan)

- Mesa NAK 25.0.7 reconstruit în cloud; `attn_back_tile_3` recompilat **identic** (SHA-256 `67c8d9a5…ba085a9c`), deci unealta e aceeași.
- `attn_back_tile_4`: SHA-256 `74f2cb81c2d7d69e9311b3fac4ff3ccd690da97f27e8875f516e1827c19eac35`, **80 de registre** (P3: DA),
  1 572 de instrucțiuni în listare (față de 1 579: bucla cu gardă rămâne pentru placa parțială, dar pe setul mare nu mai rulează).
- T196 (`crates/nvidia-ga104/tests/sim_attn_back_tile_4.rs`, 6 teste): **P1 DA** — aceiași biți ca `_3` și ca `attn_back` modul 1 pe
  10 forme; NaN deasupra diagonalei lui P / dS fără efect; −0 păstrat pe placa întreagă a diagonalei (T 64, 128) și la T 70.
  **P5:** P în dK, pasul C, diagonala sărită — prinse; placa parțială fără gardă **nu** se vede pe date întâmplătoare (rândurile
  i ≥ T au sa = sb = +0), se vede doar pe capul construit cu −0 — deci garda plăcii parțiale e necesară doar pentru semnul lui zero.
  Observație nouă: la `_3` alegerile 0 / valoare din urcare nu contau (T189); la `_4` contează (fără ele, alte dK / dV).
- **P4** (`nak_bucle.py`, `s188`): emisia modelului **0,426 → 0,328 ms** (×1,30; scris ~0,31–0,33 → la marginea de sus), lanțul
  0,336 → 0,267, suma 1,03 → 0,86. Cu factorul măsurat al lui `_3`: **~1,23 ms** (după maxim) până la **~1,34 ms** (după sumă) față de
  1,60 — scrisesem 1,1–1,3; estimarea „după sumă” iese puțin peste. Lanțul atenției ~3,43 → ~3,1–3,2 ms; bucata #513 ~123 → ~110–114 ms (DEDUS).
- `tools/check.sh`: **941 PASS**, 0 eșecuri. Pe placă: NOT TESTED. Nelegat în pas și în proba „vechi lângă nou” (pasul următor).
