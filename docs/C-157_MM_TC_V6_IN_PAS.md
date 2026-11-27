# C-157 punctul 2 — `mm_tc_v6` legat în pasul mixt rapid + seturile de timp dX / dW (sesiunea 264)

Sesiunea Cyber-GPU 264, 27 nov 2026, ~05:12 (ora României), în cloud, **fără placă**. Secțiunile 1–3 sunt scrise
**înainte** de testele T277 / T278; cifrele „MĂSURAT” vin din rapoartele plăcii (os-476); restul e CALCUL sau DEDUS.

## 1. O greșeală în planul lui v6 (găsită azi, înainte de orice măsurare a lui v6)

Am numărat ce steaguri au cele 147 de înmulțiri `mm_tc` din pasul #513 (lot 2, lista reală a pasului mixt):

| Înmulțirea din pas | Steagurile reale | GFLOP (lot 2) | În T275 („socoteala”) era socotită ca |
|---|---|---|---|
| înainte (qkv, proiecția, MLP; plus capul) | **`TB` (+ `BIAS`)** | 174 + 26 | „fără steag” (256 tacți) |
| dX = dY · W | **fără steag** | 174 + 26 | „`TB`” (640 tacți) |
| dW = Xᵀ · dY | **`TA`** | 174 + 26 | „`TA`” (1 152 tacți) — corect |

Greutățile sunt ținute `[ieșire, intrare]` (ca în PyTorch), deci pasul înainte e cel cu B transpus, iar dX e fără
transpunere. **Socoteala T275 pe steaguri rămâne corectă**; doar eticheta „care înmulțire din pas” era inversată între
înainte și dX. Urmări:

- **Proba qkv-513 (`TB | BIAS`) e cazul cu B în conflict 8 la 1** (640 de tacți pe felie, ~30 % din felie), nu cazul cel mai
  ușor. Predicția V7 din plan (×1,00–1,10) a fost scrisă pe eticheta greșită. **N-o schimb** (rămâne cum a fost scrisă și
  va fi judecată ca atare); adaug mai jos V7b, scrisă azi, tot înainte de orice rulare a lui v6 pe placă.
- **V8 (pasul întreg) rămâne cum a fost scrisă:** cele trei cazuri au aceeași greutate în pas (câte o treime), indiferent de
  etichete.

## 2. Ce leg acum (doar adăugări)

- `TcPick::V6` (numele `mm_tc_v6`).
- `train_step::pack_mixed_fast_attn6(…, v5_va, v6_va, …)`: fiecare `Tc` aliniat (`mm_tc_v6::fits` = `mm_tc_v4::fits`) merge
  la `mm_tc_v6` când e încărcat, **înaintea** lui v5; altfel regula lui `attn5` (v5 → v4 → pf → tc). QMD-ul v6 = QMD-ul v5
  cu altă adresă de program (aceleași 128 de registre, 12 KiB, același bloc, aceeași grilă, același cbuf).
  `pack_mixed_fast_attn5` = `attn6` cu `v6_va = 0`.
- `batch_ram::pack_mixed_fast_via_ram_attn6`, la fel; `attn5` prin RAM = `attn6` cu `v6_va = 0`.
- `mm_tc_probe`: două seturi de timp noi, **în afara lui `SETS`** (care rămâne neschimbat): `QKV_T_SETS`, cu indicii
  `QKV_DX_SET = 6` și `QKV_DW_SET = 7`, plus `mm_tc_probe::set(i)` (0–5 = `SETS`, 6–7 = cele noi). `launch_bytes`,
  `judge_set`, rândurile și liniile raportului primesc și indicii 6–7 (înainte îi refuzau).
  - `qkv-513-dx`: m 6 144, n 768, k 2 304, fără steag (dX al qkv pentru o bucată #513), benzi la rândurile 0 / 3 072 / 6 128.
  - `qkv-513-dw`: m 2 304, n 768, k 6 144, `TA` (dW al qkv), benzi la rândurile 0 / 1 152 / 2 288.
  - aceleași 5 lansări, aceleași gărzi, același oracol (`gpu_compute::mm_tc::judge`); valorile ca la `qkv-513` (dY în
    [−2, 2), greutățile în [−0,05, 0,05)) pentru dX; pentru dW ambele operanzi sunt activări / gradienți: [−2, 2).

## 3. Predicțiile (scrise înainte de teste și de orice rulare)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| L1 | `attn6` cu `v6 = 0` | = `attn5` octet cu octet (mini și #513; și prin RAM) | 95 % |
| L2 | cu v6 încărcat | toate cele 147 `Tc` aliniate din #513 lot 2 (câte erau la v5 / v4) → v6; pf / tc neschimbate; QMD-ul v6 diferă de cel v5 doar la adresa programului (≤ 4 octeți) | 90 % |
| L3 | antrenarea mini, 5 pași | aceiași biți ca cu v5 / v4 (pierderi, gradienți, greutăți); și cu v6 fără v5 / v4 încărcate | 90 % |
| L4 | refuzuri | adresa v6 nealiniată → refuz; QMD v6 mutat pe `mm_tc_v4` (care cere 24 KiB) → refuzat de procesor | 90 % |
| L5 | seturile noi în SIMULARE | `mm_tc` și v6 dau aceiași biți pe benzi (modelul de procesor), în margine, gărzi neatinse; seturile 0–5 neschimbate (aceleași amprente de intrare) | 85 % |
| V7b | qkv-513 (`TB`) pe placă, v6 față de v5 la 64 KiB | 1,868 ms → **1,44–1,78 ms** (×1,05–1,30) | 40 % |
| V9 | qkv-513-dx (fără steag) | v6 / v5: **×1,00–1,10** | 50 % |
| V10 | qkv-513-dw (`TA`) | v6 / v5: **×1,15–1,70**; și v5 pe dW **mai lent** decât v5 pe dX (≥ ×1,15 la același număr de operații) | 45 % / 65 % |

(V7b / V9 / V10: cât din felia de ~2 108 tacți e cale de scriere în v5 — 640 / 256 / 1 152 tacți — față de 64 în v6; plafonul,
dacă scrierile ar sta toate pe drumul critic, e ×1,37 / ×1,10 / ×2,07; banda e o parte din plafon, pentru că scrierile se
suprapun parțial cu citirile din memoria plăcii.)

## 4. Ce a ieșit (după teste)

| # | Rezultat | Stare |
|---|---|---|
| §1 | T277: 147 `mm_tc` în #513 lot 2 = 49 înainte (`TB`, `TB \| BIAS`), 49 dX (fără steag), 49 dW (`TA`), câte 199,7 GFLOP | CALCUL pe lista reală |
| L1 | `attn6` cu `v6 = 0` = `attn5` octet cu octet (mini, #513; cu și fără v5; cu și fără `tc16_v4`); prin RAM la fel | PASS (SIMULARE) |
| L2 | mini 27, #513 lot 2 147 × v5 → v6; pf / tc neschimbate; zona aceeași lungime; **1 octet** diferit pe QMD (adresa); fără v5 aceleași alegeri (v4 → v6) | PASS (SIMULARE) |
| L3 | mini 5 pași cu v6 = cu v5 = cu v4, bit cu bit (2 301 760 valori; și doar cu v6, și cu `tc16_v4`); pierderea 9,7128 → 8,8200 | PASS (SIMULARE) |
| L4 | adresă v6 nealiniată → refuz; QMD v6 mutat pe `mm_tc_v4` → „QMD dă mai puțină memorie partajată decât cere programul” | PASS (SIMULARE) |
| L5 | seturile 0–5 neschimbate; 6 / 7 judecate pe benzi: modelul în margine (3 / 2 ‰), banda mutată → NU TRECE, garda scrisă → INVALID; rândurile din jurnal citite înapoi | PASS (SIMULARE) |
| V7b, V9, V10, V8 | pe placă | NOT TESTED |

Amprentele (pentru OS): `qkv-513-dx` intrări `0x9b582cee`, ieșirea modelului `0xa2b415ed`; `qkv-513-dw` intrări
`0x4ba5998b`, ieșirea modelului `0x855f3ca3`.
