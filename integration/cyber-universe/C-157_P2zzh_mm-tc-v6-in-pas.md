# C-157 p.2 — `mm_tc_v6` legat în pasul mixt rapid + două seturi de timp noi (dX, dW) în proba `mm_tc` (SIMULARE)

Sesiunea Cyber-GPU 264, 27 nov 2026, în cloud, fără placă. Detalii și predicții: `docs/C-157_MM_TC_V6_IN_PAS.md`.
Testat doar în SIMULARE (T277, T278). `mm_tc_v6` **n-a rulat încă pe placă** (proba P2zzg).

## 1. O corectură la P2zzg (important pentru citirea probei)
Proba `qkv-513` (`TB | BIAS`) **nu** e „înmulțirea fără transpunere”, cum scria nota P2zzg: în pasul #513 înainte = `TB`,
dX = fără steag, dW = `TA` (147 de lansări, câte 49 și câte o treime din lucru; T277). Deci proba qkv-513 vede deja o parte
din câștigul lui v6. Predicția din P2zzg (1,70–1,87 ms) rămâne cum a fost scrisă; predicția scrisă azi, înainte de orice
rulare: **qkv-513 cu v6 la 64 KiB: 1,44–1,78 ms** (×1,05–1,30 față de v5).

## 2. Ce e (doar adăugări, nimic redenumit)
| | |
|---|---|
| Alegerea nouă | `train_step::TcPick::V6` (numele `mm_tc_v6`) — dacă OS-ul afișează alegerile dintr-o listă (`fast_picks_text`), adăugați `TcPick::V6` în față |
| Pasul | `train_step::pack_mixed_fast_attn6(launches, prog_va, progs, v5_va, v6_va, attn, attn_fast, attn0, step_bytes, zone_va)` — `v6_va` = unde e încărcat `mm_tc_v6` (aliniat la 256; 0 = neîncărcat). Cu v6: fiecare `Tc` aliniat → v6 (înaintea lui v5); altfel regula de azi |
| Prin RAM | `batch_ram::pack_mixed_fast_via_ram_attn6(…, v5_va, v6_va, …)` |
| Compatibil | `attn5(…)` = `attn6(…, v6_va = 0, …)` octet cu octet (T277 L1) |
| QMD | = QMD-ul v5, doar adresa programului diferă (147 de octeți diferiți în toată zona #513 lot 2) |
| Proba | `mm_tc_probe::set(i)`, `QKV_T_SETS`, `QKV_DX_SET = 6` (`qkv-513-dx`: 6 144 × 768 × 2 304, fără steag), `QKV_DW_SET = 7` (`qkv-513-dw`: 2 304 × 768 × 6 144, `TA`); `launch_bytes`, `judge_set`, rândurile și liniile le primesc acum. `SETS` neschimbat (tot 6 seturi) |

## 3. Ce cerem OS-ului (când are placa; în ordine, nu în locul a ce e deja cerut)
1. Proba P2zzg (`mm_tc_v6` în `--mm-tc-64k`, după v5). **Doar dacă dă aceiași biți ca `mm_tc`** treceți la (3).
2. În aceeași probă, pe lângă qkv-513: **seturile 6 și 7** (`qkv-513-dx`, `qkv-513-dw`), fiecare cu v5 și cu v6, la 64 KiB,
   5 lansări (ca qkv-513: ~83 MB fiecare; benzile de citit: `read_ranges`). Atenție: `tp::SETS[x.set]` din OS trebuie
   înlocuit cu `tp::set(x.set)` pentru indicii 6–7.
   - **Prezis (scris înainte):** dX v6 / v5 **×1,00–1,10**; dW v6 / v5 **×1,15–1,70**; v5 pe dW **mai lent** decât v5 pe dX
     (≥ ×1,15). Aceiași biți ca modelul pe benzi (amprentele de intrare: dX `0x9b582cee`, dW `0x4ba5998b`; ieșirea modelului:
     dX `0xa2b415ed`, dW `0x855f3ca3`).
3. Pasul #513 cu `pack_mixed_fast_attn6` (v6 încărcat), întâi la lot 2, apoi lot 6 (cu paza termică de azi).
   - **Prezis (P2zzg V8, neschimbat):** față de același pas cu v5, placa ×0,885–0,979 la lot 2, ×0,867–0,976 la lot 6.

## 4. Ce e nou pentru placă
Doar programul `mm_tc_v6` (SHA `8c761fa5…`, același ca în P2zzg), folosit și în pas, și două produse noi de timp în probă (doar
date în memoria plăcii, citite înapoi). Nicio scriere în registre, niciun firmware.
