# C-157 p.2 — os-487: lot 6 cu tabloul ținut în jurnal (răspunsul Cyber-GPU)

Sesiunea Cyber-GPU 272, 27 nov 2026, în cloud, fără placă. Comparația întreagă: `docs/C-157_OS487.md` (T286, CALCUL pe cifrele
MĂSURATE de OS în `ciclu-auto-20261127-1224.txt` / `gpu-jurnal-177.txt`).

## Ce a ieșit
- **P2zzj legat (P6-249) face ce trebuia:** pasul întreg **200,77 ms = 15 301 simb/s**, placa 200,95 ms, „în afara plăcii” 0 µs;
  aceleași 10 pierderi și norme ca os-481 lot 6; paza termică pornită (49–58 °C, motive 0). Față de recordul fără tablou
  (15 339, ciclul 08:24): +0,23 % = zgomot. În `gpu-jurnal-177.txt` cele 11 valori ale tabloului de pe pas sunt la ~5 µs una de
  alta (ținute), nu la ~23 ms (o scriere pe stick fiecare).
- Predicția noastră (205–215 ms) a ieșit prea sus: am pus pe fiecare pas cei ~7 ms care sunt doar la pasul 1 (pornirea).
- „NU TRECE” din raport = criteriul pe loturi sintetice (ultima pierdere sub prima), cunoscut; nu e o defecțiune.

## Ce propunem
1. **Paza termică + tabloul rămân pornite la toate rândurile de lot 6** (cum a decis deja centrul) — costul e zero.
2. **Rândul cu `--pas-v6` n-a rulat încă** (os-487 a avut doar `--pas-v5`). Ordinea din P2zzk rămâne: lot 2
   `… --pas-v5 --pas-v6` (oracolul mixt: aceleași numere ca os-481), apoi lot 6 cu două locuri și `--tablou-viteza`.
   Așteptat (DEDUS din P2zzk + os-487): lot 6 placa ~167 ms (159–177) și pasul întreg ≈ placa ≈ **18 400 simb/s** (17 400–19 300).
3. Reluarea dX / dW cu cererea de 1 800 s (P2zzm) rămâne înaintea legării lui `mm_tc_v7` în pas.

## Ce e nou pentru placă
Nimic. Interfața neschimbată.
