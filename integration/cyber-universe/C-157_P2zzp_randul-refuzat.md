# P2zzp — rândul P2zzl refuzat la 13:28 (greșeala noastră) și ce încape în fereastra de viteză maximă

Sesiunea Cyber-GPU 273, 27 nov 2026, ~14:15–15:00 (ora României), în cloud, FĂRĂ placă. Socoteala: T287 (DEDUS pe orarul
MĂSURAT al lui os-485, `crates/nvidia-ga104/tests/calc_c157_fereastra_randuri.rs`). **Scrisă ÎNAINTE de rezultatul rulării
repornite la 14:08.** Nou pentru placă: **nimic**.

## 1. Greșeala

Rândul din P2zzl punctul 1 (`--mm-tc --mm-tc-v5 --mm-tc-v6 --mm-tc-64k --mm-tc-dxdw`) nu e o formă pe care OS-ul o acceptă:
`--mm-tc-v5` cere `--mm-tc-v4` imediat înainte (P6-242, `ciclu.rs`). L-am scris fără să-l trec prin regulile lui `ciclu.rs`.
Nimic nu s-a scris în placă (`ciclu-auto-20261127-1328-refuzat.txt`). Rândul e corectat în P2zzl și în `docs/C-157_OS485.md`.
De acum, orice rând propus de noi e verificat întâi pe testele `board_command` din `ciclu.rs` (doar citire).

## 2. Ce va ieși din rularea repornită la 14:08 (predicție, DEDUS)

Rândul centrului: `test-placa --tablou-viteza --mm-tc --mm-tc-v4 --mm-tc-v5 --mm-tc-v6 --mm-tc-64k --mm-tc-dxdw` = **aceleași 8
programe ca os-485**, cu cererea de 600 s. Pe orarul lui os-485 (cererea la ~114 s, expirarea la ~714 s):

| Setul | În fereastră | În afara ferestrei |
|---|---|---|
| qkv-513 | toate 8 | — |
| qkv-513-dx | `mm_tc`, `@64k`, v4, v4@64k, v5, v5@64k | **v6, v6@64k** |
| qkv-513-dw | — | **toate 8** |

Deci paza din os-489 ar trebui să scrie `IN AFARA FERESTREI` la **10** seturi de timpi (2 dX + 8 dW) și lanțul să ia ~1 600 s.
Biții rămân de judecat (ei nu depind de ceas); **despre viteza lui v6 pe dX și despre dW rularea nu ne va spune nimic nou.**
Dacă ies mai puțin de 10, modelul orarului e greșit și îl refac pe cifrele noi.

## 3. Ce rând încape în 600 s — niciunul care să măsoare dW v6@64k

| Rândul | Programe pe set | Sfârșitul estimat | dW v6@64k în 600 s? |
|---|---|---|---|
| 14:08 (v4, v5, v6) | 8 | ~1 533 s | nu (toate dW afară) |
| scurt, doar din Release-ul cu P6-251: `--mm-tc --mm-tc-v6 --mm-tc-64k --mm-tc-dxdw` | 4 | **~747 s** | **nu (~33 s după)**; dW v6 simplu da |
| P2zzm (v6, v7) cu `--viteza-1800` | 6 | ~1 140 s | da, cu ~770 s rezervă |
| 4 însoțitori (v4–v7) cu `--viteza-1800` | 10 | ~1 926 s | **nu** (nici 1 800 s nu ajung) |

Partea grea e judecarea pe procesor după fiecare program (~38 s la qkv, ~45 s la dX, **~112 s la dW**), nu placa (5 lansări ≈ 5 s).

## 4. Ce propun

1. **Singura cale curată:** rândul P2zzm cu `--viteza-1800` (P6-251 e gata în OS), după „da”-ul lui Adrian în `DECIZII.md`.
   Până atunci nu mai rulați `--mm-tc-dxdw` la 600 s: nu aduce cifre noi.
2. Fără „da”: rândul scurt din tabelul de mai sus, după Release-ul cu P6-251 — măsoară dX v6, dX v6@64k și dW v6 la viteză;
   lipsește doar dW v6@64k.
3. (Pentru OS, opțional, cum era în P2zzl punctul 4) când biții unui program sunt identici cu ai lui `mm_tc`, verdictul față de
   model e același; refolosit, ar tăia cele ~112 s pe dW și atunci și rândul scurt ar încăpea în 600 s (DEDUS: chiar dacă dW ar costa cât dX, ~45 s pe program, sfârșitul ar fi ~615 s, sub 714 s).

## 5. Rezultatul (ciclul 14:42, `ciclu-auto-20261127-1442.txt`, MĂSURAT de OS) — predicția din §2 CONFIRMATĂ

- Paza os-489 a scris `IN AFARA FERESTREI` la **10** seturi, **exact cele prezise**: dX v6 (617,3 s după cerere), dX v6@64k
  (662,9 s) și toate 8 pe dW (741,7–1 426,2 s); lanțul a rulat **1 597 s** după cerere. Paza merge cum trebuie.
- Biții: aceiași ca `mm_tc` peste tot (7 / 7 pe fiecare program, `mm_tc` 8 / 8 în margine) — CONFIRMAT, a doua oară.
- Ce e în fereastră repetă os-485: v5@64k pe dX **1,617 ms** (os-485: 1,620), v6@64k pe qkv **1,260 ms** (1,260).
- Modelul orarului (T287) e cu 8–14 s mai devreme decât placa, la fiecare program (aceeași direcție) → rândul scurt din §3
  s-ar termina chiar mai târziu de ~747 s: concluzia din §3 rămâne.
- **Nimic nou despre dW sau despre v6 pe dX.** Rămâne cum scrie în §4: rândul P2zzm cu `--viteza-1800`, după „da”-ul lui Adrian.
