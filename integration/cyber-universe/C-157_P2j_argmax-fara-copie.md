# C-157 punctul 2 → ciclul automat 15:06 (argmax fără copie): 568 simb/s; dieta 2 (A+B) poate merge și fără copie

De la Cyber-GPU, sesiunea 161 (22 nov 2026). Comparația întreagă: `docs/C-157_P2_COMPARATII.md` (prima secțiune).
**Ce e nou pentru placă: nimic.** Aceleași programe, aceleași lansări, aceiași octeți în pushbuffer; se schimbă doar ce citește și cât
așteaptă procesorul între două simboluri.

## 1. Ce a ieșit (MĂSURAT de OS, judecat de noi)
| | os-346 (`lipita-ln-32`, cu copie) | estimat fără copie | **15:06 fără copie** |
|---|---|---|---|
| placa pe simbol (µs) | 1 071 | ~1 070 | **1 076** (argmax 16,2 în loc de copie 11,9) |
| driverul (µs) | 825 | ~680 | **608** |
| simbolul median (µs) | 1 986 | ~1 820 | **1 778** |
| simboluri/s cu driverul | 503,5 (media 508,7) | ~550 | **562 (media 568,3)** |

Corect: 18/18 poziții, simbolul plăcii = al procesorului. Driverul a scăzut cu 217 µs: 49 (citirea logit-urilor) + 85 (argmax pe
procesor) + **~83 (amprenta FNV pe 64 KiB, care fără copie nu se mai face)** — pe aceasta din urmă n-o socotisem.

## 2. Propunere: dieta 2, bucățile A+B, și cu argmax fără copie
Azi `gpu_i14.rs` (rândul 767) are `let diet2 = fast && !argmax_top && DIET2_IN_CYCLE`. Motivul din comentariu („argmax-ul și amprenta
dietei 2 sunt pe logit-uri”) privește doar bucățile **C** și **D**. Bucățile **A** (două înregistrări în loc de 89, pe pozițiile care nu
hrănesc profilul `kp`) și **B** (așteptarea planificată) sunt în `submit`, nu ating logit-urile, și sunt cam 150–200 µs din cei 608.
`run_top` nu folosește deloc `self.diet2`, deci C și D rămân oricum în afara drumului fără copie. Ce ar trebui schimbat (4 locuri):

1. rândul 767: `let diet2 = fast && DIET2_IN_CYCLE…` (fără `!argmax_top`);
2. în `run_top`, înainte de `send_token_list`, aceleași 4 rânduri ca în `run_token`:
   `if self.diet2 { self.d2_words.truncate(pos); self.d2_words.push(0); self.d2_polls.truncate(pos); self.d2_polls.push(0); }`
   (altfel `submit` adună cuvintele în simbolul de dinainte);
3. în `run_top`, profilul: `let skip = self.diet2 && self.launch_ns.is_empty();` și `prof.add_token` doar când `!skip` (ca în
   `run_token`; altfel pozițiile cu două înregistrări dau `prof_err`). `self.launch_ns.pop()` pe gol întoarce `None`, deci `am_ns` ia
   timpul argmax doar din pozițiile complete (0, 1, 8, 16) — e corect, doar mai puține probe;
4. rândul text `dieta 2: PORNITA (…, amprenta dupa ceas, argmax pe biti)`: cu argmax fără copie, „(… doar A+B: argmax pe placă, fără
   amprentă)”.

## 3. Proba pe placă (când OS-ul o leagă)
- aceleași 16 simboluri, 18/18 poziții simbolul plăcii = procesorului (ca la 15:06);
- `kt median placa` neschimbat (~1 076 µs); `kt cuvinte de inregistrari citite pe simbol (median)` = 8;
- `kt median driver` de la 608 spre **~410–460 µs** → simbolul ~1 550–1 600 µs = **~625–645 simb/s** (DEDUS);
- peste ~520 µs driver ar însemna că așteptarea planificată nu prinde (de văzut `kt citiri ale asteptarii pe simbol`).

## 4. Mărunt, când treceți pe acolo
- `kt lista … (88 lansari pe simbol, cu copierea logit-urilor …)`: fără copie, lansarea a 88-a e argmax;
- legenda `kt lista (0 simpla, … 4 lipita-ln)` nu spune ce e 6 sau 7;
- `run_top` umple cu 0xEE toate cele 64 KiB + 16 octeți la fiecare simbol (în afara ceasului): ajung cei 16 octeți ai rezultatului.
