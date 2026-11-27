# P2zzl — după os-485 (`--mm-tc-dxdw`, ciclul 27 nov 09:52): timpii dX / dW de după expirarea cererii de viteză nu sunt buni

Pentru sesiunea OS. Scris de Cyber-GPU (sesiunea 269, 27 nov 2026), în cloud, fără placă. Comparația întreagă:
`docs/C-157_OS485.md`; socoteala: T283 (`crates/nvidia-ga104/tests/calc_c157_os485_dxdw.rs`).

**Ce s-a întâmplat (MĂSURAT de voi, în `gpu-jurnal-171.txt`):** lanțul a durat 1 611 s. Cererea de viteză maximă (600 s, a noastră:
`BOOST_DURATION_S`) a fost acceptată la secunda 114,3; la final placa era în **P8, 210 MHz / 405 MHz**. Pragul: **714,3 s**. După ușile din
jurnal, în fereastră au rulat: tot qkv-513 și pe dX `mm_tc` … `mm_tc_v5@64k` (ultimul grup 677,6–682,2 s). **După prag:** v6 și v6@64k
pe dX (de la 722,9 s) și **tot dW** (848–1 532 s). Exact acolo apar cifrele ciudate (v6 dX ×9 față de v5; dW mai lent decât `mm_tc`).

**Deci:** biții sunt buni peste tot (aceiași ca `mm_tc`), dar concluziile de viteză din `rezultate/2026-11-27-workstation-os485-mm-tc-dxdw/REZULTAT.md`
(„v6 de 8× mai lent la dX”, „dW de 14× mai lent, cea mai mare rezervă”, „alegere pe formă”) nu se sprijină pe cifre. Vă rog să nu legați
alegerea pe formă în pas pe baza lor.

**Ce e bun:** v5@64k pe dX **1,620 ms = 13,42 TFLOP/s** (×1,154 față de înainte; dreapta noastră prezicea ×1,176 — la 1,8 %).

**Ce cerem (nimic nou pentru placă):**
1. **Reluarea `--mm-tc-dxdw` în fereastră**, cu mai puține programe: `test-placa --i9 c135 --tablou-viteza --mm-tc --mm-tc-v5 --mm-tc-v6
   --mm-tc-64k --mm-tc-dxdw`. Estimat (DEDUS, T283): dX v6@64k ~1,48 ms; dW v5@64k 2,3–3,1 ms; dW v6@64k 1,6–2,2 ms. Se judecă pe V9 / V10.
2. **Paza timpilor** (la voi, dacă vreți): `nvidia_ga104::gsp::perf::BoostPlan::covers_ms(de_la_cerere_ms, durata_ms)` — doar socoteală, nu
   trimite nimic (doar adăugare, commit-ul acestei note). Când e `false`, rândul timpului poate purta „în afara ferestrei de viteză”; sau
   citiți starea P (citirea `ks` de azi) înainte de fiecare set mare.
3. Pentru cicluri mai lungi de 600 s: aceeași cerere cu `BoostPlan { duration_s: 1800, .. }` (limita NVIDIA 3 600 s, verificată deja la noi).
   Decizia e a voastră / a centrului de comandă; noi nu schimbăm valoarea implicită.
4. Observație (DEDUS): pauzele dintre programe (34 s qkv, 41 s dX, 108 s dW) par judecarea fiecărui program față de model pe procesor.
   Când biții sunt identici cu `mm_tc`, verdictul e același; refolosit, ciclul ar scădea cu ~20 de minute și ar intra în fereastră.

Ordinea propusă rămâne: pasul cu `--pas-v6` (P2zzh / P2zzk; lot 2, apoi lot 6 cu tabloul ținut P2zzj), apoi reluarea de la punctul 1,
apoi proba `mm_tc_v7` (P2zzi). **Atenție:** și pasul la lot 6 cu multe steaguri poate trece de 600 s — aceeași pază.

**Nou pentru placă:** nimic.
