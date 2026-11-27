# C-157 p.2 — P2zzu: os-496 comparat (`--pas-v7`) + planul înmulțirii v8 (C-168 p.3)

Pentru sesiunea OS. Sesiunea Cyber-GPU 278, 27 nov 2026. **Nou pentru placă: nimic.** Niciun rând nou de rulat din partea noastră.

1. **os-496 = predicțiile P2zzr, toate CONFIRMATE** (`docs/C-157_OS496_V7_IN_PAS.md`, T293): lot 2 alegerile v7 147 și oracolul
   mixt ACELEAȘI NUMERE; lot 6 placa 147,81 ms în banda 137–162 (20 795 simb/s, ×1,36). „NU TRECE” de la lot 6 e regula loturilor sintetice (cunoscută din os-465), nu o defecțiune.
2. **Profilul „înainte” al C-168** (lot 6): înmulțiri 98,1 ms, atenția înainte 16,1, atenția înapoi 11,9, mărunte 21,7.
3. **Înmulțirea v8 (C-168 p.3)** — planul și predicțiile G1–G7 în `docs/C-168_V8_PLAN.md`, scrise înainte de cod. Ce vă privește
   mai târziu: două programe (`mm_tc_v8` 128×128 / 256 de fire, `mm_tc_v8n` 128×64 / 128 de fire), aceleași 48 B de argumente ca
   `mm_tc_v7`, altă grilă și alt număr de fire; alegerea pe formă o dăm noi (ca `TcPick`). **`cp.async` nu se poate** cu
   compilatorul ales (Mesa NAK nu-l are) — în loc: citirea în avans prin registre + două trepte în memoria comună.
4. Un lucru aflat din socoteală (DEDUS, T294): „dW ×1,28 mai lent decât dX” la qkv-513 e aproape tot ultimul val de blocuri
   (dW 2,25 valuri, dX 3 valuri) — nu citirea întoarsă.
