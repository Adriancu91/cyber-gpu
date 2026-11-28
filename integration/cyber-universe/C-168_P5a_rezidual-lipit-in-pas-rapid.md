# C-168 p.5 — P5a: reziduurile înapoi lipite (`ln_back_res`) în pasul #513 rapid (testat doar în SIMULARE)

Sesiunea Cyber-GPU 290, 28 nov 2026, în cloud, fără placă. Planul, inventarul lansărilor mărunte din os-503 și predicțiile
(scrise înainte de cod): `docs/C-168_P5_LIPIRE_PLAN.md` §1–§4; rezultatul: §6. Testul: T306
(`crates/nvidia-ga104/tests/sim_c168_p5a_ln_back_res_mixed.rs`).

## Ce e nou
**Nimic în bibliotecă** — doar o combinație nouă a lucrurilor existente:
- `Plan::with_fused_back()` (din sesiunea 192) merge și pe planul pasului rapid (`new_tiles3 … .with_mm_tile().with_colsum_split()`):
  `step_mixed` dă atunci, pe fiecare strat, `ln_back_res` (o lansare `Std`) în locul lui `ln_back` + `vadd` la „ln2” și „ln1”;
- `pack_mixed_fast_attn9` / `pack_mixed_fast_via_ram_attn9` împachetează lansările `Std` după `prog_va`, ca pe toate celelalte.
- Programul `ln_back_res` e cel care **a rulat deja pe placă** în os-415 (`--lipit-placi`, oracolul trecut).

## Ce cerem OS-ului
1. O opțiune nouă (de ex. `--pas-rezidual-lipit`, după `--pas-atentie-fwd-tc` dacă o legați întâi, altfel după `--pas-v8`): planul
   real și cel de probă cu `.with_fused_back()` în plus; încărcați în plus `plan.fused_programs()` (= `[ln_back_res]`, SHA-256 verificat
   ca la os-415) și verificați la probă că programele / `fused_programs()` sunt aceleași ca la planul real (ca în `gpu_i14.rs:8627`).
   Atenție: packerul **nu** verifică dacă `ln_back_res` e încărcat (un `prog_va` greșit nu e refuzat de el) — verificarea e a voastră.
2. **Întâi lot 2 cu oracolul**, apoi lot 6 (același rând cu `--lot-513 6`).
3. **Așteptat (scris înainte de placă, DEDUS):**
   - I2: **24 de lansări mai puțin pe pas** (profilul de azi: 491 → 467), `vadd` 48 → 24, `ln_back` 25 → 1, `ln_back_res` 24;
     alegerile înmulțirilor neschimbate (lot 6 v7 13 / v8 0 / v8n 134 cu gardă; lot 2 v7 73 / v8 24 / v8n 50);
   - I4: oracolul lot 2 **ACELEAȘI NUMERE** ca rândul fără lipire (în SIMULARE sunt aceiași biți);
   - I3: `vadd` în profil 2,48 → ~1,24 ms; câștig net pe pas **0,5–1,1 ms la lot 6** (0,2–0,4 ms la lot 2). Sub 0,3 ms la lot 6 =
     RATAT (ne trimiteți liniile `ln_back_res` / `vadd` / `ln_back` din profil).

## Ce arată SIMULAREA (T306) și ce nu
- S1: modelul mic aliniat, 5 pași rapizi (`attn9`, v7 / v8 / v8n, atenția pe circuite în ambele sensuri), cu și fără colsum împărțit:
  pierderile fiecărui pas, gradientul pasului 1 și greutățile de la sfârșit **aceiași biți** cu și fără `ln_back_res`
  (76 → 72 și 94 → 90 lansări pe pas).
- S2: #513 lot 2 și lot 6: 484 → 460 lansări (fără cele 7 copieri ale lotului), exact 24 mai puțin; restul listei identic și în
  aceeași ordine; alegerile înmulțirilor și `attn_fwd_tc` identice, în ordine; fiecare QMD `ln_back_res` arată spre program, cu
  registrele lui.
- S3: programul în plus față de azi = doar `ln_back_res`.
- Nimic din toate acestea n-a rulat pe placă în pasul rapid.

## Ce e nou pentru placă
Doar folosirea lui `ln_back_res` (binarul din os-415) în pasul rapid. Nicio scriere în registre, niciun firmware, nicio schimbare
incompatibilă.
