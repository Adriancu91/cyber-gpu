# C-168 p.3 — P3b: `mm_tc_v8` / `mm_tc_v8n` legate în pasul mixt rapid (doar SIMULARE; **nu rulați în pas înainte de proba P3a**)

Sesiunea Cyber-GPU 280, 27 nov 2026. Detalii: `docs/C-168_V8_PLAN.md` §8, testul T297.

**Ordinea pe placă:** întâi proba `--mm-tc-v8` (nota P3a). Doar dacă acolo v8 și v8n dau aceiași biți ca `mm_tc` (G4), facem noi
o notă nouă cu vitezele măsurate (`Rates`) și rândul pentru pas. Până atunci, ce e mai jos e doar ca să-l puteți lega din timp.

Ce e nou (doar adăugări, nimic redenumit):
- `nvidia_ga104::train_step::pack_mixed_fast_attn8(…, v7_va, v8: Option<V8Programs>, …)` și
  `nvidia_ga104::batch_ram::pack_mixed_fast_via_ram_attn8` — argumentele lui `attn7` + `v8` după `v7_va`. Cu `None` = `attn7` octet
  cu octet.
- `V8Programs { v8, v8n, sms, rates }`: adresele celor două programe (256-aligned, nenule), SM-urile active (48) și
  `nvidia_ga104::mm_tc_v8::RATES_PLAN` (DEDUS) până vin vitezele probei.
- `TcPick::V8` („mm_tc_v8”) și `TcPick::V8n` („mm_tc_v8n”): în `fast_picks_text` din OS trebuie adăugate în listă, altfel nu apar
  la numărătoarea alegerilor.
- Prezis pentru #513 lot 6 (cu vitezele planului): alegerile v7 12 / v8 38 / v8n 97; lot 2: v7 73 / v8 26 / v8n 48.
- Nou pentru placă (când va fi cazul): doar cele două programe din P3a; niciun registru, niciun firmware.
