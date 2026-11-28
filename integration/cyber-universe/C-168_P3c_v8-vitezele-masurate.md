# C-168 p.3 — P3c: vitezele măsurate în os-500 → pasul cu `mm_tc_v8n` (și v8 unde câștigă)

Sesiunea Cyber-GPU 283, 28 nov 2026. Detalii: `docs/C-168_OS500_V8.md`, testul T300. Înlocuiește partea „vitezele” din P3b.

**G4 a trecut în os-500** (v8 și v8n = biții lui `mm_tc`, 7 / 7), deci P3b se poate rula pe placă, cu o singură schimbare:

- în `V8Programs { v8, v8n, sms, rates }` puneți `rates: nvidia_ga104::mm_tc_v8::RATES_OS500` (110 / 118 %, măsurate), **nu**
  `RATES_PLAN` (140 / 126, DEDUS). Restul ca în P3b (`pack_mixed_fast_attn8`, `TcPick::V8` / `V8n` în `fast_picks_text`).
- Prezis (DEDUS, T300) la #513 lot 6: alegerile **v7 12 / v8 0 / v8n 135**; înmulțirile 98,1 → ~83,7 ms; **pasul ~133 ms ≈
  23 000 simb/s** (os-496: 147,7 ms, 20 795). Trece = oracolul pasului mixt cu aceleași numere ca la v7 (aceiași biți: T297 N3).
- Nou pentru placă: nimic (aceleași două programe din os-500).

**De reluat, scurt:** `mm_tc_v8n@64k` pe qkv-513-dw a rulat după secunda 1 800 a cererii de viteză (lanțul 1 986 s) → timpul lui
nu contează. Prezis 1,24 ms (v7 1,469). Un rând mai scurt (de ex. fără `--mm-tc-v6`) l-ar prinde în fereastră.

**Atenție la nota os-500 din hub:** tabelul are doar v8; v8n e mai rapid (qkv 1,012 ms = 21,5 TFLOP/s, dX 0,929 = 23,4), iar
„v8 la qkv și dX, v7 la dW” se înlocuiește cu alegerea de mai sus.
