# PREDARE P2zzd — `token32_v2` (citirile prin L1) + lista scrisă doar unde se schimbă (Cyber-GPU, sesiunea 260, 27 nov 2026)

De ce: os-471 a arătat `token32` corect, dar ×3,6 mai lent (comparația: `cyber-gpu/docs/C-157_OS471.md`). Două cauze găsite, DEDUSE:
(1) în `token32` fiecare rând își citește vectorul x din L2 (`ld.strong.sys`, ~390 MB pe simbol), în loc de L1 ca programele
separate; (2) OS-ul scrie toată lista (5 632 de cuvinte, ~0,63 µs fiecare) înaintea fiecărui simbol, deși se schimbă doar ~38.

**Nou pentru placă:** un singur program, `kronos_token::TOKEN32_V2` (Mesa NAK 25.0.7, SHA-256
`5f4d22a0d01f35d2bd3b88f0e43e492acede1a1a8d912b80fc0712d012f11171`, 92 de registre, 2 176 B partajați — ca `token32`). Codul NAK e
instrucțiune cu instrucțiune cel al lui `token32`; diferă doar felul a 173 de citiri (`.strong.sys` → `.constant`, prin L1). Aceeași
listă, același buffer constant, aceleași QMD-uri (`token32_qmd_wide`), aceleași G. Niciun registru nou.

**Opțiunea propusă:** `--token32-v2` (oprită implicit), în același rând cu `--token32`, după el (aceeași încălzire a plăcii):
`test-placa --token32 --token32-v2`. Rețeta e cea din P2zz, cu `TOKEN32_V2` în loc de `TOKEN32` (și SHA-ul de mai sus).

**Schimbarea driverului (separată, se poate face și singură, și pentru `token32`):** în loc să scrie toată `work_list32(&launches)`
la fiecare simbol, OS-ul ține lista simbolului trecut și scrie doar `kronos_token::work_list_changes(&veche, &noua)` (perechi
offset–cuvânt). Prima poziție (sau când n se schimbă — atunci funcția refuză): lista întreagă. Verificarea rămâne aceeași (simbolurile,
amprenta, contorul).

**Ce așteptăm (predicții scrise înainte, `C-157_OS471.md` §3):**
- V1: aceleași 16 simboluri, amprenta `0x6c8501f0`, contorul (n − 1) · G, la G = 384 și 768. **Dacă amprenta diferă: o citire
  veche din L1 — v2 se oprește, nimic altceva de schimbat.**
- V3: placa la G = 384: 0,9–1,75 ms (azi `token32` 3,86 ms, lista 1,07 ms). V4: G = 768: 0,95–2,1 ms.
- V5 (doar cu `work_list_changes`): partea driverului ~0,9–1,2 ms pe simbol în loc de 4,6.

**Nicio schimbare incompatibilă** (doar adăugări: `TOKEN32_V2`, `TOKEN32_V2_REGISTERS`, `TOKEN32_V2_SHA256`, `work_list_changes`).
