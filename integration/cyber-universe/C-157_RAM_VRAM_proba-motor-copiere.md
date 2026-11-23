# C-157 (punctul lui Adrian din 23 nov, a) → OS: proba vitezei RAM ↔ VRAM cu **motorul de copiere** al plăcii

De la Cyber-GPU, sesiunea 183 (23 nov 2026). Cod: `nvidia_ga104::ram_vram` + `gsp::user_space::alloc_copy_object`
(commit `9359f21`), test T192 `crates/nvidia-ga104/tests/sim_c157_ram_vram.rs` (SIMULARE). Planul și predicțiile scrise
înainte: `docs/C-157_RAM_VRAM_PROBA.md` (`477455c`). **Totul e adăugare**; nimic din ce folosește OS-ul azi nu se schimbă.

## 1. Ce e nou pentru placă (un singur lucru: motorul de copiere)
- **Un mesaj RM:** `alloc_copy_object(rm, &user_space, &work_channel)` → alocă `AMPERE_DMA_COPY_B` (`0xC7B5`, fără
  parametri) pe canalul de lucru, cu `work_channel.next_handle`; întoarce `(handle, următorul handle liber)`. Aceeași clasă
  pe care placa a acceptat-o deja la imaginea golden (pe canalul golden).
- **Metodele motorului de copiere pe subcanalul 4** (din `clc7b5.h`, MIT): `SET_OBJECT 0xC7B5`, `OFFSET_IN/OUT`,
  `LINE_LENGTH_IN`, `SET_SEMAPHORE_A/B/PAYLOAD`, `LAUNCH_DMA` (pitch → pitch, adrese virtuale; cu înregistrare:
  `FLUSH_ENABLE` + semaforul în 4 cuvinte cu ceasul). Calculul rămâne pe subcanalul 1, neschimbat.
- Niciun registru nou; soneria și fereastra VRAM ca la I11.

## 2. Ce face OS-ul (o singură trimitere)
1. După canalul de lucru: `alloc_copy_object` (o dată pe sesiune).
2. Mărimea `M`: implicit `ram_vram::DEFAULT_MAX_BYTES` = 64 MiB (multiplu de 256 KiB, cel mult 1 GiB; în QEMU 256 KiB–1 MiB).
3. Bufferele: **A** = `M` octeți RAM (PTE spre RAM, `alloc_sys_mapped`), umplut cu `transfer::fill_pattern(…, seed)`;
   **B** = `2M` RAM, umplut cu `ram_vram::FILL` (`0xEE`); **V1**, **V2** = câte `M` VRAM (`alloc_mapped`); înregistrările =
   `ram_vram::RECORD_BYTES` (31 × 16) în VRAM, ca la I11. Adresele A, B, V1, V2 aliniate la 4 KiB.
   La 64 MiB: 192 MiB RAM fixată + 128 MiB VRAM.
4. Trimiterea: `ram_vram::encode_probe(&mut pb, &Buffers { a, b, v1, v2, records }, M, first_payload)` —
   `ram_vram::probe_words()` cuvinte (≈ 600). Ultima înregistrare = a 30-a (payload `first_payload + 30`).
5. După ultima înregistrare (așteptare: la 64 MiB toată proba ≈ 0,15–0,3 s; pune 5 s): citește înregistrările →
   `ram_vram::analyze(rec, M, first_payload)`; citește B (2M) și A (M) → `ram_vram::judge(b, a, M, seed)`;
   `ram_vram::Probe { max_bytes: M, times, bytes }` → `.journal(j)` (rândurile `kt ramvram …`) și `.lines("placa")`.
   Dacă lipsește o înregistrare: `transfer::records_seen(rec, 30, first_payload)` spune până unde a ajuns placa → oprire
   curată, ca la I11.
6. `ram_vram::parse_ramvram(text)` citește rândurile înapoi din jurnalul de pe stick (pentru comparația mea).

## 3. Ce măsoară
Sus (RAM → VRAM) și jos (VRAM → RAM) la `M/64`, `M/16`, `M/4`, `M` (1, 4, 16, 64 MiB la 64 MiB), apoi 4 × `M/4` în
paralel (`PIPELINED`) în fiecare sens; fiecare de 3 ori → mediana, minimul, maximul, MB/s. Verdictul: 0 = toți octeții
corecți, 1 = B greșit (primul cuvânt greșit în rând), 2 = A atins.

## 4. Predicții (scrise înainte)
Sus 64 MiB **11–13 GB/s**, jos **11–13,5 GB/s**, 1 MiB 6–11 GB/s, în paralel ±5 % față de o copie de 64 MiB, ~1,8–2,1 × față
de `copy_b128` (6,2 GB/s azi). Constantele: `ram_vram::PREDICTED_*`.

## 5. Ce NU dovedește SIMULAREA
Modelul nostru nu verifică dacă obiectul e alocat pe canal (pe placă, fără `alloc_copy_object`, `SET_OBJECT` pe
subcanalul 4 ar fi refuzat); copiile `PIPELINED` nu se suprapun în model; forma înregistrării în 4 cuvinte (valoarea la
+0, ceasul la +8) e DEDUSĂ. **Riscul principal:** motorul de copiere pe canalul de calcul (runlist GR) — tinygrad folosește un
canal separat. Dacă placa refuză, se vede ca înregistrare lipsă / eroare de canal; atunci varianta a doua (canal separat
pe runlist-ul unui motor de copiere) cere un „da” separat.
