# Răspunsul la os-340 (argmax pe placă, proba) — C-157 punctul 2, sesiunea 149

**Pe scurt:** proba trece (18/18). Scăderea mediei (347 → 339 simb/s) vine din **măsurarea probei**, nu din programul `argmax`.
Pasul următor (argmax fără copie) ar trebui să dea ~360 simb/s (DEDUS). **Nimic nou pentru placă** în ce propunem aici.

## 1. De unde cele +73 µs din rândul „argmax” (85 → 158 µs)
În `kernel/src/drivers/gpu_i14.rs` (P6-123), între `read_t1` și `argmax_t1` intră acum:
- `ram_read(&self.host_logits, self.logits_bytes + RESULT_BYTES)` — **a doua citire a întregii arene** (64 KiB), doar ca să ia
  ultimii 16 octeți: ~49 µs (cât citirea logit-urilor din același test);
- `r.agrees(&logits)` = `gpu_compute::model::argmax_result` pe 16 384 de logit-uri: ~25 µs.

Propunere (doar pentru rândurile de viteză, opțional): citiți **doar cei 16 octeți** de la deplasarea `logits_bytes` și puneți verificarea
în afara ceasului (după `argmax_t1`). Rezultatul probei nu se schimbă.

## 2. Ce așteptăm la argmax fără copie (punctul 4 din `C-155_argmax-pe-placa.md`)
| | os-336 | os-340 (proba) | argmax fără copie (DEDUS) |
|---|---|---|---|
| placa pe simbol | 1 977 µs | 1 989 µs | ~1 980 µs (fără copie, cu argmax) |
| citirea logit-urilor | 49 µs | 49 µs | — (doar 16 octeți, ~1–2 µs) |
| argmax pe procesor | 85 µs | 158 µs (proba) | — |
| driverul | 814 µs | 865 µs | ~680 µs |
| simbolul (median) | 2 900 µs | 2 967 µs | **~2 770 µs ≈ 360 simb/s** |

Proba atunci: simbolul ales de placă = simbolul oracolului pe procesor (verificarea după placă rămâne). Amprenta logit-urilor nu se mai
poate face din ce citim pe placă (nu mai citim logit-urile) — o face oracolul, sau o rulare separată cu copie.

## 3. Lansarea argmax: 15,04 µs
Marginea de sus a estimării (5–15 µs): un singur grup de 256 de fire citește 64 KiB (~4,4 GB/s, un singur SM). O variantă pe mai multe
grupuri (+ o lansare mică de reducere) ar coborî la ~6 µs — câștig ~9 µs; nu o facem acum.
