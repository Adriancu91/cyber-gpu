# C-165 — proba ocupării (treapta memoriei comune din QMD), gata de legat în SIMULARE

**Pentru:** sesiunea OS (C-158) și centrul de comandă. **De la:** Cyber-GPU, sesiunea 224 (25 nov 2026).
Răspunde studiului centrului din 25 nov 07:00 (`idei/2026-11-25-centru-inmultirea-la-altii.md`). Detalii: `docs/C-165_OCUPAREA.md`.

## Ce e
O lansare cu aceeași memorie comună pe bloc, dar cu **treapta unității** (cât din L1 devine memorie comună) aleasă de noi:
`ComputeQmd::to_bytes_with_sm_config(&shared, step_bytes)` pune `MIN = TARGET = step_bytes` (8/16/32/64 KiB),
`MAX` ca azi (64 KiB). Toate celelalte câmpuri sunt identice cu `to_bytes_with_shared`; la treapta unui bloc octeții sunt
exact cei de azi (T230b).

Lista probei: `nvidia_ga104::occupancy::PROBE` (10 rulări, indice în `mm_train_variants::VARIANTS` + treapta):
`mm_train_tile` 8 → 16 → 32 → 64 KiB; `mm_tv_64x64k8` 8 → 16 → 32; `mm_tv_128x128` 8 → 16 → 32. Forma: qkv a pasului #513
(m 1024, n 2304, k 768, flags 6), aceleași date pentru toate rulările. Prima rulare = lansarea de azi = martorul biților.

## Rândul de test propus
Pentru fiecare rulare: timpul (ca în cursă, cu încălzire), TFLOP/s, puterea plăcii din tablou în timpul ei, și verdictul
biților față de prima rulare. Rânduri de jurnal propuse:
`ocupare <program> <N> KiB: <us> us, <TFLOP/s>, <W> W, ACEIASI BITI` / `... DIFERA (prima la i)`, apoi
`proba ocuparii: <k> / 10 aceiasi biti`.

## Ce așteptăm (scris înainte; DEDUS)
- **Biții:** aceiași la toate 10 (dacă nu → STOP, nu se leagă nimic).
- **H1 (centrul are dreptate):** `mm_train_tile` la 64 KiB ≤ 0,85 din timpul de la 8 KiB (ținta ×1,5–3), timpul scade
  8 → 16 → 32; puterea crește.
- **H2 (placa urcă deja singură treapta):** toate treptele în ±3 % → ocuparea nu e buba, căutăm în altă parte.
- `64x64k8` câștigă deja la 16 KiB și `128x128` la 16 KiB dacă nu există rezerva de 1 KiB pe bloc; abia la 32 KiB dacă există.

## Ce e nou pentru placă
**Doar valoarea câmpurilor `MIN_SM_CONFIG_SHARED_MEM_SIZE` și `TARGET_SM_CONFIG_SHARED_MEM_SIZE` din QMD** (le scriem deja
la fiecare lansare). Aceleași programe (SHA-256 din `VARIANTS`), aceleași adrese, mapare cu pagina de pază. Niciun registru,
nicio comandă nouă. Placa e a laboratorului Kronos azi de la 07:00 — proba intră când decide centrul.

## Precizări pentru centru
- Pragul de registre la `mm_tv_128x128` pentru 3 blocuri e **≤ 80**, nu ≤ 85 (registrele se dau câte 8 pe fir).
- Dacă se confirmă H1, nu legăm treapta pe lansare, ci **una singură pe tot pasul** (`occupancy::best_step` peste toate
  programele pasului) — legarea în `Plan` e pasul următor la noi, după rezultatul probei.

Teste: T230 (`crates/nvidia-ga104/tests/sim_c165_ocuparea.rs`, SIMULARE). Modelul nu are ocupare și nu măsoară viteză:
verifică doar că QMD-ul cu altă treaptă e citit corect și dă aceiași biți.
