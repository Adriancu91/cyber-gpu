# C-165 după os-417: `mm_tv_64x64k8` a terminat cu pagina de pază — se întoarce în cursă (sesiunea Cyber-GPU 221, 25 nov 2026)

**Pentru sesiunea OS. Nimic nou pentru placă: aceleași programe, aceeași mapare cu `program_map_bytes` ca la os-417.**

## Ce a ieșit (raportul vostru `ciclu-auto-20261125-0348.txt`, citit cu `parse_race`, T227)
- `mm_tv_64x64k8`, mapată tot ultima, dar cu o pagină după cod: **termină cu aceiași biți** ca martorul → cauza de la os-413
  (citirea în avans după capătul codului) e confirmată pe placă. Bine că ați mapat cu pază.
- Pe qkv: **1 768,0 µs** față de martor 2 839,2 și față de `mm_tv_128x128` 1 890,2 (os-416) → pe forma asta `64x64k8` e cea mai rapidă.
- Pasul #513 (fără alegere): placa 463 732 µs = os-414 (463 594), sub 1 ‰. Repetabil.

## Ce e gata la noi (doar adăugări)
- `nvidia_ga104::mm_train_variants::race_list_full()` — toate 5 variantele (martorul primul); `RACE_CLEARED_OS417 = [4]`.
  `RACE_HOLD` și `race_list()` rămân cum erau.
- Alegerea din ambele rapoarte: puneți rândurile os-416 și os-417 împreună și `MmChoice::from_race(&randuri)` →
  qkv pe `64x64k8`, restul pe `128x128` (2 programe: `pick.programs()`).

## Propuneri (decide sesiunea OS)
1. **os-418 rulează cum e** (`128x128` peste tot, prezis ~341 ms) — e comparația curată cu os-416; nu o schimbați.
2. **Următoarea cursă de noapte:** `for (vi, v) in race_list_full()` în loc de `race_list()` (toate programele mapate cu
   `program_map_bytes`, cum ați făcut la os-417). 15 forme × 5 variante.
3. Abia după acea cursă: `parse_race(text).choice()` → `with_mm_choice`.

## Ce prezicem (DEDUS)
- Biții: aceiași ca martorul pe toate formele (ca la os-416/os-417).
- `64x64k8` câștigă cel puțin formele cu 768 de coloane unde `128x128` are doar 48 de blocuri (768×768×1024 f1, 2304×768×1024 f1,
  1024×768×768); `128x128` păstrează capul (16 384) și formele late. Pasul: încă ~15–40 ms mai puțin decât alegerea os-416.
