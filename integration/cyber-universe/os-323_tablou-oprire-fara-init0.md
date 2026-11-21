# os-323 → tabloul de bord: oprirea fără `INIT(0)` (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 132 (21 nov 2026). Analiza: `docs/C-157_P2_COMPARATII.md`.

**Ce s-a văzut pe placă (os-323 `--tablou`):** oprirea zonei comune (`INIT(0)`) a fost refuzată cu `0x8000001f`
(`NV_ERR_INVALID_ARGUMENT`) și un ASSERT intern. Driverul NVIDIA 570.144 nu trimite niciodată această comandă.

**Schimbarea (o linie la voi):** în loc de `kd::dashboard_at_end(...)` chemați `kd::dashboard_at_end_keep_page(...)` — aceleași
argumente, același rezultat `(RusdStart, Option<Dashboard>, RusdStop)`. Opririle citirilor (masca 0) rămân; `INIT(0)` nu se mai
trimite. Jurnalul are rândul nou `kd zona comuna lasata pornita (ca driverul NVIDIA, fara INIT(0)): …`. Pagina de 4 KiB rămâne
rezervată până la FLR sau repornire (exact ca după refuzul de azi) — nu o refolosiți.

**Ce e nou pentru placă:** nimic; o comandă mai puțin (4 în loc de 5). **Așteptat:** niciun refuz în rândurile `kd`, un ASSERT mai puțin.
**Test în SIMULARE:** T140h (`crates/nvidia-ga104/tests/sim_c151_rusd.rs`). Funcțiile vechi rămân neschimbate.
Se poate lega împreună cu orice alt Release (nu e un lucru nou pentru placă).
