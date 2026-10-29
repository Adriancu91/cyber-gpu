# Pentru sesiunea OS: `gpu firmware` după C-39 (antetele reparate) — doar mutarea fixării

*29 oct 2026, sesiunea Cyber-GPU 18. Nimic de aici nu e aplicat în `cyber-universe`. Testat **doar în SIMULARE** (pe gazdă).*

Ce s-a schimbat: la fereastra din 29 oct, `gpu firmware` a dat „antet NECITIT: bin_size diferit de marimea fisierului” la `booter_load`,
`booter_unload` și `bootloader` (fișierele erau cele oficiale). Greșeala era în `nvidia_ga104::fw_hdr` și e reparată pe `g0-g2-fundatie`
(commit `ef97110`): `bin_size` nu mai e comparat cu mărimea fișierului, iar numărul de semnături Booter se citește prin adresa lui.

Ce trebuie în OS: **doar** fixarea cyber-gpu mutată de pe `2c9c90c` pe `ef97110` sau mai nou (cele 4 rânduri din `kernel/Cargo.toml`).
Codul comenzii `gpu firmware` din OS nu se schimbă. Față de `2c9c90c`: nicio funcție, structură sau câmp public redenumit ori șters;
doar adăugări (`gpu_sim::fw_stick::{booter_570_144, bootloader_570_144}`, plus modulele I9c/I9d deja adăugate în sesiunile 16–17).

Ce se schimbă la ieșire:
- `gpu firmware --sim`: fișierele de probă au acum mărimile fișierelor reale (61 304 / 41 080 / 24 684 octeți) și „HS v2: 2 semnaturi de 384
  octeti, 1 aplicatii” la ambele Booter; ultimul rând rămâne `SIMULARE: criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4)`.
  În `tests/scenarios.txt` se schimbă doar textul fixării (`cyber-gpu 2c9c90c` → noul commit).
- `gpu firmware` pe stick-ul real: așteptat `criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4)` (NOT TESTED pe stick).

Verificat aici: o copie de lucru a OS-ului (`8f51979`), cu dependențele cyber-gpu pe dosarul local: `cargo build --release` și
`--features test-gpu-sim` trec. QEMU: NOT TESTED (lipsește în această sesiune).
