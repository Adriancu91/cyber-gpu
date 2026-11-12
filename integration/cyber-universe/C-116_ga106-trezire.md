# C-116 — laptopul ASUS TUF F15 (GA106): recunoaștere doar-citire + „trezirea minimă” — ce oferă cyber-gpu OS-ului

*12 nov 2026, noaptea, sesiunea 95 cyber-gpu (cloud, FĂRĂ placă). Totul SIMULARE; nimic rulat pe laptop.* Cercetarea și faptele de pe laptop:
`docs/GA106_LAPTOP.md` (§0 ce a arătat `gpu identifica` pe 12 nov, §6 trezirea minimă). Commit-uri: `6623d3c` (pasul 1), `ccec8f9` (pasul 2).

## De ce

Pe laptop `gpu identifica` (os-295) a văzut NVIDIA `10de:2520` în D0 cu **Command 0x0000**: firmware-ul lasă funcția fără decodare de memorie, deci
BAR0 nu răspunde și nimic din placă nu se poate citi (OS-ul a scris corect „Memory Space oprit de firmware (nu il pornim)”). Ca să citim BOOT_0 / GFW / VBIOS
trebuie pornit bitul 1 din Command — o singură scriere în **configurația PCI** (nu în registre GPU), exact ce face Linux `pci_enable_device` înainte ca
orice driver să atingă BAR0. Decizia lui Adrian (C-116 pasul 2): în spatele unui „DA”, doar pe stick-ul de test.

## Funcții NOI (adăugate; cele vechi neschimbate — OS-ul compilează neschimbat la `ccec8f9`)

- `nvidia_ga104::ids::{GA106_MODELS, LAPTOP_DEVICE (0x2520), Family {Ga104, Ga106}, lookup_ga106, lookup_ampere}`, `boot::CHIP_GA106 (0x176)`.
  `Family::writes_allowed()` = doar `Ga104`. **`ids::lookup` și `probe::probe_config`/`level0` rămân GA104-only** → `gpu lant` și tot ce scrie refuză laptopul fără cod nou.
- `nvidia_ga104::probe::probe_config_ampere(cfg, addr) -> GpuResult<(Candidate, Family)>` — aceleași citiri ca `probe_config`, tabel mai larg.
  `Candidate::memory_space_enabled()` (bitul 1 din Command). `probe::MEMORY_SPACE_OFF` = textul refuzului.
- `nvidia_ga104::probe::level0_ampere(cfg, addr, bar0, journal) -> GpuResult<(Level0, Family)>` — ca `level0`, pentru GA104 sau GA106; **cu Memory Space oprit NU citește BAR0**
  și întoarce `Blocked(MEMORY_SPACE_OFF)` (în jurnal: nota + `pas nivel0: BOOT_0/BOOT_42: ESEC`). `probe_mmio_family(bar0, journal, family)`.
- `nvidia_ga104::read_tour::run_tour_ampere(...)` — turul C-54 pentru ambele familii; la GA106 adaugă faptele `chip.familie = GA106` și `chip.scrieri = interzise (...)`;
  la GA104 = `run_tour` + `chip.familie = GA104`. Se oprește înainte de prima citire BAR0 dacă Memory Space e oprit.
- **`gpu_pci::mem_space::{MemorySpacePermit, enable_memory_space(cfg, &permit, journal) -> GpuResult<u16 /* Command vechi */>, restore_memory_space(cfg, &permit, journal, old),
  check_before_enable(cfg)}`** — modelul `bus_master.rs`: o scriere, citită înapoi, în jurnal `cfg w 0x004: 0x00100000 <- 0x00000002 => 0x00100002`; refuză (0 scrieri) fără D0,
  fără BAR0 alocat, cu un BAR numit de ReBAR fără adresă, fără antet tip 0, placa absentă. `check_before_enable` e doar citire — OS-ul o poate arăta înainte de „DA”.
- **`nvidia_ga104::read_tour::run_tour_awake(cfg, addr, bar0, clock, journal, &permit) -> GpuResult<(Tour, Wake)>`** — dacă Memory Space e pornit: `run_tour_ampere`, 0 scrieri,
  `Wake::NotNeeded`; altfel: `enable_memory_space` → tur → **`restore_memory_space` întotdeauna** (și când turul pică) → `Wake::EnabledAndRestored`; faptul `pci.trezire` spune ce s-a întâmplat.
  Două scrieri în configurație în total, zero în registre GPU; Command rămâne cum l-a lăsat firmware-ul.
- `gpu_sim::pci::SimConfig::ga106_laptop_asus()` (din faptele de pe laptop: `10de:2520`, Command 0, BAR0 @0x85000000, BAR1 8 GiB ReBAR; restul SIM), `gpu_sim::mmio::SimMmio::ga106_bar0()`
  (BOOT_0/BOOT_42 cu `0x176`) — pentru testele din QEMU / `--sim`.

## Propunere pentru OS (decizia e a sesiunii OS; C-116 partea OS: „`gpu lant` / `gpu tur` acceptă GA106 doar când Cyber-GPU îl declară”)

1. `gpu identifica`, `gpu cauta`, `gpu vbios`: pe variantele `_ampere` → laptopul e identificat (`GeForce RTX 3060 Laptop`, GA106) și, cu Memory Space oprit, mesajul vine din bibliotecă
   (`MEMORY_SPACE_OFF`), fără citire din BAR0. Rândul OS-ului de azi („Memory Space oprit de firmware (nu il pornim)”) rămâne corect pe stick-ul normal.
2. **`gpu tur` pe laptop** (nou, sau `gpu tur --trezire`): doar pe stick-ul de test (`gpu-scrie`, adaptorul de configurație care poate scrie), după același „DA” ca la `gpu lant`,
   cu textul: „trezire minimă C-116: o scriere în registrul Command al PCI (Memory Space pornit), repusă la final; niciun registru GPU scris”. Apoi `run_tour_awake(...)` cu jurnalul pe stick
   (`gpu-jurnal-NNN.txt`), `trimite` după. Pe stick-ul normal (fără `gpu-scrie`): `run_tour_ampere` → refuz curat.
3. `gpu lant` / I6+ pe GA106: **NU** — `Family::writes_allowed()` e `false`; rămâne pasul 3 din C-116 (DA separat după jurnalul de citire). Nimic de schimbat acum.
4. `hw`: modelul laptopului din SMBIOS (C-116 partea OS) ajută la §3 pct. 3 din `docs/GA106_LAPTOP.md` (MUX da/nu).
5. Firmware pe stick: aceleași 4 fișiere 570.144 (`nvidia/ga106/gsp -> ../ga102/gsp` în `linux-firmware`), aceeași cale — `gpu firmware` n-are nimic de schimbat.

## Ce așteptăm din primul jurnal de pe laptop

`fapt chip.familie = GA106`, `fapt pci.trezire = Memory Space pornit de noi …`, `BOOT_0 = 0x176…`, `BOOT_42` consecvent, starea GFW, imaginile VBIOS prin PROM (dacă sunt gunoi →
citirea prin ACPI `_ROM` devine sarcină separată, cum zice C-116), FWSEC + semnături, siguranța, VRAM ~6 GiB, WPR2 = 0 la pornire rece, și cele două rânduri `cfg w 0x004: …`.
