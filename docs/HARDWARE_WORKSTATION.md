# Workstation-ul — placa țintă (date reale)

*Actualizat 23 oct 2026, sesiunea 2 (cloud). Etichete: FAPT (văzut de Adrian pe calculator), DEDUS (concluzie din surse publice),
NECUNOSCUT. Nicio valoare de aici nu a fost citită de codul nostru pe placă: sursa e Windows + BIOS.*

Surse: hub `rezultate/2026-10-23-k0-workstation/REZULTAT.md` (inventarul K0, 09:33, scriptul `tools/k0-inventar-workstation.ps1`)
și `rezultate/2026-10-23-workstation-os/REZULTAT.md` (pozele BIOS și prima pornire a OS-ului, 09:45).

## 1. Calculatorul (FAPT)
| Ce | Valoare |
|---|---|
| Calculator | Dell, placă de bază 0X75JG (ESTIMARE: Precision 5820 Tower), BIOS Dell 2.44.0 |
| Procesor | Intel Core i9-10980XE, 18 nuclee / 36 fire, Cascade Lake-X (AVX-512) |
| RAM | 128 GB (127,7 GiB) |
| VT-d (IOMMU) | **oprit** în BIOS (poza 01) |
| Secure Boot | oprit **doar pentru test** (poza 02); se repune pe Enabled pentru Windows (BitLocker) |
| Above 4G Decoding | nefotografiat; toate BAR-urile sunt sub 4 GiB → **probabil oprit** (NECUNOSCUT până la poza din BIOS) |
| Resizable BAR | **oprit** (BAR1 = 256 MiB; cu ReBAR ar fi 8 GiB) |

## 2. Placa video (FAPT)
| Ce | Valoare |
|---|---|
| Model | NVIDIA GeForce RTX 3070 Ti, GA104, 8192 MiB |
| PCI ID | `10de:2482`, subsistem `1043:8813` (ASUS), rev A1 |
| Adresa PCI | magistrala **B3** (179), dispozitiv 0, funcția 0 |
| VBIOS | 94.04.65.00.3e |
| BAR0 (registre) | `F9000000–F9FFFFFF`, 16 MiB |
| BAR1 (fereastra VRAM) | `E0000000–EFFFFFFF`, 256 MiB |
| BAR3 | `F0000000–F1FFFFFF`, 32 MiB |
| I/O | `F000–F07F`, 128 B |
| Driver în Windows | 616.56 |

**Magistrala B3 e pe altă magistrală-rădăcină** (stivele IIO ale lui Cascade Lake-X). OS-ul vedea la început doar arborele de la
magistrala 0 și nu găsea placa; sesiunea OS a reparat enumerarea (C-13, os-166+: „PCI caută pe toate magistralele-rădăcină”).
Codul nostru nu presupune nimic despre magistrală: primește `PciAddress` de la kernel.

## 3. În simulare
- `gpu-sim::workstation` — constantele de mai sus (FAPT), într-un singur loc.
- `SimConfig::workstation_rtx3070ti()` — spațiul de configurare SIMULAT construit din ele (dispunerea capabilităților rămâne SIM);
  `.without_rebar()` acoperă cazul în care placa nu expune deloc capabilitatea ReBAR când e oprită (NECUNOSCUT).
- Teste (SIMULARE): `probe::tests::level0_on_simulated_workstation_config` (ID-uri, subsistem, BAR-uri, adresa `b3:00.0`,
  zero scrieri, cu și fără ReBAR) și `bar::tests::sizing_simulated_workstation_matches_inventory` (mărimile 16/256/32 MiB, 128 B).

## 4. Depinde planul de inițializare de ReBAR sau Above-4G?
**Răspuns scurt: nu, pentru pașii I0–I8. Nu e nevoie ca Adrian să schimbe ceva în BIOS.** (DEDUS, argumentat mai jos.)

| Pas | De ce are nevoie | ReBAR | Above-4G |
|---|---|---|---|
| I0–I1 identificare | spațiul de configurare + prima pagină din BAR0 | nu | nu |
| I2 GFW_BOOT, I3 VBIOS | registre în BAR0 (0x118xxx, fereastra PROM 0x300000) — BAR0 are 16 MiB oricum | nu | nu |
| I4–I8 GSP | BAR0 (falcon, GSP) + RAM de sistem prin DMA (cozile, WPR meta); VRAM-ul folosit de GSP (WPR2) e rezervat **în placă**, nu prin BAR1 | nu | nu |
| I9–I10 canal, program | tabele de pagini GPU și date în VRAM; CPU-ul scrie în VRAM prin BAR1 (256 MiB, „fereastră mobilă” reconfigurabilă prin tabelele de pagini ale BAR1) sau prin copiere DMA | nu (doar viteză) | nu |
| Kronos pe GPU (mai târziu) | greutățile W2 ≈ 196 MB în bf16 | încap în 256 MiB; ReBAR doar ar simplifica încărcarea | nu |

Argumente:
1. Driverele publice (nouveau, NVIDIA open-gpu-kernel-modules) funcționează pe plăci cu BAR1 de 256 MiB — valoarea implicită pe
   aproape toate calculatoarele fără ReBAR. ReBAR e o optimizare a transferurilor CPU→VRAM, nu o condiție de pornire. (DEDUS)
2. Pornirea GSP folosește registrele din BAR0 și memoria de sistem prin DMA (`docs/GA104_INITIALIZATION.md` §3, pct. 7). (DEDUS din cod citit)
3. Cu Above-4G oprit, toate BAR-urile sunt sub 4 GiB → OS-ul le poate mapa fără adrese mari; e cazul cel mai simplu pentru noi.
   Dacă Adrian pornește vreodată ReBAR, BAR1 (8 GiB) va cere Above-4G și o mapare de 64 de biți; codul nostru decodează deja BAR-uri de 64 de biți.
4. **VT-d oprit = placa vede adresele fizice ale RAM-ului direct** (fără traducere IOMMU). Asta simplifică I5 (cozile GSP în RAM),
   dar înseamnă și că o greșeală de adresă DMA poate scrie oriunde în RAM → cozile GSP vor sta în blocul DMA păzit al OS-ului
   (sub 4 GiB), iar poarta M2 din `gpu-memory` rămâne obligatorie. (DEDUS)

## 5. Ce rămâne de confirmat pe placă (la întoarcerea lui Adrian)
- poza din BIOS cu „Above 4G Decoding” (doar ca să fie notat);
- dacă placa expune capabilitatea ReBAR cu ReBAR oprit (se vede în `gpu identifica`: rândul `ReBAR BAR1`);
- BOOT_0/BOOT_42 (I1), GFW_BOOT (I2), semnătura VBIOS din fereastra PROM (I3) — `docs/TEST_MATRIX.md`, partea B.
