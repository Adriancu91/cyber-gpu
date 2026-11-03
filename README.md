# cyber-gpu

Driverul de calcul GPU pentru **Cyber Universe OS** (Rust `no_std`, x86_64, UEFI/Limine).
Țintă principală: **NVIDIA GeForce RTX 3070 Ti** (GA104, Ampere, SM 8.6, 8 GB).
Licență **proprietară** („toate drepturile rezervate”, [LICENSE](LICENSE)) de la commit-ul de după `8313973`; versiunile până la `8313973`
inclusiv au fost publicate sub MIT. Cod propriu (vezi [docs/PROVENANCE.md](docs/PROVENANCE.md)); fișierele generate din antetele NVIDIA rămân MIT (NVIDIA).

> **Sesiunea 6 (26 oct seara, C-20):** crate nou `nvidia-rm570` cu valorile generate din antetele NVIDIA 570.144 (offset-urile vechi verificate:
> toate egale); procesorul de comenzi simulat rulează `add64` cap-coadă (SIMULARE, programul rulat pe procesor); proba tinygrad pregătită, nerulată.
>
> **Stare (26 oct 2026, sesiunea 5):** firmware-ul NVIDIA 570.144 e aprobat (C-18): I4 (citirea și verificarea fișierelor de pe stick) și
> structurile I5–I8 pentru GSP-RM sunt scrise și testate **doar în SIMULARE**; primul program GPU (adunarea a 64 de f32) e compilat cu Mesa NAK,
> **nerulat**. **Niciun program nu a rulat pe GPU.** Nicio linie de cod nu scrie în placă. Tabelul unic: [docs/COMPATIBILITATE_FIRMWARE.md](docs/COMPATIBILITATE_FIRMWARE.md).

## Cele trei componente, separate

| Componentă | Crate | Ce face |
|---|---|---|
| **Driverul bare-metal** | `gpu-core`, `gpu-pci`, `gpu-memory`, `gpu-command`, `nvidia-ga104`, `nvidia-rm570` (date generate din antetele NVIDIA) | contractul I1, PCI, memorie, cozi, backend GA104 |
| **Motorul de calcul / toolchain** | `tools/nak/` (Mesa NAK pe gazdă, C-18; vezi [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md)) | programe SM86, încărcare, lansare |
| **Runtime-ul tensorial** | `gpu-compute` | tensori (I2), operatorii Kronos în f32 (oracolul numeric), backend CPU |

`gpu-sim` conține **numai simulări** pentru teste pe gazdă; nu e dovadă de funcționare hardware.

## Verificare

```bash
sh tools/check.sh
```

(format, clippy `-D warnings`, teste pe gazdă, build `x86_64-unknown-none`). CI: `.github/workflows/ci.yml` (activ din 23 oct),
aceleași verificări pe stable și nightly.

## Documente

- [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) — auditul G0, ce e gata, parțial, blocat
- [docs/GA104_INITIALIZATION.md](docs/GA104_INITIALIZATION.md) — secvența de inițializare, surse, necunoscute
- [docs/MEMORY_INVARIANTS.md](docs/MEMORY_INVARIANTS.md) — invarianții M1–M6
- [docs/COMPATIBILITATE_FIRMWARE.md](docs/COMPATIBILITATE_FIRMWARE.md) — **tabelul unic**: fiecare componentă, sursa, versiunea (570.144), fișierul Rust, testul, starea
- [docs/RESEARCH_I4_I8.md](docs/RESEARCH_I4_I8.md) — cercetarea I4–I8 și argumentele pentru deciziile lui Adrian (firmware, compilator)
- [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) — formatul programelor SM86 și opțiunile legale
- [docs/RISKS.md](docs/RISKS.md) — riscuri, blocaje, politica de siguranță hardware
- [docs/HARDWARE_WORKSTATION.md](docs/HARDWARE_WORKSTATION.md) — datele reale ale workstation-ului (K0) și dependența de ReBAR/Above-4G
- [docs/TEST_MATRIX.md](docs/TEST_MATRIX.md) — rezultatele de azi și planul pentru seara de testare
- [docs/PROVENANCE.md](docs/PROVENANCE.md) — sursele, licențele, regula clean-room
- [docs/TINTA_UN_SINGUR_PROGRAM.md](docs/TINTA_UN_SINGUR_PROGRAM.md) — ținta de plan: Kronos pe placă ca un singur program pe simbol (doar plan)
- [integration/cyber-universe/README.md](integration/cyber-universe/README.md) — raportul de integrare cu OS-ul
