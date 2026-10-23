# cyber-gpu

Driverul de calcul GPU pentru **Cyber Universe OS** (Rust `no_std`, x86_64, UEFI/Limine).
Țintă principală: **NVIDIA GeForce RTX 3070 Ti** (GA104, Ampere, SM 8.6, 8 GB).
Licență MIT, cod propriu (vezi [docs/PROVENANCE.md](docs/PROVENANCE.md)).

> **Stare (23 oct 2026, sesiunea 2):** identificarea nivel 0 și pașii doar-citire I2 (GFW_BOOT) și I3 (VBIOS) sunt implementați și testați
> **doar pe dispozitive SIMULATE**. **Niciun program nu a rulat pe GPU.** Inițializarea GA104 se oprește
> explicit la pasul I4 (decizia de licență a firmware-ului semnat) și e blocată la I4–I8.
> Detalii: [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

## Cele trei componente, separate

| Componentă | Crate | Ce face |
|---|---|---|
| **Driverul bare-metal** | `gpu-core`, `gpu-pci`, `gpu-memory`, `gpu-command`, `nvidia-ga104` | contractul I1, PCI, memorie, cozi, backend GA104 |
| **Motorul de calcul / toolchain** | (nimic încă; vezi [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md)) | programe SM86, încărcare, lansare |
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
- [docs/RESEARCH_I4_I8.md](docs/RESEARCH_I4_I8.md) — cercetarea I4–I8 și argumentele pentru deciziile lui Adrian (firmware, compilator)
- [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md) — formatul programelor SM86 și opțiunile legale
- [docs/RISKS.md](docs/RISKS.md) — riscuri, blocaje, politica de siguranță hardware
- [docs/HARDWARE_WORKSTATION.md](docs/HARDWARE_WORKSTATION.md) — datele reale ale workstation-ului (K0) și dependența de ReBAR/Above-4G
- [docs/TEST_MATRIX.md](docs/TEST_MATRIX.md) — rezultatele de azi și planul pentru seara de testare
- [docs/PROVENANCE.md](docs/PROVENANCE.md) — sursele, licențele, regula clean-room
- [integration/cyber-universe/README.md](integration/cyber-universe/README.md) — raportul de integrare cu OS-ul
