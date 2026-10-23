# Starea implementării — cyber-gpu

*23 octombrie 2026, sesiunea 1. Etichete: FAPT (verificat), SIMULARE (doar pe modele software), ESTIMARE, NECUNOSCUT.*

## G0 — Audit și inventar

### Ce exista la început (FAPT)
- `Adriancu91/cyber-gpu`: repository **gol** (niciun commit, niciun ref; `git ls-remote` fără ieșire, cod 0).
- Workspace local `C:\Users\krono\Desktop\gpu`: gol (doar `.claude/`).
- Calculatorul folosit în seara asta: laptop Lenovo, i5-1135G7, Intel Iris Xe (`8086:9a49`), 8 GB. **Nu are placă NVIDIA**,
  deci **nu e workstation-ul**. Nimic nu s-a putut testa pe RTX 3070 Ti.
- Toolchain: nu exista Rust. Instalat `rustup` (winget, sursa oficială) + `stable-x86_64-pc-windows-gnu` (rustc 1.99.1)
  + ținta `x86_64-unknown-none`. Fără MSVC Build Tools (de aceea toolchain-ul GNU). Fără QEMU, fără `gh`.

### API-urile cyber-universe verificate în cod (commit `2d038de`, 22 oct)
| Afirmație din ARHITECTURA_KERNEL | Verificat în cod | Observație importantă pentru GPU |
|---|---|---|
| enumerare PCIe, BAR-uri 64 biți | DA: `pci::Device`, `bar(i)`, `first_memory_bar()` | **nu există dimensionarea BAR-urilor**; doar `read32/write32` (nu 8/16 biți) |
| `enable()` memorie + Bus Master | DA: `enable_as()`, bit cu bit, jurnalizat | nivelul 0 al GPU-ului **nu îl cheamă** |
| `map_mmio` | DA: UC (PCD+PWT) | **sare peste paginile deja mapate** → o zonă deja mapată poate rămâne cache-abilă; glue-ul nostru refuză în acest caz |
| DMA contiguu sub 4 GB | DA: `alloc_contiguous_between(n, 16<<20, 1<<32)`, `pin` | nu e zeroizată; limita 4 GB e politică, nu a GPU-ului |
| jurnal RAM + serial | DA: `info!/warn!/error!`, inel 64 KiB | |
| fără IOMMU, SMP, MSI | DA (absente) | |
| fără salvare SSE/AVX | DA | crate-urile noastre folosesc f32 în soft-float pe ținta kernelului |
| crate-uri separate, teste pe gazdă cu model | DA: tiparul `rtl8168/` (`trait Hw` + `model.rs`) | refolosit: `ConfigSpace`/`Mmio` + `gpu-sim` |

### Necunoscute hardware (rămân pentru K0 pe workstation)
BAR0/BAR1/BAR3 reale, Above-4G Decoding, Resizable BAR, starea IOMMU, Secure Boot, ID-ul subsistemului, versiunea VBIOS.
Instrumentul: `tools/k0-inventar-workstation.ps1` (Windows, doar citire; testat pe laptop).

## Stadiul pe etape

| Etapă | Stare | Dovadă |
|---|---|---|
| G0 audit | **gata** | acest document |
| G1 nucleul generic | **gata (gazdă)** | `gpu-core`: erori, I1, stări, jurnal; teste PASS |
| G2 descoperire GPU | **implementat, testat doar în SIMULARE** | `gpu-pci` + `nvidia-ga104::probe`; zero scrieri, verificat de teste; patch kernel `gpu identifica` |
| G3 cercetare/inițializare | **cercetare gata, cod oprit la I2** | `docs/GA104_INITIALIZATION.md`, `init.rs` (plan I0–I10) |
| G4 memoria GPU | **parțial** | invarianți + allocator + DMA policy + poarta IOMMU (logică); nimic pe VRAM real |
| G5 primul calcul GPU | **NU** | blocat la I4–I8 (vezi mai jos) |
| G6 toolchain | **cercetare** | `docs/TOOLCHAIN.md`; niciun binar SM86 produs |
| G7 operații tensoriale | **referință CPU gata** | `gpu-compute::reference` (add, bias, matmul, layernorm, GELU, softmax cauzal, embedding, atenție, cross-entropy), teste diferențiale f64 PASS |
| G8 pregătire Kronos | **parțial** | `TensorDesc` (f32/bf16), `OpCode` cu convenție de argumente, `CpuBackend` prin I1 |

### Componente terminate / parțiale / blocate
- **Terminate (pe gazdă):** erori structurate, contractul I1, mașina de stări, jurnalul limitat, decodarea BAR, capabilități
  PCI/PCIe/ReBAR, raportul nivel 0, ID-uri GA104, decodarea BOOT_0/BOOT_42, lista de registre permise, allocatorul de intervale,
  tabela de buffere, inelul PUT/GET, fence-urile, math `no_std`, operatorii de referință, backend-ul CPU.
- **Parțiale:** dimensionarea BAR (scrisă și testată în simulare; pe hardware e opțiune explicită `gpu dimensiuni da`),
  codificarea pushbuffer/GPFIFO (după `clc56f.h`, niciodată executată), inventarul firmware (listă + verificare structurală).
- **Blocate:** I4 (licența și transportul firmware-ului), I5 (IOMMU + DMA), I6–I8 (falcon, GSP, RPC), I9–I10 (canal, QMD, program).

## Progres (ESTIMARE argumentată)
- **Până la primul calcul GPU real: ~10 %.** Din cei 11 pași I0–I10, primii doi (cei mai ieftini, doar citire) sunt scriși;
  pașii I6–I8 (falcon + GSP-RM + RPC) sunt, după volumul de cod din nouveau/nova-core, partea cea mai mare (ESTIMARE: 70 % din efort).
  Fără I8 nu există canal de comenzi, deci nimic nu rulează.
- **Implementarea totală (driver + toolchain + runtime util lui Kronos pe GPU): ~5 %.** Runtime-ul CPU de referință e
  un câștig real pentru K6-CPU, dar motorul de calcul GPU (compilator/assembler SM86, kernels matmul) nu a început.
- **Următorul prag tehnic:** rularea `gpu identifica` pe workstation, într-o fereastră aprobată de Adrian: confirmă BOOT_0 = 0x174xxxxx,
  BAR-urile și ReBAR, fără nicio scriere. Apoi I2 (GFW_BOOT) și I3 (citirea VBIOS), ambele doar citire.
