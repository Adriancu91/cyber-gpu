# Starea implementării — cyber-gpu

## Sesiunea 4 — 24 octombrie 2026, ~01:20–01:40 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Doar calcul pe procesorul gazdei.
C-16 (integrarea în OS) a fost făcut între timp de sesiunea OS, pe commit-ul nostru `41e44ba` (Release OS `os-189`).

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| 0. Tag-ul `stabil-2026-10-23` | există local, adnotat, pe `41e44ba`; push reîncercat o dată | **BLOCKED**: HTTP 403 (politica proxy-ului cloud: tag-urile nu se pot urca). OS-ul e fixat pe commit `41e44ba`, deci nu blochează nimic |
| Compatibilitate | `git diff 41e44ba HEAD` pe `gpu-core`, `gpu-pci`, `nvidia-ga104`, `gpu-compute`: nicio linie `pub` ștearsă | PASS (doar adăugări) |
| c. Backend CPU, mai multe nuclee | `gpu-compute::rows`: `split_rows`, `softmax_rows_range` (cauzal, cu rândul absolut), `causal_attention_rows`, `cross_entropy_terms` + `cross_entropy_mean`; fără stare globală, ieșiri disjuncte | PASS (gazdă): 4 teste, rezultatul unit **identic bit cu bit** cu `reference` pentru 1, 2, 3, 4, 7, 16, 40 de părți; pe nucleele OS-ului NOT TESTED |
| b. Sprijin integrare | `integration/cyber-universe/UTILIZARE.md`: fixarea actuală, promisiunea de compatibilitate, ce e nou după `41e44ba`, exemple (identificare doar-citire, SIMULARE cu configurația workstation-ului, operație pe n nuclee cu toleranța 0); README-ul `gpu-compute` explică ce se împarte prin felii și ce cere `rows` | gata |
| CI | rularea 11 (codul nou) verde | PASS |
| Placă reală | — | NOT TESTED (Adrian plecat) |

Verificări (`tools/check.sh`, rustc 1.97.0 stable): format PASS, clippy `-D warnings` PASS, **106 teste PASS** (1 ignorat: viteza),
build `x86_64-unknown-none` PASS. Nicio scriere nouă în registre.

**Următorul pas:** fără placă și fără decizia de licență **nu mai e nimic util de făcut** (pașii a–d sunt acoperiți; I4+ depind de
firmware-ul NVIDIA, adică de decizia lui Adrian). Cu placa: testul I0–I3 (+5c FWSEC) din `TEST_MATRIX.md` partea B.
**Blocate de Adrian:** licența firmware-ului, compilatorul (NAK vs ptxas), fereastra pe workstation, urcarea tag-ului.

---

## Sesiunea 3 — 23 octombrie 2026, ~19:25–19:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Testele pe imagini ROM sunt SIMULARE; restul e aritmetică pură.
Pașii a–e din prompt erau deja gata din sesiunea 2; am făcut pașii următori notați atunci.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| C-16 (hub): punct de revenire | tag-ul adnotat `stabil-2026-10-23` creat local pe `41e44ba` | **BLOCKED**: push-ul tag-ului refuzat (HTTP 403, politica proxy-ului sesiunii cloud); ramura se urcă normal. OS-ul poate fixa dependența pe commit-ul `41e44ba492c606ba5abf15126210caf1313b9e6e` (C-16 permite „tag / commit”) |
| CI | repo-ul public (decizia lui Adrian, 19:10) → GitHub Actions merge din nou | PASS: rulările 7 și 8 verzi |
| I3b: descriptorul FWSEC | `nvidia-ga104/src/fwsec.rs`: jetonul BIT 0x70 → tabela PMU → aplicația 0x85 (FWSEC_PROD) → `FalconUCodeDescV3`; unde stau semnăturile și ucode-ul (și dacă trece dintr-o imagine FWSEC în următoarea); **ucode-ul nu e citit niciodată** (test dedicat); alegerea semnăturii după siguranță (formulă DEDUS, nefolosită pe placă) | PASS (SIMULARE), 11 teste |
| I5 pregătire: regiunile din VRAM | `nvidia-ga104/src/fb_layout.rs`: VGA, FRTS, încărcător, ELF, heap WPR2 (LIBOS3), WPR2, heap 1 MiB; pentru 8 GiB GSP ia ultimii 162 MiB (cu mărimi SIM ale firmware-ului) | PASS (aritmetică), 6 teste |
| I5 pregătire: radix3 | `nvidia-ga104/src/radix3.rs`: lista de pagini prin care GSP își găsește imaginea în RAM; verificată prin parcurgere, octet cu octet | PASS (aritmetică), 3 teste |
| Placă reală | — | NOT TESTED (Adrian plecat) |

Verificări (`tools/check.sh`, rustc 1.97.0 stable, Linux x86_64): format PASS, clippy `-D warnings` PASS, **102 teste PASS** (1 ignorat:
măsurătoarea de viteză), build `x86_64-unknown-none` PASS. Nicio scriere nouă în registre; niciun registru nou pe lista permisă.

**Următorul pas:** fără placă și fără decizia de licență **nu mai e nimic util de făcut**: următorul (mesajele RPC GSP-RM) depinde de
versiunea firmware-ului NVIDIA, deci de decizia lui Adrian. Cu placa: testul I0–I3 din `TEST_MATRIX.md` partea B (acum și pasul 5c, FWSEC).
**Blocate de Adrian:** licența firmware-ului (I4 → tot ce urmează), compilatorul (NAK vs ptxas), fereastra pe workstation, tag-ul de revenire.

---

## Sesiunea 2 — 23 octombrie 2026, ~13:50 (ora României), în cloud, FĂRĂ placă

**Nimic din această sesiune nu a fost testat pe RTX 3070 Ti.** Toate testele hardware sunt pe modelul `gpu-sim` = SIMULARE.

| Pas | Ce s-a făcut | Rezultat |
|---|---|---|
| a. Datele workstation-ului | `docs/HARDWARE_WORKSTATION.md`; `gpu-sim::workstation` (FAPT din K0) + `SimConfig::workstation_rtx3070ti()`; 2 teste noi | PASS (SIMULARE) |
| a'. Dependența de ReBAR / Above-4G | analiză: I0–I8 **nu** depind de ele; BAR1 256 MiB ajunge; VT-d oprit simplifică DMA, dar fără plasă | gata (DEDUS) |
| b. CI | mutat în `.github/workflows/ci.yml` — push acceptat; jobul `kernel-integration` scos (repo OS privat + patch depășit) | PASS: rulările 2–4 verzi pe GitHub; **rularea 5 (doar documente schimbate) BLOCKED**: joburile nu primesc calculator de la GitHub (0 pași, ~5 s, de 2 ori) — cauză probabilă: limita de minute/plată Actions pentru depozite private (neverificat) |
| c. I2 GFW_BOOT | `nvidia-ga104/src/gfw.rs`: 2 registre noi pe lista permisă (0x118128, 0x118234), limită 4 s, zero scrieri; 5 teste | PASS (SIMULARE) |
| c. I3 VBIOS | `nvidia-ga104/src/vbios.rs`: fereastra PROM (BAR0+0x300000), IFR/RFRD, lanț PCIR/NPDE, tabela BIT; limite 2 s / 300 000 citiri; 8 teste pe imagini sintetice | PASS (SIMULARE) |
| c. Planul | I2, I3 marcate implementate (doar citire); `initialize()` se oprește acum la **I4** (licența); plan de test pe placă în `TEST_MATRIX.md` | PASS (SIMULARE) |
| d. Backend CPU | `gpu-compute/src/kernels.rs`: matmul pe blocuri + matmul cu greutăți bf16 (op 9), **identice bit cu bit** cu referința; bf16 verificat exhaustiv (65 536 valori) și față de un oracol de rotunjire | PASS (gazdă) |
| e. Cercetare I4–I8 | `docs/RESEARCH_I4_I8.md` + argumentele pentru deciziile lui Adrian; fără cod care scrie în placă | gata |
| Placă reală | — | NOT TESTED (Adrian plecat; B1) |

Verificări (`tools/check.sh`, rustc 1.97.0 stable, Linux x86_64): format PASS, clippy `-D warnings` PASS, **82 de teste PASS** (1 ignorat:
măsurătoarea de viteză), build `x86_64-unknown-none` PASS.

**Următorul pas:** fără placă — extragerea descriptorului FWSEC din VBIOS (doar citire, pe imagine sintetică) și modelul aritmetic al
regiunilor WPR2/`GspFwWprMeta`; cu placa — testul I0–I3 din `TEST_MATRIX.md` (partea B), după ce OS-ul are comenzile `gpu gfw` / `gpu vbios`.
**Blocaj nou:** GitHub Actions nu mai pornește joburile (vezi rândul b) — *rezolvat în sesiunea 3: repo-ul a devenit public.*
**Blocate de Adrian:** licența firmware-ului (I4 → tot ce urmează), compilatorul (NAK vs ptxas), fereastra pe workstation.

---

*Sesiunea 1 (mai jos), 23 octombrie 2026. Etichete: FAPT (verificat), SIMULARE (doar pe modele software), ESTIMARE, NECUNOSCUT.*

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
| G3 cercetare/inițializare | **cercetare gata; sesiunea 2: I2–I3 implementate (doar citire, SIMULARE), cod oprit la I4** | `docs/GA104_INITIALIZATION.md`, `init.rs` (plan I0–I10) |
| G4 memoria GPU | **parțial** | invarianți + allocator + DMA policy + poarta IOMMU (logică); sesiunea 3: regiunile GSP din VRAM + radix3 (aritmetică); nimic pe VRAM real |
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
