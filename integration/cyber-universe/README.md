> **DEPĂȘIT (23 oct 2026):** sesiunea OS și-a scris propria comandă `gpu` (P6-11, nivel 0, pe toate magistralele-rădăcină) și
> a judecat observațiile de mai jos în `cyber-universe/docs/research/OBSERVATII-CYBER-GPU.md`. Patch-ul rămâne doar ca istoric;
> observațiile noi merg în hub, `integrare/cyber-gpu/`.

> **Pentru folosirea de acum a crate-urilor din OS, vezi [`UTILIZARE.md`](UTILIZARE.md).**

# Raport de integrare cu cyber-universe

**Nimic din acest director nu e aplicat în `Adriancu91/cyber-universe`.** Patch-ul se aplică manual, după revizuire,
conform regulii „niciun proiect nu modifică repo-ul altuia”.

## Conținut
- `kernel/src/drivers/gpu.rs` — fișier nou: glue-ul kernel ↔ crate-urile `cyber-gpu` (ECAM → `ConfigSpace`,
  prima pagină BAR0 UC → `Mmio`), comenzile `gpu identifica`, `gpu plan`, `gpu dimensiuni da`.
- `0001-C-11-gpu-nivel0.patch` — dependențele în `kernel/Cargo.toml` (git, fixate pe commit), `pub mod gpu;`, comanda `gpu` în shell.

```bash
cp <cyber-gpu>/integration/cyber-universe/kernel/src/drivers/gpu.rs kernel/src/drivers/gpu.rs
git apply <cyber-gpu>/integration/cyber-universe/0001-C-11-gpu-nivel0.patch
```

Verificat: `git apply --check` trece pe `cyber-universe@2d038de`. Compilarea kernelului cu patch-ul e făcută de jobul CI
`kernel-integration` din acest repository (în runner); local: NOT TESTED.

## D-013
O singură schimbare de comportament: comanda nouă `gpu`. Nu se execută nimic la pornire. Implicit doar citire;
singura cale care scrie (`gpu dimensiuni da`) e o opțiune explicită, cu valorile înainte/după în jurnal.

## Incompatibilități descoperite (de decis în cyber-universe, propuneri separate)
1. **`paging::map_mmio` sare peste paginile deja mapate** (`translate(page).is_some()`) fără să verifice atributele de cache.
   Pentru un GPU, o mapare cache-abilă a registrelor e incorectă. Glue-ul refuză dacă pagina era deja mapată.
   Propunere: `map_mmio` să întoarcă eroare sau să remapeze cu UC când găsește o mapare existentă cu alte atribute.
2. **Nu există dimensionarea BAR-urilor** în `pci`. Implementată în `gpu-pci::bar::size_bars` (cu decodare oprită + verificare).
3. `pci::Device` are numai acces de 32 de biți; `gpu-core::ConfigSpace` derivă 8/16 biți din citiri de 32 (corect pentru ECAM).
4. Fereastra DMA (`16 MiB – 4 GiB`) și lipsa zeroizării în `alloc_contiguous_between` — de ținut minte pentru cozile GSP.
5. `cli/sti` nu au un ajutor comun în `arch::x86_64`; glue-ul are `without_interrupts` local. Propunere: mutat în `arch`.
6. Dependențele git cer ca `cyber-gpu` să fie public și ca ramura cu commitul fixat să fie împinsă pe GitHub.
