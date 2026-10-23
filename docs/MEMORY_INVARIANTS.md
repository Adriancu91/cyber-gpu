# Invarianții de memorie (G4)

Implementați în `crates/gpu-memory`, fiecare cu teste.

| Id | Invariant | Unde | Test |
|---|---|---|---|
| M1 | Adresa fizică CPU, adresa de magistrală, offset-ul VRAM și adresa virtuală GPU sunt **tipuri diferite**; nicio conversie implicită | `addr.rs` | compilatorul |
| M2 | O adresă fizică CPU devine adresă de magistrală **numai** dacă IOMMU e cunoscut oprit/pass-through; `Unknown` → `Blocked` | `addr.rs` | `iommu_gate` |
| M3 | Intervalele alocate nu se suprapun, stau în regiune, libere + alocate = total, cele libere sunt mereu unite | `range.rs` | `randomized_invariants` (5000 de operații) |
| M4 | Orice acces printr-un buffer e verificat față de mărime, cu aritmetică fără depășire | `region.rs` | `bounds` |
| M5 | Memoria gazdei dată dispozitivului: aliniată la pagină, în fereastra DMA a OS-ului (16 MiB–4 GiB azi), fixată (`frame::pin`) | `dma.rs` | `window` |
| M6 | Handle-urile au generație; un handle eliberat nu atinge niciodată un slot refolosit | `buffers.rs` | `stale_handles_rejected` |

## Ce **nu** presupunem
- BAR1 **nu** este toată VRAM-ul (256 MiB fără ReBAR; 8 GB VRAM).
- Adresele fizice CPU **nu** sunt direct utilizabile de GPU (M2).
- Maparea MMIO a kernelului **nu** e garantat UC dacă pagina era deja mapată → glue-ul refuză (`integration/`).
- Coerența cache: OS-ul nu are PAT/WC; bufferele DMA trebuie golite explicit (CLFLUSH, ca la rtl8168) până la un mecanism mai bun.
- Sincronizarea CPU/GPU se face numai prin fence-uri monotone (`gpu-command::fence`), cu așteptări limitate.
