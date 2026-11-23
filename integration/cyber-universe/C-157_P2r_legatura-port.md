# C-157 → OS: citirea portului de deasupra plăcii video (de ce RAM ↔ VRAM merge cu ~6 GB/s)

De la Cyber-GPU, sesiunea 190 (23 nov 2026), pentru decizia lui Adrian din 23 nov 20:11 („nimic din antrenare în RAM pe placa
reală până nu e explicată și reparată viteza PCIe”). Cod: `gpu_pci::link` (commit `408c69c`), teste T198 (7, SIMULARE).
Planul și predicțiile, scrise înainte: `docs/C-157_PCIE_LEGATURA.md` (`b144292`). **Totul e adăugare**; nimic din `gpu-pci`
nu s-a schimbat la ce folosește OS-ul azi. **Doar citire: nicio scriere în placă sau în porturi.**

## 1. Ce cerem
În turul plăcii (`gpu tur` / raportul workstation-ului), imediat după rândul `pci.legatura`, două rânduri noi:

```
pci.legatura.port = gen3 x8 (max gen3 x8) la b2:00.0 8086:2030
pci.legatura.cauza = SlotNarrow: slotul are mai putine linii decat placa: doar alt slot (mai lat) ajuta; plafon azi 7.88 GB/s, cel mai bun comun 7.88 GB/s
```

Pașii (toți doar citire a configurației PCI, pe care OS-ul o citește deja pentru `lspci`):
1. Ia lista punților deja enumerate (în raport: `b2:00.0 8086:2030`, `b2:02.0 8086:2032`, `b2:03.0 8086:2033`, …).
2. `gpu_pci::link::find_upstream_port(&mut punti, bus_placa)` — alege puntea cu magistrala secundară (0x19) = magistrala plăcii
   (`0xb3` pe workstation) și care e port rădăcină / port de comutator în jos. Dacă dai câte o punte pe rând:
   `secondary_bus(&mut p) == Some(bus)` și `is_downstream_port(&mut p)`.
3. `gpu_pci::caps::pcie_link(&mut port)` și `gpu_pci::caps::pcie_link(&mut placa)` → `gpu_pci::link::diagnose(&placa, Some(&port))`.
4. Scrie `verdict` (`{:?}`), `verdict.text()`, `now_gbps` și `best_gbps` (plafonul brut pe sens, GB/s).

Dacă portul nu se găsește: `pci.legatura.port = negasit` și `diagnose(&placa, None)` (`NoPort`).

## 2. Ce prezicem (de comparat)
- **P1:** `b2:00.0` (8086:2030, portul A al unui grup de 16 linii împărțit x8 + x4 + x4 — porturile C 2032 și D 2033 există,
  B 2031 nu) are **max gen3 x8** → verdictul **`SlotNarrow`**: placa stă în **slotul 1** al lui Dell 5820 (cablat x8, 50 W).
- **P3 (dacă iese altfel):** port **x16** cu legătură x8 → `TrainedNarrow` = placă prost așezată / contacte / prelungitor.
- În repaus legătura e `gen1 x8`: verdictul tot `SlotNarrow` (lățimea nu se schimbă în repaus).

## 3. Nou pentru placă
Nimic: doar citiri ale configurației PCI a portului rădăcină, aceleași ca la `lspci`. Poate intra în orice ciclu.
