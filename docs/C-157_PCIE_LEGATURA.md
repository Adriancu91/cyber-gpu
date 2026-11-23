# C-157 — de ce RAM ↔ VRAM merge cu ~6 GB/s și cum se repară (decizia lui Adrian, 23 nov 2026 20:11)

**Cerut (DECIZII 23 nov 20:11, nota centrului de comandă din C-157 și C-158):** nimic din antrenare nu se mută în RAM pe placa
reală până nu e explicată și reparată viteza PCIe (~6 GB/s în os-376, față de ~12 prezis). Prioritate: citirea legăturii
(LnkCap / LnkSta) și cauza. Acest document: planul și predicțiile, scrise **înainte** de cod (sesiunea 190).

## 1. Ce știm deja (citit, nu dedus)

- Rândul turului nostru, la fiecare pornire pe workstation: `pci.legatura = gen3 x8 (max gen4 x16)` — placa **poate** gen4 x16
  (LnkCap al plăcii), dar s-a înțeles cu slotul pe **gen3 x8** (LnkSta). În repaus coboară la `gen1 x8` (economie normală de
  energie; lățimea rămâne x8, deci lățimea e semnul de încredere, nu viteza).
- os-376: sus 5,97 GB/s (cel mai bun 6,26), jos 6,76 GB/s.
- Lista `lspci` a OS-ului (rapoartele workstation-ului din 21–23 nov): placa video la **b3:00.0**, discul NVMe la **b4:00.0**,
  deasupra lor trei porturi rădăcină ale procesorului pe magistrala b2: **b2:00.0 (8086:2030), b2:02.0 (8086:2032),
  b2:03.0 (8086:2033)**. Nu există b2:01.0.
- Procesorul: Core i9-10980XE (48 de linii PCIe ale procesorului, toate Gen3).
- Manualul Dell Precision 5820 (tabelul „PCIe slots”, coloana procesoarelor i9 / Xeon): **slotul 1 = x8** (50 W, cel mai aproape
  de procesor), **slotul 2 = x16** (300 W), slotul 3 = x1 (din chipset), **slotul 4 = x16** (300 W), slotul 5 = x4 (din chipset),
  slotul 6 = PCI vechi. „Toate sloturile sunt Gen3 (8 GT/s)”. Fișa tehnică: „2 × PCIe x16, 1 × PCIe x16 cablat ca x8, …”.

## 2. Ce deducem (DEDUS — de confirmat cu o citire)

Porturile 8086:2030 / 2032 / 2033 sunt porturile A, C și D ale unui grup de 16 linii al procesorului (în familia Skylake-SP /
Cascade Lake, un grup de 16 linii se poate împărți x16, x8+x8 sau x8+x4+x4 / x4×4). Fără portul B (2031) și cu C și D prezente,
grupul e împărțit **x8 + x4 + x4**: portul A are cel mult 8 linii. Placa stă sub portul A (b3), NVMe-ul sub altul (b4).
Sloturile x16 ale lui Dell (2 și 4) au fiecare un grup întreg; grupul împărțit x8 + x4 + x4 se potrivește cu **slotul 1 (x8)**.

**Cauza probabilă: placa video stă în slotul 1, cablat x8.** Nu e un contact prost, nu e o setare: slotul are doar 8 linii.

## 3. Predicții (scrise înainte de citire)

| # | Predicție | Cum se verifică |
|---|---|---|
| P1 | Portul rădăcină de deasupra plăcii (b2:00.0) are în LnkCap **max gen3 x8** (nu x16) | OS-ul citește configurația portului (doar citire), rândul `pci.legatura.port` |
| P2 | Diagnosticul automat spune „slotul are doar 8 linii” (nu „placa s-a înțeles mai îngust decât poate slotul”) | rândul `pci.legatura.cauza` |
| P3 | Dacă P1 iese **x16**, cauza e alta: placa prost așezată / murdărie pe contacte / prelungitor — atunci se reașază placa, nu se mută | același rând |
| P4 | Viteza (gen3) nu se poate repara pe 5820: toate sloturile sunt Gen3; plafonul real e gen3 x16 | manualul Dell |
| P5 | Mutată în slotul 2 sau 4: `pci.legatura = gen3 x16 (max gen4 x16)` și os-376 refăcut ~11,5–13 GB/s în ambele sensuri (×1,8–2,0 față de azi) | os-376 rulat din nou după mutare |

Plafonul brut al legăturii, pe sens: gen3 = 8 GT/s × 128/130 ÷ 8 ≈ 0,985 GB/s pe linie → x8 ≈ 7,88 GB/s, x16 ≈ 15,75 GB/s.
os-376 = 79 % (sus) / 86 % (jos) din plafonul brut x8 — normal pentru PCIe (antetele pachetelor și confirmările iau ~10–20 %).

## 4. Ce scriem (doar citire, nicio scriere în placă sau în porturi)

1. `gpu-pci`, modul nou `link` (adăugare): `link_bandwidth_gbps(speed, width)` (plafonul brut), `diagnose(placa, port)` → un
   verdict (`Full`, `SlotNarrow`, `TrainedNarrow`, `SlotSlow`, `TrainedSlow`, `Idle`, `NoPort`) cu textul în română pentru
   tur, și `find_upstream_port(...)` — alegerea portului din amonte după magistrala secundară (registrul 0x19 al punții).
2. Teste (T198): verdictul pe cazul de azi (placă gen4 x16, legătură gen3 x8, port gen3 x8 → `SlotNarrow`); pe cazul P3 (port
   x16, legătură x8 → `TrainedNarrow`); repausul gen1 x8 nu se confundă cu o problemă de viteză; plafonul brut; citirea portului
   pe o configurație simulată nu face nicio scriere.
3. Nota pentru OS: `integration/cyber-universe/C-157_P2r_legatura-port.md` — în tur, după `pci.legatura`: găsește puntea cu
   magistrala secundară = magistrala plăcii, citește-i capabilitatea PCIe (doar citire) și scrie `pci.legatura.port` și
   `pci.legatura.cauza`. Nimic nou trimis plăcii.
4. Pentru Adrian (fizic, fără cod): pașii de mutare a plăcii în slotul 2 sau 4 — în raportul din hub.

## 5. Rezultat

Sesiunea 190: codul `gpu_pci::link` (`408c69c`), T198 7 teste PASS (SIMULARE), `tools/check.sh` 952 PASS; nota pentru OS
`integration/cyber-universe/C-157_P2r_legatura-port.md`. **Citirea portului pe placă: NOT TESTED** (P1–P3 așteaptă un ciclu OS
cu rândurile `pci.legatura.port` / `pci.legatura.cauza`). P5 (mutarea plăcii) așteaptă pe Adrian.

**Pentru Adrian (fizic, fără cod, când vrea):** calculatorul oprit și scos din priză; placa video mutată din slotul 1 (cel mai
aproape de procesor) într-unul din sloturile **x16 întregi: 2 sau 4** (manualul Dell îi numerotează; sunt cele de 300 W), cu
ambele cabluri de alimentare puse la loc. Înainte, în Windows, GPU-Z → „Bus Interface” arată azi `PCIe x16 3.0 @ x8 3.0`;
după mutare trebuie să arate `@ x16 3.0`. Apoi un os-376 din nou (așteptat ~11,5–13 GB/s).
Atenție: slotul 4 poate fi acoperit de o placă dublă din slotul 2 sau de răcitor — se alege slotul liber în care încape placa.

## Surse
- PCI Express Base Specification: Link Capabilities (offset 0x0c al capabilității PCIe), Link Status (0x12), tipul portului
  (PCIe Capabilities, biții 7:4), registrul „Secondary Bus Number” al punții (0x19).
- [Dell Precision 5820 Tower — manualul proprietarului (tabelul „PCIe slots”)](https://dl.dell.com/topicspdf/precision-5820-workstation_Owners-Manual2_en-us.pdf)
- [Dell Precision 5820 Tower — sloturi (pagina manualului)](https://www.dell.com/support/manuals/en-us/precision-5820-workstation/precision_5820_om_pub/pcie-slots)
- interne: `docs/C-157_RAM_OFFLOAD.md` §5, `docs/C-157_P2_COMPARATII.md` (os-376), rapoartele workstation-ului din hub (`lspci`).
