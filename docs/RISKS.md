# Riscuri, blocaje și politica de siguranță hardware

## Politica (aplicată în cod)
1. **Nivelul 0 nu scrie nimic** în placă: fără `enable()`, fără Bus Master, fără scrieri BAR0. Testele verifică zero scrieri
   (`probe::tests::level0_on_simulated_board_is_write_free`, `regs::tests::allow_list_enforced`).
2. BAR0 se citește **numai** la registrele din `regs::READ_ALLOWED` (azi: BOOT_0, BOOT_42). Nu există cale de scriere BAR0 în crate.
3. Dimensionarea BAR-urilor cere `SizingPermit` explicit, oprește decodarea, restaurează și verifică; la eșec → placa e „Faulted”.
   În kernel e o opțiune separată (`gpu dimensiuni da`), cu întreruperile oprite. Preferăm ReBAR (doar citire) pentru BAR1.
4. `reset_or_recover(Full)` → `NotPermitted`: GPU-ul principal nu se resetează fără protocol de recuperare.
5. Nicio adresă CPU nu ajunge la GPU fără poarta IOMMU (M2).
6. Revenirea la normal: nivelul 0 nu schimbă starea plăcii, deci repornirea normală în Windows rămâne neatinsă. Stick-ul OS
   pornește separat; Windows/Kronos nu sunt atinse (Secure Boot trebuie oprit temporar pentru stick — decizia lui Adrian).

## Blocaje (cu ce le deblochează)
| Id | Blocaj | Deblocare |
|---|---|---|
| B1 | Workstation-ul e al lui Kronos | fereastră aprobată de Adrian (30 min, Kronos oprit) |
| B2 | ~~IOMMU / Above-4G / ReBAR necunoscute~~ **rezolvat în mare (K0, 23 oct):** VT-d oprit, ReBAR oprit, BAR-uri sub 4 GiB; planul nu depinde de ele (`HARDWARE_WORKSTATION.md` §4) | rămâne doar poza Above-4G din BIOS |
| B3 | Licența firmware NVIDIA în imaginea OS | decizia lui Adrian pe `LICENCE.nvidia` §2.1.2 (OS MIT = OSI) |
| B4 | Protocolul RPC GSP-RM nedocumentat oficial, legat de versiune | fixarea unei singure versiuni (propunere: 570.144, ca nova-core; vezi `RESEARCH_I4_I8.md`) |
| B5 | Lipsa unui compilator SM86 | decizie NAK (MIT, pe gazdă) vs ptxas (EULA) — `TOOLCHAIN.md` |
| B6 | Kernelul: `map_mmio` nu garantează UC pe pagini deja mapate | glue-ul refuză; propunere de patch separat în OS (`integration/`) |
| B7 | Fereastra DMA a OS-ului < 4 GiB, fără zeroizare | suficientă pentru cozile GSP (0x40000); de revizuit |
| B9 | GitHub Actions nu mai alocă calculatoare joburilor (rularea 5, 23 oct, de 2 ori) — probabil limita de minute pentru depozite private | Adrian verifică „Billing and plans → Actions” pe GitHub; până atunci `tools/check.sh` local e verificarea |
| B8 | Kernel fără SSE/AVX salvat | crate-urile merg în soft-float; performanța CPU cere XSAVE (ARHITECTURA §14.7) |

## Riscuri
| Risc | Impact | Măsură |
|---|---|---|
| Citire BAR0 cu placa în D3 | 0xffffffff, eventual eroare PCIe | verificăm D0 înainte; oprire la prima valoare 0xffffffff |
| Dimensionarea BAR pe GPU-ul principal | ecran negru / blocaj dacă se accesează placa în fereastră | întreruperi oprite, opțiune explicită, restaurare verificată |
| Efort GSP-RM subestimat | luni | K6-CPU merge înainte (backend CPU gata), experimente AMD în paralel |
| Codificări neverificate (push.rs) | comenzi greșite la primul test | marcate „neverificat pe GPU”; primul test real e cel mai mic posibil |
