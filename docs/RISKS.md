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
| B3 | ~~Licența firmware NVIDIA în imaginea OS~~ **rezolvat 26 oct 17:10 (C-18):** DA, 570.144, fișier separat pe stick, niciodată în repo | — |
| B4 | Protocolul RPC GSP-RM nedocumentat oficial, legat de versiune | **versiunea fixată: 570.144 (C-18)**; structurile după antetele MIT `open-gpu-kernel-modules@570.144`; orice altă versiune e refuzată de `firmware::verify` |
| B5 | Lipsa unui compilator SM86 | **decis: Mesa NAK pe gazdă (C-18)**; rămâne de construit unealta — `TOOLCHAIN.md` |
| B6 | Kernelul: `map_mmio` nu garantează UC pe pagini deja mapate | glue-ul refuză; propunere de patch separat în OS (`integration/`) |
| B7 | Fereastra DMA a OS-ului < 4 GiB, fără zeroizare | suficientă pentru cozile GSP (0x40000); de revizuit |
| B9 | ~~GitHub Actions nu mai alocă calculatoare joburilor~~ **rezolvat 23 oct 19:10:** repo-ul e public, CI gratuit | — |
| B8 | Kernel fără SSE/AVX salvat | crate-urile merg în soft-float; performanța CPU cere XSAVE (ARHITECTURA §14.7) |

## Riscuri
| Risc | Impact | Măsură |
|---|---|---|
| Citire BAR0 cu placa în D3 | 0xffffffff, eventual eroare PCIe | verificăm D0 înainte; oprire la prima valoare 0xffffffff |
| Dimensionarea BAR pe GPU-ul principal | ecran negru / blocaj dacă se accesează placa în fereastră | întreruperi oprite, opțiune explicită, restaurare verificată |
| Efort GSP-RM subestimat | luni | K6-CPU merge înainte (backend CPU gata), experimente AMD în paralel |
| Codificări neverificate (push.rs) | comenzi greșite la primul test | marcate „neverificat pe GPU”; primul test real e cel mai mic posibil |

## Riscuri noi după decizia din 26 oct (firmware 570.144)
| Risc | Impact | Măsură |
|---|---|---|
| Fișier firmware greșit / corupt / altă versiune pe stick | GSP refuză pornirea sau, mai rău, pornește alt ABI decât cel codat | `firmware::verify`: nume exact, mărime exactă și SHA-256 față de valorile din `firmware.rs`; altfel `VerifyFailed`, nu se încarcă nimic |
| ABI-ul RPC r570 citit greșit din antete | mesaje respinse de GSP-RM, placa blocată până la repornire | structurile au teste de mărime/offset față de antete; primul test pe placă se face cu Adrian de față, cu repornire planificată |
| Firmware-ul NVIDIA ajunge din greșeală în repo | încălcarea regulii proiectului (și a deciziei C-18) | `.gitignore` pentru `*.bin` în `firmware/`; repo-ul conține doar nume, mărimi și sume |
| Cod care scrie în placă rulat din greșeală | stare nedefinită a plăcii care ține ecranul | tot ce scrie în placă pentru I5–I8 stă în spatele `HardwareWritePermit` (oprit implicit) și nu există în OS până la fereastra pe workstation |
