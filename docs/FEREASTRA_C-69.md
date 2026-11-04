# Ferestrele cu Adrian după I6: I7 → I8 → I9/I10 → primul calcul (C-69)

*4 nov 2026, sesiunea Cyber-GPU 47 (cloud, fără placă). Listă de plan, nu sarcină: fiecare fereastră o face Adrian, cu el de față,
și pornește doar cu o comandă nouă în hub. **Nimic de aici nu a rulat pe RTX 3070 Ti.** Tot codul de mai jos e scris și testat
doar în SIMULARE (`gpu-sim`). Regulile din `docs/FEREASTRA_C-49.md` rămân valabile (pornire rece, oprire la primul rezultat neașteptat,
nicio reîncercare în aceeași fereastră, firmware-ul nu intră niciodată în repo sau în hub).*

**Ordinea e fixă.** O fereastră începe doar dacă cea dinainte a dat TRECE, verificat de noi pe jurnalul de pe stick.

## Fereastra 0 — I6 o dată, pornire rece (C-66, deja planificată)
- OS: Release `os-224`. Pași: ca în `docs/FEREASTRA_C-49.md` (secțiunea C-63).
- Trece dacă: „rezultat I6: TRECE”, rândul „WPR2 citit dupa I6: … (capat: ultimul bloc de 128 KiB)”.
- Noi, după: `verifica_jurnal_i6` pe jurnal → TRECE. **Abia atunci** cerem în hub comanda pentru fereastra 1.

## Fereastra 1 — I7: Booter pe SEC2 + pornirea GSP-RM
**Ce trebuie să existe înainte (le face sesiunea OS, după o comandă nouă — propunerea e în `integration/cyber-universe/C-69_i7-fereastra.md`):**
1. Firmware-ul 570.144 de pe stick, verificat (I4, deja în OS): `booter_load-570.144.bin`, `bootloader-570.144.bin`, `gsp-570.144.bin`.
2. Pregătirea în memoria calculatorului (I5, fără nicio scriere în placă): imaginea GSP-RM în pagini (radix3), argumentele LibOS,
   cozile de mesaje (2 × 256 KiB), `GspFwWprMeta` (256 B) — codul e în `nvidia-ga104::gsp` (`wpr_meta`, `libos`, `msgq`, `radix3`).
3. Booter-ul pregătit cu semnătura aleasă după siguranța SEC2 (`fuse`, registrul `0x824140 + 4·(id−1)`, doar citire).

**Pașii ferestrei** (pornire rece; aceeași pornire a OS-ului pentru tot):

| # | Pas | Ce se scrie | Dovada citită | Trece dacă |
|---|---|---|---|---|
| 1 | `gpu pregatire-i6` (doar citire) | nimic | ca la fereastra 0 | „WPR2 coborat”, 0 scrieri |
| 2 | Bus Master pornit (I5g) | 1 scriere în configurație | COMMAND recitit | bitul 2 pus |
| 3 | I6 (FWSEC-FRTS) | ~720 scrieri, 18 registre GSP (lista din FEREASTRA_C-49) | WPR2 citit după I6 | TRECE, forma de 128 KiB |
| 4 | ⚠️ **I7 (Booter pe SEC2)** | **18 registre, toate în fereastra SEC2** `0x840000–0x843FFF` (lista de mai jos); ~66 de scrieri în simulare, pe placă ~3 pe fiecare bloc de 256 B al Booter-ului | `read_after_i7`: 3 citiri, fără permis | cutia SEC2 0 = `0`, GSP RISC-V **ACTIV** |
| 5 | Bus Master oprit la loc | 1 scriere | COMMAND recitit | bitul 2 șters |
| 6 | Oprire completă (cablul scos 30 s) → Windows | — | `nvidia-smi`, Kronos pornește | da |

**Cele 18 registre ale lui I7** (fixate de testul `gsp::hs_load::tests::i7_writes_exactly_the_listed_sec2_registers`, T72): aceleași 18 ca
la I6, dar pe SEC2: `0x8403C0` reset (1, apoi 0) · `0x841668` nucleul falcon · `0x840624` FBIF_CTL · `0x84010C` DMACTL = 0 ·
`0x840600` TRANSCFG(0) · `0x840110`, `0x840128` sursa DMA · `0x840114`, `0x84011C`, `0x840118` câte un bloc · `0x841210`, `0x84119C`,
`0x841198`, `0x841180` verificarea semnăturii · `0x840104` adresa de pornire · **`0x840040`, `0x840044` = adresa `GspFwWprMeta`** (singura
diferență de valoare față de I6, unde sunt 0) · `0x840100` pornirea.

**Ce face placa singură după pasul 4:** Booter-ul (semnat NVIDIA) încarcă GSP-RM în zona protejată WPR2 și pornește procesorul RISC-V al
GSP. De aici GSP-RM **citește și scrie singur în memoria calculatorului** (cozile, prin DMA) → bufferele de la pregătire rămân rezervate
și neatinse până la oprirea completă. Nimic nu se scrie pe disc. Totul dispare la oprirea completă.

**NU în fereastra 1:** soneria GSP și primul mesaj (I8). I7 se oprește la „GSP RISC-V ACTIV”.

**Noi, după:** `cargo run --release -p nvidia-ga104 --example verifica_jurnal_i7 -- gpu-jurnal-00N.txt <adresa GspFwWprMeta>`
→ ordinea I6, WPR2, ordinea I7 (cutiile = adresa), RISC-V activ, 2 scrieri în configurație → „TRECE” sau „DE ANALIZAT”.

## Fereastra 2 — I8: primele mesaje cu GSP-RM
Pași 1–4 ca în fereastra 1, apoi:
- I8a–c: `SetSystemInfo` + `SetRegistry` în coada de comenzi, **soneria** (o scriere: `NV_PGSP_QUEUE_HEAD(0)`, în fereastra GSP);
- I8d: secvențiatorul cerut de GSP-RM — fiecare comandă trece prin `GuardedBar0` (o comandă în afara ferestrelor GSP/SEC2 = oprire, nu execuție);
- așteptarea `GSP_INIT_DONE` (cu termen-limită).
- Trece dacă: „GSP_INIT_DONE” primit; jurnalul are doar scrieri GSP/SEC2. **De pregătit înainte:** lista exactă a scrierilor I8 (ca T59/T72)
  și verificarea ei în jurnal — pas următor pentru noi.

## Fereastra 3 — I8e + I9: „golden image” și canalul de lucru
- Golden image (`gsp::golden`): client, dispozitiv, canal, contextele GR promovate — **doar mesaje RPC**, nicio scriere în registre;
- I9b–I9e: tabelele de pagini scrise în VRAM prin fereastra BAR0 (`vram_window`: 4 registre, jurnalizate), clientul de lucru, grupul de canale.
- Trece dacă: fiecare răspuns RM = 0 (succes); citirea înapoi a tabelelor din VRAM = ce am scris.

## Fereastra 4 — I10: primul calcul pe placă + măsurătoarea 1 („un singur program”)
- I10: lansarea `add64` (64 de adunări f32, compilat cu Mesa NAK) → rezultatul citit din VRAM, comparat bit cu bit cu `gpu-compute::reference`.
  **Abia atunci** scriem „executat pe GPU”.
- **Măsurătoarea 1 a țintei „un singur program pe simbol” (pregătită acum, T75):** în aceeași fereastră, după ce `add64` a trecut:
  un marcaj de start + 16 lansări `add64`, fiecare urmată de o eliberare de semafor **cu marcă de timp** (`launch_cost::encode`).
  Placa își scrie singură ora (ns) după fiecare lansare; `launch_cost::analyze` dă mediana/minimul/maximul costului unei lansări.
  Ce verificăm întâi pe placă: că marca de timp e la +8 în înregistrarea de 16 B (DEDUS; dacă nu, oprim măsurătoarea, nu ghicim).
- Trece dacă: `add64` corect + 17 înregistrări cu valorile în ordine și timpul crescător.
- Ce decide cifra: dacă o lansare costă puțin față de un strat Kronos, rămâne „câte un program pe calcul”; dacă costă mult, merităm
  „un singur program” (`docs/TINTA_UN_SINGUR_PROGRAM.md`, pașii 2–3).

## Fereastra 5 și după
Treptele matmul (T55) cu viteză măsurată, apoi un strat Kronos (comparat cu procesorul), apoi Kronos pe placă în OS — fiecare cu comandă nouă.

## Ce rămâne neschimbat în toate ferestrele
- Orice cod care scrie în placă e în spatele opțiunii `hw-write` (oprită implicit) și cere `HardwareWritePermit` + „DA” de la Adrian.
- Jurnalul fiecărui acces se scrie pe stick pe loc; după fiecare fereastră: poze + `raport/*.txt` + `raport/gpu-jurnal-*.txt` → hub `rezultate/`.
- Revenirea, oricând: butonul de pornire 10 s → cablul scos 30 s → Windows → `nvidia-smi` + Kronos.
