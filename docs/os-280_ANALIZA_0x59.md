# După os-280 — ce spun cele 844 de înregistrări NOCAT și ce îi lipsea lui GSP-RM (`bDmemStack`)

Aplică **regula „trei surse”** (DECIZII.md, 11 nov 14:41) și comanda **C-107**: NVIDIA RM 570.144, tinygrad, nouveau — plus **nova-core** (Rust, Linux, tot 570.144).

Sesiunea 87, 11 noiembrie 2026 (cloud, fără placă). Jurnalul analizat: `Cyber-hub/rezultate/os-automat/2026-11-11/gpu-jurnal-008.txt`
(2438 rânduri, testul `os-280` de la ~14:20, `rezultate/2026-11-11-workstation-os280/REZULTAT.md`). Unealta: `verifica_jurnal_lant <jurnal> 0x9c66000`.

Etichete: **OBSERVAT** = scrie în jurnalul de pe placă; **SPECIFICAȚIE** = citit în codul public (MIT: open-gpu-kernel-modules 570.144, tinygrad; nouveau
doar citit, nimic copiat); **DEDUS** = concluzia noastră, neprobată pe placă.

## 1. Ce spune jurnalul (OBSERVAT)

- I6 și I7 trec (a 7-a oară); secvențiatorul: toate cele 420 de comenzi, `CORE_RESUME` prin `CPUCTL_ALIAS`, Booter a rulat (SCRATCH_14 bit 26 pus după start),
  GSP RISC-V activ (`0x80`), GSP MAILBOX0 `0xcb` după resume.
- `GSP_INIT_DONE`: `rpc_result = 0x59` = `NV_ERR_OPERATING_SYSTEM`, la fel ca la os-279. Eroarea vine din `GSP_INIT_DONE` (nu din coadă).
- **844 de înregistrări NOCAT** (`0x1020`), **toate de tip 5 = ASSERT**, `source = "ASSERT"`, `subsystem = 0`, **toate ÎNAINTE de cererea
  secvențiatorului** (rândurile 1498–1625; „cererea GSP-RM, comenzile urmeaza” la 1626), **niciuna după**. 16 jurnalizate în întregime (fiecare alt
  `errorCode`), 828 numărate ca „feluri noi peste limită” → **toate cele 844 au adrese diferite**.
- Cele 16 adrese jurnalizate (față de începutul imaginii GSP-RM) sunt **crescătoare**: `0x019ef3c6`, `0x019efd78`, `0x019efdb8`, `0x019efdf0`, `0x019efe2e`,
  `0x019efe54`, `0x019efe7a`, `0x019efea2`, `0x019efee4`, `0x019eff4c`, `0x019eff74`, `0x019effb4`, `0x019effee`, `0x019f0014`, `0x019f003a`, `0x019f0060`
  (pași de 0x26–0x9b2 octeți). `linia = 0` la toate. **Stiva de apeluri e goală** (toate cele 10 adrese 0) și `count = 1` — de aceea jurnalul nu are rânduri „apel #n”.
  Marcajele de timp GSP cresc cu ~1,1 ms între înregistrări consecutive.
- **2 × `UCODE_LIBOS_PRINT`** de la `ENG_PMU` (`ucodeEngDesc 0xf3d72200`), 16 și 24 de octeți, chiar înainte de `GSP_INIT_DONE` (după `CORE_RESUME`).

## 2. Ce înseamnă forma înregistrărilor (SPECIFICAȚIE: `journal.c`, `nvassert.c` din 570.144)

- Pe GSP (`NVOS_IS_LIBOS`), `NV_ASSERT`/`NV_ASSERT_OK` care pică apelează `nvAssertFailed()` / `nvAssertOkFailed(status)` → `rcdbRmAssert(…)` cu
  **`NV_RM_ASSERT_UNKNOWN_LINE_NUM` (= 0)** și `ip = adresa de întoarcere` (`portUtilGetReturnAddress`). De aici `linia = 0` la toate: **nu e o
  eroare de decodare**, firmware-ul nu raportează linii. Codul `NV_STATUS` al unui `NV_ASSERT_OK` eșuat merge în câmpul `level` al înregistrării din jurnalul RM,
  **nu** în înregistrarea NOCAT → nu-l putem vedea.
- `_rcdbNocatReportAssert`: `errorCode = (ip − loadAddress) & 0xffffffff | linia << 32`; `diagBuffer = { callStack[10] − loadAddress; count }`. Pe GSP stiva
  nu e completată (toate 0) — consistent cu jurnalul. Deduplicarea („același assert ca ultimul”) compară stivele și cere intrarea din jurnalul NOCAT local; pe GSP
  înregistrările pleacă oricum prin RPC, deci nu se pierd.
- **Adrese diferite, crescătoare, 38–100 de octeți una de alta** = **locuri de apel diferite, în ordinea codului**: o porțiune lungă de cod „în linie dreaptă” în care
  verificările cad una după alta (nu o buclă pe același loc — aceea ar da aceeași adresă). Ce funcție e: **nu se poate spune fără simbolurile GSP-RM**, pe care nu le
  deschidem (regula proiectului; RISKS.md).
- **nouveau r570 ignoră implicit înregistrările NOCAT** (`r570_gsp_drop_post_nocat_record`: handler gol pentru `GSP_POST_NOCAT_RECORD` și `GSP_LOCKDOWN_NOTICE`
  sub nivelul de depanare). Un driver care merge nu și-ar pune un „coș de gunoi” pentru un mesaj care nu vine → **și pe pornirile bune GSP-RM 570 trimite
  înregistrări NOCAT**. DEDUS: cele 844 de asserturi **pot fi zgomot normal al firmware-ului 570**, nu neapărat cauza lui 0x59. Nu le mai urmărim ca piste principale;
  rămân jurnalizate.

## 3. Comparația câmp cu câmp: ce trimitem noi față de cele patru drivere care pornesc GSP-RM 570 (SPECIFICAȚIE)

Referințe: **RM** = open-gpu-kernel-modules 570.144 (`rpc.c` `rpcGspSetSystemInfo_v17_00`, `kernel_gsp.c` `kgspPopulateGspRmInitArgs`,
`kgspSetupLibosInitArgs`, `kernel_gsp_tu102.c` `kgspCalculateFbLayout_TU102`, `osinit.c`); **nouveau** = Linux `nvkm/subdev/gsp/rm/r570/gsp.c`, `tu102.c`,
`rm/r535/gsp.c` (570.144 pe GA10x e varianta 1 în `ga102.c`); **tinygrad** = `runtime/support/nv/ip.py` (pornește GSP-RM 570.190 pe GA102/AD102 fără modul nucleu);
**nova** = Linux `drivers/gpu/nova-core` (`gsp/fw.rs`, `gsp/fw/commands.rs`, `gsp/commands.rs`, `gsp.rs`; firmware 570.144). nouveau și nova sunt GPL: doar citite, nimic copiat.

| Structură / câmp | RM | nouveau r570 | tinygrad | nova-core | noi (până la sesiunea 86) | verdict |
|---|---|---|---|---|---|---|
| `GSP_ARGUMENTS_CACHED.bDmemStack` | **1** (`RmGspStackPlacement` implicit 0 → stiva în DMEM; 0 doar cu cheia `=1`) | **1** (`args->bDmemStack = 1`) | **1** (`bDmemStack=True`) | **1** (`bDmemStack: 1`) | **0** | **singura diferență față de toate patru → reparat (sesiunea 87)** |
| `GSP_ARGUMENTS_CACHED` restul (cozi, SR, gpuInstance 0, profiler 0) | la fel | la fel | la fel | la fel | la fel | ok |
| Regiuni LibOS | LOGINIT, LOGINTR, LOGRM, (LOGMNOC), RMARGS | LOGINIT, LOGINTR, LOGRM, RMARGS (64 KiB fiecare, RMARGS 4 KiB) | + LOGMNOC, LOGKRNL | ca nouveau | ca nouveau | ok (nouveau și nova merg cu 4) |
| `GspFwWprMeta` așezare | formula din `kgspCalculateFbLayout_TU102` | identică (`tu102_gsp_oneinit`) | heap fix 129 MiB | aceeași formulă (`fb.rs`) | identică cu nouveau (heap 127 MiB = 22+8+1+96, `gsp_fw_heap.h`) | ok |
| `GspFwWprMeta.flags` | `CLOCK_BOOST` (bit 0) pe TU/GA/AD | 0 | 0 | 0 | 0 | ok (nouveau/tinygrad/nova merg cu 0); opțiune de rezervă |
| `GspFwWprMeta.pmuReservedSize` | doar GH100+ | doar GB10x/GB20x | doar `fmc_boot` | 0 pe GA10x | 0 | ok pe GA104 |
| Registru | multe chei + `RMSecBusResetEnable=1`, `RMForcePcieConfigSave=1` | cele două + `RMDevidCheckIgnore=1` | cele două | cele trei (+ `RMSetSriovMode` doar la vGPU) | cele două | ok |
| `GspSystemInfo` BAR0/1/3, BDF, `maxUserVa`, oglinda PCFG `0x88000/0x1000`, ID-uri PCI | la fel | la fel (`pci_dev_id` = bus<<8\|devfn) | la fel | la fel (`maxUserVa` = 2^47 − 4096, ca noi) | la fel | ok |
| `GspSystemInfo.acpiMethodData` | completat (DOD/JT/CAPS/…) | completat dacă `CONFIG_ACPI` | **0** | **0** | **0** | ok (tinygrad și nova merg cu 0) |
| `GspSystemInfo.bIsPassthru` | după platformă (0 pe desktop) | 0 | **1** | 0 | 0 | ok (nova: fără ACPI și `bIsPassthru = 0`, ca noi); `= 1` rămâne doar o rezervă |
| `GspSystemInfo.bIsPrimary` | după VGA | după VGA | 0 | 0 | după VGA (1 pe workstation) | ok (merge și cu 0, și cu 1) |
| `GspSystemInfo` chipset (`clSyncWithGsp`, `pcieConfigReg.linkCap`, FLR, 64b BAR0) | completat | 0 | 0 | 0 | 0 | ok |

## 4. Decizia (DEDUS) și ce urmează

1. **`bDmemStack = 1`** (`nvidia-ga104::gsp::libos::encode_gsp_args`, octetul 48, verificat cu bindgen) — exact ce fac toate cele patru drivere. Stiva sarcinii RM
   de pe GSP stă în DMEM în loc de „stiva LibOS” (varianta pe care RM o alege doar la cerere, prin cheia de registru). `NV_ERR_OPERATING_SYSTEM` e codul generic al
   stratului „sistem de operare” al GSP-RM (LibOS); o problemă de stivă ar fi exact genul de eroare raportat așa — dar **nu avem dovada**: se vede la următorul `gpu lant`.
   Pe placă nu se schimbă nimic altceva: aceeași așezare VRAM, aceleași scrieri în registre, același jurnal.
2. Dacă `0x59` rămâne: nu mai există nicio diferență de **conținut** față de nova/nouveau în ce trimitem → se caută în altă parte: ordinea/momentul (ce e în coadă la
   sonerie), starea plăcii dinainte (VBIOS/devinit, FRTS), memoria de sistem (adresele DMA reale ale regiunilor). Rezerve, câte **una** pe rulare: `bIsPassthru = 1`
   (tinygrad), `GspFwWprMeta.flags |= CLOCK_BOOST` (RM pe GA104), `RMDevidCheckIgnore = 1` (nouveau, nova).
3. Ce **nu** facem: nu deschidem imaginea GSP-RM pentru simboluri (nici pentru a traduce adresele asserturilor, nici pentru jurnalele LibOS/PMU) — regula proiectului.
   Dacă Adrian vrea totuși numele funcțiilor, e decizia lui, consemnată în hub; până atunci adresele rămân numere.

## 5. Ce vedem la os-282 (lista pentru fereastră)

- `gpu lant` → `DA`. Dacă `bDmemStack` era cauza: după `CORE_RESUME`, rândul `valoare GSP_INIT_DONE: rpc_result … = 0x00000000` și lanțul merge la **I8e (golden)**
  — prima dată când GSP-RM răspunde la un apel RM pe placă. Poze la `rezultat lant: …`.
- Dacă nu: același `0x59`; se compară numărul de înregistrări NOCAT (844?) și adresele primelor 16 cu cele de mai sus — dacă sunt **identice**, asserturile sunt
  independente de stivă (zgomot), și căutăm în afara conținutului mesajelor (§4 punctul 2).
