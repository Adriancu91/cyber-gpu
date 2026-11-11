# După os-283 — `bDmemStack` nu era cauza; ce diferă e **ordinea**: RPC-urile de pornire ajungeau la GSP-RM după ce pornise

Sesiunea 88, 11 noiembrie 2026 (cloud, fără placă). Jurnalul analizat: `Cyber-hub/rezultate/os-automat/2026-11-11/gpu-jurnal-009.txt`
(2438 rânduri, testul `os-283` de la ~17:25, cyber-gpu `2bbd0b3`; `rezultate/2026-11-11-workstation-os283/REZULTAT.md`).
Aplică regula „trei surse” (DECIZII.md, 11 nov 14:41), comanda **C-107** (continuare) și pregătește **C-108** (variantele pentru joi).

Etichete: **OBSERVAT** = scrie în jurnalul de pe placă; **SPECIFICAȚIE** = citit în codul public (MIT: open-gpu-kernel-modules 570.144 `8ec351a`,
tinygrad `e51dc1d`; GPL, doar citite: nouveau și nova-core din Linux `067c5c2`); **DEDUS** = concluzia noastră, neprobată pe placă.

## 1. Ce spune jurnalul (OBSERVAT)

- `gpu-jurnal-009.txt` (os-283, `bDmemStack = 1`) e **identic** cu `gpu-jurnal-008.txt` (os-280, `bDmemStack = 0`) rând cu rând, în afara
  marcajelor de timp (`diff` după tăierea `[ secunde]`: doar rândul 1 — commit-ul — și cele 16 `timestamp` NOCAT). Aceleași 844 de asserturi, aceleași 16 adrese,
  aceleași 2 jurnale PMU, `GSP_INIT_DONE: rpc_result = 0x59`. `verifica_jurnal_lant … 0x9c66000`: „OPRIT CURAT la i8-gsp-init”.
- Concluzie (OBSERVAT): **`bDmemStack` nu schimbă nimic**. Cum §3 din `os-280_ANALIZA_0x59.md` nu mai găsise nicio diferență de *conținut* față de cele patru
  drivere, s-a comparat **ordinea** (ce e în coada de comenzi în momentul în care GSP-RM pornește).
- Momentele din jurnal (ora OS-ului): Booter pornit pe SEC2 la `99.908` (`0x840100 ← 2`), `FALCON_OS` scris la `100.145`, I7 „ok” la `100.158`;
  **soneria I8 (`0x110c00 ← 0`) la `100.284`** — adică `SetSystemInfo` + `SetRegistry` au intrat în coadă **≈ 0,3–0,4 s după ce Booter pornise GSP-RM**.
  Prima înregistrare NOCAT e citită la `100.296` (era deja în coada de stare), cererea secvențiatorului la `102.884`.

## 2. Ce fac cele patru drivere (SPECIFICAȚIE)

| Driver | Unde pune `GSP_SET_SYSTEM_INFO` + `SET_REGISTRY` | Unde pornește GSP-RM (Booter) | Soneria (`NV_PGSP_QUEUE_HEAD(0) = 0`) |
|---|---|---|---|
| **RM 570.144** (MIT) | `kgspInitRm_IMPL` → `kgspQueueAsyncInitRpcs` (`kernel_gsp.c`), **înainte** de `_kgspBootGspRm` → `kgspBootstrap_HAL` | după | la fiecare trimitere (`_kgspRpcSendMessage` → `kgspSetCmdQueueHead_HAL`), deci și înainte de pornire |
| **nouveau** (GPL, citit) | `r535_gsp_oneinit` (`rm/r535/gsp.c`: `set_system_info`, apoi `r535_gsp_rpc_set_registry`) | `tu102_gsp_init` → `tu102_gsp_booter_load` → `r535_gsp_init` (poll `GSP_INIT_DONE`), **mai târziu** | la fiecare push (`rpc.c`: `nvkm_falcon_wr32(&gsp->falcon, 0xc00, 0)`) |
| **tinygrad** (MIT) | `GSP.init_sw`: „`# Prefill cmd queue with info for gsp to start.`” → `rpc_set_gsp_system_info()`, `rpc_set_registry_table()` | `init_hw` (FRTS, Booter), apoi `gsp.init_hw` → `wait_resp(GSP_INIT_DONE)` | la fiecare trimitere (`NV_PGSP_QUEUE_HEAD[0].write(0)`) |
| **nova-core** (GPL, citit) | `Gsp::boot` (`gsp/boot.rs`): `cmdq.send_command_no_wait(SetSystemInfo)`, apoi `SetRegistry`, **înainte** de `hal.boot(...)` | după, în aceeași funcție | la fiecare trimitere (`cmdq.rs`: `NV_PGSP_QUEUE_HEAD`) |
| **noi (os-278…os-283)** | **I8**, după ce I7 a pornit GSP-RM | I7 | la I8 |

Și de ce contează, în cuvintele NVIDIA (570.144, MIT):
- `kernel_gsp.c`, deasupra `kgspQueueAsyncInitRpcs_IMPL`: „*Prepare and place RPCs in message queue that GSP-RM will process **in early boot before OBJGPU is
  created**.*”
- `kernel_gsp_gh100.c`, `_kgspEstablishSpdmSession` (calea cu calcul confidențial, unde mesajele nu pot fi puse dinainte): „*SPDM will not continue with boot until we send
  ack after sending these RPCs, else **we create a race condition where GSP-RM may skip these RPCs if not sent in time**.*”

DEDUS: GSP-RM citește coada de comenzi o dată, la pornirea timpurie; dacă `SetSystemInfo` nu e acolo, **sare peste el** și își construiește OBJGPU fără BAR-uri,
fără ID-uri PCI, fără `maxUserVa` → verificările cad una după alta (cele 844 de asserturi, toate înainte de secvențiator, în cod „în linie dreaptă” — exact forma
din os-280), iar inițializarea se încheie cu `NV_ERR_OPERATING_SYSTEM`. Mesajele noastre, sosite după, sunt tratate ca RPC-uri obișnuite, prea târziu.
Se potrivește și cu faptul că jurnalul e identic de la os-278 la os-283 indiferent de ce am schimbat în *conținut*.

## 3. Ce se schimbă (C-108, `gsp::variants`)

- **Varianta 1 (`rpc-inainte`, cea mai probabilă):** `SetSystemInfo` + `SetRegistry` + soneria intră în coadă **la I7, înainte de resetul GSP și de Booter**
  (ordinea RM: mesaje → sonerie → reset în RISC-V → cutiile LibOS → Booter). Conținutul rămâne cel de la os-283. În OS: `Settings::queue_init_rpcs` în pregătirea I7,
  `falcon::ring_doorbell` înaintea lui `booter_load_dma_libos`; I8 verifică cu `init_rpcs_ready_for_i8` și nu mai trimite nimic.
- Variantele 2–8 adaugă la varianta 1, câte una sau ca set al unui driver, singurele valori pe care cele patru drivere le completează diferit (tabelul C-107 +
  trei câmpuri verificate acum în surse): `bIsPassthru` (tinygrad 1), `bIsPrimary` (tinygrad/nova 0), `hostPageSize` (RM 4096 = `osGetPageSize`; nouveau r570,
  tinygrad, nova nu-l pun = 0), `RMDevidCheckIgnore = 1` (nouveau, nova), `GspFwWprMeta.flags = CLOCK_BOOST` (RM pe GA10x), `bDmemStack = 0` (RM cu
  `RmGspStackPlacement = 1`). Variantele 9–10 păstrează ordinea veche + seturile tinygrad/nova, pentru cazul în care ordinea nu era cauza.
- Lista completă, cu numele, ce schimbă și sursa: `crates/nvidia-ga104/src/gsp/variants.rs` (`VARIANTS`); în OS: `gpu variante`.
- **SIMULARE (T122):** modelul GSP-RM din `sim_peer` (`SimGsp::start_early_boot`) consumă la pornire ce e în coadă și răspunde la `GSP_INIT_DONE` cu `0x59` dacă
  cele două RPC-uri nu erau acolo — ordinea os-283 dă `0x59` ca pe placă, variantele 1–8 dau `0`, 9–10 dau `0x59`. Modelul e **scris după sursele RM, nu după placă**:
  dovada vine joi.

## 4. Ce NU s-a schimbat (verificat în surse, sesiunea 88)

- Câmpurile `GspSystemInfo` pe care RM le completează (`rpc.c` `rpcGspSetSystemInfo_v17_00`, lista întreagă): BAR0/1/3, `nvDomainBusDeviceFunc`
  (`gpuEncodeDomainBusDevice(domain, bus, slot)`; nouveau `pci_dev_id` = bus<<8|devfn — egale pentru funcția 0, cum e placa), `oorArch`, ID-uri PCI, oglinda PCFG,
  `maxUserVa`, `hostPageSize`, ACPI, `bIsPassthru`, `bIsPrimary`, restul 0 la nouveau/tinygrad/nova — ca la noi. `upstreamAddressValid`/porturile PCIe: 0 la toate trei
  driverele deschise (RM le completează doar pe CPU; `clInitDeviceInfo` → `NV_ERR_OPERATING_SYSTEM` e o cale de pe CPU, nu de pe GSP).
- Registrul: RM trimite multe chei, dar valorile lor vin din sistemul de operare; cele cunoscute public și folosite de driverele care pornesc GSP-RM sunt cele 2 (+1).
- Secvențiatorul `CORE_RESUME`: `kgspExecuteSequencerCommand_GA102` = reset GSP în RISC-V, cutiile LibOS, **start SEC2** (Booter rămas în IMEM), așteptare
  SCRATCH_14 bit 26, `FALCON_OS = appVersion`, RISC-V activ — exact ce face `sequencer.rs` (confirmat în os-278…os-283: toate cele 420 de comenzi trec).
- Nu s-a deschis imaginea GSP-RM; adresele asserturilor rămân numere (regula proiectului).

## 5. Ce vedem joi (lista pentru fereastră, C-108)

`gpu lant --varianta 1` → `DA`. Dacă ordinea era cauza: rândurile `valoare varianta I8 (C-108): numarul = 0x00000001`, apoi după `CORE_RESUME`
`valoare GSP_INIT_DONE: rpc_result … = 0x00000000` și lanțul merge la **I8e (golden)**. De urmărit și: numărul de înregistrări NOCAT (mult sub 844 sau 0) și dacă
mai apar asserturi înainte de secvențiator. Dacă rămâne `0x59` cu varianta 1, următoarele variante, în ordine; jurnalele (`gpu-jurnal-0NN.txt`, cu numărul variantei)
se compară cu `verifica_jurnal_lant <jurnal> 0x9c66000`.
