# După os-279 — GSP-RM pornește, dar refuză inițializarea (0x59); jurnalul decodează acum ce a trimis înainte

Pentru sesiunea OS (cyber-universe). Commit cyber-gpu: `5cfe523` pe `g0-g2-fundatie` (sau mai nou). **Doar adăugări** — interfața publică nu se schimbă.

## Ce a spus `gpu-jurnal-007.txt` (11 nov 12:20, os-279) — OBSERVAT

- Lanțul de pornire merge **până la capăt**: FWSEC-FRTS (I6) → Booter (I7) → toate cele 420 de comenzi ale secvențiatorului → la `#419 CORE_RESUME`
  Booter-ul a rulat de-adevăratelea (SCRATCH_14 `0x13100000` → `0x17100000`, bitul 26 pus abia după `0x840130 ← 2`; SEC2 MAILBOX0 = 0 → fără eroare),
  GSP RISC-V activ (`RISCV_CPUCTL = 0x80`), GSP MAILBOX0 = 0.
- GSP-RM **a pornit și a răspuns**: `GSP_INIT_DONE` cu `rpc_result = 0x59` = `NV_ERR_OPERATING_SYSTEM` („Generic operating system error”, `nvstatuscodes.h`).
  Driverul NVIDIA face exact același lucru cu acest câmp (`kgspWaitForRmInitDone_IMPL`: `NV_ASSERT_OK_OR_RETURN(RPC_HDR->rpc_result)`), deci nu e o
  interpretare a noastră: GSP-RM însuși spune că inițializarea lui a eșuat.
- Înainte de asta, GSP-RM a trimis **843 de mesaje cu funcția `0x1020`** (cele mai multe înainte de cererea către secvențiator) și **2 cu `0x100c`** (chiar înainte de
  `GSP_INIT_DONE`). Jurnalul vechi le număra, dar nu le citea.

## Ce sunt cele două mesaje (antetele MIT 570.144, citite ca specificație)

- `0x1020` = **`GSP_POST_NOCAT_RECORD`**: o înregistrare de diagnostic (NOCAT) pe care RM-ul de pe GSP o trimite driverului ca s-o pună în jurnalul lui
  (`kernel_gsp.c`, `_kgspRpcGspPostNocatRecord`). Conține `recType` (1 BUGCHECK, 2 ENGINE, 3 TDR, 4 RC, 5 ASSERT), `source` (text), `subsystem`,
  `errorCode`, `faultingEngine` (text), `diagBuffer`. La **ASSERT** (`journal.c`, `_rcdbNocatReportAssert`): `source = "ASSERT"`, `errorCode` = adresa
  assert-ului față de începutul imaginii RM + linia din sursă în biții de sus, `diagBuffer` = stiva de apeluri (10 adrese) + de câte ori.
  **843 de astfel de înregistrări înseamnă că RM-ul de pe GSP s-a plâns de 843 de ori în timpul propriei inițializări** — probabil asserturi; textul lor
  (`source`, codurile) spune de ce. Pe o pornire sănătoasă nu apare niciuna.
- `0x100c` = **`UCODE_LIBOS_PRINT`**: jurnalul altui microcod (PMU) trimis prin GSP-RM (`_kgspRpcUcodeLibosPrint`). Conținutul e în format LibOS binar,
  decodabil doar cu tabelele de simboluri ale microcodului — pe care nu le deschidem (RISKS.md). Se jurnalizează descriptorul motorului și mărimea.

## Ce face `5cfe523` (SIMULARE; T116, T117)

Toate **doar citiri** din mesaje deja primite; nicio scriere nouă în placă. Tip nou de rând în jurnal, `Event::Text`, afișat `text <nume> = "..."`
(OS-ul folosește `_ =>` și `Display` → nimic de schimbat).

- fiecare `GSP_POST_NOCAT_RECORD` e decodat; **primele 4** și **fiecare fel nou** (`recType`, `errorCode`, `subsystem`) ajung în jurnal în întregime
  (până la 16 feluri), restul sunt numărate:
  ```
  valoare NOCAT: inregistrarea nr. = 0x00000001
  valoare NOCAT: recType (1 BUGCHECK, 2 ENGINE, 3 TDR, 4 RC, 5 ASSERT) = 0x00000005
  text NOCAT: recType = "ASSERT"
  valoare NOCAT ASSERT: adresa assert-ului (fata de inceputul GSP-RM) = 0x........
  valoare NOCAT ASSERT: linia din sursa RM = 0x........
  text NOCAT: source = "ASSERT"
  valoare NOCAT: subsystem = 0x........
  valoare NOCAT ASSERT: apel #0 (adresa fata de inceputul GSP-RM) = 0x........   (până la 5)
  valoare NOCAT: timestamp (32 de biti de jos) = 0x........
  ```
  (la celelalte tipuri: `errorCode` cu numele `NV_STATUS` dacă e unul cunoscut, `faultingEngine`, `bugcheck`, `tdrReason`, `diagBufferLen`);
- fiecare `UCODE_LIBOS_PRINT` (primele 8): `ucodeEngDesc`, `libosPrintBufSize`;
- orice alt mesaj sărit: numărul **și numele** (`text I8 mesaj GSP-RM sarit (numele) = "..."`), din tabelul generat din `rpc_global_enums.h`;
- la `GSP_INIT_DONE` cu eroare: `text GSP_INIT_DONE: rpc_result (nume NV_STATUS) = "NV_ERR_OPERATING_SYSTEM"`;
- la orice sfârșit al așteptării (bine, eroare, timp expirat, Esc), **rezumatul**: `I8 mesaje GSP-RM sarite … total`, `I8 NOCAT (0x1020): inregistrari de
  diagnostic primite`, `I8 NOCAT: feluri distincte`, `I8 UCODE_LIBOS_PRINT (0x100c): jurnale de microcod primite`.

## Ce are OS-ul de făcut

1. Legarea `cyber-gpu` la `5cfe523` (sau mai nou pe `g0-g2-fundatie`). `Event` are o variantă nouă (`Text`); `gpu_core::diag::TEXT_LEN` nou;
   `nvidia_rm570::{status_name, status_text, rpc_name}` și `nvidia_rm570::names` noi — toate adăugări.
2. Release nou (os-280) și `gpu lant` pe workstation, cu Adrian de față. **Nu e o reparație**: I8 se va opri probabil tot la 0x59 — important e ce scriu
   rândurile `NOCAT` (mai ales `text NOCAT: source`, `recType`, `errorCode` / adresa + linia asserturilor și `subsystem`) și rezumatul de la final.
3. Abia cu acele rânduri se poate spune ce îi lipsește lui GSP-RM (parametrii din `GSP_SET_SYSTEM_INFO`, registrul, argumentele de pornire, memoria) —
   o „reparație” acum ar fi inventată.

`verifica_jurnal_lant` ignoră rândurile noi (verificat pe `gpu-jurnal-007.txt` cu rândurile `text …`/`valoare …` adăugate artificial: același verdict
„OPRIT CURAT la i8-gsp-init (Bus Master pus la loc)”, I8 326 scrieri + 1 sonerie).
