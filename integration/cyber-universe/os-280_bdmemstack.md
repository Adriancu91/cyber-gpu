# După os-280 — `bDmemStack = 1` în argumentele GSP (candidatul pentru 0x59)

Pentru sesiunea OS (cyber-universe). Commit cyber-gpu pe `g0-g2-fundatie`: vezi `docs/IMPLEMENTATION_STATUS.md`, sesiunea 87 (sau mai nou).
**Nimic de schimbat în codul OS**: `gpu_i7.rs` apelează deja `libos::encode_gsp_args(&ql, …)`; funcția are aceeași semnătură, doar octetul 48 al
structurii (`GSP_ARGUMENTS_CACHED.bDmemStack`) e acum 1. Trebuie doar **legarea commit-ului nou + Release `os-282`** (`os-281`, 14:39, are codul vechi, `bcc314e`).

## De ce

Jurnalul `gpu-jurnal-008.txt` (os-280) arată aceeași oprire ca os-279: GSP-RM pornește, trimite 844 de asserturi (toate înainte de secvențiator, adrese diferite,
fără linie și fără stivă — forma normală pe GSP, `docs/os-280_ANALIZA_0x59.md` §2) și răspunde `GSP_INIT_DONE` cu `0x59` = `NV_ERR_OPERATING_SYSTEM`.
Comparând câmp cu câmp ce trimitem cu cele patru drivere care pornesc GSP-RM 570 (NVIDIA RM, nouveau r570, tinygrad, nova-core — regula „trei surse”, C-107), **singura diferență față de toate patru**
e `bDmemStack`: ele pun 1 (stiva sarcinii RM în DMEM), noi puneam 0 (varianta „stiva LibOS”, pe care RM o alege doar la cerere). `NV_ERR_OPERATING_SYSTEM` e codul
generic al stratului LibOS al GSP-RM. **DEDUS, neprobat**: se vede la `gpu lant` cu os-282. Pe placă nu se schimbă altceva (aceleași scrieri, aceeași așezare VRAM).

De reținut: nouveau **ignoră implicit** înregistrările NOCAT (handler gol) → ele apar și pe porniri bune; nu sunt neapărat cauza.

## În plus (cerut 15:00, „C-102 continuare 2”): conținutul celor 2 jurnale PMU ajunge pe stick

Mesajele `UCODE_LIBOS_PRINT` (de la `ENG_PMU`, 16 și 24 de octeți la os-280) se jurnalizează acum și ca conținut, **exact cum au venit**, cuvinte de 4 octeți LE
(până la 64 de octeți): rânduri `valoare UCODE_LIBOS_PRINT: continut, octetii 0..3 (asa cum au venit, LE) = 0x........` etc. Nu se interpretează (format binar LibOS;
ar cere simbolurile microcodului — nu se deschid). `verifica_jurnal_lant` le tipărește sub „jurnale LibOS jurnalizate”. Tip de rând deja existent (`Event::Value`) → nimic
de schimbat în OS.

## Ce se urmărește în jurnal la os-282

- Reușită: `valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x00000000`, apoi pasul **I8e (golden)** începe — primul apel RM pe placă.
- Eșec: tot `0x59`. Atunci se compară cu os-280: dacă tot 844 de înregistrări NOCAT și aceleași prime 16 adrese (`0x019ef3c6 …`), asserturile sunt zgomot;
  și conținutul mesajelor noastre nu mai diferă de nova/nouveau → se caută în ordinea/momentul lor și în starea plăcii; rezerve, câte una pe rulare:
  `bIsPassthru = 1` (tinygrad), `CLOCK_BOOST` (RM), `RMDevidCheckIgnore` (nouveau/nova).
- Ca de obicei: `raport` + `trimite`; `verifica_jurnal_lant <jurnal> 0x9c66000` tipărește rezumatul NOCAT.
