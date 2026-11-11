# După os-280 — `bDmemStack = 1` în argumentele GSP (candidatul pentru 0x59)

Pentru sesiunea OS (cyber-universe). Commit cyber-gpu pe `g0-g2-fundatie`: vezi `docs/IMPLEMENTATION_STATUS.md`, sesiunea 87 (sau mai nou).
**Nimic de schimbat în codul OS**: `gpu_i7.rs` apelează deja `libos::encode_gsp_args(&ql, …)`; funcția are aceeași semnătură, doar octetul 48 al
structurii (`GSP_ARGUMENTS_CACHED.bDmemStack`) e acum 1. Trebuie doar **legarea commit-ului nou + Release `os-281`**.

## De ce

Jurnalul `gpu-jurnal-008.txt` (os-280) arată aceeași oprire ca os-279: GSP-RM pornește, trimite 844 de asserturi (toate înainte de secvențiator, adrese diferite,
fără linie și fără stivă — forma normală pe GSP, `docs/os-280_ANALIZA_0x59.md` §2) și răspunde `GSP_INIT_DONE` cu `0x59` = `NV_ERR_OPERATING_SYSTEM`.
Comparând câmp cu câmp ce trimitem cu cele trei drivere care pornesc GSP-RM 570 (NVIDIA RM, nouveau r570, tinygrad), **singura diferență față de toate trei**
e `bDmemStack`: ele pun 1 (stiva sarcinii RM în DMEM), noi puneam 0 (varianta „stiva LibOS”, pe care RM o alege doar la cerere). `NV_ERR_OPERATING_SYSTEM` e codul
generic al stratului LibOS al GSP-RM. **DEDUS, neprobat**: se vede la `gpu lant` cu os-281. Pe placă nu se schimbă altceva (aceleași scrieri, aceeași așezare VRAM).

De reținut: nouveau **ignoră implicit** înregistrările NOCAT (handler gol) → ele apar și pe porniri bune; nu sunt neapărat cauza.

## Ce se urmărește în jurnal la os-281

- Reușită: `valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x00000000`, apoi pasul **I8e (golden)** începe — primul apel RM pe placă.
- Eșec: tot `0x59`. Atunci se compară cu os-280: dacă tot 844 de înregistrări NOCAT și aceleași prime 16 adrese (`0x019ef3c6 …`), asserturile sunt zgomot;
  candidatul următor e `GspSystemInfo.bIsPassthru = 1` (ce face tinygrad, singurul driver fără date ACPI care merge) — câte o schimbare pe rulare.
- Ca de obicei: `raport` + `trimite`; `verifica_jurnal_lant <jurnal> 0x9c66000` tipărește rezumatul NOCAT.
