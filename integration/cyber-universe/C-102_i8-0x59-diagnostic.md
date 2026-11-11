# C-102 — I8 „eroare hardware 0x59” la os-278: jurnalul spune acum de unde vine

Pentru sesiunea OS (cyber-universe). Commit cyber-gpu: `d31ad8d` pe `g0-g2-fundatie`. **Doar adăugări** — interfața publică nu se schimbă.

## Ce s-a aflat din `gpu-jurnal-006.txt` (11 nov, os-278)

- Toate cele 420 de comenzi ale secvențiatorului au trecut, inclusiv `#419 CORE_RESUME` pe calea nouă (C-99): `0x840130 ← 2` (SEC2 pornit prin CPUCTL_ALIAS),
  „SEC2 (Booter) gata: răspuns după 2 µs”, `0x110080 ← 0x0008b320` (FALCON_OS), „GSP RISC-V activ: răspuns după 2 µs”. Nu există rândul
  „I8 secventiator: comanda de mai jos s-a oprit” → eroarea **nu** e din secvențiator.
- ~150 ms mai târziu: `pas i8-gsp-init: ESEC`, „eroare hardware 0x59”. În codul nostru, după secvențiator, `0x59` poate veni din exact două locuri:
  1. `GSP_INIT_DONE` sosit cu `rpc_result = 0x59` — în antetele publice NVIDIA 570.144 (`nvstatuscodes.h`, MIT) `0x59` = **`NV_ERR_OPERATING_SYSTEM`**
     („Generic operating system error”); nouveau tratează și el orice `rpc_result ≠ 0` ca eroare.
  2. pointerul de scriere al cozii de stare citit **89** (= 0x59), în afara cozii (0..62) — ar însemna că memoria cozii a fost stricată după resume.
- Jurnalul vechi nu scria nicio citire în I8, deci nu se poate spune care din cele două a fost. **Reparația „la sigur” nu e posibilă fără acest răspuns.**
- Observația din `REZULTAT.md` rămâne: 2 µs e prea puțin ca Booter să fi rulat; posibil ca bitul 26 din `NV_PGC6_BSI_SECURE_SCRATCH_14` să fi fost deja pus de la I7.
  Driverul NVIDIA (`kernel_gsp_ga102.c`, CORE_RESUME) **nu scrie și nu șterge** acest registru — face exact pașii noștri — deci nu scriem nici noi în el.

## Ce face `d31ad8d` (SIMULARE; T113–T115)

Rânduri noi în jurnal, toate **doar citiri**, tip nou `Event::Value` afișat `valoare <nume> = 0x........`:
- la `CORE_RESUME`: „nota: I8 CORE_RESUME: starea inainte de pornirea SEC2” + 7 valori (SCRATCH_14, SEC2 CPUCTL/MAILBOX0/MAILBOX1, GSP RISCV_CPUCTL, GSP MAILBOX0/1),
  apoi același set după „predarea etapei 3” („starea dupa predarea etapei 3”);
- fiecare mesaj GSP-RM sărit în așteptarea `GSP_INIT_DONE`: „valoare I8 mesaj GSP-RM sarit (functia) = 0x....” (+ `rpc_result` dacă ≠ 0);
- la `GSP_INIT_DONE`: „valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x....”; dacă ≠ 0: `rpc_result_private`, „nota: I8: eroarea vine din GSP_INIT_DONE” + cele 7 valori;
- eroare din coada de stare: „valoare I8 coada de stare: pointer in afara cozii (0..62) = 0x....”, „nota: I8: eroarea vine din coada de stare” + cele 7 valori;
- timp expirat / Esc: notă + cele 7 valori.

## Ce are OS-ul de făcut

1. Legarea `cyber-gpu` la `d31ad8d` (sau mai nou pe `g0-g2-fundatie`); `Event` are o variantă nouă (`Value`) — OS-ul folosește `_ =>` și `Display`, deci nimic de schimbat.
2. Release nou (os-279) și `gpu lant` pe workstation, cu Adrian de față.
3. În jurnalul următor se citesc, după `secv #419 CORE_RESUME`: valorile dinainte de pornirea SEC2 (**dacă SCRATCH_14 are deja bitul 26 = `0x04000000` înainte de `0x840130 ← 2`,
   „2 µs” era bitul vechi de la I7**), apoi rândul „I8: eroarea vine din …” care spune ce e `0x59`.

`verifica_jurnal_lant` ignoră rândurile noi (verificat pe `gpu-jurnal-006.txt` cu rândurile adăugate: același verdict „OPRIT CURAT la i8-gsp-init”).
