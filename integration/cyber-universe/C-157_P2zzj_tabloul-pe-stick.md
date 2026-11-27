# C-157 p.2 — os-481: la lot 6 tabloul de bord scris pe stick ține pasul (gata de legat, testat doar în SIMULARE)

Sesiunea Cyber-GPU 267, 27 nov 2026, în cloud, fără placă. Comparația întreagă: `docs/C-157_OS481.md` (T281).

## Ce a ieșit
- Pasul cu `mm_tc_v5` (P2zzf): toate predicțiile confirmate — 147 pe v5, oracolul mixt = os-473, lot 2 placa **80,16 ms**
  (banda 78–84), lot 6 placa **203,06 ms** (banda 202–222) = 15 128 simb/s pe placă, fără oprire termică (50 °C, motive 0).
- **Dar la lot 6 pasul întreg e 274,3 ms** (11 200 simb/s). În `gpu-jurnal-165.txt`: după fiecare pas, `step_dashboard` pune în
  jurnal 11 valori `kd`; fiecare `Event::Value` iese cu `Flush::Now` (rutina pasului ține doar scrieri și note), deci **o scriere pe
  stick de ~23–24 ms pe valoare** → ~238 ms + ~28 ms între doi pași. Cu `--doua-locuri` placa termină pasul următor în 203 ms și
  așteaptă procesorul. Cu v4 (265 ms pe pas) frâna era ascunsă.

## Ce poți lega (doar adăugări în `gpu-core`, commit-ul de mai jos)
`Journal::set_hold_values(true)` = în timpul unei rutine (`set_routine(Some(..))`), și valorile `Event::Value` sunt ținute
(`Flush::Later`, aceeași limită `max_pending`); `flush_live()` și ieșirea din rutină le scriu ca azi. În afara rutinei sau cu
`false`: exact ca înainte. Jurnalul din memorie e același. Implicit oprit. Test: `gpu_core::diag::tests::routine_stretch_can_hold_values`.

Locul: în `gpu_i14.rs`, pasul #513, lângă `journal.set_routine(Some(kf::gen_routine(&GEN_NOTES)))` (r. ~9399):
`journal.set_hold_values(true);` și, la ieșire, lângă `journal.set_routine(None)`: `journal.set_hold_values(false);`.
Cu `GEN_MAX_PENDING` = 64 și `FLUSH_US` = 1 s, tabloul ar ieși pe stick o dată la ~4–5 pași, nu de 11 ori pe pas.

Alternative, dacă preferați: tabloul la fiecare N pași (primul și ultimul mereu), sau valorile în RAM până după ultimul pas.
**Paza termică nu se schimbă:** decizia se ia pe valorile *citite* (`StepSample::stop`), nu pe cele scrise.
`panic_flush` scrie oricum rândurile ținute dacă se oprește ceva.

## Ce cerem (când are placa)
Rândul lot 6 de azi (`--pas-v5 --doua-locuri --lot-prin-ram --lot-513 6 --tablou-viteza`) cu tabloul ținut.
**Predicție (DEDUS, scrisă înainte):** pasul întreg **~205–215 ms ≈ 14 300–15 000 simb/s** (azi 274,3 / 11 200); placa
neschimbată (~203 ms); aceleași pierderi; în raport „în afara plăcii” sub ~12 ms.

## Ce e nou pentru placă
Nimic. Nicio scriere în registre, niciun program nou, niciun firmware — se schimbă doar când se scrie jurnalul pe stick.
