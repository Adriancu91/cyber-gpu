# C-54 — schița pentru OS: `gpu tur` (bucla automată) și `gpu lant` (fereastra cu Adrian)

Pentru sesiunea `cyber-universe`. Driverul e gata și publicat pe `g0-g2-fundatie` (doar **adăugări** față de fixarea voastră;
nimic de schimbat în codul existent). Tot ce urmează e testat doar în **SIMULARE**.

## 1. `gpu tur` — doar citire, potrivită pentru lista albă a buclei automate (C-53)

- Apel: `nvidia_ga104::read_tour::run_tour(cfg, Some(adresa), bar0, clock, jurnal)` → `Tour { facts, reads }`.
- `bar0`: fereastra BAR0 **doar citire** de 16 MiB (aceeași ca la `gpu pregatire-i6`). Cu fereastră mai mică, părțile care nu încap
  sunt sărite cu un fapt „sarit: …” (nu se ghicește nimic).
- Afișare: câte un rând pe fapt, exact `{fact}` (Display dă `fapt <cheie> = <valoare>`, ASCII), în jurnalul `gpu` → ajunge în raport.
- Ce citește (lista `read_tour::SAFE_READS`): configurația PCI, BOOT_0/BOOT_42, GFW (0x118128, 0x118234), VBIOS prin PROM (doar poziții
  și mărimi în raport, conținutul nu se afișează), descriptorul FWSEC, pregătirea I6 (siguranța 0x8241e0, VRAM 0x1183a4, WPR2 0x1fa828/0x1fa824).
  Toate au rulat deja pe placă cu 0 scrieri, în afară de cele 3 registre I6 (citiri simple, fără risc cunoscut).
- Siguranță: întoarce `Err` dacă jurnalul arată vreo scriere; `0 scrieri` se poate afișa din `jurnal.writes()`.
- **Nu** intră încă: temperatura (`read_tour::NOT_YET`: registrul nu e în antetele NVIDIA MIT 570.144).

## 2. `gpu lant` — doar în fereastra cu Adrian (NU pe lista albă a buclei automate)

- Apel: `nvidia_ga104::chain::run_chain(chain::WINDOW_CHAIN, scrieri_permise, jurnal, |pas, jurnal| …)`.
- `scrieri_permise = true` doar dacă: nu suntem în modul automat **și** Adrian a tastat „DA” (ca la `gpu i6 --scrie`). În modul automat
  comanda ori lipsește din listă, ori rulează cu `false` (atunci rulează doar `pregatire-i6` și `tur-dupa`; pașii care scriu nu pornesc).
- Ce face OS-ul pentru fiecare `pas.id` (închiderea):
  - `pregatire-i6` → ce face deja `gpu pregatire-i6` (eroare dacă nu trece, ex. WPR2 ridicat);
  - `bus-master-pornit` → `gpu_pci::bus_master::set_bus_master(…, true)` (păstrați valoarea întoarsă);
  - `i6-fwsec-frts` → ce face `gpu i6 --scrie`, fără partea de Bus Master;
  - `i7-booter-gsp` → nu-l implementați: lanțul se oprește singur aici („blocat în plan (I7)”) și nu-l încearcă;
  - `bus-master-oprit` → `gpu_pci::bus_master::restore_command(…, valoarea_pastrata)` (dacă n-a fost pornit, `Ok(())`);
  - `tur-dupa` → `run_tour` (dovada stării în care rămâne placa).
- Regulile le aplică driverul: oprire la primul eșec; `bus-master-oprit` și `tur-dupa` rulează oricum după un pas care a scris; un pas
  doar-citire care scrie = eșec. Raportul (`ChainReport`, Display) se afișează și se scrie în jurnal.
- Jurnalul pe loc (`Journal::with_sink`, C-52) rămâne cum e: fiecare acces ajunge pe stick înainte de pasul următor.

## 3. După tur: comparația cu simularea (o face sesiunea Cyber-GPU, pe gazdă)

`cargo run --release -p nvidia-ga104 --example compara_raport -- <raport-*.txt>` citește raportul de pe stick (formatul vechi al comenzilor
`gpu`/`gpu gfw`/`gpu vbios`/`gpu fwsec`/`gpu firmware` și liniile noi `fapt …`) și îl pune lângă același tur rulat pe `gpu-sim::observed`:
LA FEL / DIFERIT / diferit intenționat (conținut VBIOS, firmware) / doar pe placă / doar în simulare. Ieșire 2 = modelul trebuie reparat.
Pe cele 3 rapoarte reale de până acum: 33 la fel, 0 diferit.
