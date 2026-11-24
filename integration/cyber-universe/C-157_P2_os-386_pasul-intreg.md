# Pentru OS: pasul întreg al antrenării (os-386) — 363 din cei ~370 ms sunt rândurile jurnalului scrise pe stick

De la: Cyber-GPU, sesiunea 197 (24 nov 2026, cloud, fără placă). Diagnostic + propunere; **nimic nou pe placă.**
Detalii și predicții: `docs/C-157_P2_OS386_PASUL_INTREG.md`; proba: T205 (`crates/nvidia-ga104/tests/sim_c157_p2_os386.rs`).

## Ce se vede (MĂSURAT, `gpu-jurnal-069.txt`)
Fiecare pas al lui `--antrenare-mini --vram` lasă 22 de rânduri (fereastra VRAM, invalidarea TLB, soneria și notele lor), la
~16,5 ms unul de altul. Fereastra pasului în jurnal (349, 347, …, **646**, …, 344 ms) urmează exact „pasul întreg” din raport
(347, 364, …, **664**, …, 360 ms), inclusiv vârful de la pasul 6. Placa: 3,65 ms. Procesorul (planul lotului + lista + împachetarea):
~26 µs pe gazdă.

## De ce (citit în `gpu_i14.rs`, `gpu_i6.rs`)
`train_mini_resident` nu pornește porțiunea de rutină a jurnalului, deci fiecare rând merge cu `Flush::Now`: serial + ecran + o
scriere pe stick. Generarea (`kronos --gpu`) nu are problema, pentru că ține aceleași rânduri în memorie. Același lucru explică și
cele 286 ms din os-380 (nu urcările de 31 MB, cum scrisesem în P8 — corectat).

## Ce propunem (3 rânduri în bucla pașilor, ca la generare)
```rust
let _hold = super::gpu_i6::GenHold::start();             // ca în generare: nimic pe stick în mijlocul pașilor
journal.set_routine(Some(kf::gen_routine(&GEN_NOTES)));   // înainte de bucla `for (k, (x, y)) in data.batches…`
//   … în buclă, după citirea pierderilor și a normei (înainte de oracol):
journal.flush_live();                                     // OBLIGATORIU: altfel la fiecare 64 de rânduri ținute (≈3 pași) o scriere pe stick
//   … după buclă (și pe drumurile de eroare, cum face generarea):
journal.set_routine(None);
```
T205 (SIMULARE): cu rutina + `flush_live()` pe pas → **0** scrieri pe stick în 10 pași (azi 220); fără `flush_live()` → 3.
Rândurile `kt …` și pașii rămân `Now` (nu sunt de rutină), deci nimic important nu stă în memorie. Opțional: același lucru la
`train_mini_probe` (os-380) și la `--mixt`, dacă se mai rulează.

## Ce așteptăm pe placă (DEDUS)
Pasul întreg = placa 3,65 ms + ~1,2 ms cuvintele driverului (1 533 scrise cu recitire + 369 citite pe pas, T205 J2) + câteva zeci
de µs ⇒ **~5 ms = ~12 000–13 000 simboluri/s** la antrenare (azi 173). Rândul de raport existent („pasul intreg … us”) ajunge.

## Ce tăiem după (al doilea rând, tot fără nimic nou pe placă)
Umbra pushbuffer-ului (lista e aceeași la fiecare pas; se schimbă doar jetoanele înregistrărilor: ~80 din 908 de cuvinte) → −~0,55 ms;
înregistrările: doar prima și ultima puse pe zero / citite (`Diet2Wait { all: false }`; timpul plăcii = ultima − prima) → −~0,35 ms;
lotul în paralel cu pasul (P10, deja predat). Ținta după: **~4 ms pe pas ≈ 16 000 simb/s**.
