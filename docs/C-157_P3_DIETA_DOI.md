# C-157 punctul 3 — „dieta, partea a doua”: driverul de la 759 µs spre ~350 µs pe simbol

Sesiunea 136, 21 nov 2026, în cloud, **fără placă**. Ținta din coadă: driverul sub 0,5 ms pe simbol, **nimic nou trimis la placă**.

Etichete: **MĂSURAT** = din jurnalul de pe placă; **DEDUS** = calcul sau interpretare a noastră; **SIMULARE** = doar pe `gpu-sim` /
pe procesorul gazdei din cloud.

Surse: `Cyber-hub/rezultate/2026-11-21-workstation-os328-p3/REZULTAT.md` și `rezultate/os-automat/2026-11-21/gpu-jurnal-026.txt`
(os-328, lista lipită, #513, viteza maximă); unealta `timpul_driverului` pe același jurnal; codul OS citit (doar citit):
`cyber-universe` `13582db`, `kernel/src/drivers/gpu_i14.rs` (`BoardRunner::{submit, run_token}`), `gpu_i11.rs`
(`run_records_inner`), `gpu_i10.rs` (`kick_and_wait`), `kernel/.cargo/config.toml` (ținta nucleului); măsurarea pe gazdă
`crates/nvidia-ga104/examples/dieta_doi.rs`.

## 1. Unde se duc azi cei 759 µs ai driverului (os-328, median pe simbol)

`kt median driver (us)` = 759 (MĂSURAT) = ceasul OS-ului pe simbol − timpul plăcii − jurnalul (61 µs). Împărțirea după fereastra VRAM
(`timpul_driverului`, 40 de simboluri constante, MĂSURAT): fereastra înregistrărilor `0x1b2f` 2 587 µs (placa 2 421 + **~166**),
pushbuffer-ul `0x10e6` 74, lansările `0x1b2d` 38, inelul `0x101f` 16, procesorul fără fereastră 928 (din care jurnalul 61 și ~96 în
afara ceasului simbolului).

| Parte | µs (OS) | Ce e | Etichetă |
|---|---:|---|---|
| citirea celor **89 de înregistrări** prin fereastra VRAM | ~166 | 356 de citiri de câte 4 octeți (~0,47 µs fiecare), după fiecare simbol; folosite pentru profilul `kp` pe fel | MĂSURAT (fereastra) ; ce e în ea DEDUS |
| **așteptarea** ultimei înregistrări | ~50 | `poll_until(…, 100, …)`: o citire la 100 µs → simbolul se termină în medie ~50 µs după placă | DEDUS din cod |
| **amprenta FNV-1a** a logit-urilor | ~100–150 | 65 536 de octeți, unul câte unul, în ceasul simbolului; e un ajutor de probă, nu face parte din generare | gazda: 90 µs (SIMULARE); OS DEDUS |
| **argmax** pe numere | 85 | nucleul OS e compilat **soft-float** (`x86_64-unknown-none`): fiecare comparație `>` e un apel în emularea float | MĂSURAT (`kt median argmax`) |
| citirea logit-urilor din RAM | 48 | 64 KiB → `Vec<f32>` | MĂSURAT |
| construirea lansărilor, QMD-urile, umbrele, pushbuffer-ul | ~130 | 88 de lansări construite și comparate cu umbra la fiecare simbol | gazda: 37 µs (SIMULARE); OS DEDUS ×1,6–3,5 |
| ferestrele pushbuffer + lansări + inel | 128 | cuvintele schimbate (130) scrise și recitite | MĂSURAT |
| restul | ~50 | copiile logit-urilor (`last_logits`, `kept`), profilul, viteza | DEDUS |

Măsurarea pe gazdă (`cargo run --release -p nvidia-ga104 --example dieta_doi`, SIMULARE, 400 de simboluri cu forma #513): lansările
8 µs, QMD 9, umbra lansărilor 16, pushbuffer 4, `analyze_sequence` 0,4, citirea logit-urilor 30, argmax 17, **FNV 90**, copiile 3.
Raportul OS/gazdă e ~1,6 pentru lucru cu întregi (citirea logit-urilor: 48 față de 30) și ~5 pentru comparațiile float (argmax: 85
față de 17) — de aceea înmulțirea „×4,9” tipărită de unealtă e prea mare pentru partea cu întregi (DEDUS).

## 2. Dieta 2 — patru tăieturi, nimic nou pentru placă (toate DEDUS)

| # | Ce | Cum (de la noi, doar adăugări) | Câștig pe simbol (median) |
|---|---|---|---:|
| A | **Două înregistrări în loc de 89** pe 7 din 8 poziții | `kronos_diet2::RecordPlan::DIET` (toate doar la pozițiile 0, 1 și din 8 în 8 — profilul `kp` rămâne median) + `ends_span` (prima și ultima înregistrare: timpul plăcii **identic** cu suma duratelor, T147) | −150…−165 µs |
| B | **Așteptarea planificată** | `kronos_diet2::WaitPlan` + `poll_planned`: doarme cât cel mai scurt timp recent al plăcii − 100 µs, apoi citește la 5 µs | −45 µs |
| C | **Amprenta în afara ceasului** | aceeași funcție FNV-1a, calculată după ce ceasul simbolului s-a oprit (amprenta rămâne comparabilă cu os-324/os-328) | −100…−150 µs |
| D | **argmax pe biți** | `gpu_compute::model::argmax_bits`: aceeași alegere ca `argmax`, doar comparații de întregi (fără emularea float) | −50…−70 µs |
| | **Total** | | **−350…−430 µs → driverul ~330–410 µs** |

Fără C (dacă OS-ul vrea amprenta în ceas): ~480–560 µs — la limita țintei. Cu argmax-ul pe placă (C-155 punctul 3, deja predat;
acela e nou pentru placă) dispar și citirea logit-urilor + argmax-ul (~48 + ~25 µs).

**Simbolul întreg (#513, median, DEDUS):** 3 284 µs → ~2 880 µs (~347 simb/s), placa neschimbată (2 421 µs).

## 3. Ce NU e dovedit
- Nimic pe RTX 3070 Ti. Câștigurile sunt calcule din cifrele os-328 și din citirea codului OS.
- Pentru B: precizia lui `delay_us` în OS nu o știm; dacă o întârziere scurtă durează mai mult, câștigul e mai mic (nu mai mare).
- Pentru A: pe pozițiile fără citire completă nu știm durata fiecărei lansări (doar a simbolului); de aceea profilul `kp` se face din
  pozițiile cu citire completă. Verificarea „fiecare lansare și-a scris înregistrarea” rămâne pe pozițiile complete; pe celelalte,
  ultima valoare cerută dovedește că toate s-au terminat (canalul le rulează în ordine).

## 4. Ce s-a făcut în sesiunea 136 (doar adăugări)
- `gpu_compute::model::{order_key, argmax_bits}` + testul `argmax_bits_is_argmax` (4 000 de vectori la întâmplare cu −0/+0,
  subnormale, MAX/MIN, NaN, infinit: aceeași alegere ca `argmax`).
- modulul `nvidia_ga104::kronos_diet2`: `RecordPlan` (`ALL`, `DIET`, `FULL_EVERY`, `reads_all`, `words`), `ends_span`, `WaitPlan`
  (`new`, `observe`, `forget`, `first_sleep_us`, `FINE_US`, `MARGIN_US`, `COARSE_US`, `RECENT`), `poll_planned` → `WaitStats`,
  `Diet2Stats` (`add`, `rows`, `journal`), `ParsedDiet2`, `parse_diet2`.
- `tests/sim_c157_diet2.rs` (T147, SIMULARE): un text de 16 poziții cu lista lipită și dieta 1: la fiecare poziție, timpul din
  două înregistrări = suma duratelor = `end − start` = `busy`; `argmax_bits` = `argmax`; o înregistrare lipsă sau un ceas înapoi e
  prins și cu două înregistrări; cuvinte citite 1 664 → 416.
- exemplul `dieta_doi` (măsurarea pe gazdă de mai sus).

## 5. Proba pe placă (după ce OS-ul o leagă, în spatele unei opțiuni)
Aceleași 16 simboluri, **amprenta logit-urilor identică** cu os-328 (`0xda9ae084`) — aceleași programe, aceiași octeți;
`kt median placa` neschimbat (~2 421 µs); `kt median driver` de la 759 spre **~350–400 µs**; rândurile noi
`kt cuvinte de inregistrari citite pe simbol (median)` = 8 și `kt citiri ale asteptarii pe simbol (median)` ~20.
