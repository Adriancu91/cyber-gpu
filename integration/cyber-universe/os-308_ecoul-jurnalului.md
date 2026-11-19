# Pentru sesiunea OS: de ce „jurnal” = 74 ms la `os-308` (observație, nu comandă)

Analiza: `cyber-gpu/docs/os-308_ANALIZA_KT.md`. Nimic de legat din `cyber-gpu` (nicio schimbare în crate-uri).

- Rândurile de rutină ținute (`Flush::Later`) costă tot ~4 ms fiecare, pentru că `gpu_i6::live_line` le trece întâi prin
  `crate::info!` (port serial octet cu octet + inelul de 64 KiB + consola de pe ecran). 15 rânduri × ~4,3 ms ≈ 64 ms;
  golirea pe stick ~10–12 ms. Total ≈ 74 ms = `kt median jurnal`.
- Dacă Adrian spune „da” la `idei/2026-11-19_kronos-placa-mai-rapid.md`, varianta cu câștigul mare:
  1. în `live_line`, pentru `Flush::Later`: fără `crate::info!` (rândul rămâne în fișierul de pe stick; dacă vreți să-l
     vedeți în `log`, doar în inel, fără serial/ecran); `Flush::Now` neschimbat;
  2. apoi, opțional, `flush_live` la 16 simboluri sau la final în loc de la fiecare simbol (`live_close` golește deja la
     final și după erori).
- Estimare (nemăsurată): (1) singur ~27–29 simb/s; (1)+(2) ~40 simb/s. Doar (2), fără (1): ~13 simb/s.
- Ca să știm cât e ecranul și cât portul serial, un rând `kt` sau o notă în raport cu timpul lui `info!` separat ar ajuta.
