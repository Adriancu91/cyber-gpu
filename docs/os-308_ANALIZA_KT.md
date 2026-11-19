# os-308: cifrele `kt` față de estimarea C-145 (sesiunea 118, 19 nov 2026)

Doar analiză (fără cod). Date: `hub/rezultate/os-automat/2026-11-19/gpu-jurnal-013.txt` și `ciclu-auto-20261119-1437.txt`
(ciclul automat `os-308`, 14:37, fără Adrian), comparate cu `gpu-jurnal-011.txt` (`os-307`). Codul OS citit (doar citire):
`cyber-universe/kernel/src/drivers/gpu_i6.rs` (`live_line`, `live_flush`, `write_held`) și `kernel/src/log.rs` (`_log`).

## 1. Ce s-a măsurat (MĂSURAT de OS, pe placă)

| | os-307 | os-308 | estimarea noastră (C-145 §4) |
|---|---|---|---|
| placa (lansările) | 22,9 ms | 23,0 ms | 23 |
| driver (rescrierea lansărilor) | ~41 ms | **0** | ~1 |
| jurnal | ~246 ms | **73,9 ms** | **~17** |
| total pe simbol | 294,8 ms (3,10/s) | **86,8 ms (10,99/s)** | ~45 (20–22/s) |

Umbra lansărilor a lucrat exact cum trebuia (driver 41 → 0 ms; recitirea de la final: identică). **Jurnalul a ieșit de
~4 ori mai scump decât am estimat.** De aici diferența 11 față de 20–22.

## 2. De ce jurnalul costă încă ~74 ms (DEDUS din orele rândurilor)

- Un simbol are 15 rânduri de rutină (mutări ale ferestrei, nota permisului, cele 3 scrieri de control, nota și scrierea
  soneriei). Toate sunt **ținute** (nu se mai scriu pe stick unul câte unul): raportul spune 646 de rânduri ținute, 24 de
  goliri.
- Totuși, între două rânduri ținute trec **~4,0–4,3 ms** (media 4,9 ms pe 239 de intervale din generare; 188 sub 4,5 ms).
  La `os-307`, cu fiecare rând sincronizat pe stick, pasul era ~16–18 ms. Deci stick-ul costa ~12 ms pe rând, iar
  **~4 ms pe rând rămân și fără stick**.
- Cauza, din codul OS: `live_line` face întâi `crate::info!("gpu: jurnal: {}", e)` pentru **fiecare** rând, și abia apoi
  îl ține în memorie. `info!` scrie rândul pe portul serial (octet cu octet, prin sondare), în inelul de 64 KiB și **pe
  ecran** (consola grafică). Timpul acesta e numărat la „jurnal” (ceasul pornește înaintea lui `info!`), deci
  „ecran 0 ms” din `kt` nu-l include. Costul e același pentru rânduri lungi și scurte (4,0 ms), deci cel mai probabil e
  derularea consolei pe ecran; portul serial poate contribui. **Care din ele — nemăsurat.**
- Golirea o dată pe simbol: o scriere de ~1,2 KB pe stick; raportul ciclului dă pentru scrieri în medie ~12 ms
  (sincronizarea propriu-zisă ~0 ms).
- Socoteala: 15 × ~4,3 ms (ecoul) ≈ 64 ms + ~10–12 ms (golirea) ≈ **74–76 ms** = cât a măsurat `kt` (73,9 ms).

**Ce am greșit în estimare:** am presupus că un rând ținut nu costă nimic; n-am citit că OS-ul îl afișează și în jurnalul
nucleului înainte să-l țină.

## 3. Ce înseamnă pentru propunerea `idei/2026-11-19_kronos-placa-mai-rapid.md` (ESTIMARE, nemăsurată)

| Varianta (în OS) | ms pe simbol | simboluri/s |
|---|---|---|
| azi (`os-308`) | 86,8 | 10,99 |
| doar golirea la final (pasul 1 din idee, ecoul rămâne) | ~75 | **~13** |
| fără ecoul rândurilor ținute, golire pe simbol | ~23 + ~12 ≈ 35 | **~27–29** |
| fără ecou + golire la 16 simboluri sau la final | ~24–25 | **~40** |
| plafonul (doar lansările) | 22,9 | 43,8 |

Deci câștigul mare nu vine din „golirea la final”, ci din **a nu mai afișa fiecare rând ținut** în jurnalul nucleului.
Rândurile rămân toate în fișierul de pe stick (aceeași ordine, cu ora la care s-au întâmplat), iar pașii, `kg`/`kv`/`kt`,
citirile și erorile rămân afișate imediat, ca azi.

## 4. Ce ar face OS-ul (propunere; doar cu „da”-ul lui Adrian pe idee)

În `gpu_i6::live_line`: pentru `Flush::Later` fără `crate::info!` pe ecran/serial (eventual doar în inelul din memorie, ca
`log` din shell să le arate), pentru `Flush::Now` exact ca azi. Golirea la 16 simboluri sau la final e un al doilea pas,
mic (`Journal::flush_live` se cheamă unde vrea OS-ul; golirea la erori și la închidere există deja: `live_close`).
Nimic nu se schimbă în ce primește placa; `cyber-gpu` nu are nimic de schimbat pentru asta. Detalii pentru sesiunea OS:
`integration/cyber-universe/os-308_ecoul-jurnalului.md`.

## 5. Siguranță

- Socoteala din §2 se potrivește la ~2 ms cu cifra măsurată; cauza (ecoul prin `info!`) e citită din cod. Siguranță ~80 %.
- Ce nu știm: cât din cele ~4 ms e ecranul și cât portul serial; dacă golirea rară face o singură scriere mare mai lentă
  (cea mai lungă scriere USB din raport: 771 ms — una singură, la deschiderea fișierului).
