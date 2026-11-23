# C-157 — AdamW cu m, v în RAM, strat cu strat (SIMULARE)

Pasul 2 din `docs/C-157_RAM_OFFLOAD.md` §4. Planul și predicțiile sunt scrise **înainte** de cod (sesiunea 184, 23 nov 2026).

## Ce schimbăm

Azi `train_step::Plan` ține pe placă patru vectori de mărimea modelului: greutățile `w`, gradientul `g` și mediile AdamW `m`, `v`
(16 B pe parametru). `m` și `v` sunt citite și scrise **doar** de ultima lansare a pasului (`opt_step`). Le mutăm în RAM:

- `Plan::new_tiles_mv_ram` / `Plan::new_p_only_tiles_mv_ram` (adăugare): aceeași arenă ca planurile pe plăci, **fără** `m`, `v`
  (planul cel mai bun de azi, cel pe care OS-ul îl rulează);
- modulul nou `nvidia_ga104::adam_ram`: bucățile (tabelele; fiecare strat cu cele 12 tensori ai lui; LN-ul final), un loc de
  așteptare pe placă de mărimea celei mai mari bucăți (m + v), și pentru fiecare bucată, în ordine:
  1. motorul de copiere: `m`, `v` ale bucății RAM → locul de pe placă, înregistrarea motorului (cu `FLUSH`);
  2. canalul așteaptă înregistrarea (`SEM_EXECUTE` `ACQ_STRICT_GEQ`, 32 de biți) — calculul nu pornește înainte de date;
  3. `opt_step` pe bucată (`g`, `w` = felia bucății; `m`, `v` = locul de pe placă), apoi înregistrarea cu timp și WFI (ca azi);
  4. motorul de copiere: locul de pe placă → RAM, înregistrarea motorului.
- `opt_step` lucrează element cu element: pe bucăți dă **aceiași biți** ca pe tot vectorul (aceleași operații, aceleași valori).
  Norma și factorul de tăiere rămân pe tot gradientul, pe placă, neschimbate.

Ordinea motor de copiere ↔ calcul: așteptarea pe înregistrarea motorului (pasul 2) și WFI-ul de după lansare (pasul 3) — aceleași
metode pe care placa le-a primit deja (C-69, os-* `ram_vram`). Că WFI-ul canalului așteaptă și scrierile calculului înainte ca
motorul de copiere să citească locul: **DEDUS** (aceeași memorie L2; tinygrad și NVK pun tot o barieră între cozi).

## Predicții (înainte de cod)

1. **Bit cu bit:** pasul de aur mic (MIC, 2 × 32), `adamw_513`, 5 pași, plan pe plăci: pierderile, normele, greutățile după pasul 1
   și după ultimul pas, și `m`, `v` (din RAM) = cele ale planului de azi, **identice bit cu bit** (SIMULARE). La fel 4 straturi.
2. Lansări pe pas: aceleași ca azi, minus 1 (`opt:pas`), plus una pe bucată: L + 2 (MIC: 4 în loc de 1).
3. Greșeli plantate prinse: o bucată sărită (greutăți diferite), `m` și `v` inversate la o bucată, o copiere înapoi lipsă (pasul 2
   diferă).
4. Placa economisește `8·N − 8·max_bucată` octeți: la #513 (12 × 768, 16 384 de simboluri) ~0,73 GiB − 0,10 GiB ≈ **0,63 GiB**; la
   #651 (24 straturi) ~1,36 − 0,10 ≈ **1,26 GiB** (bucata cea mai mare = tabelele, ~12,6 M parametri).
5. Timpul de legătură în plus pe pas (DEDUS, cu 11–13 GB/s din predicțiile `ram_vram`, fără suprapunere): 16 B/param →
   ~0,25 s la #513, ~0,45 s la #651 (de ~2 ori cât scrie studiul, care presupunea suprapunerea pe jumătate). Suprapunerea (bucata
   următoare urcă în timp ce calculul lucrează) = pasul următor, după cifra reală a legăturii.

## Ce NU face

- pasul cu loss scaling (`step_mixed`) — refuzat pe un plan cu m, v în RAM (pasul următor, dacă e nevoie);
- nicio suprapunere încă (copierile și calculul unul după altul);
- nimic pe placă: proba se face prin OS, după comandă.

## Surse

ZeRO-Offload (Ren et al., 2021, arXiv 2101.06840) — ideea mutării stărilor optimizatorului; `clc7b5.h` / `clc56f.h` (NVIDIA, MIT) —
metodele; tinygrad `ops_nv.py` (MIT) citit ca specificație pentru așteptarea pe semafor.

## Rezultat (sesiunea 184, după cod) — SIMULARE

| Predicție | Ce a ieșit | |
|---|---|---|
| 1. bit cu bit (MIC 5 pași, ambele planuri pe plăci; 4 straturi 3 pași) | identice: pierderi, norme, greutăți după pasul 1 și ultimul, m, v | DA |
| 2. lansări pe pas L + 2 în loc de 1 | 75 → 78 (P păstrat), 76 → 79 (P refăcut) | DA |
| 3. greșeli plantate prinse | bucată repetată, m ↔ v la pasul 2, copiere înapoi pierdută: prinse. **Aflat:** m ↔ v inversate la fiecare pas lasă greutățile identice (inversarea se anulează); se vede doar în m, v din RAM — testul o verifică așa | DA, cu o observație |
| 4. economie pe placă 0,63 / 1,26 GiB | 0,63 / 1,27 GiB (#513 1,98 → 1,25 GiB + 0,10 locul; #651 3,67 → 2,31 + 0,10) | DA |
| 5. timpul de legătură pe placă | — | NOT TESTED |

Nota pentru OS: `integration/cyber-universe/C-157_ADAM_RAM_m-v-in-RAM.md`. `tools/check.sh`: 924 de teste (+7 ignorate), PASS.
