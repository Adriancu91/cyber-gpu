# Ordinea 20:11, pasul 3 — lipirea operațiilor „înapoi” (plan + predicții, scrise ÎNAINTE de cod)

**Sesiunea 192, 23 nov 2026, cloud, fără placă.** Tot ce urmează e SIMULARE sau DEDUS; nimic nu rulează pe RTX 3070 Ti.

## 1. Ce înseamnă „lipirea” aici

Pasul de antrenare e o listă de lansări (programe mici pe placă). Între două lansări, rezultatul intermediar se scrie în
memoria plăcii și se citește înapoi. „Lipirea” = un singur program face două operații vecine, iar rezultatul intermediar
rămâne în registre. Câștigul: mai puțin trafic de memorie și mai puține lansări. Condiția noastră: **aceiași biți** ca
pasul de azi (altfel nu mai putem spune „e același antrenament, doar mai repede”).

## 2. Inventarul părții „înapoi”, pe strat (din `Plan::forward_backward`)

| # | Lansarea de azi | Ce citește / scrie (n = rânduri, C = lățimea, F = 4C) | Se poate lipi cu |
|---|---|---|---|
| a | `ln_back` (ln2) → `vadd` „rezidual-ln2” | `ln_back` scrie dx2 (nC); `vadd` citește dx și dx2, scrie dx2 (3nC) | **a1: `ln_back_res`** — dx2 = dx + ln_back(…) în același program |
| b | `ln_back` (ln1) → `vadd` „rezidual-ln1” | la fel | **a1** (același program) |
| c | `mm` „proj-dX” → `gelu` înapoi | `mm` scrie dwide (nF); `gelu` citește u și dwide, scrie dwide2 (3nF) | b1: GELU' în coada înmulțirii (program `mm_train` nou) |
| d | `colsum` (bias) pe dx / dx2 / dwide2 / dqkv | citește o dată fiecare | nu acum (sume pe coloane, alt tipar de citire) |

## 3. Ordinea

1. **a1 — `ln_back_res` (sesiunea asta).** Cel mai mic program nou, înlocuiește 2 lansări `vadd` pe strat.
2. b1 — GELU' în coada lui `mm_train` (sesiunea următoare: câștig de ~4× mai mare, dar înmulțirea e programul cel mai delicat).
3. d — de reevaluat după ce măsurăm pe placă a1/b1.

## 4. a1: programul `ln_back_res`

`ln_back.comp` neschimbat, plus un argument `dres` (gradientul care vine pe calea reziduală):
`dx[k] = dres[k] + rstd·((g − m1) − x̂·m2)`.

- Ca să iasă **aceiași biți**: pasul vechi rotunjește întâi `rstd·(…)` (scris în memorie), apoi `vadd` face `fadd.ftz(a, b)`.
  În programul lipit, NAK ar putea contopi înmulțirea și adunarea într-un singur `ffma` (o rotunjire în loc de două) — alți
  biți. De aceea produsul și suma sunt marcate `precise` în GLSL (interzice contopirea).
- Adunarea e comutativă exact în IEEE (fără NaN), deci ordinea `dres + r` față de `vadd(dx, dx2)` nu contează.
- Fără alias: ieșirea (dx2 la ln2, dx la ln1) e alt buffer decât `dres` și decât `dy` (dtmp) — verificat în `Plan`.
- În bibliotecă: `Prog::LnBackRes` (doar adăugare), `Plan::with_fused_back()` (opțiune, oprită implicit),
  `Plan::fused_programs()` (ce mai trebuie încărcat în plus față de `programs()`). `PROGRAMS_ALL` și celelalte liste
  **nu se schimbă** (OS-ul își face tabla simulată din `PROGRAMS_ALL`).

## 5. Predicțiile (scrise înainte de cod)

- **P1 (biți):** pasul cu `ln_back_res` dă **exact aceiași biți** ca pasul de azi (`new_tiles3`): pierderea la fiecare pas,
  gradienții după pasul 1, greutățile după pașii 1 și 10, pe pasul de aur mic (SGD și AdamW) și pe 4 straturi cu lățimi mai mari.
- **P2 (codul NAK):** în ultima trecere a lui `ln_back_res` apar un `fmul` urmat de un `fadd` (nu un `ffma` cu rstd);
  registrele ≤ 24 (ln_back are 19); aceeași memorie partajată (64 B) și o barieră.
- **P3 (lansări):** pe pas, exact **2·L lansări mai puține** (L = straturi): toate `vadd` „inapoi:*” dispar, cele 2·L din
  „inainte:*” rămân; restul listei identic (aceleași etichete, în aceeași ordine, fără cele 2·L).
- **P4 (greșeli plantate, prinse):** (i) modelul cu `ffma` în loc de `fmul`+`fadd` → alți biți decât pasul vechi pe cel puțin un
  gradient; (ii) `dres` dat greșit (alt buffer) → verdictul față de PyTorch pică.
- **P5 (câștig, DEDUS):** la #513 (12 straturi, C 768, 6 144 de rânduri pe bucată) fiecare lipire scutește ~2·n·C·4 B ≈ 37,7 MB
  de trafic; 24 pe bucată ≈ **0,9 GB → ~2–2,5 ms pe bucată** (la ~380–450 GB/s practici) + 24 de lansări (~0,1 ms). Mic
  (~1 % din pas), dar fără niciun risc numeric; pe placă se va vedea ca diferență de timp între „vechi” și „lipit”.

## 6. Ce NU face

Nu schimbă nimic din înainte, din AdamW sau din atenție. Nu atinge placa reală: pe placă, `ln_back_res` e un program nou
care se probează abia după `--antrenare-mini` F32 și os-352, cu Adrian de față.

## 7. Rezultatul

(se completează după test)
