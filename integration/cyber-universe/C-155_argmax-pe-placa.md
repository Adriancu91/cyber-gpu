# Pentru sesiunea OS: argmax pe placă (C-155 punctul 3) — gata de legat, după LayerNorm nou

**De la Cyber-GPU, sesiunea 134 (21 nov 2026). Stare: SIMULARE** (T145, `crates/nvidia-ga104/tests/sim_c155_argmax.rs`). Nimic din ce
scrie aici nu a rulat pe RTX 3070 Ti. Totul e **adăugat** (nimic redenumit sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.

## De ce
Azi fiecare simbol se termină cu copierea logit-urilor în RAM (64 KiB la #513: 16 384 de numere), OS-ul le citește (os-324: mediana
**47 µs**) și alege simbolul pe procesor (**85 µs**). Cu programul nou, **placa alege simbolul**: un grup de 256 de fire citește
logit-urile din VRAM și scrie în RAM **un singur rezultat de 16 octeți**; OS-ul citește 16 octeți. E și primul pas spre etapa 5
(simbolul rămâne pe placă pentru pasul următor).

Nu se calculează nimic altfel: alegerea maximului nu face nicio rotunjire, deci rezultatul e **exact** `cm::argmax` al acelorași
logit-uri (cel mai mic index la egalitate; −0,0 = +0,0). Comparațiile sunt pe chei întregi făcute din biți, nu pe numere cu virgulă:
NAK compilează comparațiile de numere cu FTZ (numerele foarte mici luate drept 0), procesorul nu — cheile dau exact ordinea procesorului.

## Rezultatul (16 octeți, 4 cuvinte)
| Cuvânt | Ce e |
|---|---|
| 0 | simbolul (indexul celui mai mare logit); `0xffffffff` dacă vreun logit e NaN/infinit — atunci nu există simbol |
| 1 | biții celui mai mare logit finit (−0,0 scris ca +0,0) |
| 2 | câte logit-uri sunt NaN sau infinite |
| 3 | n (vocabularul), cum l-a citit programul — dacă nu e vocabularul, rezultatul nu e al acestei lansări (umplutura `0xEE` rămasă) |

## Ce e nou la noi (ramura `g0-g2-fundatie`)
- `nvidia_ga104::argmax`: `ARGMAX` (7 888 octeți), `ARGMAX_SHA256`, `REGISTERS` = 25, `BLOCK` = 256, `SHARED` (96 B, **o barieră**),
  `RESULT_BYTES` = 16, `launch(logits_va, rezultat_va, n)`, `ArgmaxResult::{from_bytes, token(n), agrees(logits), top, nonfinite}`.
  `token(n)` dă `Ok(Some(i))`, `Ok(None)` (logit-uri nefinite) sau **eroare** (rezultat nescris / alt n / index în afara
  vocabularului) — niciodată un simbol ghicit.
- `kronos_layer::{Prog::Argmax, PROGRAMS_ARGMAX}` (cele 9 de la LayerNorm nou + `Argmax`). `Launch::qmd` pune singur 256 de fire,
  25 de registre, 96 B partajată, `BARRIER_COUNT` = 1.
- `kronos_gen::{TopOutput, TopRunner, Generation::accept_top, Generation::run_top}`: aceeași generare, cu simbolul ales pe placă
  (aceleași opriri: `LogitiNefiniti`, `CeasInapoi`, `Eroare`).
- `gpu_compute::model::{argmax_result, ARGMAX_NONE}` (oracolul, cuvânt cu cuvânt); `gpu_sim::cmdproc::{argmax_on_cpu, ARGMAX_NEEDS}`.
- `kronos_token::work_list` refuză acum `LayerNorm256` și `Argmax` (programul unic pe simbol nu le are; OS-ul nu-l folosește).

## Ce legi (Release separat, după LayerNorm nou; opțiune oprită implicit, de ex. `--argmax-placa`)
1. Programele: cu opțiunea pornită, încarcă și `Prog::Argmax` (de ex. `kl::PROGRAMS_ARGMAX` în loc de `list.programs()`).
2. RAM: 16 octeți în plus în arena logit-urilor, aliniați la 16 — de ex. imediat după bufferul logit-urilor
   (`host_logits.va + logits_bytes`; `logits_bytes` e deja multiplu de 16). Umplut cu `0xEE` înaintea fiecărui simbol, ca logit-urile.
3. **Proba (prima rulare, testul obișnuit):** `extra = [copy_launch(...), argmax::launch(vas.logits, rezultat_va, m.vocab)]` — copia
   ȘI argmax-ul în același simbol (o lansare în plus). La fiecare poziție: `ArgmaxResult::from_bytes(16 octeți).agrees(&logits)` trebuie
   să fie `true` și `token(m.vocab)` = `cm::argmax(&logits)`; oracolul pe procesor, ca acum. Rânduri propuse:
   `kt argmax placa = procesor: 16/16` și `kt median argmax placa (us)` (timpul lansării, din înregistrări).
4. **Viteza maximă (după ce proba trece):** `extra = [argmax::launch(...)]` fără copie; în loc de `TokenRunner` +
   `Generation::run`, `TopRunner` + `Generation::run_top` (citești doar cei 16 octeți). Amprenta logit-urilor (`fnv1a`) nu mai există
   în această rulare: rândul de comparație cu os-322 rămâne în rularea cu copie (punctul 3). Rândul `kt median citirea rezultatului (us)`.
5. Dieta (C-154), tăieturile, `kp`: o lansare în plus (punctul 3) sau aceeași lungime ca azi (punctul 4: copia înlocuită de argmax).
   Felurile `kp` nu au încă felul „argmax” — dacă profilul refuză simbolul, ajunge un fel nou (îl adaug la cerere).

## Ce trebuie să iasă
- Proba: **16/16 poziții la care cele 4 cuvinte = `argmax_result` al logit-urilor copiate**, simbolurile ca în os-322
  `[44, 10, 73, 2524, 10766, 1438, 1505, 1428, 1765, 99, 1547, 3644, 44, 10, 15879, 1505]`. Prag: **0** (bit cu bit).
- Viteza (DEDUS, neverificat): citirea 47 µs + argmax-ul pe procesor 85 µs ≈ **130 µs mai puțin pe simbol**, minus timpul lansării
  argmax (un singur grup citește 64 KiB: estimez ~5–15 µs, cam cât copia pe care o înlocuiește). Adică ~4,08 → ~3,95 ms pe simbol
  (~253 simb/s mediană). Câștig mic; rostul mare e etapa 5.

## Ce e nou pentru placă
**Un program nou** (`argmax`, SHA-256 `4d67cd34…0572`, Mesa NAK 25.0.7): 256 de fire, `shfl.bfly`, o barieră (ca `layernorm256`),
8 citiri `ld.global.constant` pornite una după alta, **o scriere de 16 octeți în RAM** (ca scrierile copiei de azi, în aceeași
arenă). Nicio scriere nouă în registre, nimic nou prin GSP. Dacă proba nu trece: rularea cu copie singură (cum e azi) rămâne la îndemână.
