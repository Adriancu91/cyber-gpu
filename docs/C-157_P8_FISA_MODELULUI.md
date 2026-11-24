# C-157 punctul 8 — fișa modelului și „pachetele de calcul” (documentul de proiectare)

**Stare:** documentul de proiectare + pașii 1 și 2 de cod (pasul 2: fișele de referință, T211 PASS, `e87fa5d`). Totul pe procesor / în SIMULARE; nimic nou trimis la placă.
**Deciziile lui Adrian (hub `DECIZII.md`):**
- **20 nov 19:41** — driverul și Kronos merg împerecheați: tot ce e dedicat lui Kronos (lanțul, programele lipite,
  mărimile) se construiește **din fișa modelului** din KGR1, nu din numere scrise în cod; o arhitectură nouă = fișă nouă
  + aceeași probă „identic cu procesorul”, nu un driver nou.
- **20 nov 19:45** — **o singură bază** (pornirea plăcii, GSP, memoria, siguranța) + **pachete de calcul** pe fel de
  model (programele pentru placă + lanțul pregătit), alese la pornire după fișă; fără pachet potrivit → **refuz cu mesaj
  clar**, nu calcul greșit; fiecare pachet trece proba „identic cu procesorul” înainte de folosire.

## 1. Ce e azi (citit din cod, 24 nov)

| ce | unde | problema față de decizii |
|---|---|---|
| Cheile KGR1 citite | OS: `kernel/src/drivers/gpu_i14.rs` (cititorul lui), testele noastre: `tests/common/kgr1.rs` | fiecare își face verificările lui; driverul nu are un „tip” pentru fișă |
| Formele lui #513 / #651 | `nvidia_ga104::kronos_speed::{K513_*, K651_*}` | numere scrise în cod (folosite la limite și la probe) |
| Lista de lansări | `kronos_diag::TokenList` (5 liste), `kronos_batch::TokenList` (2) | alegerea listei o face OS-ul după opțiunile din linia de comandă |
| Limitele unei rulări | `kronos_speed::ModelLimits::of_lists(shape, layers, vocab, context, …)` | primesc deja numerele ca argumente — bun, doar că nu dintr-o fișă |
| Condițiile programelor | împrăștiate: `kronos_layer::token_launches` (E multiplu de 32 și de capete, d ≤ 128, poziția < 512), `kronos_model` (context ≤ 512), `layernorm256` (LayerNorm nou doar la E ≤ 1024, altfel cel vechi), OS (1–64 de straturi, capul legat, Q8_0) | un model nepotrivit e prins abia la prima lansare, cu un mesaj de program, nu de model |

## 2. Forma propusă

```
KGR1 (pe stick) ──► FIȘA (gpu_compute::card::ModelCard)          ← doar antetul + cheile, fără tensori
                        │
                        ▼
              ALEGEREA (nvidia_ga104::packs::select)
                 │                         │
         niciun pachet potrivit     pachetul P + dovada lui
                 │                         │
     REFUZ cu motivele, în română   ModelLimits din fișă → lanțul, VRAM, zona de comenzi
                                     dovada: „pe placă pentru forma asta” sau „PROBA NECESARĂ înainte”
```

**Baza** (neschimbată, aceeași pentru orice model): pornirea plăcii, GSP, memoria, canalul de lucru, ceasul termic,
resetul. Nu citește fișa.

**Fișa** (`ModelCard`, crate-ul `gpu-compute`, `no_std`, fără `unsafe`): arhitectura (`kronos_gpt`), felul greutăților
(`q8_0` / `f32`), straturi, capete, n_emb, context, vocabular, `ln_eps`, capul legat. Se citește din primii octeți ai
fișierului KGR1 (antetul + cheile; tensorii nu), așa că OS-ul o poate citi înainte să încarce sutele de MiB.
Cheia `tokenizer.bpe` e sărită (e a procesorului, nu a plăcii).

**Pachetul** (`Pack`, crate-ul `nvidia-ga104`): date fixe, nu cod nou pe model:
- numele + versiunea (ex. `kronos-gpt-q8-generare` v1);
- ce fișe primește: arhitectura, felul, condițiile programelor lui (toate cele de mai sus, într-un singur loc);
- ce rulează: lista de lansări (`kronos_diag::TokenList::FusedLn32` = `lipita-ln-32`, implicită pe placă din os-346)
  și programele ei (SHA-256 verificate deja de cod la urcare);
- **dovezile**: formele pentru care proba „identic cu procesorul” a trecut pe placa reală, cu rularea care o arată.

**Alegerea** (`select(&fișă)`): ia pe rând pachetele; primul care primește fișa câștigă. Rezultatul:
- **refuz** — toate motivele pentru care niciun pachet nu se potrivește (ex. „capete de 192 de valori; pachetul
  `kronos-gpt-q8-generare` merge până la 128”); OS-ul îl tipărește și nu pornește calculul pe placă;
- **pachet + „dovedit pe placă”** — forma fișei e una din dovezi (azi #513 și #651);
- **pachet + „PROBA NECESARĂ”** — fișa se potrivește condițiilor, dar forma n-a fost niciodată probată pe placă: OS-ul
  rulează întâi proba „identic cu procesorul” (cea de azi, `kronos --gpu` cu verificarea pe procesor) și abia apoi
  lucrul. Așa o arhitectură „vecină” (alt număr de straturi, alt n_emb) nu cere un driver nou, dar nici nu e crezută
  pe cuvânt.

**Limitele** vin din fișă: `Choice::limits(piece)` = `ModelLimits::of_lists` cu numerele fișei și lista pachetului.
`K513_*` / `K651_*` rămân (interfața publică nu se strică); devin doar fișele de referință din teste.

## 3. Ce nu face (încă)

- Pachetul de **antrenare** (`train_step::Plan`, atenția pe plăci d ≤ 64, pasul mixt f16): e doar în SIMULARE și pe placă
  prin probele centrului de comandă; intră ca al doilea pachet când OS-ul îl folosește pentru antrenare reală.
- Pachete „instalate” separat (fișiere de pe stick): azi pachetul = programele deja compilate în driver; formatul de
  fișier vine când există al doilea pachet de generare (altă arhitectură).
- Nu schimbă nimic din ce trimite OS-ul la placă: fișa doar alege și verifică înainte.

## 4. Primul pas de cod (planul + predicțiile, scrise ÎNAINTE de cod)

Doar adăugări:
1. `gpu_compute::card` — `ModelCard`, `Arch`, `Quant`, `CardError` (cu `line()` în română), `ModelCard::from_kgr1(&[u8])`,
   `ModelCard::shape()`.
2. `nvidia_ga104::packs` — `Pack`, `Proof`, `Choice`, `Refusal`, `PACKS`, `select`, `Choice::limits`.

Predicțiile testului (T208):
| fișa | prezis |
|---|---|
| #513 (12 straturi, 12 capete, 768, context 512, vocabular 16 384, Q8_0, cap legat) | `kronos-gpt-q8-generare`, **dovedit pe placă** (os-346) |
| #651 (24 de straturi, restul ca #513) | același pachet, **dovedit pe placă** (os-355) |
| fișa minusculă a vectorilor KGR1 (2 straturi, 4 capete, n_emb 32) | același pachet, **PROBA NECESARĂ** |
| `fel = f32` | refuz: felul greutăților |
| n_emb 768, 4 capete (d = 192) | refuz: d > 128 |
| context 1024 | refuz: context > 512 |
| n_emb 100 | refuz: nemultiplu de 32 |
| `arhitectura = llama` | refuz: arhitectura |
| cap nelegat | refuz: capul |
| antet stricat / tăiat / cheie lipsă / `vocabular` ≠ … | eroare de fișă, nu panică |
| limitele din fișa #651 | = `ModelLimits::k651_lists(&[FusedLn32])` |

## 5. Pasul 2 de cod — fișele de referință #513 / #651 (planul + predicțiile, scrise ÎNAINTE de cod, sesiunea 204)

Ținta: numerele lui #513 / #651 să aibă **o singură sursă** — fișa. Azi ele stau de trei ori: constantele `K513_*` /
`K651_*`, formele scrise de mână în dovezile pachetului (`(12, 12, 768, 512, 16 384)`) și cheile din teste. Doar adăugări:
1. `gpu_compute::card::ModelCard::kronos(layers, heads, e, context, vocab)` — fișa Kronos „cum o scrie
   `exporta_greutati.py`” (kronos_gpt, Q8_0, cap legat, eps 1e-5), fără octeți KGR1.
2. `nvidia_ga104::kronos_speed::{card_k513, card_k651}` — fișele de referință făcute din `K513_*` / `K651_*`.
3. `ModelLimits::of_card(&card, piece)` (= `of`) și `ModelLimits::of_card_lists(&card, piece, lists, extra)`
   (= `of_lists`); `Choice::limits_with(piece, extra)` (OS-ul adaugă `Prog::Argmax` după listă).
4. `k513()` / `k651()` / `k651_lists` rămân neschimbate (interfața nu se strică); testul arată că dau aceleași cifre.

Predicțiile testului (T211):
| ce | prezis |
|---|---|
| `card_k513()` = `from_kgr1(kgr1_header(kronos_keys(12, 12, 768, 512, 16 384)))` | egale câmp cu câmp |
| `card_k651()` la fel, cu 24 de straturi | egale |
| `ModelLimits::of_card(card_k513, 8 MiB)` / `card_k651` | = `k513()` / `k651()` |
| `of_card_lists(card_k651, 8 MiB, [FusedLn32], [])` | = `k651_lists(&[FusedLn32])` (172 de lansări) |
| `select(card_k513)` / `select(card_k651)` | dovedit pe placă, os-346 / os-355 |
| formele scrise în dovezile pachetului | = `card_k513().form()` și `card_k651().form()` (dacă cineva schimbă o constantă, testul cade) |
| `Choice::limits_with(piece, [Argmax])` pentru #651 | o lansare în plus față de `limits(piece)` |
| `ModelCard::kronos` cu n_emb 100 | fișa se face (e doar date), dar `select` o refuză (nemultiplu de 32) |

Ce e nou pentru placă: nimic.
