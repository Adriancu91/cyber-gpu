# C-160 punctul 5 — pasul de aur mic cap-coadă pe placa simulată

Sesiunea 153, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce s-a făcut
Pasul întreg de antrenare al modelului mic (`pas_de_aur_mic`: 2 straturi, C 64, 4 capete, vocabular 256, lot 2 × 32) rulează
acum **numai ca listă de lansări** pe placa simulată: fiecare program trece prin pushbuffer-ul + QMD-ul nostru, prin procesorul
de comenzi, iar toate tensoarele intermediare stau în memoria (simulată) a plăcii. Ordinea unui pas:

1. `emb_fwd` (încorporările);
2. pe fiecare strat: `ln_fwd` → `mm_train` (in_proj) → `attn_fwd` → `mm_train` (out_proj) → `vadd` (reziduul) → `ln_fwd` →
   `mm_train` (fc) → `gelu_tr` (înainte) → `mm_train` (proj) → `vadd`;
3. `ln_fwd` final → `mm_train` (capul legat, logit-urile) → `xent` (eroarea pe rânduri + gradientul ei, scara 1/N);
4. înapoi: `mm_train` (dW-ul capului, scris direct în gradientul tabelei de simboluri) → `mm_train` (dhf) → `ln_back` + `colsum`
   (LayerNorm final); pe fiecare strat, invers: 4 × (`mm_train` dX, `mm_train` dW, `colsum` deplasarea), `gelu_tr` (înapoi),
   2 × (`ln_back` + `colsum`), `attn_back` (rândurile lui dS, apoi dQ | dK | dV), 2 × `vadd` (reziduul);
5. `emb_back` (pozițiile — scriere; simbolurile — adunare peste dW-ul capului);
6. `adamw_513`: `gnorm` (două lansări: sume parțiale, apoi norma + factorul de tăiere) → `opt_step`; `sgd`: factorul 1 scris
   de driver → `opt_step`.

**75 de lansări pe pas** la `adamw_513` (73 la `sgd`): `mm_train` 27, `colsum` 13, `vadd` 8, `ln_fwd` 5, `ln_back` 5,
`gelu_tr` 4, `attn_back` 4, `attn_fwd` 2, `emb_back` 2, `gnorm` 2, `emb_fwd` 1, `xent` 1, `opt_step` 1 = **31 pe strat + 13**.
Memoria plăcii folosită: ~2,8 MiB.

Procesorul face doar: încărcarea o dată a lotului și a greutăților inițiale, zeroizarea o dată a bufferului de gradienți (rândurile
lui `emb_pozitie` după lungimea textului nu le scrie nimeni și trebuie să rămână 0), listele de segmente ale lui `emb_back` (numere
întregi, din simboluri, o dată pe lot), oprirea pasului dacă norma nu e finită, și citirea înapoi a ce comparăm. Media erorii se
adună pe procesor **doar pentru raport** (pasul nu o folosește).

## Ce a ieșit (SIMULARE) — T165, `crates/nvidia-ga104/tests/sim_c160_p5_golden.rs`
Față de **PyTorch** (`reteta.json` + tensoarele din hub), în toleranțele hub-ului (aceleași ca la oracolul pe procesor, T151):

| Rețeta | eroarea, 10 pași (tol. 1e-5 rel.) | gradientul pasului 1 (tol. 1e-5 abs., 1e-3 rel.) | greutăți după pasul 1 / 10 | toleranța | în plus |
|---|---|---|---|---|---|
| `sgd` | ≤ 1,09e-7 | 1,49e-7 abs., 5,4e-7 rel. | 6,0e-8 / 2,4e-7 | 1e-6 | |
| `adamw_513` | ≤ 8,8e-8 | 1,49e-7 abs., 5,4e-7 rel. | 2,4e-6 / 1,28e-5 | 1e-4 | normele celor 10 pași ≤ 8,9e-8 rel. |

Față de oracolul pe procesor (`gpu_compute::train`): gradientul pasului 1 ≤ 3,6e-7; `sgd` greutățile ≤ 1,2e-7 după 1 și 10 pași;
`adamw_513` ≤ 2,2e-6 / 8,1e-6; erorile la `adamw_513` au ieșit aceleași numere f32 la toți cei 10 pași (diferențele sunt sub
o unitate f32 la ~44, deci se rotunjesc la fel), la `sgd` ≤ 1,0e-7.

Eroarea scade la fel ca în PyTorch: `sgd` 44,19 → 33,89; `adamw_513` 44,19 → 43,21 în 10 pași.

**Greșeli plantate în lista driverului, prinse de comparație** (`adamw_513`): reziduul uitat la întoarcere (gradient 0,23 abs.),
scara erorii 2/N în loc de 1/N (gradient 1,29), dW-ul capului scris peste suma simbolurilor (gradient 0,079) — toate după un
pas; **tăierea uitată** — după un pas nu se vede (2,7e-5 < 1e-4: AdamW la primul pas e aproape insensibil la o scară constantă a
gradientului, m/√v ≈ ±1), după 10 pași da (2,7e-4 > 1e-4). Notat ca să nu ne bazăm pe un singur pas când verificăm tăierea.

## `pas_de_aur_513` „cât permite simularea”
În hub există doar `pas_de_aur_513/reteta.json` (erorile PyTorch pe 10 pași); greutățile lui #513 (2,7 GB) nu sunt în cloud, deci
**pasul de aur 513 adevărat nu se poate compara aici.** Ce am făcut în locul lui: aceeași listă de lansări la **lățimile lui #513**
(C 768, 12 capete de 64, vocabular 16 384, un text de 64) cu **greutăți inventate** (deterministe), un pas `adamw_513`, față de
oracolul pe procesor, în aceleași toleranțe:
- **1 strat** (test obișnuit, ~40 s): 44 de lansări, ~320 MiB pe placa simulată; eroarea = oracolul (aceeași f32), norma 3,8e-8
  rel., gradientul 5,2e-7 abs. / 2,7e-6 rel., greutățile 6,1e-6 abs. — **trece**;
- **12 straturi** (toate straturile lui #513; test `#[ignore]`, rulat de mână, ~3 min): **385 de lansări** (31 × 12 + 13), ~1 545 MiB pe
  placa simulată (greutăți + gradienți + m + v ≈ 1,46 GiB, cum spunea studiul, + activările unui text de 64); eroarea = oracolul,
  norma 1,5e-7 rel., gradientul 5,1e-7 abs. / 3,7e-6 rel., greutățile 6,9e-6 abs. — **trece**.

Pasul de aur 513 adevărat (aceleași numere ca PyTorch pe greutățile reale) rămâne pentru placa reală / OS, unde greutățile există
(Kronos, C-162; OS-ul are deja `antrenare proba` cu oracolul).

## Ce NU e dovedit
Nimic pe placă. Programele de antrenare (toate, în afară de `vadd`) n-au rulat niciodată pe RTX 3070 Ti; unitățile aproximative
(`MUFU`) sunt modelate rotunjite corect, cu loc 2⁻²² în margini. Viteza: nemăsurată (75 de lansări × ~6 µs pe lansare ≈ 0,45 ms
doar costul lansărilor la modelul mic — DEDUS din P8).

## Ce e nou pentru placă când se predă (C-160 punctul 6)
Ordinea propusă rămâne: **întâi un singur program „înapoi”** (cel mai simplu: `colsum` în modul deplasare, sau `mm_train` în
modul dW), apoi pasul întreg al modelului mic = această listă de 75 de lansări, comparată pe placă cu oracolul și cu PyTorch.

**Predat (sesiunea 155):** lista e acum în bibliotecă (`nvidia_ga104::train_step`, un pas = o trimitere); nota pentru OS:
`integration/cyber-universe/C-160_P6b_pasul-de-aur-pe-placa.md`.

## Surse
Ordinea operațiilor: `gpu_compute::train::loss_and_grad` (scris de noi după `integrare/kronos/PAS_DE_AUR.md` și PyTorch 2.14 —
licență BSD — ca specificație). Niciun cod străin.
