# C-157 p.2 — `token32_v2c` pe placă (os-495, ciclul 27 nov 18:06) comparat cu predicțiile C1–C8

Sesiunea Cyber-GPU 276, 27 nov 2026, în cloud, fără placă. Sursa (MĂSURAT de OS): hub
`rezultate/os-automat/2026-11-27/ciclu-auto-20261127-1806.txt` (rândurile 83–140; rândul de test
`test-placa --i9 c135 --tablou-viteza --token32 --token32-v2 --token32-v2c`). Predicțiile erau scrise înainte, în
`docs/C-157_TOKEN32_CEAS.md` §3. Socoteala: T291 (`crates/nvidia-ga104/tests/calc_c157_token32_v2c_placa.rs`, CALCUL pe procesor).
Viteza e valabilă: cererea de viteză maximă acceptată (600 s), lanțul a mai rulat 51 s după ea, 0 seturi în afara ferestrei, P0 la
final. Cifrele `kc` sunt ale **ultimului simbol** (placa 1 741,6 µs; mediana pe 18 simboluri: 1 676 µs).

## 1. Ce s-a prezis și ce a ieșit

| # | Prezis (scris înainte) | Ieșit (MĂSURAT) | Verdict |
|---|---|---|---|
| C1 | aceiași biți ca `token32_v2` | 16 simboluri identice, amprenta `0x6c8501f0`, contorul 18 / 18 | CONFIRMATĂ |
| C2 | ceasul costă ≤ +3 % | 1 676 față de 1 666 µs = **+0,6 %** | CONFIRMATĂ |
| C3 | ceasul SM-ului 1,4–2,0 GHz | **1 951 MHz** | CONFIRMATĂ |
| C4 | bariera „curată” 0,14–0,40 ms | **0,172 ms** = 2,0 µs pe barieră (proba barierei fără muncă: 3,11) | CONFIRMATĂ |
| C5 | înmulțirile ≥ 0,40 ms peste lista fără porniri (B1) | 0,696 ms = +0,18 ms; **egale cu lista din același lanț** (cu porniri, ×0,99) | NECONFIRMATĂ |
| C6 | cele 25 de `layernorm256` 0,05–0,20 ms | **0,459 ms** (18,6 µs fiecare, lista ~4 µs) | NECONFIRMATĂ — de ×2,3 peste capăt |
| C7 | ciocănitul ≥ 20 de citiri pe grup și barieră | **26,5** (cel mai harnic grup: 2 433 pe simbol) | CONFIRMATĂ |
| C8 | atenția ≤ ×1,3 față de lista fără porniri (~0,20 ms) | fereastra **0,424 ms = ×2,12** | NECONFIRMATĂ |

Repetabilitatea (același program, alt ciclu): `token32_v2` la G = 768 **1 666 µs** față de 1 668 la 10:01; la G = 384 1 936 față
de 1 969. Lista obișnuită 1 067 µs, ca la 10:01.

## 2. Unde stă „restul” (CALCUL, T291)

Ultimul simbol față de lista obișnuită din același lanț (pe fel, cu pornirile lansărilor incluse în listă):

| Fel | `token32_v2c` — munca celui mai lent grup | Lista | Diferența |
|---|---|---|---|
| LayerNorm (ln1 + ln2 + ln final, 25) | 459 µs | 98 µs | **+361 µs** |
| Atenția (12) | 403 µs | 250 µs | **+153 µs** |
| Bariera (86, după cel mai lent grup) | 172 µs | — | **+172 µs** |
| Înmulțirile (qkv, out, fc, proj, logit) | 696 µs | 702 µs | −6 µs |
| Restul (embed, copia logit-urilor) | 11 µs | 12 µs | −1 µs |
| **Total** | 1 742 µs | 1 067 µs | **+680 µs** (5 µs = rotunjirea medianelor listei) |

Ce înseamnă (DEDUS din cifre și din cod):
- **B2 e cauza principală, nu B1.** `layernorm256` rulează pe **un singur grup** de 32 de fire care joacă pe rând cele 8 warp-uri ale
  programului separat (de 8 ori mai multe citiri în șir, fiecare cu drumul ei); media pe grupuri (4,06 µs) = munca celui singur / 768
  + ~0,3 µs bucla goală a celorlalte 767 — deci chiar unul singur lucrează.
- **Atenția e și ea „pe puțini”:** 12 blocuri (câte unul pe cap) din 768 de grupuri; din medie ies ~11–12 grupuri care lucrează.
  Lista folosește aceleași 12 blocuri, dar le termină în ~21 µs pe lansare cu pornire cu tot; aici 33,6 µs. Diferența (~×1,6) e
  probabil ciocănitul celorlalți 756 de grupuri pe L2 (C7: 26 de citiri atomice pe grup și barieră) — nedovedit, aceeași bănuială
  ar explica o parte din LayerNorm.
- **B1 nu frânează înmulțirile:** cu tot ciocănitul, munca lor e cât în listă; dezechilibrul între grupuri (B3) costă ~84 µs, cam cât
  câștigă lipsa pornirilor.
- **Bariera** costă 2,0 µs × 86 — mai puțin decât în proba fără muncă (3,11), pentru că grupurile ajung la ea pe rând.

## 3. Ce tai, ce las

- **Tai** bănuiala B1 pentru înmulțiri (nu mai caut acolo). `token32_v2c` rămâne doar program de măsurat (nu intră la generare).
- **Las** G = 768 și `token32_v2` ca bază.
- **Prag:** v2 egalează lista cu driverul la placa ~1,47 ms (T285); fără „rest” (LayerNorm + atenție la costul din listă) ar fi
  ~1,10–1,30 ms → **~520–580 simb/s** cu driverul, față de 484 azi.

## 4. Următorul pas (pentru o sesiune viitoare; planul complet + predicțiile se scriu ÎNAINTE de cod)

1. **LayerNorm fără grupul singur** (cel mai mare câștig, ~0,36 ms): fiecare grup al înmulțirii care urmează (ln1 → qkv, ln2 → fc,
   ln final → logit) își calculează singur media și abaterea pe cei 768 de x, în exact aceeași ordine a adunărilor ca
   `layernorm256` (aceiași biți), și normalizează doar ce citește → 25 de operații și 25 de bariere mai puțin. Bandă DEDUSĂ (T291 D1):
   placa **1,28–1,38 ms** pe simbol. Riscul: fiecare din cele 768 de grupuri recitește x (3 KiB) — din L1, după `cctl`.
2. **Atenția pe mai multe grupuri** (~0,15 ms): mai multe grupuri pe cap (felii de poziții, apoi o adunare a feliilor) — cere o
   barieră în plus sau o ordine fixă a adunărilor ca biții să rămână cei ai listei; de studiat. Bandă DEDUSĂ cu pasul 1 (T291 D2):
   **1,10–1,30 ms** → ~520–580 simb/s cu driverul.
3. Abia după acestea: bariera care ciocăne mai rar (C7 a ieșit, dar pe înmulțiri nu se vede efectul).

**Nou pentru placă:** nimic. Rândul `--token32-v2c` nu mai trebuie rulat (răspunsul l-am primit); `--token32-v2` la G = 768 rămâne
proba de bază.
