# C-157: `token32` pe placă la G = 384 și G = 768 — predicțiile (scrise ÎNAINTE de orice rulare pe placă)

Sesiunea Cyber-GPU 256, 26 nov 2026, în cloud, **fără placă**. Nimic din ce e mai jos n-a rulat pe placă. Programul `token32`, QMD-ul
lărgit și împărțirea pe grupuri există deja (sesiunile 239–240, `C-157_TOKEN32.md`, `C-157_TOKEN32_G_MARE.md`, T248 / T249 în
SIMULARE). Ce lipsea: ce așteptăm de la placă, în cifre, înainte să cerem testul. Calculul: T266
(`crates/nvidia-ga104/tests/calc_c157_token32_g384.rs`, doar pe procesor).

## 1. De la ce pornesc (MĂSURAT pe placă de OS)

- Un simbol al listei de azi (`lipita-ln-32`, 88 de lansări): **~1 066 µs** pe placă (os-468, raportul de la 20:03). Pe feluri:
  atenția 250, fc 224, qkv 175, proj 143, logit 87, out 73, ln1 48, ln2 46, restul 16 µs.
- Pornirea unei lansări: **3,87 µs** (os-459) → 88 × 3,87 = **341 µs pe simbol doar porniri** (o treime).
- O barieră pe toată placa: **2,03 µs la G = 384**, **3,11 µs la G = 768** (os-465). `token32` are 87 de bariere pe simbol.
- Cele trei LayerNorm sunt practic doar pornire (48 / 12 = 4,0 µs pe lansare, pornirea 3,87).
- Înmulțirile matrice × vector (`q8_matvec32` / `q8_fused32`) lansează azi un warp pe rând, grila = rândurile → **16 warp-uri pe SM**
  (limita de blocuri). Munca lor + a atenției, fără porniri: **716 µs** pe simbol.

## 2. Întrebarea

`token32` taie pornirile (341 → 4 µs) dar adaugă barierele (176 µs la G = 384, 271 la G = 768) și, la G = 384, lasă **doar 8 warp-uri
pe SM** în loc de 16. Dacă înmulțirile sunt limitate de câte citiri sunt în zbor (azi ~100 B pe warp, puțin), jumătate din warp-uri
înseamnă până la dublul timpului lor. Notez cu **m** cât de mult se lungește munca de 716 µs.

## 3. Predicțiile

| # | Ce | Predicție | Încredere |
|---|---|---|---|
| T1 | aceleași 16 simboluri ca lista de azi, aceeași amprentă a logit-urilor `0x6c8501f0`, la G = 384 și la G = 768 | DA, bit cu bit (aceleași contopiri în listare, T248) | mare (DEDUS din listare) |
| T2 | programul termină (bariera se deschide) la ambele G; contorul barierei = (n − 1) · G pe simbol (n = operațiile listei, ~88) | DA (G = 384 / 768 au trecut în proba barierei, os-465; treptele 32 / 64 KiB țin 10 / 16 blocuri `token32` pe SM) | medie |
| T3 | placa pe simbol la **G = 384** | **0,90–1,62 ms**; cel mai probabil **~1,26 ms (mai LENT decât azi)**; câștigă doar dacă m ≤ 1,22 | mică |
| T4 | placa pe simbol la **G = 768** | **0,92–1,10 ms**; cel mai probabil **~1,0 ms (−6 %)** — aceleași warp-uri pe SM ca azi, barierele mai scumpe | mică |
| T5 | ordinea | G = 768 ≤ G = 384 (cu ≥ 0,1 ms) | medie |
| T6 | timpul driverului pe simbol (azi 941 µs: pregătirea și trimiterea a 88 de lansări) | scade mult cu o singură lansare; fără cifră — nu știu ce parte din cele 941 µs ține de numărul lansărilor | mică, de măsurat |

**Ce schimbă predicțiile față de sesiunea 254:** acolo am ales G = 384 (plafon ~0,16 ms pe simbol) socotind că munca rămâne aceeași.
Cu înmulțirile pe un warp pe rând, munca depinde de warp-urile pe SM, iar G = 768 e singurul care le păstrează. De aceea testul
cere **ambele G**, G = 384 întâi (bariera mai ieftină), G = 768 după (doar dacă G = 384 a trecut T1–T2).

## 4. Cum se judecă (pentru OS)

- Trece = T1 și T2 la G-ul respectiv. Cifra de comparat = timpul pe simbol al plăcii și cel cu driverul, lângă lista de azi
  rulată în același lanț (aceeași încălzire a plăcii — vezi `C-157_OS468.md` §2: deriva de 0,5 % între rulări).
- Dacă nu termină: limita de timp a OS-ului oprește lanțul; raportul spune la ce G.
- Nimic nou pentru placă în afară de programul `token32` (o singură lansare pe simbol); niciun registru nou.

## 5. Ce urmează

PREDARE către OS: `integration/cyber-universe/C-157_P2zz_token32-pe-placa.md` (rețeta, opțiunea oprită implicit). Când vine
rezultatul: comparația cu T1–T6 aici.
