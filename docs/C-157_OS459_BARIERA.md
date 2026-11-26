# C-157 după os-459: proba barierei pe toată placa — comparația cu B1–B5 și predicțiile pentru proba largă (sesiunea 249, 26 nov 2026, cloud, FĂRĂ placă)

**Stare:** comparație cu un rezultat MĂSURAT de OS pe RTX 3070 Ti (os-459, 26 nov 11:01) + predicții W1–W6 pentru pasul următor,
scrise **înainte** ca OS-ul să-l ruleze. Planul inițial: `docs/C-157_PROBA_BARIERA.md` (B1–B5, sesiunea 237).

## 1. Ce a măsurat OS-ul (raportul `rezultate/os-automat/2026-11-26/ciclu-auto-20261126-1101.txt` din hub)

| G | R = 1 | R = 1 001 | contorul | o barieră (DEDUS din MĂSURAT, `grid_barrier::barrier_ns`) |
|---|---|---|---|---|
| 48 (1 pe SM) | 3 968 ns, TRECE | 1 130 624 ns, TRECE | 48 048 | **1 126 ns** |
| 96 (2 pe SM) | 3 456 ns, TRECE | 1 146 592 ns, TRECE | 96 096 | **1 143 ns** |

48 SM-uri active citite corect (24 TPC × 2). Niciun cuvânt vechi, nicio blocare, programul `grid_barrier` cu suma `535f3537…`.

## 2. Comparația cu predicțiile B1–B5

| | Predicție (sesiunea 237) | Placa | Verdict |
|---|---|---|---|
| B1 | G = 48, R = 1 001 termină, contorul 48 048, 0 runde greșite | exact așa | **CONFIRMATĂ** |
| B2 | G = 96 termină la fel | contorul 96 096, 0 greșite | **CONFIRMATĂ** |
| B3 | o barieră la G = 48: 1–3 µs | 1,126 µs | **CONFIRMATĂ** (aproape de capătul de jos) |
| B4 | la G = 96 mai scumpă decât la 48, dar sub ×2 | 1 143 / 1 126 = ×1,015 | **CONFIRMATĂ** (abia: aproape la fel) |
| B5 | dacă bariera ≤ 2 µs, câștigul la generare cere bariera sub ~1,5 µs sau operații lipite | 1,13 µs < 1,5 µs | condiția îndeplinită (vezi §3) |

**Ce înseamnă diferența mică 48 → 96:** bariera e limitată de **drumul dus-întors** până la L2 (adunarea atomică + citirile
contorului), nu de câte grupuri trag de același contor. Dublarea grupurilor a adăugat doar ~17 ns. DEDUS: un drum până la L2 și
înapoi pe GA104 e de ordinul a câteva sute de ns; o barieră are ~2–3 astfel de drumuri (adunarea, citirea care vede ultimul sosit,
`membar.sc.gpu`).

**Ce NU spune:** nimic despre G = 384 / 768. Acolo 384–768 de grupuri citesc **în buclă** același cuvânt cu adunări atomice; L2 le
servește pe rând pe aceeași adresă, deci la un moment dat costul crește cu G (§4).

## 3. Aritmetica pentru generare (DEDUS din MĂSURAT)

- Lista de azi (`lipita-ln-32`, os-459): **88 de lansări pe simbol**. Costul fix al unei lansări (cea mai mică măsurată, os-346):
  **3,87 µs** → ~0,34 ms pe simbol doar în pornirile lansărilor.
- Într-un singur program pe simbol (`token32`), 88 de operații = **87 de bariere**. La 1,13 µs: **~0,10 ms** pe simbol.
  Câștigul de plafon: ~0,24 ms pe simbol.
- **Dar** la G = 48 / 96 fiecare SM are 1–2 warp-uri; înmulțirile (cea mai mare parte a generării) au nevoie de multe citiri în zbor
  (lansările separate țin până la 16 blocuri pe SM). De aceea pasul următor e proba la G mare, nu `token32` la G = 48.
- Pragul: `token32` câștigă doar dacă bariera la G-ul ales rămâne sub **~3,9 µs** (costul unei lansări) — mai sus, cele 87 de
  bariere costă cât sau mai mult decât cele 88 de porniri de azi.

## 4. Predicțiile pentru proba largă (G = 384 și 768), scrise înainte de rulare

Aceeași probă, același program (nimic nou pentru placă în afară de numărul de grupuri): G din
`grid_barrier::groups_for_wide_sms(48, 8)` = 384 și `(48, 16)` = 768. **Nu** `groups_for_wide(&gr, …)`: aceea refuză mereu
(os-456 — `GrShape` dă maximul de proiect, 84). Al doilea G doar dacă primul trece.

| | Predicție | Încredere |
|---|---|---|
| W1 | G = 384 (8 pe SM), R = 1 și 1 001: TRECE, contorul 384 384, 0 runde greșite | ~85 % |
| W2 | G = 768 (16 pe SM): TRECE la fel, contorul 768 768 | ~75 % (riscul: placa ține mai puțin de 16 blocuri de 32 de fire pe SM → BLOCATĂ la limita de 100 ms, fără altă stricăciune) |
| W3 | o barieră la G = 384: **1,2–2,5 µs** | ~60 % |
| W4 | o barieră la G = 768: **1,5–4,0 µs**, mai mare decât la 384 | ~60 % |
| W5 | R = 1 (lansarea + o barieră) la 384 / 768: sub 15 µs | ~70 % |
| W6 | aritmetică: dacă W4 iese ≤ 2 µs, `token32` la G = 768 are ~0,17 ms în bariere pe simbol față de ~0,34 ms porniri azi → merită pasul (c); dacă iese ≥ 3,9 µs, `token32` nu câștigă și nu-l cerem; între ele: decide proba `token32` | regulă |

## 5. Ce urmează

1. PREDARE către OS: `integration/cyber-universe/C-157_P2zr_bariera-larga.md` (opțiunea nouă, rândurile tipărite, limita de timp).
2. Când vine rezultatul: comparația cu W1–W6 aici, apoi (dacă W6 o permite) `token32` cu G ales, sub o opțiune oprită implicit.

Testul care fixează cifrele de mai sus: T259 (`crates/nvidia-ga104/tests/sim_c157_grid_barrier.rs`, pe procesor; cifrele plăcii
sunt MĂSURATE de OS, restul DEDUS; proba largă în SIMULARE).
