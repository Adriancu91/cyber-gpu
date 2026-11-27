# C-157 p.2 — `token32_v2` pe placă (ciclul 27 nov 10:01) comparat cu predicțiile V1–V5

Sesiunea Cyber-GPU 271, 27 nov 2026, în cloud, fără placă. Sursa (MĂSURAT de OS): hub
`rezultate/2026-11-27-workstation-token32-v2/REZULTAT.md` și `rezultate/os-automat/2026-11-27/ciclu-auto-20261127-1001.txt`
(rândurile 54, 66–97). Predicțiile erau scrise înainte, în `docs/C-157_OS471.md` §3. Socoteala: T285
(`crates/nvidia-ga104/tests/calc_c157_token32_v2_placa.rs`, CALCUL pe procesor). Viteza e valabilă: cererea de viteză maximă
acceptată la pornirea lanțului (600 s), P0 la final, lanțul întreg 126 s.

## 1. Ce s-a prezis și ce a ieșit

| # | Prezis (scris înainte) | Ieșit (MĂSURAT) | Verdict |
|---|---|---|---|
| V1 | aceiași biți: 16 simboluri, amprenta `0x6c8501f0`, contorul (n − 1) · G, la 384 și 768 | DA la amândouă (18 / 18) | CONFIRMATĂ — după `cctl` placa nu ține linii vechi în L1 |
| V3 | G = 384: 0,9–1,75 ms (probabil ~1,2) | **1,969 ms** | NECONFIRMATĂ — cu ~12 % peste capătul lent |
| V4 | G = 768: 0,95–2,1 ms | **1,668 ms** | CONFIRMATĂ |
| V5 | partea driverului ~0,9–1,2 ms (în loc de 4,6) | **~0,61 ms** (609 / 613 µs) | NECONFIRMATĂ — mai bine decât am prezis |

Față de `token32` din același lanț: placa ×1,96 (384) și ×2,46 (768) mai repede; cu driverul ×3,3 / ×3,8. Față de lista obișnuită
(1 067 µs placa, 2 083 µs cu driverul): placa încă ×1,56 mai lentă la 768, dar **cu driverul doar ×1,10** (2 281 µs = 438 simb/s față
de 480), pentru că partea driverului e acum **sub** a listei (0,61 față de 1,02 ms).

## 2. Ce spun cifrele (DEDUS, T285)

- **Cauza 1 (x citit din L2) e confirmată la G = 768.** Doar ea ar fi dat ~1,57 ms; placa 1,668 (+6 %).
- **La G = 384 e altă frână:** cauza 1 ar fi dat ~1,31 ms, placa 1,97 (+0,65 ms). Cu 8 warp-uri pe SM, așteptările citirilor nu
  mai sunt acoperite. De aceea ordinea s-a întors: la v1 768 era mai lent, la v2 768 e mai rapid. **G = 768 rămâne singurul de folosit.**
- **„Restul” din os-471 a rămas:** la 768 multiplicatorul muncii m = 1,94 (lista = 1); ~0,8 era rest și la v1, acum ~0,94 →
  **~0,67 ms din 1,668 nu sunt explicate** (L1 nu le-a atins). Candidați (neverificați): împărțirea fixă a rândurilor pe grupuri
  (lansările separate au mult mai multe warp-uri de distribuit), plafonul de 16 warp-uri pe SM (blocuri de 32 de fire, cel mult 16
  blocuri pe SM pe sm_86 — 768 = 48 × 16 e deja maximul), citirile atenției.
- **Cuvintele scrise:** „348 pe simbol” e media pe 18 simboluri cu primul scris întreg (5 632). Celelalte 17 au scris **37–38 de
  cuvinte** fiecare = exact T271 (38). Cost ~24 µs pe simbol.
- **De ce V5 a ieșit mai bine:** am scos din predicție doar rescrierea listei; dar o lansare în loc de 88 face și pushbuffer-ul
  și înregistrările mult mai mici.

## 3. Ce tai, ce las, ce urmează

- **Tai:** `token32` (v1) — v2 e mai bun peste tot; rândurile de placă pot rula doar `--token32-v2`. **Tai G = 384** pentru v2.
- **Prag:** v2 la 768 egalează lista (cu driverul) când placa coboară la ~1,47 ms (m ≤ 1,67); dacă „restul” dispare (m = 1):
  ~1,61 ms ≈ **620 simb/s** față de 480 azi (×1,29). DEDUS.
- **Următorul pas (pentru o sesiune viitoare, nimic predat azi):** să aflăm unde stă „restul” — ceasul pe operație în interiorul
  lansării unice (grupul 0 notează ceasul plăcii după fiecare barieră) sau blocuri de 64 de fire (mai multe warp-uri pe SM). Ambele
  cer un program nou și predicții scrise înainte.
- **Nou pentru placă:** nimic. Nota pentru OS: `integration/cyber-universe/C-157_P2zzn_token32-v2-placa.md`.
