# C-157, punctul lui Adrian din 23 nov (b): ce putem ține în RAM în loc de VRAM — studiu scurt

Sesiunea 183, cloud, fără placă. Planul de rezervă cerut de Adrian: calcule și antrenare cu o parte din memorie mutată în
RAM (workstation: 8 GB VRAM, 128 GB RAM). Toate cifrele de mai jos sunt **DEDUSE** (socoteală), nu măsurate; legătura
RAM ↔ VRAM o măsoară proba din `docs/C-157_RAM_VRAM_PROBA.md` (am pus aici **~12 GB/s**, predicția ei — se înlocuiește cu
cifra de pe placă).

## 1. Ce fac alții (surse, citite ca idei)

| Sursă | Ce ține în RAM | Ideea de reținut |
|---|---|---|
| **ZeRO-Offload** (Ren et al., 2021, arXiv 2101.06840; DeepSpeed, MIT) | greutățile „master” F32, mediile Adam (m, v) și **pasul optimizatorului** (pe procesor, „CPU-Adam” vectorizat); pe placă rămân greutățile 16 biți, înainte / înapoi | pe legătură trec doar gradientul (jos) și greutățile noi (sus), **o dată pe pas**, nu pe lot; ~10× model mai mare pe o placă (13 B pe un V100 de 32 GB) |
| **ZeRO-Infinity** (Rajbhandari et al., 2021, arXiv 2104.07857) | și greutățile straturilor, aduse pe placă strat cu strat, chiar și de pe NVMe | aducerea stratului următor **în timp ce** placa lucrează la cel curent (suprapunere) |
| **FlexGen** (Sheng et al., ICML 2023, arXiv 2303.06865) | la generare: greutăți, memoria cheie/valoare, activări, împărțite GPU / RAM / disc după o socoteală | ordinea „zig-zag”: același strat pentru multe loturi, ca o greutate adusă o dată să servească mult calcul |
| **llama.cpp** `--n-gpu-layers` (MIT) | straturile care nu încap rămân în RAM și **se calculează pe procesor** | simplu, dar procesorul e de zeci de ori mai lent: bun la generare, nu la antrenare |

## 2. Ce putem muta, cât costă (pentru #513 și #651)

Pe parametru, antrenarea noastră F32 ține 16 octeți: greutate 4, gradient 4, m 4, v 4 (`tools/buget-antrenare/buget.py`).
#513 = 98 M parametri (1,46 GiB din cele patru), #651 = 183 M (2,73 GiB). Pasul PyTorch: #513 ~1,25 s (0,80 pași/s),
#651 ~3,1 s (0,32 pași/s) — `integrare/kronos/FISA_ANTRENARE.md`.

| Ce mutăm în RAM | Cât eliberăm (#513 / #651) | Trafic pe pas | Cost la ~12 GB/s (#513 / #651) | Verdict |
|---|---|---|---|---|
| **m și v ale AdamW** (pasul AdamW rămâne pe placă, m/v aduse și duse strat cu strat) | 0,73 / 1,36 GiB | 16 B/parametru (8 sus, 8 jos) | ~0,13 s (10 %) / ~0,24 s (8 %); **~0 dacă se suprapune** cu ultimul strat înapoi | **primul pas, merită** |
| m, v + greutăți master + **pasul pe procesor** (ZeRO-Offload) | + gradient F32 doar pe placă | 8 B/param (grad jos, greutate sus) | ~0,07 / ~0,12 s | bun doar dacă procesorul din OS face AdamW repede (mai multe nuclee + SIMD — azi nu știm; de măsurat) |
| **activările de păstrat** (intrările straturilor, „checkpoint”) | la #651, 6 × 512: ~0,23 GB pe lot mic | 2 × 0,23 GB pe lot mic × 16 | ~0,6 s pe pas (19 %) / suprapus ~0 | doar dacă lotul mai mare câștigă mai mult |
| **greutățile straturilor care nu lucrează** (ZeRO-Infinity / FlexGen) | aproape toate greutățile | 4 B/param pe lot mic (16 biți, înainte + înapoi) + gradientul | #651: ~0,12 s pe lot mic față de ~0,2–0,26 s calcul | **doar pentru modele mai mari decât #651**; la #513 / #651 încap pe placă |

Concluzia pentru #513 / #651: **nu avem nevoie de RAM ca să încapă** (la #651 cu P refăcut încape și azi); câștigul ar fi un
**lot mai mare** pe placă (m, v mutate = +1,4 GiB la #651 ≈ +2–4 secvențe de 512, după cât ține o secvență: ~0,65 GiB în F32, ~0,4 GiB în 16 biți fără P). Prima mutare utilă = m, v ale AdamW.

## 3. Ce mărime de Kronos ar încăpea cu 8 GB VRAM + 128 GB RAM

Socoteala: în RAM tot (16 B/parametru, ~110 GB folosibili), pe placă doar stratul curent + următorul (16 biți) + activările
unui lot mic; viteza de calcul = cea efectivă a lui PyTorch pe #651 (~17 TFLOP/s, adică exact 15 500 simb/s la 183 M);
legătura ~12 GB/s; lot mic 6 × 512 = 3 072 simboluri.

| Model (12·d²·L) | Parametri | RAM (16 B/p) | Un strat 16 biți | Legătura / calculul pe lot mic | Simboluri/s (plafon calcul) |
|---|---|---|---|---|---|
| #651 (24 × 768) | 183 M | 2,9 GB | 14 MB | 0,12 / 0,26 s | ~15 500 (măsurat PyTorch) |
| 24 × 1024 | 319 M | 5,1 GB | 25 MB | 0,21 / 0,46 s | ~8 900 |
| 24 × 1536 | 705 M | 11 GB | 57 MB | 0,47 / 1,0 s | ~4 000 |
| 24 × 2048 | 1,24 B | 20 GB | 101 MB | 0,83 / 1,8 s | ~2 300 |
| 32 × 2560 | 2,56 B | 41 GB | 157 MB | 1,7 / 3,7 s | ~1 100 |
| 32 × 4096 | 6,5 B | 104 GB | 403 MB | 4,3 / 9,4 s | ~430 |
| 40 × 5120 | 12,7 B | 203 GB — **nu încape** | | | |

- **Încape:** până la ~6 B parametri (RAM-ul e limita, nu VRAM-ul). Legătura nu e frâna: la lotul mic de 3 072 simboluri
  calculul durează de ~2 ori mai mult decât aducerea greutăților, deci se pot suprapune.
- **Frâna e calculul plăcii:** la 1,2 B ~2 300 simb/s → un miliard de simboluri în ~5 zile; la 6,5 B ~430 simb/s → ~27 de
  zile. Practic: **~0,3–1,3 B** e zona în care RAM-ul ajută și antrenarea rămâne de ordinul zilelor.
- Toate astea cer motorul nostru la viteza lui PyTorch (16 biți, înmulțiri pe circuitele speciale); azi motorul nostru e în
  F32 și mai lent.

## 4. Ce urmează (propunere, nimic pe placă fără comandă)

1. Cifra reală a legăturii (proba `ram_vram`, predată OS-ului) → refac tabelele cu ea.
2. AdamW cu m, v în RAM, strat cu strat, în SIMULARE, bit cu bit egal cu varianta de azi (aceleași operații, doar alt loc).
3. Abia apoi greutățile aduse strat cu strat, pentru un model mai mare decât #651 — decizie a lui Adrian (mărimea lui
   Kronos).

## Surse
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/pdf/2101.06840v1)
- [ZeRO-Infinity: Breaking the GPU Memory Wall](https://arxiv.org/pdf/2104.07857)
- [FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU](https://arxiv.org/abs/2303.06865)
- [llama.cpp: hybrid CPU and GPU inference (discuție)](https://github.com/ggml-org/llama.cpp/discussions/12126)
- interne: `integrare/kronos/FISA_ANTRENARE.md` (hub), `tools/buget-antrenare/buget.py`, `docs/C-157_RAM_VRAM_PROBA.md`.
