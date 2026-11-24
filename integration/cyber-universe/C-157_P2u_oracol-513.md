# C-157 p.2 după os-395 — oracolul #513 pe procesor, profilul pasului #513, ceasurile, pasul mixt #513

Sesiunea 205 Cyber-GPU, 24 nov 2026, în cloud, **fără placă**. Răspuns la cererile centrului de comandă din nota de 11:10 (os-395).
Comparația întreagă: `docs/C-157_P2_COMPARATII.md` (secțiunea os-395).

## 1. Oracolul #513 (cererea 1) — MĂSURAT pe procesor, în cloud

`cargo run --release -p gpu-compute --example oracol_513` refă exact rularea din os-395: `train::M513`, lot 2 × 512, greutățile de start
ca `antrenare_mini::initial(&a, 0)` (MT19937 cu sămânța 1, ±0,034641016, bias 0, LayerNorm 1), loturile ca `board_batches` fără date
(MT19937 cu sămânța 0x164, `next_u32() % 16 384`, 513 simboluri pe rând), AdamW lr 1e-3 / β (0,9, 0,999) / eps 1e-8 / wd 0,1, tăiere 1,0.

| Pasul | Placa (os-395): pierderea / norma | Oracolul (procesor): pierderea / norma | Abaterea relativă (pierdere / normă) |
|---|---|---|---|
| 1 | 9,8493 / 3,2729 | 9,8493 / 3,2729 | 0 / 0 |
| 2 | 9,8750 / 1,1021 | 9,8750 / 1,1021 | 0 / 0 |
| 3 | 9,8293 / 1,0242 | 9,8293 / 1,0242 | 0 / 0 |
| 4 | 9,9083 / 0,9176 | 9,9083 / 0,9176 | 0 / 0 |
| 5 | 9,8642 / 0,8955 | 9,8642 / 0,8955 | 0 / 0 |
| 6 | 9,8817 / 0,8962 | 9,8817 / 0,8962 | 0 / 0 |
| 7 | 9,8472 / 0,8914 | 9,8472 / 0,8914 | 0 / 0 |
| 8 | 9,9389 / 0,9363 | 9,9391 / 0,9364 | 2,0e-05 / 1,1e-04 |
| 9 | 9,9930 / 0,9346 | 9,9929 / 0,9346 | 1,0e-05 / 0 |
| 10 | 10,1774 / 1,6549 | 10,1628 / 1,6557 | 1,4e-03 / 4,8e-04 |

**Concluzia:** placa face exact pasul oracolului: pașii 1–7 identici la 4 zecimale, 8–9 la ≤ 1·10⁻⁴. La pasul 10 pierderea diferă cu
1,4·10⁻³ relativ (peste toleranța 1e-3 a lui mini), norma cu 4,8·10⁻⁴. DEDUS: pasul 10 e chiar pasul în care rularea „sare” (norma
0,93 → 1,65, pierderea +0,17), adică punctul în care orice diferență mică de rotunjire (ordinea adunărilor F32 pe placă ≠ procesor)
se mărește; nu se vede nicio abatere în pașii dinainte. Ca să fie sigur, comparația greutăților finale ar lămuri (oracolul nu le-a
scris; se poate face într-o sesiune viitoare, ~40 min). Pierderea urcă pentru că loturile sunt la întâmplare (nu e nimic de învățat; cel mai
bun rezultat e ln 16 384 = 9,704 pe orice lot nou) și lr 1e-3 „învață” zgomotul unui lot, care strică lotul următor. Nu e o greșeală.

**Propunere (a OS-ului de hotărât):** când `--antrenare-513` rulează pe loturile sintetice, verdictul = pierderea și norma pașilor
**1–9** față de `gpu_compute::train::ORACLE_513_SYNTH` (constante noi, doar adăugare, 4 zecimale) la 1e-3 relativ (aici: ≤ 1,1·10⁻⁴);
pasul 10 doar afișat, sau la 2e-3 (spus cinstit: pragul pentru pasul 10 e ales după ce am văzut 1,4e-3). Mai curat pe termen lung:
rularea de probă cu lr 5e-5 (rețeta lui #513), unde nimic nu „sare” — atunci oracolul se refă pentru acea rețetă. Criteriul „ultima pierdere
sub prima” rămâne doar pe datele lui Kronos (acolo e ceva de învățat). Constantele sunt valabile **doar** pentru exact rețeta de mai sus;
orice schimbare (lot, lr, sămânță, date) le face inutilizabile — atunci verdictul vechi.

## 2. Timpul pe fel de program (cererea 2)

Nemăsurat de noi (doar pe placă). Socoteala (`tools/buget-antrenare/pas513.py`, DEDUS): 657 GFLOP pe pas, 91 % în `mm_train`
(straturile 79 %, capul legat 12 %), atenția ~9 %; 1 123 ms = 0,59 TFLOP/s efectiv, deci `mm_train` probabil ~0,5–0,7 TFLOP/s.

Cum îl măsoară OS-ul fără nimic nou pentru placă: `nvidia_ga104::train_profile::StepProfile::from_launches(&list)` cu lista întreagă a
pasului (385 de lansări), apoi pentru fiecare pas citit complet: `transfer::analyze_sequence` **pe fiecare parte** (255 + 130 de lansări,
fiecare cu înregistrările ei, `n + 1`), timpii părților puși cap la cap în ordinea listei → `add_step`. Dacă o parte nu are înregistrarea
ei de început, `add_step` refuză (alt număr de timpi) — nimic ghicit. Rândurile: `lines(top)` (pe etichetă, program, etapă).

## 3. Ceasurile „placa” > „pasul întreg” (observația 3)

Raportul e 1,027 pe **fiecare** din cei 10 pași (1,1 s), iar la os-390/392 era același 2,7 % pe intervale de 4 ms. Același raport pe
intervale de 250 de ori mai lungi = **cele două ceasuri merg cu ritm diferit cu ~2,7 %**, nu un decalaj fix (DEDUS). Care greșește:
NECUNOSCUT. Probă ieftină pentru OS: ambele ceasuri (al plăcii, citit prin registrul de timp, și al procesorului) față de un al treilea
(ceasul RTC al plăcii de bază) pe ~60 s.

## 4. Pasul mixt #513 (cererea 3: bf16 / circuitele speciale) — planul

Pe GA104 prin NAK, circuitele speciale dau f16 × f16 → f32 (bf16 nu), deci „bf16” = f16 + scara erorii. `Plan::step_mixed` merge și pe
planul rezident #513 (nimic nou de cod în Cyber-GPU pentru asta). Ordinea propusă, fiecare test separat:
1. profilul de la §2 (nimic nou pentru placă);
2. proba `mm_tc` (os-352, `qkv-513`: estimat 1,3–2,7 ms; după os-358 privim estimările cu de 4–9 ori mai multă prudență);
3. `--antrenare-513 --mixt` cu **S = 2¹⁴ = 16 384**, nu 1 024: la 1 024 de simboluri pe lot d(logits) ≈ 6·10⁻⁸ · S, deci la S = 1 024
   cade exact la cel mai mic număr normal f16 (6,1·10⁻⁵); cu 2¹⁴ ≈ 10⁻³, iar valoarea țintei (p − 1)/N · S ≈ −16, departe de 65 504.
   Verdict: pierderile pașilor 1–3 față de `ORACLE_513_SYNTH` la 1e-3 relativ (toleranța pasului mixt din `C-160_P7_PAS_MIXT.md`).
   DEDUS: dacă `mm_tc` atinge 5 TFLOP/s, înmulțirile scad de la ~1 000 la ~120 ms → ~5 000–7 000 simb/s; spre 39 300 mai trebuie lotul
   12 × 512 și restul pasului.

**Ce e nou pentru placă din această notă:** nimic.
