# C-165 / C-157 p.2 — os-423 (ocuparea) și os-431 (treapta pe tot pasul): modelul „valurilor” și alegerea la 64 KiB

Sesiunea 232 (25 nov 2026, ~16:20; în cloud, FĂRĂ placă). Toți timpii de mai jos sunt ai plăcii reale, măsurați de OS
(`rezultate/os-automat/2026-11-25/ciclu-auto-20261125-1119.txt` rândurile 214–232 și `…-1419.txt` rândurile 160–255,
copiate neschimbat în `crates/nvidia-ga104/tests/rapoarte/os423-ocupare.txt` și `os431-pas.txt`). Nimic de aici nu rulează
pe placă. Cinste: rezultatele os-423 și os-431 le-am văzut înainte să scriu modelul — deci potrivirea modelului pe ele
(§2) e o **explicație**, nu o predicție. Singura predicție adevărată e §3 (cursa / pasul la 64 KiB, încă nerulate).

## 1. os-423 față de predicțiile din `docs/C-165_OCUPAREA.md` §3

| | predicția | pe placă | |
|---|---|---|---|
| P1 | aceiași biți în toate 10 | 10 / 10 | ✓ |
| P2 | H1: `mm_train_tile` 64 / 8 KiB ≤ 0,85 | **0,366** (×2,73) | ✓ H1 |
| P3 | monoton 8 → 16 → 32; 32 → 64 câștig mic | 2 840 → 1 569 → 1 097 → 1 041 µs | ✓ |
| P4 | `64x64k8` câștigă la 8 → 16 (R0) | 1 769 → 1 114 µs | ✓ |
| P5 | `128x128` câștigă la 8 → 16 sub R0; sub R1 abia la 32 | 1 890 → **1 278** → 1 276 | ✓ **R0** |
| P6 | puterea crește cu ocuparea | ~79 W la toate (media paginii, nefolosibilă) | NECUNOSCUT |

**Nou din P5:** la 16 KiB, sub R1 (8 KiB + 1 KiB rezervă pe bloc) ar încăpea un singur bloc de `128x128`, deci niciun
câștig; placa a câștigat ×1,48 → **rezerva de 1 KiB pe bloc nu se aplică lansărilor noastre (R0)**, ca la Mesa NVK.
(DEDUS din timpi, nu dintr-un registru citit.) Pentru pasul #513 nu schimbă nimic: atenția cere oricum 64 KiB.

## 2. os-431 — de ce ×1,32 pe înmulțiri, nu ×1,48 (explicație)

Pasul #513 cu `--mm-alegere --treapta-pas`: 266,8 ms (os-418: 338,4); `mm_tv_128x128` 223,2 ms (os-418: 294,9) = ×1,32;
restul plăcii 32,7 ms (neschimbat). Pe etichete (cele 7 prezente în ambele rapoarte):

| eticheta | forma (M×N×K) | blocuri 128×128 | os-418 → os-431 (µs pe lansare) | raport |
|---|---|---|---|---|
| inainte:fc | 1024×3072×768 | 192 | 2 376 → 1 235 | ×1,92 |
| inapoi:proj-dX | 1024×3072×768 | 192 | 2 292 → 1 206 | ×1,90 |
| inapoi:proj-dW | 768×3072×1024 | 144 | 2 285 → 1 569 | ×1,46 |
| inapoi:fc-dW | 3072×768×1024 | 144 | 2 271 → 1 545 | ×1,47 |
| inapoi:qkv-dW | 2304×768×1024 | 108 | 2 270 → 1 523 | ×1,49 |
| inainte:proj | 1024×768×3072 | **48** | 2 277 → 2 281 | **×1,00** |
| inapoi:fc-dX | 1024×768×3072 | **48** | 2 277 → 2 278 | **×1,00** |

**Modelul valurilor:** placa are 48 de unități (SM). Cu `b` blocuri pe unitate, o lansare de `g` blocuri merge în
⌈g / 48b⌉ „valuri”; un val durează cam cât un singur bloc (la `128x128` val cu 1 bloc pe unitate = 630 µs, cu 2 = 638 µs
la K = 768, os-423 — adică unitatea e pe jumătate goală cu un singur bloc). La 8 KiB `128x128` are 1 bloc pe unitate, la
64 KiB are 2 (registrele). Deci: 192 de blocuri → 4 valuri → 2 (×2); 144 → 3 → 2 (×1,5); 108 → 3 → 2 (×1,5);
**48 → 1 → 1 (×1)**. Exact rapoartele de mai sus. Formele cu N = 768 (ieșirea 1024 × 768) au doar 48 de blocuri de
128 × 128 — câte unul pe unitate — și nu se pot grăbi cu treapta: **ele sunt frâna rămasă** (6 din 15 forme, ~97 ms din
223 în os-431, DEDUS din model).

## 3. Predicția (scrisă înainte de cod): alegerea pe formă la 64 KiB

Modelul (`nvidia_ga104::wave_model`, doar adăugare): timpul pe val al fiecărei variante, la K = 768, din os-423
(`mm_train_tile` b = 1/2/4/6; `64x64k8` b = 2/4/6, b = 1 dedus cu raportul lui `mm_train_tile`; `128x128` b = 1/2),
interpolat liniar la b = 3, 5; timp ∝ K; val parțial = ⌈rest / 48⌉ blocuri pe unitate; pasul = cursa × 0,936
(raportul pas / cursă din os-418, V3). `mm_tv_128x64`, `mm_tv_64x128` și `mm_tv_128x128t512` nu au timpi la treptele
noi → în afara modelului (cursa le judecă).

| | predicția | dacă iese altfel |
|---|---|---|
| V1 | modelul refăcut pe cursa os-416 (b = 1): fiecare din cele 15 forme `128x128` la ≤ 11 % | modelul nu ține nici pe datele lui |
| V2 | modelul pe pas: os-418 (b = 1) și os-431 (b = 2) — totalul `mm_tv_128x128` la ≤ 4 %; fiecare din cele 7 etichete la ≤ 7 % | valurile nu explică os-431 |
| V3 | **la 64 KiB, cea mai bună pe formă:** `128x128` păstrează formele cu 192 de blocuri și capul (logit, cap-dW); `mm_train_tile` / `64x64k8` iau cele 6 forme cu N = 768 și formele cu 108–144 de blocuri | — |
| V4 | **pasul #513 cu alegerea V3: înmulțirile ~161 ms, pasul ~204 ms ≈ 5 000 simb/s (interval 195–215 ms)**, aceiași biți (toate variantele au trecut „=” în os-416/417) | > 215 ms: valurile parțiale costă mai mult decât credem; < 195: unitățile câștigă și din alt loc |
| V5 | cursa refăcută la 64 KiB (când OS-ul o leagă): pe fiecare formă, câștigătoarea = cea din V3 sau la ≤ 5 % de ea | altă câștigătoare cu > 5 % → modelul greșește pe acea formă; notăm care |

## 4. Rezultat (sesiunea 232, T238, pe procesor)

(completat după cod)

## Surse
- `docs/C-165_OCUPAREA.md` (predicțiile os-423), `docs/C-165_CURSA_MM.md` §„os-418” (raportul pas / cursă 0,936).
- Tabelul public de ocupare pentru capabilitatea 8.6 (ghidul CUDA, doar idei) — deja în `nvidia_ga104::occupancy`.
- „Valurile” (wave quantization) sunt o idee generală din documentația publică NVIDIA despre performanța înmulțirilor
  (GEMM); formula de aici e scrisă de noi.
