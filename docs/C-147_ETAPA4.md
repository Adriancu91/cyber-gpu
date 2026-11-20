# C-147 punctul 3 — etapa 4: `q8_matvec32` (toate cele 32 de fire) + cursa pe placă (sesiunea 121, 20 nov 2026, cloud, FĂRĂ placă) — SIMULARE

## Ce s-a făcut
Un program nou pentru placă, `tools/nak/q8_matvec32.comp` → `tools/nak/out/q8_matvec32.sm86.sass` (Mesa NAK 25.0.7, SM 8.6, reconstruit în cloud;
`q8_matvec` și `q8_fused` au ieșit **identice**, aceleași sume, deci unealta e aceeași).

| | `q8_matvec` (azi) | `q8_matvec32` (nou) |
|---|---|---|
| Unitatea de lucru a unui fir | un bloc (32 de coloane) | un sfert de bloc (8 coloane) |
| Fire care lucrează la 768 de intrări | **24 din 32** | **32 din 32** (3 sferturi fiecare) |
| Ce citesc cele 32 de fire într-un pas | 32 de cuvinte la 36 de octeți distanță | 8 blocuri întregi, unul după altul (288 octeți, fără goluri) |
| Citirea lui x | 32 de citiri de 4 octeți pe bloc | 2 citiri de **16 octeți** (`ld.global.constant.b128`) pe sfert |
| Instrucțiuni / registre / octeți | 503 / 19 / 8 048 | **199 / 22 / 3 184** |
| Memorie partajată, `BAR.SYNC` | 128 B, niciuna | 128 B, niciuna (același arbore de adunare) |
| Argumente (bufferul constant) | W 0x28, x 0x30, y 0x38, rânduri 0x40, blocuri 0x44 | **aceleași**; în plus x trebuie aliniat la 16 |

SHA-256 `q8_matvec32.sm86.sass`: `86ae584bd079454150a585968c2ea63ef2b7dadec31054d21a6f59d6afc7c303`.

**Citiri de 128 de biți pentru greutăți: nu.** Blocurile KGR1 au 36 de octeți, deci nu sunt aliniate la 16; o citire de 16 octeți din W ar cere
altă așezare a greutăților (de ex. scalele separat și numerele în 32 de octeți aliniați) — ar însemna rescrierea greutăților la urcare. Nu am făcut-o:
KGR1 rămâne cum îl scrie Kronos. E candidatul pentru pasul următor dacă cursa arată că citirea lui W e frâna.

**Vectorul de intrare în memoria partajată: nu, încă.** Cu un singur grup de 32 de fire pe rând, fiecare element din x e citit o singură dată pe
rând, deci memoria partajată nu scutește nimic. Ar ajuta doar cu mai multe rânduri pe grup (x citit o dată pentru 4–8 rânduri) — o variantă pentru
după cursă, dacă `q8_matvec32` câștigă.

## De ce trebuie să dea aceiași biți ca oracolul
Listarea NAK (`q8_matvec32.sm86.txt`, blocul 5): pentru fiecare sfert, NAK a transformat `q0·x0 + q1·x1` în `FFMA.FTZ(q0, x0, FMUL.FTZ(q1, x1))`,
apoi 6 × `FFMA.FTZ` în ordinea coloanelor, apoi `part = FFMA.FTZ(scale, s, part)`; arborele de adunare e cel al lui `q8_matvec` (`FADD.FTZ`, pas 16, 8,
4, 2, 1). Oracolul `gpu_compute::q8::row_gpu_order_w32` urmează exact aceste operații (prima mea variantă presupunea `FMUL` pe primul termen — am
corectat-o după listare).

Rezultatul **nu** e bit cu bit cel al lui `q8_matvec` (altă ordine a adunărilor), dar e în aceeași marjă a lui I12 față de motorul de pe procesor.
Judecătorul lui `q8_matvec` vede rezultatul nou „în marjă”, deci pe placă se vede care program a rulat.

## Cursa (`nvidia_ga104::q8_race`) — pregătită pentru OS
- **Benzile** = formele lui #513: qkv 2304×768, out 768×768, fc 3072×768, proj 768×3072 (fiecare de 12 ori pe simbol), logits 16 384×768 (o dată).
- Pe fiecare bandă: 4 serii **A B B A** (A = `q8_matvec`, B = `q8_matvec32`), fiecare cu `RACE_LAUNCHES` = 16 lansări și o marcă de timp după fiecare
  (metoda `launch_cost`, ceasul plăcii, fără procesor în buclă); după fiecare serie y e judecat de judecătorul programului.
- **Regula, scrisă înainte de orice rulare:** o bandă trece pe programul nou doar dacă acesta **trece proba** pe placă și mediana lui e mai mică cu
  cel puțin **50 ‰**; altfel rămâne `q8_matvec`. Egalitatea (ca în SIMULARE, unde orice lansare costă la fel) păstrează programul de azi.
- Rânduri `kr …` în jurnal (mediana, minimul, „trece” pe program și bandă; câștigătorul și câștigul în ‰ pe bandă; suma pe simbol azi / cu
  câștigătorii) + cititorul `parse_journal`.
- **Nu intră în cursă:** programele de înmulțire de matrice f32 din C-20 E (`matmul_naive` … `matmul_vec4`). Pentru un singur simbol ar cere
  greutățile despachetate în f32 (de 4 ori mai mulți octeți: ~443 MB pe simbol în loc de ~111) și o „matrice” cu o singură coloană. Locul lor e la
  etapa 6 (mai multe răspunsuri deodată).

## Cod (doar adăugări)
`gpu_compute::q8::{row_gpu_order_w32, matvec_gpu_order_w32, busy_threads, W32_QUARTERS, W32_UNIT}`;
`gpu_sim::cmdproc::{q8_matvec32_on_cpu, Q8_MATVEC32_NEEDS}`;
modulul nou `nvidia_ga104::q8_matvec32` (`Q8_MATVEC32`, `Q8_MATVEC32_SHA256`, `Q8_MATVEC32_REGISTERS`, `Q8_MATVEC32_BLOCK`, `Q8_MATVEC32_SHARED`,
`X_ALIGN`, `cbuf`, `qmd`, `judge`, `reads_per_row`);
modulul nou `nvidia_ga104::q8_race` (`Entrant`, `ENTRANTS`, `Lane`, `K513_LANES`, `Heat`, `heats`, `encode_heat`, `HeatResult`, `Standing`, `Winner`,
`rows`, `journal`, `ROW_NAMES`, `ParsedRace`, `ParsedLane`, `parse_journal`, `MIN_GAIN_PERMILLE`, `RACE_LAUNCHES`).
Nimic schimbat în ce primește placa la `kronos --gpu` azi (lista de 196 și lista lipită de 88 folosesc tot `q8_matvec`).

**Reparat în bancul de test (nu în driver):** `tests/common::dma_pages` (paginile RAM ale OS-ului, în simulare) folosea pasul 7 fix, deci la un număr
de pagini multiplu de 7 (de ex. 651) paginile se repetau și o urcare se suprascria singură. Acum pasul e prim cu numărul de pagini (7, altfel 11 sau 13).
Toate testele vechi trec la fel.

## Teste (SIMULARE) — `tests/sim_c147_w32.rs` + teste unitare, T137
- `q8_matvec32` pe canalul de lucru (urcare → produs → coborâre, ca I12): **identic bit cu bit** cu oracolul la 1×32, 100×96, 37×1056, 3072×768,
  768×3072; în marja I12 față de `q8_matvec` și de motorul procesorului; la formele lui Kronos judecătorul vechi îl vede „în marjă”;
- sub modelul „date vechi” (C-139): identic;
- greșeli plantate: x nealiniat la 16 (refuzat de driver; scris de mână → modelul refuză lansarea, marca lipsește), x cu un sfert mai încolo,
  grila cu un rând lipsă (numit), un număr schimbat după calculul oracolului (rândul numit), modelul rulând ordinea veche (nu „identic”) — niciuna nu trece;
- cursa pe două benzi micșorate: serii A B B A, mărcile citite, y judecat după fiecare serie, rândurile `kr` scrise și citite înapoi; egalitate → rămâne `q8_matvec`;
- regula câștigului (20 % → schimbă; 3 % → rămâne; programul nou picat → rămâne; fără date → rămâne).

## Ce NU e dovedit
Nimic pe RTX 3070 Ti. Câștigul (planul estimează 1,3–3× pe partea de calcul) se vede doar în cursa pe placă. Proba pe placă: `q8_matvec32` identic
bit cu bit cu oracolul lui; în lista de simbol, aceleași 16 simboluri ca procesorul (logit-urile se schimbă în ultimii biți față de lista de azi).
