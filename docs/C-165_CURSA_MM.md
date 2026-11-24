# C-165 — cursa de noapte a programelor: variantele lui `mm_train_tile`

**Sesiunea 211, 24 nov 2026, în cloud, FĂRĂ placă.** Planul și predicțiile sunt scrise **înainte** de compilare și de teste.

## Ce cere comanda (DECIZII 24 nov 17:35, H3)
Pentru formele de înmulțire ale pasului #513: mai multe variante ale aceluiași program (mărimea plăcii, fire pe bloc, mărimi
fixe), rulate într-o pornire pe placa reală; raportul scrie timpul median pe variantă și formă și păstrează cea mai rapidă.
Condiții: fiecare variantă = **aceiași biți în SIMULARE** ca un program deja trecut pe placă sau ca oracolul; SHA-256 verificat;
**niciun registru, nicio comandă nouă** (doar alt cod de program + alt QMD, ca la orice lansare de azi); paza de temperatură.
Prima cursă: abia după ce `mm_train_tile` a trecut o dată singur pe placă (os-410, `--mm-placi`).

## Partea Cyber-GPU (ce facem azi)
1. Un singur corp de program, `tools/nak/mm_train_tv.glsl`, cu geometria plăcii ca parametri (BM × BN bucata din C pe bloc,
   BK felia lui K, TM × TN rezultate pe fir; fire pe bloc = (BM/TM)·(BN/TN)); fiecare variantă e un fișier mic
   `tools/nak/mm_tv_<nume>.comp` care fixează cele 5 numere și include corpul. Același bufer constant ca `mm_train`.
2. Primele 4 variante (toate cu 256 de fire = 16 × 16, ca `mm_train_tile`):

| Variantă | Bucata din C | BK | Pe fir | Memorie partajată | Pentru ce formă |
|---|---|---|---|---|---|
| (`mm_train_tile`, martorul) | 64 × 64 | 16 | 4 × 4 | 8 KiB | toate |
| `mm_tv_128x64` | 128 × 64 | 16 | 8 × 4 | 12 KiB | multe rânduri, puține coloane |
| `mm_tv_64x128` | 64 × 128 | 16 | 4 × 8 | 12 KiB | multe coloane (capul, 16 384) |
| `mm_tv_128x128` | 128 × 128 | 8 | 8 × 8 | 8 KiB | formele mari (siboehm kernel 5) |
| `mm_tv_64x64k8` | 64 × 64 | 8 | 4 × 4 | 4 KiB | aceeași bucată, felie mai mică (mai multe blocuri pe SM) |

3. Modelul pe procesor: o singură funcție cu geometria ca parametru (`gpu_sim::cmdproc::mm_train_tv_on_cpu`), care urmează
   plăcile programului (încărcarea pe fire, lanțul de FMA pe fir) — și câte o funcție mică pe variantă pentru registrul de programe.
4. În `nvidia-ga104`: modulul nou `mm_train_variants` (doar adăugare): tabelul variantelor (nume, cod, SHA-256, registre,
   bloc, memorie partajată, grilă) + formele #513 ale pasului, ca OS-ul să aibă lista pentru `test-placa --cursa`.
5. „Mărimi fixe compilate” (K cunoscut la compilare, fără verificări de margine): **nu azi** — un program care primește altă
   formă ar da în tăcere alte numere; vine cu o pază pe procesor (`fits(m, n, k)`) într-un pas separat.

## Predicții (scrise înainte)
1. **Biții (SIMULARE):** fiecare variantă = `mm_train` = `mm_train_tile` = `reference::matmul_t_fma_ftz` **bit cu bit**, pe
   toate cele 8 combinații de opțiuni, pe margini (M, N, K care nu se împart la 128 / 16 / 8), cu −0,0 păstrat (nicio
   adunare de zerouri la ultima felie).
2. **Codul NAK:** câte TM·TN·BK `ffma.ftz` pe felie (desfăcute) + TM·TN `fadd.ftz` (bias), o barieră, **0 memorie locală**.
   Registre: `128x64` și `64x128` ~45–64; `128x128` ~80–110 (64 de sume); `64x64k8` ~34 (ca martorul). Dacă NAK pune
   sume în memoria locală → varianta iese din cursă (QMD-ul nostru nu pregătește memorie locală).
3. **Formele #513** (B 2 × T 512 = 1 024 de rânduri; 12 straturi, lățimea 768, MLP 3 072, vocabular 16 384): toate laturile se
   împart la 128 → nicio placă parțială în cursa #513.
4. **Viteza (DEDUS, NETESTAT):** `128x128` câștigă la capul (1 024 × 16 384 × 768 și gradienții lui) și la MLP; pe formele
   768 × 768 are doar 48 de blocuri = un bloc pe SM (48 de SM) → probabil pierde acolo în fața lui `mm_train_tile`
   (192 de blocuri). De aceea cursa alege **pe formă**, nu o variantă pentru tot. Câștig total estimat pe înmulțiri:
   ×1,2–1,8 față de `mm_train_tile` (siboehm: kernel 5 față de 4 × 4 ≈ ×1,5–2 pe A6000). NECUNOSCUT până la placă.

## Ce urmează după (nu azi)
- OS: rândul `test-placa --cursa <listă>` (partea sesiunii OS, nota din `integration/cyber-universe/`).
- Cyber-GPU: `Plan::with_mm_choice` — pentru fiecare formă, varianta câștigătoare din raportul cursei (aceiași biți → aceeași
  amprentă a pasului); apoi variantele pentru `mm_train_gelu` și atenție.

## Rezultat (sesiunea 211, după compilare și teste)
Mesa NAK 25.0.7 reconstruit în cloud; martorul `mm_train_tile` reconstruit = `f8acce21…` (neschimbat).

| Variantă | SHA-256 | Instrucțiuni / octeți | Registre (prezis) | Partajată | Bucla lui p | Memorie locală |
|---|---|---|---|---|---|---|
| `mm_tv_128x64` | `deb06f297f42fedca493d565010511987f7f413b3261d3b1ffb96857709c7a61` | 1 812 / 28 992 | **54** (45–64) | 12 KiB | păstrată (32 `ffma`) | 0 |
| `mm_tv_64x128` | `e5bc59e3ad9af1109c85cd3f895e4f2b26749b460d5bb6740dd12f71df27f582` | 1 788 / 28 608 | **54** (45–64) | 12 KiB | păstrată (32 `ffma`) | 0 |
| `mm_tv_128x128` | `5d3affd4fd8e9e63edd2719fc204cb7ad197f18d9c0c5c453a73cc0cb9e23ebc` | 2 762 / 44 192 | **90** (80–110) | 8 KiB | păstrată (64 `ffma`) | 0 |
| `mm_tv_64x64k8` | `880884afec4b4beb8f7dedbd7d2145e9c1981dc1c007c3ef98c0b238fbb8f903` | 1 023 / 16 368 | **34** (~34) | 4 KiB | desfăcută (128 `ffma`) | 0 |

**Predicția 2 parțial greșită:** NAK **nu** a desfăcut bucla lui p la cele trei variante mari (o singură trecere de TM·TN `ffma`
în cod, plus salt); la martor și la `64x64k8` a desfăcut-o. Biții nu se schimbă (aceeași ordine a FMA-urilor), doar viteza:
câteva instrucțiuni în plus pe pas al buclei (DEDUS: ~5–15 % pierdere la variantele mari). Dacă cursa le arată aproape de martor,
pasul următor e o variantă cu bucla desfăcută de mână (`BK` fix în cod).

| # | Predicție | Ieșit (T217, SIMULARE) | Stare |
|---|---|---|---|
| 1 | Biții = `mm_train` = martorul = oracolul | 5 programe × 8 combinații × 7 forme cu plăci parțiale (K 1, 9, 17, 33…) bit cu bit; −0,0 păstrat la K = 19 | PASS |
| 2 | 0 memorie locală, registre în intervale, `fadd` = TM·TN | toate; vezi mai sus pentru buclă | PASS (cu nota) |
| 3 | Formele #513 se împart la 128 | **15 forme, 147 lansări, 599,1 GFLOP pe pas**; toate laturile ÷ 128; copii micșorate ale fiecărei forme = oracolul pe toate variantele | PASS |
| — | Refuzuri | opțiune 8, partajată prea mică cu 256 B, fără barieră, blocuri 16 × 16 → refuzate, 0 scrieri | PASS |
| — | Greșeala plantată: grila altei variante | grilă prea mică → C cu goluri, prinsă de comparație; grilă prea mare → aceiași biți | PASS |

**Ce arată formele (DEDUS, pentru cursă):** 9 din 15 forme au N = 768 → `128x128` ar avea doar 36–48 de blocuri pe 48 de SM
(un val, câte un bloc pe SM); acolo martorul sau `128x64` au șanse mai bune. Capul (1 024 × 16 384 × 768 și gradientul lui
16 384 × 768 × 1 024: 51,5 GFLOP din 599) e locul unde `128x128` / `64x128` ar trebui să câștige. Forma 1 024 × 768 × 16 384
(dX al capului, K = 16 384) are puține blocuri la orice variantă → candidat pentru K împărțit (nu azi).

**Viteza pe placă: NOT TESTED.** Cursa o face OS-ul (`test-placa --cursa`), după ce os-410 arată `mm_train_tile` trecut pe placă.
