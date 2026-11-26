# C-157 punctul 2 — planul și predicțiile pentru `mm_tc_v5` (înmulțirile = 79 % din pasul #513)

Sesiunea Cyber-GPU 258, 26 nov 2026, în cloud, **fără placă**. Totul de mai jos e scris **înainte** de cod și de compilare.
Cifrele „MĂSURAT” vin din rapoartele plăcii trimise de OS în hub (os-423, os-443, os-468); tot ce iese din ele e DEDUS.
Calculul e fixat în testul T268 (`crates/nvidia-ga104/tests/calc_c157_mm_tc_v5_plan.rs`, doar pe procesor).

## 1. Unde suntem

- Pasul #513, lot 6 (os-468): **273,1 ms**, din care `mm_tc_v4` **216,6 ms (79,3 %)**, 147 de lansări.
- Cele 147 de înmulțiri ale pasului fac **1,797 TFLOP** → în pas `mm_tc_v4` merge cu **8,30 TFLOP/s**. Proba izolată qkv-513
  (os-443, 2 blocuri pe SM) dădea **8,29 TFLOP/s**: pasul merge exact cu viteza probei. Deci ce câștigă proba, câștigă și pasul.
- Vârful circuitelor speciale (f16 cu sume f32, pe GeForce la jumătate de viteză) la 1 950 MHz: **~47,9 TFLOP/s**. Folosim **17 %**.

## 2. De ce e lent (DEDUS)

Din listarea NAK a lui `mm_tc_v4` (MĂSURAT pe text): bucla pe o felie de K = 32 are ~418 instrucțiuni pe grup de fire, din care
16 `hmma`, 8 citiri de 128 de biți, 32 conversii f32 → f16, 32 scrieri de 2 octeți în memoria comună, 16 citiri din ea + 16 `movm`,
2 bariere. La 8,3 TFLOP/s:
- circuitele speciale sunt ocupate ~17 % din timp, emiterea instrucțiunilor ~14 %;
- memoria plăcii: ~0,1–0,25 TB/s cerut din 0,61 (A și B se refolosesc din L2) — nu e frâna;
- **rămâne așteptarea**: la treapta 64 KiB intră doar **2 blocuri pe SM** (24 KiB fiecare, al treilea nu încape) = 8 grupuri de fire
  pe SM, 2 pe fiecare sfert de SM. Prea puține ca să acopere citirile din memorie și cele două bariere pe felie.

Dovada din placă: când s-a dublat numărul de blocuri pe SM, viteza pe bloc a crescut aproape de două ori:

| Program (măsurat pe placă) | 1 → 2 blocuri pe SM | 2 → 4 blocuri pe SM |
|---|---|---|
| `mm_tc_v4` (os-443) | **×1,81** | — (nu încap 4) |
| `mm_train_tile` (os-423) | ×1,81 | **×1,43** |
| `mm_tv_64x64k8` (os-423) | — | **×1,59** |

`mm_tc_v4` s-a purtat la 1 → 2 exact ca `mm_train_tile`. Pasul următor firesc: **4 blocuri pe SM**.

## 3. Ce schimbăm: `mm_tc_v5` = `mm_tc_v4` cu ieșirea mai mică

Din cei 24 KiB ai unui bloc, **16 KiB sunt `cs_`** — placa 64 × 64 de f32, folosită **doar la final**, ca să se scrie rezultatul.
- `mm_tc_v5`: fiecare grup de fire își scrie cele 4 bucăți 16 × 16 **pe rând** printr-o zonă proprie de 16 × 16 f32 (1 KiB):
  `coopMatStore` în zonă → sincronizare în grup → cele 32 de fire citesc câte 8 valori, adaugă `bias[col]` (aceeași singură
  adunare f32), scriu în memorie → sincronizare → bucata următoare.
- Memoria comună: 4 + 4 + 4 × 1 = **12 KiB** în loc de 24 → **4 blocuri pe SM** la treapta 64 KiB, cu sau fără rezerva de 1 KiB
  (limita devine registrul: 128 × 128 fire).
- **Bucla nu se schimbă deloc** (o singură schimbare, ca să se vadă clar ce aduce ocuparea). Aceleași citiri, aceleași felii,
  același lanț `coopMatMulAdd` → **aceiași biți ca `mm_tc` / `mm_tc_v4`**.
- Aceeași lansare, același cbuf, aceeași regulă `fits` ca `mm_tc_v4`; doar QMD-ul cere 12 KiB.

Tăiat acum (de reluat după placă): memoria comună dublă (încă 8 KiB → doar 3 blocuri), scrierile f16 împerecheate și citirile
`ldmatrix` (schimbă bucla; pasul de după, dacă ocuparea nu ajunge).

## 4. Predicțiile (scrise înainte)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| V1 | registre NAK | **≤ 128**, 0 memorie locală, 12 288 B memorie comună | 70 % |
| V1b | dacă ies > 128 registre | 3 blocuri pe SM → câștig doar ×1,2–1,35 | — |
| V2 | bucla | aceleași instrucțiuni ca v4 (±3 %): 16 `hmma`, 8 citiri `b128`, 32 `f2f`, 32 scrieri u16, 2 bariere | 85 % |
| V3 | finalul | 4 `coopMatStore` pe grup în zona de 1 KiB, fără `bar.sync` în plus pe tot blocul | 60 % |
| V4 | SIMULARE | aceiași biți ca `mm_tc` pe toate produsele aliniate; nealiniatele refuzate | 90 % |
| V5 | pe placă: biții | aceiași ca `mm_tc` pe toate seturile aliniate, gărzile neatinse | 85 % |
| V6 | ocuparea | 4 blocuri pe SM la 64 KiB (sub ambele citiri ale rezervei) | 85 % |
| V7 | proba qkv-513 la 64 KiB | 2,62 ms → **1,59–1,94 ms** (×1,35–1,65; central ×1,45 = 1,81 ms) = **11,2–13,7 TFLOP/s** | 55 % |
| V8 | pasul #513 lot 6 | înmulțirile 216,6 → **131–160 ms**; pasul 273 → **188–217 ms** (central ~206); **~14 200–16 400 simb/s** (azi 11 256) | 45 % |

Prag scris: dacă proba nu e măcar **×1,2** (sub 2,19 ms), ocuparea nu e frâna → următorul pas e bucla (scrierile f16 împerecheate,
`ldmatrix`), nu mai multe blocuri.

Coada valurilor nu strică socoteala: la 4 blocuri pe SM doar dW pentru qkv (432 de blocuri) și proj (144) au ultimul val incomplet
(9,6 % din calcul); celelalte forme intră în valuri întregi.

## 5. Ce urmează (pașii, câte unul, fiecare cu test)

1. `tools/nak/mm_tc_v5.comp` (din `mm_tc_v4.comp`, doar finalul și `cs_`), compilat cu `build2.sh` (Mesa 26.3.1) → V1–V3.
2. Rust (doar adăugări): `nvidia_ga104::mm_tc_v5` (binarul, SHA, `REGISTERS`, `SHARED` = 12 KiB, `fits` = al lui v4);
   `gpu_sim::cmdproc::mm_tc_v5_on_cpu`; testul în SIMULARE (V4).
3. Nota pentru OS: un rând în `--mm-tc-64k` după `mm_tc_v4` (V5–V7), apoi pasul #513 cu v5 (V8).

**Nou pentru placă:** încă nimic (doar plan). Când va fi: doar programul `mm_tc_v5`, aceeași lansare; nicio scriere în registre,
niciun firmware.
