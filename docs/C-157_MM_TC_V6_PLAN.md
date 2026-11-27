# C-157 punctul 2 — planul și predicțiile pentru `mm_tc_v6` (bucla lui `mm_tc_v5`: scrierile f16 împerecheate)

Sesiunea Cyber-GPU 263, 27 nov 2026, ~03:10–04:28 (ora României), în cloud, **fără placă**. Totul de mai jos e scris
**înainte** de orice cod de program GPU și de compilare. Cifrele „MĂSURAT” vin din rapoartele plăcii trimise de OS în hub
(os-468, os-473, os-476); tot ce iese din ele e DEDUS. Calculul e fixat în testul T275
(`crates/nvidia-ga104/tests/calc_c157_mm_tc_v6_plan.rs`, doar pe procesor).

## 1. De ce bucla, acum

- `mm_tc_v5` (os-476) a adus ×1,41 pe proba qkv-513 doar din ocupare (4 blocuri pe SM). Ocuparea e acum la plafon: v5 are
  exact **128 de registre** (limita pentru 4 blocuri), iar memoria comună nu mai e frâna.
- Proba qkv-513 e o înmulțire **fără transpunere** (pasul înainte). În pasul #513, cele 147 de înmulțiri se împart în trei
  părți **egale** ca număr de operații: înainte, dX = dY·Wᵀ (B transpus) și dW = Xᵀ·dY (A transpus). Proba nu vede deloc
  cele două treimi transpuse.

## 2. Ce e lent în scrierile din buclă (CALCUL, exact, T275)

În fiecare felie de K = 32, fiecare fir convertește 4 vectori de A și 4 de B în f16 și îi scrie în memoria comună **câte o
valoare de 2 octeți** (32 de scrieri pe felie, ca în listarea NAK). Am numărat, după regulile publice CUDA pentru 8.6
(32 de bănci de 4 octeți), câte treceri („wavefronts”) cere fiecare scriere a unui grup de 32 de fire:

| Înmulțirea din pas | A (treceri pe grup, pe felie) | B | Total pe bloc (4 grupuri) | Ca timp, la viteza os-476 |
|---|---|---|---|---|
| înainte (fără transpunere) | 32 (conflict 2 la 1) | 32 (2 la 1) | **256** tacți | 12 % |
| dX (B transpus) | 32 | **128 (8 la 1)** | **640** | 30 % |
| dW (A transpus) | **256 (16 la 1)** | 32 | **1 152** | 55 % |
| **`mm_tc_v6`, orice caz** | **8** | **8** | **64** | 3 % |

(„Ca timp”: o felie de bloc durează ~2 108 tacți ai SM-ului la 11,63 TFLOP/s; procentul = cât din acest timp ar ține
ocupată doar calea de scriere în memoria comună, dacă dX / dW ar merge la fel de repede ca proba.)

Concluzia (DEDUS): la înainte câștigul posibil e mic (plafon ×1,10 chiar dacă scrierile ar sta toate pe drumul critic);
la dW scrierile ocupă singure peste jumătate din timpul unei felii — acolo e de câștigat.

## 3. Ce schimbăm: `mm_tc_v6` = `mm_tc_v5` cu scrierile împerecheate

- Cele 4 valori ale unui vector se convertesc la fel (`f16vec4(v)` = 4 × aceeași conversie f32 → f16, rotunjire la cel mai
  apropiat) și se scriu **într-o singură scriere de 64 de biți**. Tablourile `as_` / `bs_` devin `f16vec4` (aceiași 4 KiB
  fiecare); `coopMatLoad` le citește cu pasul împărțit la 4.
- **Fără transpunere:** aceeași așezare ca în v5 (cele 4 valori erau deja una lângă alta). Citirile neschimbate.
- **Cu transpunere:** op(A) se ține pe coloane (`[p][r]`, pas 64), op(B) pe coloane (`[c][p]`, pas 32), ca vectorul de 4 să
  rămână continuu; `coopMatLoad` le citește cu `ColumnMajor`. Aceeași matrice ajunge în circuitele speciale.
  Alegerea între cele două așezări e o ramură pe argumentul `flags` (aceeași pentru tot blocul, fără divergență).
- Restul **neschimbat**: aceleași citiri din memoria plăcii, aceleași felii, același lanț `coopMatMulAdd`, același final
  (zona de 1 KiB pe grup, de la v5), aceeași lansare, același cbuf, aceeași regulă `fits` ca `mm_tc_v4` / v5,
  12 KiB de memorie comună → **aceiași biți ca `mm_tc`**.
- Efect secundar așteptat la citiri: pentru op(B) pe coloane (dX), forma e cea pe care circuitele o vor direct →
  cele 16 `movm` (transpunerile) ar trebui să dispară; pentru op(A) pe coloane (dW) apar transpuneri în plus.
  Înainte: nimic schimbat la citiri.

Tăiat acum (de reluat după placă): rânduri cu umplutură contra conflictelor la **citiri**, `ldmatrix`, memoria comună dublă,
programe separate pe fiecare combinație de transpunere (rămâne varianta de rezervă V1b).

## 4. Predicțiile (scrise înainte)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| V1 | registre NAK | **≤ 128**, 0 memorie locală, 12 288 B memorie comună | 55 % |
| V1b | dacă ies > 128 | 3 blocuri pe SM → în loc de v6 cu ramură: 4 programe fixe (`nn`, `nt`, `tn`, `tt`), fiecare mai simplu | — |
| V2 | scrierile în buclă | 32 scrieri u16 → **8 scrieri de 64 de biți** pe felie | 75 % |
| V2b | conversiile | 32 `f2f` → 16 (conversie pereche `f2fp`) | 45 % |
| V3 | citirile | înainte: aceleași 24 citiri + 16 `movm`; dX: `movm` scad; dW: `movm` cresc | 50 % |
| V4 | SIMULARE | aceiași biți ca `mm_tc` pe toate produsele aliniate × toate opțiunile; nealiniatele refuzate | 90 % |
| V5 | pe placă: biții | aceiași ca `mm_tc` pe toate seturile aliniate (inclusiv `-ta`, `-ta-tb`), gărzile neatinse | 85 % |
| V6 | ocuparea | 4 blocuri pe SM la 64 KiB (dacă V1 trece) | 80 % |
| V7 | proba qkv-513 (înainte) la 64 KiB | 1,868 ms → **1,70–1,87 ms** (×1,00–1,10) | 60 % |
| V8 | pasul #513 față de același pas cu v5 | înmulțirile **×1,03–1,20** (central ×1,10) → placa **×0,885–0,979** la lot 2, **×0,867–0,976** la lot 6 | 40 % |

Prag scris: dacă în pas înmulțirile câștigă sub **×1,03**, scrierile nu stau pe drumul critic → pasul următor e la citiri
(umplutură / `ldmatrix`) sau memoria comună dublă, nu mai departe la scrieri.

Notă: V8 se măsoară **față de pasul cu v5**, care încă n-a rulat pe placă (P2zzf, prima rulare la lot 2). Dacă vine întâi
pasul cu v5, banda absolută se scrie atunci, din cifra plăcii, nu acum.

## 5. Ce urmează (pașii, câte unul, fiecare cu test)

1. `tools/nak/mm_tc_v6.comp` (din `mm_tc_v5.comp`, doar scrierile din buclă și cele două `coopMatLoad`), compilat cu
   `build2.sh` (Mesa 26.3.1, trebuie reconstruit în cloud, ~20 min) → V1–V3.
2. Rust (doar adăugări): `nvidia_ga104::mm_tc_v6` (binarul, SHA, `REGISTERS`, `SHARED` = 12 KiB, `fits` = al lui v4);
   modelul în `gpu_sim::cmdproc`; testul în SIMULARE (V4), inclusiv o greșeală plantată (o felie transpusă pe rânduri).
3. Probă: pe lângă qkv-513, două seturi de timp **transpuse** (dX și dW ale qkv: 6 144 × 768 × 2 304 cu B transpus,
   768 × 2 304 × 6 144 cu A transpus), adăugate ca listă nouă (nu schimb `mm_tc_probe::SETS`), ca să se vadă câștigul acolo
   unde e.
4. Nota pentru OS: un rând în `--mm-tc-64k` după `mm_tc_v5` (V5–V7), apoi pasul #513 cu v6 (V8), întâi la lot 2.

**Nou pentru placă:** încă nimic (doar plan și socoteală). Când va fi: doar programul `mm_tc_v6`, aceeași lansare; nicio
scriere în registre, niciun firmware.

Sursa ideii (scrieri vectoriale în memoria comună, conflictele de bancă, așezarea pe coloane): ghidul public CUDA
(„shared memory”, capacitatea 8.6) și specificația publică `GL_KHR_cooperative_matrix` (ordinea `ColumnMajor`). Codul și
socoteala sunt ale noastre.

## 6. Ce a ieșit (sesiunea 263, în cloud, după predicții)

Unealta: Mesa 26.3.1 reconstruit în cloud; `mm_tc_v5` recompilat a ieșit **identic** (SHA `f34f4f93…`), deci unealta e aceeași.
`mm_tc_v6.comp` a trecut `glslangValidator` + `spirv-val` și NAK din **prima** compilare:
SHA-256 `8c761fa50321ee891f510a6b70ffac88778312be6b399747411cbfa637df980f`.

Bucla, numărată din listare pe fiecare drum (ramura pe `flags`; v5 are același drum pentru toate: 513 instrucțiuni):

| Drumul | instrucțiuni / felie | citiri `lds` | `ldsm` | `movm` |
|---|---|---|---|---|
| v5 (orice caz) | 513 | 16 | 4 | 16 |
| v6 înainte (fără transpunere) | **474** | 16 | 4 | 16 |
| v6 dX (B transpus) | **448** | 0 | 8 | **0** |
| v6 dW (A transpus) | **508** | 32 | 0 | **32** |
| v6 A și B transpuse | 482 | 16 | 4 | 16 |

| # | Prezis | Ieșit | Verdict |
|---|---|---|---|
| V1 | ≤ 128 registre, 0 locală, 12 288 B | **128 registre**, 0 memorie locală, **12 288 B**; NAK: „Max warps/SM: 16” = 4 blocuri | PASS (MĂSURAT pe text) |
| V2 | 32 scrieri u16 → 8 de 64 de biți | **8 `st.shared.b64`**, 0 scrieri u16 în buclă | PASS |
| V2b | 32 `f2f` → 16 `f2fp` | **16 `f2fp.pack_ab`**, 0 `f2f` | PASS |
| V3 | înainte la fel; dX `movm` scad; dW cresc | înainte aceleași 16 + 4 + 16; dX 16 `movm` → 0 (NAK folosește `ldsm`); dW 16 → 32 | PASS |
| V4 | SIMULARE: aceiași biți | T276: scrierile și citirile transcrise fir cu fir dau fiecărui fragment 16 × 16 exact elementele din v5, pe toate cele 4 combinații (greșeala plantată, felia transpusă citită pe rânduri, prinsă); fiecare jumătate a feliei scrisă o dată; textul GLSL din afara scrierilor și citirilor = al lui v5; prin procesorul de comenzi **aceiași biți ca `mm_tc`** pe toate produsele aliniate × opțiuni; nealiniatele refuzate | PASS (SIMULARE) |
| V5–V8 | pe placă | — | NOT TESTED |

Ce n-am prevăzut: drumul dW are cu 16 citiri + 16 `movm` mai mult (A pe coloane trebuie întoarsă la citire), deci câștigul la dW
depinde de cât costau conflictele la scriere față de aceste citiri în plus (citirile de 32 de biți pe coloane pot avea și ele
conflicte; nu le-am numărat — pasul următor dacă V8 iese sub prag). Predicțiile V7–V8 rămân cum au fost scrise.
