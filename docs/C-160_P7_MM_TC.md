# C-160 punctul 7, partea a doua — `mm_tc`: înmulțirea antrenării pe circuitele speciale (SIMULARE)

Sesiunea 157, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE** sau **DEDUS**: nimic nu a rulat pe RTX 3070 Ti.

## 1. Ce e
Un program de placă nou, `mm_tc` (`tools/nak/mm_tc.comp` → a doua unealtă NAK, Mesa 26.3.1 → `tools/nak/out2/mm_tc.sm86.sass`),
care face **aceeași treabă ca `mm_train`** — C = op(A)·op(B) (+ deplasare), cu aceleași opțiuni `TA`, `TB`, `BIAS` și aceleași buffere
f32 — dar pe **circuitele speciale** (tensor cores):
- fiecare element din op(A) și op(B) e **rotunjit la 16 biți** (f16, „la cel mai apropiat, la egalitate spre par”) când e copiat în
  memoria partajată;
- produsele se adună în **f32** de circuitele speciale (`HMMA.16816.F32`, aceeași instrucțiune care a trecut pe placă la P8, os-335);
- deplasarea se adună o dată, în f32, la sfârșit.

Deci rezultatul **nu** e „aceleași numere” ca `mm_train` sau PyTorch-F32: diferența e prețul rotunjirii la f16 (≈ 3·10⁻⁴ relativ în
probele de mai jos). Cu el se pot face qkv, out_proj, fc, proj și capul legat, înainte și înapoi (la fel ca `mm_train`).

| Ce | Valoare (din listarea NAK, `tools/nak/out2/mm_tc.sm86.txt`) |
|---|---|
| Binar | 2 632 de instrucțiuni, **42 112 octeți**, SHA-256 `f60cd944ab67ae17ddc0d61931085791dc25f76397de82070fe0b0f1c3821ee3` (identic la două compilări; `mma16_f16` refăcut în aceeași unealtă a ieșit identic, `f8c0db68…a1ef9`) |
| Registre | 64 pe fir |
| Bloc | 128 de fire (4 grupuri de 32); fiecare bloc face o bucată 64×64 din C; grila (⌈n/64⌉, ⌈m/64⌉, 1) |
| Memoria partajată | 24 KiB (felia 64×32 din op(A) și 32×64 din op(B) în f16 + bucata 64×64 f32 de la final), o barieră |
| Argumentele | bufferul constant 0, **de la 0x700** (ca la P8): a/b/c/bias la 0x700/0x708/0x710/0x718, m/n/k/opțiuni la 0x720/0x724/0x728/0x72c; baza grupurilor la 0x0/0x4 (lăsată 0) |
| Instrucțiunile importante | **16 × `hmma.m16n8k16.f32`** pe felia de 32 (8 înmulțiri 16×16×16, fiecare = 2 instrucțiuni), `f2f.f16.f32.re` (rotunjirea la f16, la cel mai apropiat), `ldsm.16.m8n8.x4` (A din memoria partajată), `movm.16.m8n8.trans` (B întors), `fadd.ftz` pentru deplasare |

## 2. Cum se folosește (cod: doar adăugări, nimic redenumit — OS-ul compilează neschimbat)
- `nvidia_ga104::mm_tc` — binarul + SHA-256, `REGISTERS`, `BLOCK`, `SHARED`, `Args::cbuf` (0x800 octeți), `Args::grid`, `Args::flop`.
- `gpu_compute::mm_tc` — `reference` (f64 pe intrările rotunjite), `margin`, `rounding_margin`, `judge` → `Verdict`, `worst_vs_f32`
  (față de oracolul F32 de antrenare), `model` (ipoteza a ce calculează placa).
- `gpu_sim::cmdproc::{mm_tc_on_cpu, MM_TC_NEEDS}` — programul pe procesor, în procesorul de comenzi simulat.

## 3. Marginea, scrisă înainte de rulare
**Față de răspunsul exact pe intrările rotunjite:** |d| ≤ (K + 3)·2⁻²³·(Σ|â·b̂| + |bias|) + 2⁻¹²⁶.
De unde: modelul adunării circuitelor speciale potrivit pe placa reală (os-335: 8 produse + acumulatorul pe pas, tăiere spre zero, un bit
în plus) greșește pe pas cel mult 9·2⁻²⁴ + 2⁻²³ ≈ 5,5·2⁻²³ din suma pasului, deci pe tot K cel mult (K/8)·5,5·2⁻²³·Σ; o adunare
„la cel mai apropiat” greșește cel mult K·2⁻²⁴·Σ. Marginea le acoperă pe amândouă cu loc (pentru K = 16 e strânsă ca toleranța P8, 17·2⁻²³).
**Față de oracolul F32 de antrenare** (intrările nerotunjite, `train::linear` / `linear_back`): în plus prețul rotunjirii,
Σ(|a|·|b|·(2⁻¹⁰ + 2⁻²²) + 2⁻²⁵·(|a| + |b|) + 2⁻⁵⁰).

## 4. Ce s-a verificat (T168, `crates/nvidia-ga104/tests/sim_mm_tc.rs`, 7 teste + 4 + 2 în biblioteci) — SIMULARE
| Test | Rezultat |
|---|---|
| Binarul = cel compilat; listarea are `hmma`, bariera, argumentele la 0x700 | PASS |
| 5 forme (cu blocuri parțiale: 70×33×65, 130×96×17, 5×1×3, 16×200×16, 64×32×64) × toate 8 combinațiile de opțiuni, prin pushbuffer + QMD | toate în margine, cel mai rău **46 ‰** din margine |
| Pe greutățile reale `in_proj` ale stratului 0 din modelul mic (ca T152) | Y = X·Wᵀ + b: 22 ‰ din margine; față de F32 190 ‰ din margine + preț, abaterea relativă **2,7·10⁻⁴** |
| | dX = dY·W: 10 ‰; față de F32 110 ‰, abaterea **3,3·10⁻⁴** |
| | dW = dYᵀ·X: 30 ‰; față de F32 206 ‰, abaterea **2,9·10⁻⁴** |
| Transpusele chiar contează; un rezultat judecat cu transpusa greșită e prins | PASS |
| Refuzuri (opțiuni necunoscute, memorie partajată prea mică, fără barieră, bloc greșit, grila prea mică, o intrare > 65 504): **nicio scriere** | PASS |
| Greșeli plantate: rezultatul F32 (fără rotunjire) **nu trece** marginea f16 — verdictul deosebește cele două programe; modelul `mm_train` pus pe acest binar e refuzat; un element mutat cu 1 % e prins exact acolo | PASS |
| Limita cinstită a f16: un gradient de 1·10⁻⁸ devine **0**; înmulțit întâi cu 2¹⁶ supraviețuiește (eroare < 2·10⁻³) | PASS |

`tools/check.sh`: format, clippy `-D warnings`, **786 de teste** (+7 ignorate), build no_std — PASS.

## 5. Ce înseamnă pentru antrenare (DEDUS)
- **Gradienții mici se pierd în f16** (sub ~6·10⁻⁸ devin 0, sub ~6·10⁻⁵ pierd cifre). De aceea antrenarea „mixtă” înmulțește eroarea cu
  un factor mare înainte de drumul înapoi și împarte gradienții la sfârșit („loss scaling”, Micikevicius et al. 2018, „Mixed Precision
  Training”). PyTorch folosește bf16 (domeniu mare, nu are nevoie), dar NAK pe GA104 dă doar f16 → **pasul următor: scalarea erorii**
  în lista pasului, apoi proba „aceeași eroare pe cei 10 pași în toleranța scrisă atunci” (estimarea, §4.3).
- Greutățile, gradienții adunați și AdamW rămân în F32 (ca la PyTorch); doar intrările înmulțirilor sunt rotunjite.

## 6. Viteza: estimarea revăzută după listare (DEDUS, scrisă înainte de orice măsurătoare)
Estimarea din `docs/C-160_P7_VITEZA_ESTIMARE.md` (17–30 TFLOP/s) presupunea un program bun. Listarea acestei prime versiuni arată unde
va frâna:
- **memoria plăcii:** intrările sunt f32 (4 octeți) și plăcile doar 64×64 → ~16 operații pe octet citit; la ~608 GB/s asta ar da cel mult
  ~10 TFLOP/s dacă nimic nu vine din L2 (o parte vine, dar nu știm cât);
- **citirile:** câte un `ld.global` de 4 octeți pe element, cu ramificare pe element (marginile), plus `f2f` + `st.shared.u16` pe element;
- **B întors cu `movm`** (16 pe felie) — se poate evita ținând B „pe coloane” în memoria partajată.

**Estimare pentru `mm_tc` așa cum e: ~8–17 TFLOP/s** la formele lui #513 (adică ~0,4–0,8 din plafonul F32, cam cât un F32 bun).
Sub 5 = altceva frânează (QMD, ocuparea, ceasul) — de lămurit întâi. Ca să ajungă la ~30: plăci 128×128, citiri de 16 octeți,
B ținut pe coloane, și activările păstrate direct în f16 (jumătate din trafic). Asta vine **după** ce versiunea simplă e corectă pe placă.

## 7. Ce e nou pentru placă (când va fi predat — nu acum)
Nu e predat: pe placă întâi `colsum`, apoi pasul de aur mic (deja predate). Pentru `mm_tc` va fi o probă separată (un singur lucru nou):
binarul `mm_tc` (al doilea program al uneltei Mesa 26.3.1 după `mma16_f16`, care a trecut în os-335), **memoria partajată 24 KiB**
(cea mai mare de până acum: 2–4 KiB la celelalte), bloc de 128, argumentele la 0x700. Proba de pregătit: câteva seturi fixe + forma
qkv a lui #513 pentru timp, judecate cu `gpu_compute::mm_tc::judge`.

## Surse
- Aranjamentul (bucăți 64×64, felii de K în memoria partajată, 4 grupuri × 2×2 acumulatoare): idei din scrierile publice despre GEMM pe
  tensor cores (NVIDIA CUTLASS, BSD-3, doar ideea; „How to Optimize a CUDA Matmul Kernel”, siboehm, MIT). Codul GLSL e al nostru.
- `coopMatLoad` / `coopMatMulAdd`: specificația `GL_KHR_cooperative_matrix` (Khronos).
- Modelul adunării: `gpu_compute::mma::GA104_OS335` (amprenta măsurată pe placa reală în os-335, sesiunea 140).
- Scalarea erorii: Micikevicius et al., „Mixed Precision Training”, ICLR 2018 (idee).
