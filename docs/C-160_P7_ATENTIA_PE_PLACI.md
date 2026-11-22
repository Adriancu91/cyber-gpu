# C-160 punctul 7 — atenția „pe plăci”, întâi jumătatea „înainte” (`attn_fwd_tile`)

Sesiunea 163, 22 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Totul despre viteză e **DEDUS**; corectitudinea se verifică doar în **SIMULARE**.

## De ce nu „un warp pe rând”
Prima idee din sesiunea 162 era: rândul i împărțit între cele 32 de fire ale unui warp. Socoteala arată că **nu câștigă**: un warp care
ține un singur rând face ~2 000 de iterații, dar 32 de rânduri costă tot ~66 000 de iterații de warp — exact cât programul vechi, unde
un warp ține 32 de rânduri (câte unul pe fir). Numărul de instrucțiuni pe produs rămâne același (o citire, un `ffma`, indici, salt);
se schimbă doar cine le face. Câștigul adevărat vine din **refolosire**: o valoare citită o dată în memoria partajată să intre în mai
multe produse (ca la `matmul_2d`). Nici „atenția = două `mm_train`” nu ajută acum: `mm_train` e estimat la ~2–3 TFLOP/s și ar calcula
și jumătatea de deasupra diagonalei (~116 GFLOP pe bucată → ~45 ms, mai rău decât blocurile de 128).

## Planul (scris ÎNAINTE de cod)
Program nou `tools/nak/attn_fwd_tile.comp` (Mesa NAK 25.0.7, ca celelalte programe de antrenare F32), cu aceleași argumente ca
`attn_fwd` (același cbuf: qkv / P / att, T, C, H, d, scale), alături de cel vechi (OS-ul compilează neschimbat).
- **Blocuri de 256 de fire (16 × 16), fiecare bloc = 64 de rânduri** (b, h, i0 … i0 + 63); grila [⌈T/64⌉, H, B]. Firul (tx, ty) ține
  rândurile ty + 16·a și coloanele tx + 16·k (a, k = 0…3): **4 × 4 rezultate pe fir**. Cere d ≤ 64 (Kronos: d = 64; modelul mic 16).
- **Faza A (scorurile):** placa lui Q (64 × d) o dată în memoria partajată; pentru fiecare placă de 64 de coloane până la diagonală:
  placa lui K în memoria partajată, apoi pentru e = 0 … d − 1 cele 16 `ffma` (8 citiri partajate → 16 produse). s = acc · scale,
  scris în P (doar j ≤ i); fiecare fir ține maximul pe rândurile lui; maximul rândului = maximul celor 16 fire (exact, orice ordine).
- **Faza B (exp + suma):** fiecare fir își recitește propriile s (același fir le-a scris → fără barieră), x = exp(s − m), scris în P;
  suma parțială a firului tx = lanțul `fadd` din 0 peste j ≡ tx (mod 16), j crescător, j ≤ i; **suma rândului = lanțul celor 16
  parțiale, tx = 0 … 15, din 0** (o ordine fixă, deci aceiași biți la fiecare rulare); inv = 1/sumă.
- **Faza C (P și att):** pe fiecare placă până la diagonală, firul își recitește x-urile, P = x · inv (0 deasupra diagonalei), scrie P
  și îl pune în memoria partajată; placa lui V în memoria partajată; att = lanțul `ffma` din 0 peste j crescător, **doar j ≤ i**
  (aceeași ordine ca `attn_fwd`). Plăcile de deasupra diagonalei: P = 0.
- Memoria partajată: Q, K/P, V ≈ 48 KiB + 4 KiB pentru maxime / sume → 1–2 blocuri pe SM.
- **Modelul pe procesor:** `gpu_compute::layer::attn_fwd_tile_gpu` — **același lanț ca `attn_fwd_row_gpu`** pentru s, m, x și att;
  **diferă doar suma** (16 parțiale). Simulatorul: `gpu_sim::cmdproc::attn_fwd_tile_on_cpu` (refuză blocuri ≠ 256, d > 64, grile
  care trec de T / H).

## Predicții (scrise acum)
1. **Rândurile i ≤ 15** au cel mult un element în fiecare sumă parțială, deci suma = lanțul vechi: **P și att bit cu bit egale cu
   `attn_fwd`** pe aceste rânduri, pe toate formele.
2. Pe rândurile i ≥ 16: **s și m bit cu bit** ca la `attn_fwd`; P și att diferă cel mult cu câteva unități în ultima poziție:
   |ΔP| ≤ 2⁻²⁰ · P relativ, |Δatt| ≤ 2⁻²⁰ · Σ|P·V|. Față de oracolul f64 `train::attention_rows`, în **aceeași margine ca T160**,
   cel mai rău ≤ 20 % din margine (T160: 13,6 % P, 6,5 % att).
3. Lanțul `attn_fwd_tile` → `attn_back` (programul înapoi neschimbat) față de P-ul în f64: ≤ 2⁻¹⁴, ca la T160.
4. Registre ≤ 64 (16 acumulatori + 4 maxime / sume + indici), cel mult 2 bariere distincte; nicio `atomic`.
5. Greșeli plantate prinse: grila cu ⌈T/64⌉ − 1 blocuri (rânduri nescrise → gărzile NaN), blocuri ≠ 256 refuzate, d > 64 refuzat,
   modelul cu suma veche în locul celei pe 16 parțiale (diferă pe rândurile ≥ 16 la o formă lungă).
6. **Viteza (DEDUS, doar pe placă; `tools/buget-antrenare/fir_limitat.py`, `print_tiled`):** la #513, 5,4 GFLOP calculate pe lansare
   și ~0,5 GB de P mutat → limitat de memorie: **~19 ms pe bucată** (interval 12–40), față de ~37 (blocuri de 128) și ~90 (de 32).
   P trece de 5 ori prin memorie; varianta fără P păstrat (recalculat în „înapoi”) ar fi pasul următor.

## Ce NU face
Nimic pe placă. Nu schimbă `attn_fwd` / `attn_fwd128` / `attn_back`. Jumătatea „înapoi” pe plăci = pasul următor, plan separat.

## Surse
- Ideea plăcilor în memoria partajată cu mai multe rezultate pe fir: `matmul_2d` / `matmul_vec4` (pașii noștri, după siboehm
  SGEMM_CUDA, MIT, doar ideea); atenția pe plăci: FlashAttention (Dao et al., articol public) și llm.c `attention_forward` (MIT) —
  doar ideea; noi păstrăm P (pentru `attn_back`) și o ordine fixă a sumelor.
- Limitele SM 8.6: ghidul public CUDA (capacitatea 8.6).
