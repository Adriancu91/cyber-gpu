# C-160 punctul 7 — atenția „înapoi” pe plăci, întâi modul 1 (coloanele): `attn_back_tile`

Sesiunea 164, 22 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Totul despre viteză e **DEDUS**; corectitudinea se verifică doar în **SIMULARE**.

## De ce modul 1 întâi
`tools/buget-antrenare/fir_limitat.py`: la #513 modul 1 al lui `attn_back` (dQ | dK | dV) e cel mai lent program al pasului:
~133 ms pe bucată cu blocuri de 32, ~44 ms cu blocuri de 128 (`attn_back128`). Un warp ține o singură poziție și citește
coloanele lui dS și P „pe sărite” (pas de T elemente), fără nicio refolosire. Modul 0 (~34 ms) vine după, plan separat.

## Planul (scris ÎNAINTE de cod)
Program nou `tools/nak/attn_back_tile.comp` (Mesa NAK 25.0.7), **aceleași argumente ca `attn_back`** (același cbuf de 0x68 octeți:
qkv / P / dO / dS / dqkv, T, C, H, d, modul, scala), alături de cel vechi. Face **doar modul 1**: cu modul ≠ 1 iese fără să scrie
nimic (modul 0 rămâne `attn_back` / `attn_back128`, neschimbat). Ordinea pe placă: `attn_back` modul 0 → `attn_back_tile`.
- **Blocuri de 256 de fire (16 × 16), un bloc = 64 de poziții** (b, h, t0 … t0 + 63) și toate cele d ≤ 64 coloane; grila
  [⌈T/64⌉, H, B]. Firul (tx, ty) ține pozițiile ty + 16·a și coloanele tx + 16·k (a, k = 0…3): 4 × 4 rezultate pe fir.
- Trei treceri, fiecare o înmulțire mică pe plăci de 64 × 64 în memoria partajată (două plăci = **32 KiB**):
  - **dQ**: pentru fiecare placă de coloane jt = 0 … (blocul): placa dS[t0.., jt·64..] și placa K[jt·64.., e]; apoi pentru
    jj = 0 … 63, j crescător, **doar j ≤ t**: dq = ffma(dS_tj, K_je, dq).
  - **dK**: pentru fiecare placă de rânduri it = (blocul) … ⌈T/64⌉ − 1: placa dS[it·64.., t0..] (rândul i al lui dS citit
    contiguu: coloanele t0 … t0 + 63) și placa Q; i crescător, **doar t ≤ i < T**: dk = ffma(dS_it, Q_ie, dk).
  - **dV**: la fel, cu P în loc de dS și dO în loc de Q.
- **Fiecare rezultat e același lanț `ffma` ca în `attn_back` modul 1:** din 0, același index crescător, aceiași termeni, aceleași
  margini (j ≤ t; t ≤ i < T). Doar cine îl calculează și de unde citește operanzii (memoria partajată) se schimbă. Fără MUFU, fără
  `atomic`, fără sume împărțite.
- **Modelul pe procesor:** nu e nevoie de unul nou — fiecare element = `gpu_compute::layer::attn_back_col_gpu`. Simulatorul:
  `gpu_sim::cmdproc::attn_back_tile_on_cpu` (blocuri de 256, d ≤ 64, grila nu trece de ⌈T/64⌉ / H; modul ≠ 1 → nimic scris).
  Modulul `nvidia_ga104::attn_back_tile` (doar adăugări; folosește `attn_back::Args`).

## Predicții (scrise acum)
1. **dqkv bit cu bit egal cu `attn_back` modul 1** (blocuri de 32), pe toate formele cu d ≤ 64: modelul mic, T 70 / C 96 / H 3,
   d 40, d 1, T 5 / B 3, softmax ascuțit, gradienți mici, plus forme lungi (T 200 cu d 64, T 129, T 64 exact). Deci și lanțul
   `attn_back` modul 0 → `attn_back_tile` = `attn_back_gpu` bit cu bit, iar față de oracolul f64 aceleași cifre ca T159.
2. P și dS deasupra diagonalei (NaN, 1e30 acolo) nu schimbă niciun bit (masca = marginile buclelor, ca în `attn_back`).
3. Registre ≤ 64 (16 acumulatori + indici; fără buclele de copiere desfăcute, ca la `attn_fwd_tile`), memorie partajată 32 768 B,
   nicio `atomic`, niciun MUFU, nicio memorie locală.
4. Greșeli plantate prinse: grila cu ⌈T/64⌉ − 1 blocuri (poziții nescrise → gărzile NaN), blocuri ≠ 256 refuzate, d > 64 refuzat,
   grila peste H refuzată, modul 0 → nimic scris; un model cu marginea dK greșită (i > t în loc de i ≥ t) diferă.
5. **Viteza (DEDUS, doar pe placă; `fir_limitat.py`, `print_tiled_back`):** la #513, 8,2 GFLOP calculate pe lansare și ~255 MB
   de dS + P citiți → **~15–24 ms pe bucată** (interval 10–50), față de ~44 (blocuri de 128) și ~133 (de 32). Cu atenția înainte
   pe plăci (~19) și modul 0 (~34), atenția întreagă ar scădea de la ~257 la ~75 ms pe bucată; modul 0 pe plăci = pasul următor.

## Rezultatul
(se completează după cod)

## Ce NU face
Nimic pe placă. Nu schimbă `attn_back` / `attn_back128` / `attn_fwd*`. Modul 0 pe plăci = pasul următor, plan separat.

## Surse
- Plăcile în memoria partajată cu 4 × 4 rezultate pe fir: `matmul_2d` / `attn_fwd_tile` (pașii noștri, după siboehm SGEMM_CUDA,
  MIT, doar ideea); drumul înapoi al atenției pe plăci: FlashAttention-2 (Dao, articol public) și llm.c `attention_backward`
  (MIT) — doar ideea; noi păstrăm P și dS și ordinea fixă a sumelor din `attn_back`.
- Limitele SM 8.6: ghidul public CUDA (capacitatea 8.6).
