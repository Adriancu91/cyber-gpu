# Ordinea 20:11, pasul 3, partea a doua — GELU' lipit în înmulțirea „proj-dX” (plan + predicții, scrise ÎNAINTE de cod)

**Sesiunea 192, 23 nov 2026, cloud, fără placă.** Doar planul; codul vine în sesiunea următoare. Tot ce e aici e DEDUS.

## 1. Ce se lipește

Pe partea „înapoi” a MLP-ului, azi (pe strat):
1. `mm_train` „inapoi:proj-dX”: `dwide = dx · W_proj` (n × F, F = 4C), scris în memorie;
2. `gelu_tr` mod 1 „inapoi:gelu”: citește `u` și `dwide`, scrie `dwide2 = dwide · (Φ(u) + u·φ(u))`.

Lipit: un program nou **`mm_train_gelu`** = `mm_train` (aceeași buclă, aceleași plăci, aceeași ordine a FMA-urilor) care, în loc să
scrie `acc`, citește `u[row, col]` și scrie `acc · (Φ(u) + u·φ(u))` cu exact operațiile lui `gelu_tr` mod 1.
`mm_train` și `gelu_tr` rămân neschimbate (alți octeți ar cere o nouă probă pe placă).

## 2. De ce e al doilea, nu primul

Câștigul e de ~4× mai mare decât la `ln_back_res`, dar înmulțirea e programul cel mai sensibil, iar în simulare biții sunt dovediți doar
dacă **codul NAK al părții GELU din `mm_train_gelu` e același șir de instrucțiuni ca în `gelu_tr`** — asta se verifică pe listare, nu
în model (modelul ar fi compus din funcțiile existente și ar „trece” oricum).

## 3. Predicțiile

- **G1 (biți):** pasul cu `mm_train_gelu` = pasul de azi bit cu bit (pasul de aur mic 10 pași `sgd` / `adamw_513`; 4 straturi;
  `antrenare mini` F32), cu sau fără `ln_back_res`.
- **G2 (listarea):** partea GELU din `mm_train_gelu` are aceleași instrucțiuni ca `gelu_tr` mod 1 (un `MUFU.EX2`, un `MUFU.RCP`,
  aceleași `ffma` / `fmul`); înmulțirea `acc · (…)` e un `fmul` separat (acc nu se contopește cu nimic). Registre ≤ 32.
  Dacă NAK contopește altfel, se folosește aceeași metodă ca la `add_exact` (o funcție cu `precise` doar unde trebuie) sau se renunță.
- **G3 (lansări):** L lansări mai puține pe pas (toate `gelu_tr` „inapoi:gelu”), restul listei identic.
- **G4 (greșeli plantate):** `u` al altui strat → alți biți / verdictul pică; GELU' înainte de sumă (pe A în loc de C) → refuzat sau alți biți.
- **G5 (câștig, DEDUS):** la #513 (n = 6 144, F = 3 072) azi 4 treceri de 75,5 MB pe strat (scrierea `dwide`, citirile `u` și `dwide`,
  scrierea `dwide2`), lipit 2 → **~1,8 GB și 12 lansări mai puțin pe bucată, ~4–5 ms** (la ~380–450 GB/s). Împreună cu `ln_back_res`:
  ~2,7 GB, ~6–7 ms pe bucată (~2–3 % din pas).
- **Pasul mixt:** acolo „proj-dX” merge pe `mm_tc`; lipirea pe `mm_tc` vine separat, după ce G1–G4 trec pe `mm_train`.

## 4. Rezultatul (sesiunea 193, 23 nov 2026, cloud, fără placă) — SIMULARE

| Predicție | Rezultat |
|---|---|
| G1 (biți) | **IEȘIT**: bit cu bit cu pasul nelipit — pasul de aur mic 10 pași `sgd` / `adamw_513` (`new`, `new_tiles3`, și cu `ln_back_res`), 4 straturi 3 pași AdamW, `antrenare mini` F32 5 pași; verdictul PyTorch trece |
| G2 (listarea) | **IEȘIT abia la a treia compilare.** (1) `acc * (cdf + u * pdf)` scris ca în `gelu_tr`: NAK a contopit altfel — `fmul`(u, pdf) apoi `ffma`(1 + erf, 0,5, ·), alți biți. (2) Cu `precise` doar pe 0,5·(1 + erf): `fmul` + `fadd`, fără `ffma` — tot alți biți. (3) Cu `fma(u, pdf, cdf)` scris explicit: **aceeași secvență ca `gelu_tr` mod 1** (23 de operații, aceleași constante, `ffma`(u, pdf, Φ) apoi `fmul`(acc, ·)). Bucla e identică cu `mm_train` (16 `ffma`, aceeași ordine). 274 de instrucțiuni, 19 registre, SHA-256 `ee7e3145…ad0b053`. Cauza: în `gelu_tr` Φ e folosit și de modul 0, deci NAK îl rotunjește separat; aici nu |
| G3 (lansări) | **IEȘIT**: L lansări mai puține (73 → 71 / 75 → 73 la pasul mic; cu `ln_back_res` 73 → 67 / 75 → 69; 4 straturi 137 → 125), restul listei identic, aceeași arenă |
| G4 (greșeli plantate) | **IEȘIT**: `u` al stratului următor → alți biți și grad_abs 8,9·10⁻² față de PyTorch; GELU' uitat (lansat ca `mm_train` simplu) → grad_abs 1,1·10⁻¹; epilogul fără `ffma` (riscul de la G2) → 158 622 / 218 496 gradienți alți biți |
| G5 (câștig) | DEDUS, neschimbat (~1,8 GB și 12 lansări pe bucata #513) — **se vede doar pe placă** |

**Pasul mixt:** `step_mixed` refuză `with_fused_gelu` (lipirea pe `mm_tc` vine separat).
