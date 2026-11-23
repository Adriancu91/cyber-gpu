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
