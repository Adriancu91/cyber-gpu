# C-157 punctul 2, după os-364 — `attn_*_3`: aceiași biți, citiri fără ramură

Programele `_2` (os-364) au ieșit identice bit cu bit cu cele vechi și de ×1,17–×1,77 mai rapide, dar comparația
(`docs/C-157_P2_COMPARATII.md`, os-364) a arătat că schimbarea F3 („citirile emise deodată”) n-a câștigat nimic: NAK pune fiecare
citire globală păzită de un `if` în zona ei (`bssy` / `bsync`) și **așteaptă rezultatul la ieșirea din zonă**, deci citirile au rămas
una după alta, câte o latență întreagă a memoriei. Remediul (predicția 4 de acolo): **citiri fără ramură** — adresa strânsă în
buffer, citirea mereu, apoi alegerea 0 / valoare. Aceleași sume în aceeași ordine → aceiași biți.

Câte un program, cu plan și predicții întâi. Primul: `attn_back_tile0_3` (cel mai mare câștig prezis: trecerea a doua are 32 de
citiri în șir pe placă).

## `attn_back_tile0_3` — planul și predicțiile (sesiunea 179, 23 nov 2026, scrise ÎNAINTE de cod)
Program nou **lângă** `attn_back_tile0` și `attn_back_tile0_2` (neschimbate). Pornește de la `attn_back_tile0_2.comp`: aceleași
argumente (cbuf-ul lui `attn_back`, modul 0), aceeași grilă [⌈T/64⌉, H, B], blocuri de 256, aceeași memorie partajată (33 024 B),
același model pe procesor (`attn_back_row_gpu`), fiecare lanț în aceeași ordine. Schimbări:
- **G1 — trecerea a doua (dS = P·(dP − dot)·scale) împărțită pe feluri de plăci:**
  - plăcile de **sub diagonală** (`jt < bx`): acolo `j < i0 ≤ i`, deci garda `j <= i` e mereu adevărată; citirile lui dS și P
    **fără ramură**, cu rândul strâns `ic = min(i, T − 1)`; scrierea doar dacă `i < T` (o scriere nu se așteaptă);
  - placa **diagonalei** (`jt == bx`): citirile la `(ic, min(j, ic))` (mereu în buffer), apoi `y = (i < T && j <= i) ? valoarea : 0,0`;
    scrierea dacă `i < T && j < T` (ca înainte);
  - plăcile de **deasupra diagonalei** (`jt > bx`): acolo `j > i` mereu, vechiul scria 0,0 după o ramură; acum **scrie 0,0 fără nicio
    citire** (aceeași valoare, +0).
- **G2 — urcarea plăcilor dO, V și P fără ramură:** adresa strânsă (`min(i, T − 1)`, `min(x, d − 1)`, `min(j, ic)`), citirea mereu,
  apoi `cond ? valoarea : 0,0` (aceeași valoare scrisă în `sa` / `sb` ca la `_2`). Tot câte 8 citiri, apoi cele 8 scrieri partajate.
- Restul (bucla lui dP, lanțul `dot`, scrierea lui dP în dS) = `attn_back_tile0_2`, neschimbat.
- De ce aceiași biți: fiecare valoare care intră într-o sumă sau într-o scriere e aceeași (aceeași adresă când condiția e adevărată,
  0,0 când e falsă); valorile citite la adresele strânse și aruncate nu intră nicăieri (nici NaN-urile de deasupra diagonalei lui P).

Predicții:
1. **Aceleași operații:** în listare 0 `mufu`, nicio atomică, nicio memorie locală; felurile de operații aritmetice ⊂ cele din
   `attn_back_tile0_2` plus alegeri / minime pe întregi (`sel`, `imnmx` sau ce scrie NAK); `fmul` / `fadd` doar în trecerea a doua.
2. **Registre ≤ 80** (3 blocuri de 256 pe SM, ca `_2`). Dacă ies peste, cobor citirile emise deodată în trecerea a doua (de la 16
   elemente = 32 de citiri la 8 elemente) și notez.
3. **Buclele (`nak_bucle.py`, coloana `sir`):** fiecare iterație de urcare a unei plăci (8 citiri) **1** așteptare a memoriei (la `_2`:
   8); trecerea a doua, plăcile de sub diagonală: **≤ 2** așteptări pe iterație (la `_2`: 32 pe fiecare placă, și deasupra diagonalei);
   0 zone `bssy` în buclele de urcare și în trecerea a doua de sub diagonală.
4. **SIMULARE:** emularea literală a GLSL-ului nou = `attn_back` modul 0 **bit cu bit** pe cele 10 forme din T175 / T182; lansarea =
   `attn_back_tile0` bit cu bit; NaN / 1e30 deasupra diagonalei lui P și NaN în dO / V dincolo de `d` sau de `T` (unde citirile
   strânse ajung) nu schimbă niciun bit. Greșeli plantate prinse: alegerea scoasă la placa P (valoarea strânsă intră în sumă), alegerea
   scoasă pe diagonala trecerii a doua, plăcile de deasupra diagonalei sărite (fără scrierea lui 0), memoria partajată veche în QMD.
5. **Pe placă (într-o probă „vechi lângă nou” ca os-364, mai târziu):** dS bit cu bit = `attn_back_tile0` (~85 %); timpul pe setul
   mare **~0,5–0,9 ms** față de 1,54 la `_2` (DEDUS: modelul pe bucle cu o așteptare pe iterație 0,50 → 0,20 ms, înmulțit cu factorul
   măsurat ×3,12 → ~0,63 ms; factorul ×3 rămâne NECUNOSCUT, de aici intervalul).

## Ce NU face
- Nu schimbă `attn_back_tile0`, `attn_back_tile0_2`, `train_step` sau proba „vechi lângă nou” din os-364; programul `_3` se leagă
  în probă după ce sunt gata toate trei `_3` (sau doar el, dacă fwd / back1 întârzie).
- Nimic nu rulează pe placă din sesiunea asta. Nicio scriere în registre.

## Surse
- `docs/C-157_P2_COMPARATII.md` (os-364, predicția 4), `tools/buget-antrenare/nak_bucle.py` (`waits`, `model364`).
- Ideea „citire strânsă + alegere în loc de ramură”: ghidul public CUDA (divergența ramurilor, „predication”); nicio linie copiată.
