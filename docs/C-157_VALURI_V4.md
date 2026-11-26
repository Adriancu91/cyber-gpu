# C-157 — valurile `mm_tc_v4` pe pasul mixt #513: unde se duc cele 106 ms și dacă lansarea dublă dX + dW mai merită

Sesiunea 241, 26 nov 2026, în cloud, **fără placă**. Cifrele plăcii sunt MĂSURATE de OS (os-431, os-443, os-445); tot ce e
calculat aici e **DEDUS** (aritmetică pe procesor). Nimic nu a rulat pe RTX 3070 Ti în sesiunea asta.

## 1. De ce acum

Studiul programului persistent (sesiunea 236, `docs/C-157_PROGRAM_PERSISTENT.md` §5, punctul 2) a lăsat deschis: „lansarea dublă
dX + dW se va repeta pe `mm_tc_v4` (pasul mixt), unde înmulțirile devin mai scurte și cozile cântăresc mai mult — de măsurat”.
Pasul mixt rapid a rulat pe placă (os-445: placa **106,0 ms**, pasul întreg 116,9 ms), dar fără profil pe lansări, deci nu știm
cât din cele 106 ms sunt înmulțiri și cât restul. Niciun test nou pe placă de la os-445.

## 2. Ce avem (citit, nu presupus)

- `mm_tc_v4`: blocuri de 128 de fire, plăci 64 × 64, 128 de registre, 24 KiB memorie comună → la treapta 64 KiB **2 blocuri pe
  unitate** (96 de locuri pe 48 de unități), la 32 KiB 1 bloc (48 de locuri).
- os-443, qkv-513 (1024 × 2304 × 768 = 576 de blocuri): **2 623 680 ns la 64 KiB** (6 valuri), **4 737 216 ns la 32 KiB** (12 valuri).
- os-431 (pasul F32 #513, aceeași listă de programe în afara înmulțirilor, aceeași treaptă 64 KiB): placa 255 896 µs, din care
  înmulțirile (`mm_tv_128x128`, 147 lansări) **223 234 µs** → restul **32 662 µs**.
- Lista pasului mixt (`Plan::new_tiles3(M513, 2, 512).with_mm_tile().with_colsum_split().step_mixed`): 484 de lansări, 147 `mm_tc`
  în 15 forme (cozile valurilor: 1024 × 768 = 192 de blocuri = exact 2 valuri; `out-dW` 768 × 768 = 144 = 1,5 valuri).

## 3. Predicțiile (scrise ÎNAINTE de cod și de calcule)

- **V1.** Înmulțirile pasului mixt pe placă (DEDUS: 106,0 ms − restul os-431) ≈ **73 ms**; modelul valurilor `mm_tc_v4` (un timp pe
  val, calibrat doar pe qkv-513, liniar în K) le reface **în ±10 %**.
- **V2.** Pe plăcile 64 × 64 cozile valurilor sunt mici: locurile goale din cele 147 de lansări **sub 5 %** din timpul înmulțirilor.
- **V3.** Lansarea dublă dX + dW pe `mm_tc_v4` (podeaua, blocuri împachetate perfect) câștigă **sub 2 %** din placa pasului mixt
  (pe F32 era 3,1 % peste alegerea pe formă) → nu merită un program nou.
- **V4.** Restul (~32,7 ms) devine **~31 %** din placă; cea mai mare bucată a lui e atenția (os-431: `attn_*` 20,6 ms ≈ 19 % din
  placa mixtă) → următoarea tăietură mare e atenția, nu cozile înmulțirilor.

## 4. Ce se face în sesiune

Cod (doar adăugare) în `nvidia_ga104::wave_model`: timpii pe val ai `mm_tc_v4` din os-443 și predicția unei liste de produse;
în `persistent_study`: podeaua lansării duble pentru `mm_tc_v4`. Test T250 pe procesor (rândurile os-443 / os-445 / os-431 citite
din `tests/rapoarte/`). Nimic nou pentru placă; la final, o cerere opțională către OS: profilul pasului mixt (aceeași comandă cu
profilul pornit), ca să înlocuiască DEDUS-ul de aici cu o măsurătoare.
