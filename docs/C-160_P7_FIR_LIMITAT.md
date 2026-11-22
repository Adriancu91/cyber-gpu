# C-160 punctul 7 — cine e limitat de un fir care face prea mult (după `colsum`)

Sesiunea 162, 22 nov 2026, în cloud, **fără placă**. Totul aici e **DEDUS** (din codul programelor, listele NAK și cifre publice despre
GA104); nimic nu e măsurat pe placă. Planul și predicțiile pentru pasul următor sunt scrise **înainte** de cod.

## De ce
os-349 a arătat că `colsum` e corect pe placă, dar un fir care adună singur toate rândurile ar costa la #513 mai mult decât tot pasul
PyTorch (`docs/C-160_P7_COLSUM_IMPARTIT.md`). Comanda C-160 p.7 cere aceeași privire pentru `ln_back`, `attn_back` și AdamW; am făcut-o
pentru toate programele pasului, în afară de înmulțiri (`mm_train` / `mm_tc`, estimate în `C-160_P7_VITEZA_ESTIMARE.md`) și `colsum`.

## Cum (socoteala: `tools/buget-antrenare/fir_limitat.py`)
Formele lui #513: 12 straturi, C 768, 12 capete, d 64, V 16 384, 98 032 128 de parametri; o bucată = 12 texte × 512 = 6 144 de rânduri,
pe care PyTorch o face în **156 ms** (39 265 simb/s, MĂSURAT de Kronos, `FISA_ANTRENARE.md`). Pentru fiecare program:
- câte fire și ce bloc → câte „valuri” pe placă (GA104: 48 SM; cel mult 48 de warp-uri și **16 blocuri** pe SM, 64 K registre — tabelele
  publice CUDA pentru capacitatea 8.6);
- cel mai lung lanț al unui fir (iterații ale buclelor; NAK nu desface buclele: o citire pe iterație, ~6 instrucțiuni) × cât așteaptă o
  iterație (L1 ~35, L2 ~200, memoria plăcii ~450 de cicluri — cifre publice de microbenchmark pentru Ampere, nemăsurate la noi), dar nu
  mai puțin decât îi lasă ceilalți warp-i ai planificatorului;
- octeții mutați, la ~450 GB/s (aceeași cifră ca în `colsum_timp.py`).
Timpul unei lansări = cel mai mare dintre cele două. **Nesiguranța: un factor 2–3.**

## Rezultatul (DEDUS)

| Program | Lansări pe bucată | Ce limitează | ~ms pe bucată |
|---|---|---|---|
| `ln_back` | 25 | memoria (un bloc de 256 pe rând, 3 elemente pe fir) | ~5 |
| `opt_step` AdamW | 1 | memoria (un fir pe parametru, 28 de octeți) | ~6 |
| `gnorm` (2 lansări) | 1 | memoria | ~1 |
| `xent` | 1 | memoria (un bloc pe rând) | ~3 |
| `gelu_tr` înapoi | 12 | memoria (un fir pe element) | ~6 |
| `emb_back` simboluri | 1 | **firul**, dar puțin: lanțul = de câte ori apare cel mai des simbol în bucată (~500 presupus) | ~12 |
| **`attn_fwd`** | 12 | **firul**: un fir pe rând (b, h, i), lanț până la 512 × 64 × 2 iterații, rândul lui P citit de 64 de ori | **~90** |
| **`attn_back` modul 0** | 12 | la limită fir / memorie: un fir pe rând, lanț 512 × 64 | **~34** |
| **`attn_back` modul 1** | 12 | **firul**: un warp pe poziție, coloanele lui dS și P citite pe sărite | **~133** |
| **Total (fără înmulțiri și `colsum`)** | | | **~290 = ~185 % din toată bucata PyTorch** |

**Concluzia:** `ln_back`, AdamW și norma **nu** sunt limitate de un fir — au destule fire și stau aproape de podeaua memoriei (împreună
~12 ms, ~8 % din bucată). **Atenția este**: ~257 ms din cei ~290, mai mult decât toată bucata PyTorch. Două cauze:
1. **Blocuri de 32 de fire.** GA104 ține cel mult 16 blocuri pe SM → doar 16 warp-uri din 48 posibile; la `attn_back` modul 1 asta
   înseamnă 96 de valuri în loc de 32.
2. **Un fir pe rând** (`attn_fwd`, `attn_back` modul 0): firul i face (i + 1) × 64 de pași în lanț; rândurile lungi (i ≈ 511) țin
   placa ocupată cât rândurile scurte s-au terminat de mult.

Ținta pentru atenție, ca înmulțiri pe plăci (2 înainte + 4 înapoi, doar triunghiul): 174 GFLOP pe bucată → **~12–17 ms**.
`emb_back` (~12 ms) depinde de text (cât de des apare cel mai frecvent simbol) — de privit după atenție.

## Pasul următor (plan scris ÎNAINTE de cod): atenția cu blocuri de 128
Cea mai ieftină schimbare, fără risc pentru numere: **același cod pe fir, aceeași ordine a fiecărei sume → aceiași biți**, doar mai multe
fire într-un bloc.
- `attn_fwd.comp` și `attn_back.comp`: `local_size_x = 128` (4 warp-uri).
  - `attn_fwd` și `attn_back` modul 0: grila devine [⌈T/128⌉, H, B]; firul i = `gl_GlobalInvocationID.x` ca azi, `i ≥ T` iese.
  - `attn_back` modul 1: patru capete într-un bloc — h = 4·`WorkGroupID.x` + (lid >> 5), e = lid & 31, …; grila [⌈H/4⌉, T, B];
    `h ≥ heads` iese (fără barieră, deci ieșirea timpurie e sigură; modelul mic are H = 4, #513 H = 12).
- Programe noi, cu nume noi (`attn_fwd128`, `attn_back128`), lângă cele vechi (OS-ul compilează neschimbat; `attn_fwd` / `attn_back`
  rămân, cu SHA-256 lor). Mesa NAK 25.0.7, ca originalele.
- Modelul pe procesor **nu se schimbă** (`attn_fwd_gpu`, `attn_back_gpu`): fiecare ieșire e tot lanțul unui singur fir.

**Predicții (scrise acum):**
1. Pe toate formele din T160 (atenția înainte) și T159 (înapoi), placa simulată cu programele de 128 = **bit cu bit** cele de 32
   (P, att, dS, dqkv) — certitudine aproape deplină, fiindcă aritmetica fiecărui fir e aceeași.
2. Registrele: tot ~25 (aceleași calcule; doar indicii se schimbă) — **≤ 32**, ca 48 de warp-uri pe SM să încapă.
3. Greșelile plantate din T159/T160 sunt prinse și de programele noi; plus o grilă prea mică (de ex. [⌈T/256⌉, …], sau la modul 1
   [H/4] rotunjit în jos) lasă rânduri / capete nescrise → prinse de gărzile NaN din ieșire.
4. Viteza (DEDUS, doar pe placă): `attn_fwd` ~90 → ~37 ms pe bucată, `attn_back` modul 1 ~133 → ~44 ms, modul 0 neschimbat (~34);
   atenția ~257 → ~115 ms. Tot de ~7–10× peste țintă → apoi atenția „pe plăci” (pasul de după, plan separat).

## Ce NU face
Nimic pe placă. Nu schimbă programele vechi și nici ordinea pe placă (pasul de aur mic F32 → proba `mm_tc` → `colsum_part`).
Programele de 128 intră pe placă doar când ajungem la viteza pasului, ca probă separată.

## Surse
- Limitele SM 8.6 (16 blocuri, 48 de warp-uri, 64 K registre pe SM): ghidul public de programare CUDA, tabelul capacităților de calcul.
- Latențele L1 / L2 / memorie: ordine de mărime publice din articolele de microbenchmark pentru Ampere (de ex. „Dissecting the Ampere
  GPU Architecture through Microbenchmarking”); nemăsurate pe placa noastră.
- Formulele atenției: ca în `C-160_P4_ATTN_FWD.md` / `C-160_P4_ATTN_BACK.md` (PyTorch BSD, llm.c MIT — doar ideile).
