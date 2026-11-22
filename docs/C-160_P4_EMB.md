# C-160 punctul 4 — programul 8 al antrenării: încorporările (`emb_fwd` înainte, `emb_back` înapoi)

Sesiunea 150, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Primele și ultimele calcule ale pasului de antrenare: intrarea în stratul 0 și gradientul celor două tabele de încorporare.

**Înainte — `emb_fwd`:** x[r, k] = emb_token[simbol[r], k] + emb_pozitie[i, k], pentru rândul r = b·T + i al lotului.
Blocuri de 64 pe k, grila [⌈C/64⌉, T, B]: poziția i = y-ul blocului, textul b = z-ul lui → **nicio împărțire**. O singură
adunare `fadd.ftz` pe element, deci programul dă **exact aceiași biți** ca procesorul.
`tools/nak/emb_fwd.comp` → Mesa NAK 25.0.7: **672 de octeți**, 42 de instrucțiuni, 19 registre; SHA-256
`247e0e42871932822aa201cc6ee8e254ae9e657bec4ce6e30bbd3b691cd5bebb`. Bufferul constant 0: simboluri / emb / poz / x la
0x28 / 0x30 / 0x38 / 0x40, C 0x48, T 0x4c; baza grupurilor 0x0 / 0x4 / 0x8 = 0.

*De ce fără împărțire:* prima variantă lua poziția ca r mod T. NAK a făcut împărțirea cu unitatea aproximativă a plăcii
(`mufu.rcp` + corecții). Am scris pe procesor exact pașii NAK și i-am încercat cu RCP-ul mișcat cu ±2 unități din ultima
cifră: la +2 dă un rest greșit (r = 6, T = 2). Exact rămâne doar dacă RCP-ul plăcii greșește cu cel mult o unitate în sus —
probabil adevărat, dar nedovedit. Am scos împărțirea în loc să ne bazăm pe asta.

**Înapoi — `emb_back`, fără `atomic`:** gradientul unei tabele = sumele rândurilor lui dx care au folosit fiecare rând.
Pe procesor, driverul face o listă de **segmente** (`gpu_compute::layer::EmbSegments`): segmentul s adună rândurile
`rows[off[s] .. off[s+1]]` ale lui dx și pune suma în rândul `dst[s]` al gradientului. Un fir pe (segment, k), un lanț de
adunări de la 0 în ordinea listei: aceiași biți la fiecare rulare; destinațiile sunt distincte, deci două fire nu scriu
niciodată același loc.
- `emb_pozitie`: un segment pe poziție i < T (rândurile b·T + i), modul „scrie”; rândurile ≥ T le pune driverul pe 0;
- `emb_token`: un segment pe **fiecare simbol diferit** din lot (rândurile crescător), modul „adună” peste gradientul
  capului de ieșire legat (dW din `mm_train`) — exact ordinea din `loss_and_grad`: întâi suma, apoi + capul.

`tools/nak/emb_back.comp` → Mesa NAK 25.0.7: **1 280 de octeți**, 80 de instrucțiuni, 19 registre; SHA-256
`fca7b007aebb0b8fbfbdbb209f6d4a7162e579212132b47a85da0524a12af0be`. Bufferul constant 0: dx / off / dst / rows / g la
0x28 / 0x30 / 0x38 / 0x40 / 0x48, C 0x50, modul 0x54 (0 scrie, 1 adună); baza grupurilor 0x0 / 0x4 = 0. Aritmetica: doar
`fadd.ftz` (lanțul, de la 0, și adunarea modului 1); fără MUFU, memorie partajată, barieră → modelul = codul, bit cu bit.

Adăugări (nimic redenumit, OS-ul compilează neschimbat): `nvidia_ga104::{emb_fwd, emb_back}` (`EMB_FWD`, `EMB_BACK`, SHA-256,
`REGISTERS`, `BLOCK`, `CBUF_BYTES`, `Args::{cbuf, grid}`, `emb_back::Mode`), `gpu_compute::layer::{emb_fwd_gpu, EmbSegments,
emb_back_seg_gpu, emb_back_gpu}`, `gpu_compute::train::{embed_rows, embed_back_rows, embed_grad_inputs}` (oracolul =
liniile din `loss_and_grad`, mutate în funcții; T151 neschimbat), `gpu_sim::cmdproc::{emb_fwd_on_cpu, emb_back_on_cpu}`.

## Marginea pentru „înapoi”, scrisă înainte de măsurare (T161)
u = 2⁻²⁴. Pentru un element sumat din m rânduri x₁..x_m: 1,01·(m − 1)·u·Σ|x_j| (lanțul f32; prima adunare 0 + x e exactă)
+ 2,01·u·|valoarea| (rotunjirea oracolului) + (m + 1)·FTZ; în modul „adună” încă 2,01·u·|valoarea| + FTZ (capul adunat o dată
de fiecare). „Înainte” nu are margine: trebuie să fie bit cu bit.

## Ce a ieșit (SIMULARE, programele lansate prin procesorul de comenzi față de oracol)
| Caz | `emb_token` (cât din margine) | `emb_pozitie` (cât din margine) |
|---|---|---|
| forma modelului mic: B 2, T 32, C 64, V 256 | 31,1 % | 0 % |
| C 70 (trece de blocul de 64), B 3, T 5 | 0 % | 48,8 % |
| C 1, B 4, T 9 | 0 % | 0 % |
| un text: B 1, T 17, C 130 | 0 % | 0 % |
| vocabular 3 (fiecare simbol în ~100 de rânduri) | 1,4 % | 25,0 % |
| vocabular 5 000 > lotul (cele mai multe simboluri nefolosite) | 0 % | 0 % |
| gradienți mari (1e6) | 43,4 % | 39,4 % |
| gradienți mici (1e-30, lângă FTZ) | 38,1 % | 33,6 % |

**Pasul de aur mic, cu datele adevărate:** dx și dW-ul capului din oracol (`embed_grad_inputs`), apoi cele două lansări →
față de **gradientul lui PyTorch** de la pasul 1 (`pas_de_aur_mic/sgd/grad_pas1`): abaterea maximă **1,5e-8** la ambele
tabele (toleranța T151: 1e-6); față de oracol 31,3 % / 0 % din margine. „Înainte” pe tabelele și lotul adevărat: **bit cu bit**
egal cu oracolul, la fel pe toate cele 8 forme.

În plus: lansarea = modelul, bit cu bit; două rulări = aceiași biți; simbolurile nefolosite păstrează exact dW-ul capului;
pozițiile ≥ T rămân 0; modul „scrie” ignoră ce era în g (NaN), rândurile din niciun segment nu sunt atinse; grila scurtă
scrie doar elementele ei (la ambele programe); cele două programe pe același procesor de comenzi; refuzuri (simbol în afara
vocabularului, tabel de poziții scurt, destinație repetată, deplasări greșite, rând în afara lui dx, bloc greșit, baza
grupurilor ≠ 0: nimic scris). **8 greșeli plantate prinse** (capul uitat, capul de două ori, pozițiile doar din primul text,
poziția r în loc de i, simbolul deplasat cu unu, ultimul rând din segment pierdut, coloana inversată, media în loc de sumă).

**Ce NU e dovedit:** nimic pe placă. Viteza: un fir pe (segment, coloană), bun pentru pașii de aur; la #651 (vocabular mare,
lot 6 × 512) lista de segmente se face pe procesor la fiecare pas (câteva mii de rânduri — ieftin, DEDUS, nemăsurat).

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Două programe mici, fără nicio unitate aproximativă: `emb_fwd` (grilă 3D, ca `attn_fwd`) și `emb_back` (o buclă cu margini
citite din memorie, o citire indirectă `rows[j]`, o citire-scriere în același loc în modul „adună”). Driverul trebuie să
verifice simbolurile și listele înainte de lansare (`EmbSegments::check`): programele au încredere în ele.

## Ce urmează (punctul 4, mai departe)
Norma și tăierea gradientului, apoi pasul AdamW. Cu ele, punctul 5: pasul de aur mic cap-coadă pe placa simulată.
