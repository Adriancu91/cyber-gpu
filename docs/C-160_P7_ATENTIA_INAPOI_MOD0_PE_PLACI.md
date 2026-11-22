# C-160 punctul 7 — atenția „înapoi” pe plăci, modul 0 (rândurile lui dS): `attn_back_tile0`

Sesiunea 165, 22 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Totul despre viteză e **DEDUS**; corectitudinea se verifică doar în **SIMULARE**.

## De ce
Cu `attn_fwd_tile` (~19 ms) și `attn_back_tile` modul 1 (~15–24 ms), modul 0 al lui `attn_back` (~34 ms pe bucată la #513,
`fir_limitat.py`) rămâne cea mai mare parte a atenției. Un fir ține un rând întreg (b, h, i): pentru fiecare j ≤ i citește tot
rândul lui V (d valori) și tot rândul lui dO, fără refolosire între fire; lanțul cel mai lung are 512 × 64 iterații.

## Planul (scris ÎNAINTE de cod)
Program nou `tools/nak/attn_back_tile0.comp` (Mesa NAK 25.0.7), **aceleași argumente ca `attn_back`** (același cbuf de 0x68
octeți). Face **doar modul 0**: cu modul ≠ 0 iese fără să scrie nimic. Ordinea pe placă: `attn_back_tile0` (modul 0) →
`attn_back_tile` (modul 1); `attn_back` / `attn_back128` rămân neschimbate.
- **Blocuri de 256 de fire (16 × 16), un bloc = 64 de rânduri** (b, h, i0 … i0 + 63); grila [⌈T/64⌉, H, B]. Firul (tx, ty)
  ține rândurile i0 + ty + 16·a și coloanele jt·64 + tx + 16·k (a, k = 0…3): 4 × 4 valori dP pe fir. Două plăci de 64 × 64 în
  memoria partajată (**32 KiB**).
- **Trecerea 1, pentru fiecare placă de coloane jt = 0 … bx (până la diagonală):**
  1. placa dO[i0.., e] și placa V[jt·64.., e] în memoria partajată (zerouri în afara lui T și d);
  2. dP_ij = lanț `ffma` din 0, **e = 0 … d − 1 crescător** (bucla se oprește la d, nu la 64: un `ffma(0, 0, −0)` ar schimba
     semnul lui zero), 4 × 4 pe fir; se scrie în dS (buffer global) doar pentru j ≤ i < T;
  3. barieră; placa dP (din registre) și placa P[i0.., jt·64..] (citită pe rânduri, contiguu) trec în memoria partajată;
     barieră; **64 de fire (câte unul pe rând) continuă lanțul dot = ffma(P_ij, dP_ij, dot)**, jj = 0 … 63 crescător, doar
     j ≤ i. Plăcile vin în ordinea jt crescătoare, deci lanțul e exact cel din `attn_back` (din 0, j = 0 … i).
- **Trecerea 2:** dot-ul fiecărui rând trece prin memoria partajată (barieră); apoi, pentru fiecare placă jt = 0 … ⌈T/64⌉ − 1,
  **fiecare fir își recitește propriile valori dP** (aceeași împărțire ca la scriere, deci fără dependențe între fire prin
  memoria globală) și scrie dS_ij = `fmul`(`fmul`(P_ij, `fadd`(dP_ij, −dot)), scala) pentru j ≤ i, 0 pentru i < j < T.
- **Fiecare valoare e aceeași ca în `attn_back` modul 0:** același lanț pentru dP, același lanț pentru dot, aceleași trei operații
  pentru dS. Fără MUFU, fără `atomic`, fără sume împărțite.
- **Modelul pe procesor:** nu e nevoie de unul nou — fiecare rând = `gpu_compute::layer::attn_back_row_gpu`. Simulatorul:
  `gpu_sim::cmdproc::attn_back_tile0_on_cpu` (blocuri de 256, d ≤ 64, grila nu trece de ⌈T/64⌉ / H; modul ≠ 0 → nimic scris).
  Modulul `nvidia_ga104::attn_back_tile0` (doar adăugări; folosește `attn_back::Args`).
- **Verificarea:** ca la T174, testul are și o **emulare literală a codului GLSL** pe procesor (blocuri, fire, plăci, barierele
  ca faze, aceleași bucle și margini), comparată bit cu bit cu `attn_back` modul 0 — simulatorul singur ar repeta doar afirmația.

## Predicții (scrise acum)
1. **dS bit cu bit egal cu `attn_back` modul 0** (blocuri de 32), pe toate formele cu d ≤ 64: modelul mic, T 70 / C 96 / H 3,
   d 40, d 1, T 5 / B 3, softmax ascuțit, gradienți mici, T 129, T 200 cu d 64, T 64 exact — inclusiv semnul zerourilor. Deci
   lanțul `attn_back_tile0` → `attn_back_tile` = `attn_back_gpu` bit cu bit (dS și dqkv).
2. P deasupra diagonalei (NaN, 1e30 acolo) nu schimbă niciun bit; zerourile de deasupra diagonalei sunt +0.
3. Registre ≤ 80 (16 acumulatori + indici; ca `attn_back_tile`, 73), deci 3 blocuri de 256 pe SM; memorie partajată 32 768 B;
   nicio `atomic`, niciun MUFU, nicio memorie locală; exact 2 `ffma` în bucle (dP și dot) dacă bucla 4 × 4 nu e desfăcută, altfel
   16 + 1.
4. Greșeli plantate prinse: grila cu ⌈T/64⌉ − 1 blocuri (rânduri nescrise → gărzile NaN), blocuri ≠ 256 refuzate, d > 64
   refuzat, grila peste H refuzată, modul 1 → nimic scris; o emulare cu bucla e până la 64 (nu la d) diferă la semnul zerourilor
   pe o formă construită cu d < 64 în care produsele unui rând sunt sub-normale (aduse la −0 de FTZ, deci dP = −0); o emulare cu lanțul dot în altă ordine (pe plăci, apoi adunate) diferă.
5. **Viteza (DEDUS, doar pe placă; `fir_limitat.py`, `print_tiled_back0`):** la #513, ~2,7 GFLOP calculate pe lansare și ~490 MB
   mutați (dP scris și recitit, P citit de două ori pe triunghi, dS scris întreg) → **~12–20 ms pe bucată** (interval 8–40), față
   de ~34 azi. Lanțul dot (64 de `ffma` dependente pe placă, doar 2 warp-uri din 8) adaugă cel mult ~25 % la calcul. Cu toate
   trei plăcile, atenția întreagă ar scădea de la ~257 la ~50–65 ms pe bucată.

## Rezultatul (SIMULARE, T175, `crates/nvidia-ga104/tests/sim_attn_back_tile0.rs`; codul după planul `80968c3`)
- **Binarul** (Mesa NAK **25.0.7**, reconstruit în cloud; `attn_back_tile` a ieșit identic, `1da80d3f…`):
  `attn_back_tile0.sm86.sass` SHA-256 **`7107ecc6eceddd05b0e7c0d4360d5fe01b79e5ffa2bbf306afafe2ab06530a9a`**, 21 024 de octeți
  (1 314 instrucțiuni), **50 de registre**, 32 768 de octeți partajați (în QMD), **17 `ffma`** (16 în bucla dP, unul pe rezultat,
  + 1 în lanțul dot), 16 × (`fadd`, `fmul`, `fmul`) în trecerea 2 — aceleași trei operații ca `attn_back`, NAK nu le-a contopit;
  nicio `atomic`, niciun MUFU, nicio memorie locală.
- **Adresele „rotite” în memoria partajată** (rândul r rotit cu r poziții): lanțul dot citește o coloană din 32 de rânduri și
  scrierile pe rânduri cad pe 32 de bănci diferite. Schimbă doar adresele, nu valorile; emularea din test folosește aceleași adrese.
- **Predicția 1 (bit cu bit = `attn_back` modul 0): DA** pe toate cele 10 forme (modelul mic, T 70 / d 32, d 40, T 5 / B 3, d 1,
  softmax ascuțit, gradienți 1e-4, T 129, #513 cu T 200 și d 64, T 64 exact): emularea literală a GLSL = `attn_back` = lansarea
  prin procesorul de comenzi; dS-ul emulat dat modului 1 dă dqkv-ul lui `attn_back` bit cu bit; față de oracolul f64 cu P în
  f64, fiecare parte ≤ 2⁻¹⁶ din cea mai mare valoare (ca T159 / T174).
- **Predicția 2: DA** — NaN și 1e30 deasupra diagonalei în P nu schimbă niciun bit; zerourile de acolo sunt +0.
- **Predicția 3: DA** — 50 de registre (≤ 80; 4 blocuri pe SM după registre, 3 după cei 32 KiB partajați → 24 de warp-uri pe
  SM), 32 KiB, fără `atomic` / MUFU / memorie locală; 17 `ffma` (varianta „16 + 1” din predicție).
- **Predicția 4: DA** — grila cu un bloc lipsă lasă exact rândurile 128 și 129 nescrise (2 × 2 × 130 de gărzi pe rând, ca
  emularea); blocuri 128 / 16 × 16, grila peste ⌈T/64⌉ sau peste H, d = 65 refuzate; modul 1 în cbuf → nimic scris (și `shape_ok`
  îl refuză); lanțul dot adunat pe plăci dă alți biți; pe rândul construit (d 32, produse −5·10⁻³⁹, aduse la −0) `attn_back` și
  programul dau dS = −0, bucla până la 64 dă +0 — prinsă.
- `tools/check.sh`: **826 de teste** (+7 ignorate), format, clippy `-D warnings` (Rust 1.97), build no_std — PASS.
- **Viteza: nemăsurată** (doar pe placă). `fir_limitat.py` (`print_tiled_back0`): ~2,7 GFLOP și ~491 MB pe lansare → **~18 ms pe
  bucată** la #513 (interval 8–40, DEDUS), față de ~34 azi; limitat de memorie, nu de calcul. Cu toate trei plăcile atenția ar fi
  ~19 + ~15–24 + ~18 ≈ **52–61 ms pe bucată** în loc de ~257 (DEDUS).

Nepredat pentru placă (intră la viteza pasului, ca probă separată, după pasul de aur mic și `mm_tc`, împreună cu `attn_fwd_tile`
și `attn_back_tile`).

## Ce NU face
Nimic pe placă. Nu schimbă `attn_back` / `attn_back128` / `attn_back_tile` / `attn_fwd*`.

## Surse
- Plăcile în memoria partajată cu 4 × 4 rezultate pe fir: `matmul_2d` / `attn_fwd_tile` / `attn_back_tile` (pașii noștri, după
  siboehm SGEMM_CUDA, MIT, doar ideea); dS = P ∘ (dP − rowsum(P ∘ dP)): FlashAttention-2 (Dao, articol public), llm.c
  `attention_backward` (MIT) — doar formula; noi păstrăm ordinea fixă a sumelor din `attn_back`.
- Limitele SM 8.6: ghidul public CUDA (capacitatea 8.6).
