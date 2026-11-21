# C-160 punctul 4 — programul 5 al antrenării: GELU înainte și înapoi (`gelu_tr`)

Sesiunea 147, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Programul care trece pasul de antrenare prin GELU-ul din MLP-ul lui Kronos, în ambele direcții, cu un singur binar:
- **modul 0 (înainte):** y = u·Φ(u) — activarea, păstrată pentru drumul înapoi prin `proj`;
- **modul 1 (înapoi):** y = g·(Φ(u) + u·φ(u)) — du din gradientul activării (dact), prin derivata lui GELU.

Φ vine din erf după Abramowitz & Stegun 7.1.26 (ca `gelu.comp` al generării), iar φ = e^(−u²/2)/√(2π) refolosește **aceeași
exponențială** e^(−a²), a = |u/√2|: un singur `MUFU.RCP` și un singur `MUFU.EX2` pe element. Modul 0 nu citește g. Un fir pe element,
blocuri de 256, fără memorie partajată și fără barieră.

`tools/nak/gelu_tr.comp` → Mesa NAK 25.0.7: **912 octeți**, 57 de instrucțiuni, 19 registre; SHA-256
`52dd5f9a81fd88a864d77ebbeab5a3cc3ab537b89362f1dd36424e21c4ae54f1`. Bufferul constant 0: u / g / y la 0x28 / 0x30 / 0x38, n la 0x40,
modul la 0x44 (0x48 octeți).

Ordinea (citită din `tools/nak/out/gelu_tr.sm86.txt`; modelul `gpu_compute::layer::gelu_tr_gpu` o urmează): z = `fmul`(u, 1/√2);
a = |z|; t = `MUFU.RCP`(`ffma`(a, 0,3275911, 1)); polinomul A&S prin `ffma` de la c5; p = `fmul`(t, ·); ex = `MUFU.EX2`(−(a·log₂e)·a);
e = `ffma`(−p, ex, 1); erf = ±e după semnul lui z; Φ = `fmul`(`fadd`(erf, 1), 0,5); înainte: `fmul`(u, Φ); înapoi:
`fmul`(g, `ffma`(u, `fmul`(ex, 1/√2π), Φ)). Toate cu FTZ. Unitățile MUFU ale plăcii sunt aproximative, deci modelul le ia corect rotunjite
și **placa nu se compară bit cu bit**, ci în margine. Oracolul: `gpu_compute::train::gelu_rows` / `gelu_back_rows` (adăugate; exact
liniile din `loss_and_grad`, care au dat numerele PyTorch la pasul de aur mic, T151).

## Marginea, scrisă înainte de măsurare (T158)
u = 2⁻²⁴. t: 4u + 2⁻²² (loc pentru RCP-ul plăcii); polinomul: |P′(t)|·t·Δt + 12u·Σ|cᵢ|tⁱ; ex: 5u·a² (argumentul) + u + 2⁻²² (loc pentru
EX2-ul plăcii); erf: **1,5e-7 (eroarea formulei A&S)** + ΔP·ex + P·ex·Δex + u + deplasarea argumentului; Φ: ½(Δerf + u·|1 + erf|);
înainte: |u|·ΔΦ + u·|y|; înapoi: |g|·(ΔΦ + |u|·φ·(Δex + 2u) + u·|d|) + u·|y|; plus FTZ.

## Ce a ieșit (SIMULARE, modelul față de oracol)
| Caz | înainte (cât din margine) | înapoi (cât din margine) |
|---|---|---|
| u ∈ [−12, 12], 10 007 puncte | 75,6 % | 69,1 % |
| u ∈ [−1, 1], 3 001 puncte | 12,6 % | 14,8 % |
| valori speciale (0, ±subnormale, ±14, ±20, ±50, ±1e4) | 51,4 % | 37,6 % |
| 4 096 de valori ca ieșirea `fc` (\|u\| ≤ 6) | 75,3 % | 75,6 % |
| lanțul înainte → înapoi (două lansări, 16 401 elemente) | 75,5 % | 81,0 % |

Cea mai mare abatere absolută pe [−12, 12] (g = 1): **4,8e-7 înainte, 1,8e-7 înapoi**. Partea mare a marginii e chiar eroarea formulei A&S
(aceeași ca la generare). **Modelul încape și în marginea fără locul pentru unitățile plăcii** (≤ 76 %), deci o placă ale cărei `MUFU.RCP` /
`MUFU.EX2` sunt în 2⁻²² de cele exacte rămâne în marginea întreagă.

Și: modelul lansat = modelul pe element, bit cu bit; două rulări = aceiași biți; modul 0 dă aceiași biți oricare ar fi g; grila scurtă lasă
coada neatinsă; refuzuri (bloc greșit, grilă 2D: nimic scris); 7 greșeli plantate prinse (derivata fără u·φ; φ fără 1/√2π; GELU cu tanh în loc
de erf; modurile inversate în ambele sensuri; Φ în loc de u·Φ; erf fără semn) + modelul altui program înregistrat pentru acest cod.

**Găsit pe drum:** oracolul `train::erf64(0)` nu se oprea niciodată (toți termenii seriei sunt 0, testul de oprire nu era atins). Antrenarea
nu a dat de el (u nu e exact 0 în practică), dar un test cu u = 0 rămânea agățat. Reparat: întoarce ±0. Numerele pasului de aur nu se schimbă.

**Ce NU e dovedit:** cât de exacte sunt `MUFU.RCP` și `MUFU.EX2` pe placa noastră (marginea lasă 2⁻²² fiecăreia); viteza (un element pe fir,
citiri și scrieri alăturate — neglijabil la pașii de aur).

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Un program nou, `gelu_tr`, cel mai simplu fel de QMD (ca `gelu` / `vadd`: 256 de fire, fără memorie partajată, fără barieră; bufferul
constant 0x48 octeți), un `MUFU.RCP` + un `MUFU.EX2` pe element (aceleași unități ca `gelu` al generării, care a rulat deja pe placă).

## Ce urmează (punctul 4, mai departe)
Atenția înapoi, apoi încorporările, norma + tăierea, AdamW.
