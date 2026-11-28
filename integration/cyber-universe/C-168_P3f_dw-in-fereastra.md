# C-168 p.3 — dW-ul lui v8 / v8n / v8r în fereastra de viteză (cererea centrului din 28 nov 11:10, după os-505)

Sesiunea Cyber-GPU 292, 28 nov 2026, în cloud, fără placă. Socoteala: `docs/C-168_OS505_OS507.md` §3 și T308
(`crates/nvidia-ga104/tests/calc_c168_os505_fereastra.rs`, CALCUL pe orarul măsurat din `gpu-jurnal-200.txt`; nimic pe placă).

## Ce s-a întâmplat
os-505 a rulat 5 însoțitori (v6, v7, v8, v8n, v8r), fiecare simplu și @64k, pe qkv / dX / dW. Fiecare program pe setul dW ține
~114 s (procesorul judecă rezultatul), deci lanțul a ținut 2 429 s, iar ultimele 5 seturi dW (v8@64k, v8n, v8n@64k, v8r, v8r@64k)
au ieșit din fereastra de 1 800 s. Regulile de rând (P6-263: v7 cere v6, v8 cere v7, v8r cere v8) nu lasă scoaterea însoțitorilor.

## Ce cerem OS-ului (doar o opțiune nouă în probă, nimic nou pe placă)
**Varianta A (propunerea noastră):** un fanion nou, de ex. `--mm-tc-doar-64k`, valabil doar cu `--mm-tc-64k`: însoțitorii
(`v4` … `v8r`) rulează **doar la treapta 64 KiB** (fără rularea lor simplă); `mm_tc` rămâne simplu + @64k, ca reper pentru biți.
Pasul lansează totul la 64 KiB, deci nu pierdem nicio cifră folosită. Rândul:

`test-placa --tablou-viteza --viteza-1800 --mm-tc --mm-tc-v6 --mm-tc-v7 --mm-tc-v8 --mm-tc-v8r --mm-tc-64k --mm-tc-dxdw --mm-tc-doar-64k`

**Varianta C (dacă A nu vă convine):** un fanion `--mm-tc-doar-dw` (după seturile mici, doar `qkv-513-dw`), într-un rând separat.
Varianta „dW întâi” (B) scoate în afară toate cele 10 seturi dX ale însoțitorilor, deci n-o recomandăm.

## Așteptat (scris înainte)
- **W1:** 0 seturi în afara ferestrei; ultimul dW gata la ~1 270 s după cererea acceptată; lanțul întreg ~1 430 s (azi 2 429 s). DEDUS din os-505.
- **W2:** qkv / dX la @64k ca în os-505 în ±3 % (v8r 0,981 / 0,933 ms; v8n 1,015 / 0,948 ms).
- **W3:** qkv-513-dw v8r@64k **1,40–1,60 ms** (aceeași bandă ca V4).
- **W4:** qkv-513-dw v8n@64k 1,35–1,55 ms.

## Ce e nou pentru placă
Nimic: aceleași programe, aceleași SHA-256, mai puține rulări. Nicio scriere în registre, niciun firmware. Nicio schimbare la noi
(doar documentul și testul de calcul).
