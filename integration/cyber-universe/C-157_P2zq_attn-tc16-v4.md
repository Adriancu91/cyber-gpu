# PREDARE C-157 P2zq → cyber-universe: `attn_back_tile_tc16_v4` și `_v4pf` (atenția înapoi f16, citiri de 128 de biți)

**Ce e.** Leacul pentru viteza văzută la os-457 (tc16 0,87 ms la `atentie-513-b4`, > 90 % așteptare de citiri). Două programe noi,
**aceeași aritmetică și aceleași valori în memoria partajată ca `attn_back_tile_tc16`** → trebuie să dea **aceleași cuvinte** ca tc16:

| Program | Modul | Registre | Partajată | Ce schimbă |
|---|---|---|---|---|
| `attn_back_tile_tc16_v4` | `nvidia_ga104::attn_back_tile_tc16_v4` | 72 | 16 384 B, 1 barieră | 4 + 4 citiri de 128 de biți pe fir pe placă (în loc de 32 scalare), 5 în zbor |
| `attn_back_tile_tc16_v4pf` | `nvidia_ga104::attn_back_tile_tc16_v4pf` | 104 | 16 384 B, 1 barieră | ca v4 + placa următoare citită înainte de produse |

Același cbuf ca tc16 (argumentele la 0x700: `attn_back_tile_tc::cbuf`, sau modulul `::cbuf`), aceeași grilă [⌈T/64⌉, H, B], blocuri de
256. Fișierele: `tools/nak/out2/attn_back_tile_tc16_v4{,pf}.sm86.sass`, SHA-256 în constantele `*_SHA256` ale modulelor.

**Regula de lansare — obligatorie:** doar când `fits(&args)` (sau `fits_cbuf(&cb)`) spune da: T și d = C / H multipli de 4, adresele
qkv, p, dout, dS multipli de 16 octeți. Altfel tc16. O citire de 128 de biți nealiniată ar fi o eroare pe placă. `cbuf()` întoarce
`None` pentru formele nealiniate. Din proba voastră `--atentie-tc` iau 5 / 6 seturi (nu `t130-d64`, T = 130), dacă bufferul setului
începe la o adresă multiplu de 16 (testul nostru folosește începutul de pagină).

**Ce vă rugăm (proba „vechi lângă nou”, ca la os-457):** pe seturile care trec `fits`, lansați tc16, v4 și v4pf pe aceleași intrări și
scrieți: (1) cuvintele lui v4 și v4pf = ale lui tc16? (prezis DA, V6); (2) timpul pe lansare la `atentie-513-b4` (mediana a 5)
pentru tc16 / v4 / v4pf. Prezis (DEDUS, V7): **0,35–0,55 ms** față de 0,87; v4 cel puțin cât v4pf (55 %). Dacă niciunul nu e
×1,3 față de tc16, cauza nu erau citirile — ne spuneți și căutăm altundeva.

**Ce e nou pentru placă:** doar aceste două programe. Nicio scriere în registre, niciun firmware, aceeași lansare ca tc16.
**O diferență de citire (nu de valori):** un vector care atinge diagonala lui P / dS citește cel mult 3 numere de deasupra ei, în
același rând (în buffer); sunt aruncate prin alegere, deci și un NaN acolo dă aceiași biți (testat în SIMULARE).

**Nicio schimbare de interfață:** doar adăugări (`attn_back_tile_tc16_v4`, `attn_back_tile_tc16_v4pf`, în `gpu-sim`
`attn_back_tile_tc16_v4{,pf}_on_cpu` + `ATTN_BACK_TILE_TC16_V4{,PF}_NEEDS`). Încă nelegate în pasul mixt (`TcPick`): după
măsurătoarea voastră legăm varianta câștigătoare.

**Test la noi:** T258 (`crates/nvidia-ga104/tests/sim_c157_attn_tc16_v4.rs`), PASS — SIMULARE. **Nota întreagă:**
`docs/C-157_ATTN_BACK_TC16_V4.md`. Nimic de aici n-a rulat pe RTX 3070 Ti.
