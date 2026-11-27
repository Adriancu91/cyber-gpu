# C-157 p.2 — P2zzt: `token32_v3`, LayerNorm calculat în înmulțirea care urmează

Sesiunea Cyber-GPU 277, 27 nov 2026, în cloud, fără placă. Planul și predicțiile L1–L7 (scrise înainte de cod, commit `4650561`):
`docs/C-157_TOKEN32_V3_LN.md`. Teste: T292 (`crates/nvidia-ga104/tests/sim_c157_token32_v3.rs`, SIMULARE + listare) și testele din
`nvidia_ga104::token_ln`.

## Ce e
`token32_v2` (aceleași citiri prin L1, aceeași barieră) cu **cele 25 de `layernorm256` lipite în înmulțirea care le citește ieșirea**
(ln1 → qkv, ln2 → fc, ln final → logit-uri). os-495 a arătat că acolo stau ~0,46 ms din 1,67: un singur grup făcea fiecare LayerNorm.
Acum fiecare grup își calculează singur media și abaterea (exact adunările lui `layernorm256`, aceeași ordine) și ține în registre
cele 24 de valori normalizate pe care le folosește la toate rândurile lui; grupul 0 scrie și ieșirea LayerNorm-ului (aceleași valori).
Lista: **88 → 63 de operații** pe simbol la #513 (25 de bariere mai puțin).

Compilat cu Mesa NAK 25.0.7 (`token32_v2` reconstruit identic, `5f4d22a0…`): **10 548 de instrucțiuni, 97 de registre** (v2: 92),
0 memorie locală, aceeași memorie partajată (2 176 B) → 16 grupuri pe SM încap (G = 768 pe 48 de SM);
SHA-256 `8ecf4fed9ec987af4de86aeba024556f00ce1a7310842e46fa0e10b6e5c5b719`. În listare: bariera aceeași (4 `cctl`, 2 adunări
atomice), +80 `shfl.bfly` și +1 `mufu.rsq` (= un LayerNorm în plus, la început), toate scrierile tot `strong.sys`.

## Ce cerem OS-ului (opțiune nouă, oprită implicit)
`--token32-v3`, doar imediat după `--token32-v2`; rulează după `token32_v2` din același lanț, la G = 384 și G = 768 (ca v2: 768
doar după 384), aceleași texte / simboluri. Rândul: `test-placa --tablou-viteza --token32 --token32-v2 --token32-v3`.

Pentru fiecare simbol (`nvidia_ga104::token_ln`, doar adăugări; `kronos_token` și `Prog` neschimbate):
1. programul `TOKEN32_V3` (verifică `TOKEN32_V3_SHA256`);
2. lista: **`let (list, n) = token_ln::work_list_ln(&launches)?`** — `n` (numărul de operații, 63 la #513) e cel care merge în
   bufferul constant și în ținta contorului, **nu** `launches.len()`; listele a două simboluri au aceeași lungime, deci
   `kronos_token::work_list_changes` merge la fel ca la v2;
3. bufferul constant: `kronos_token::token_cbuf_for(list_va, sync_va, n, groups)` (ca la v2);
4. QMD: **`token_ln::token32_v3_qmd_wide(prog_va, cbuf_va, groups, sms)`** — NU `kronos_token::token32_qmd_wide` (acela scrie 92 de
   registre; programul folosește 97);
5. trece = aceleași simboluri și aceeași amprentă ca `token32_v2`, contorul `(n − 1) · G`. Dacă amprenta diferă, `token32_v3` se
   oprește (ca v2) și scrie pe ce poziție.

Rândul de jurnal dorit (ca la v2): `kt token32_v3 G=… placa … us pe simbol (mediana), operatii n=63, …`.

## Ce așteptăm (DEDUS, `docs/C-157_TOKEN32_V3_LN.md` §2)
L1 aceiași biți (16 simboluri, `0x6c8501f0`), contorul 62 · G; L3 placa pe simbol la G = 768 **1,18–1,38 ms** (v2: 1,666);
L4 cu driverul ~505–560 simb/s (lista 484); L5 G = 768 mai rapid decât G = 384; L6 puține cuvinte schimbate între simboluri.

## Ce e nou pentru placă
Doar programul `token32_v3` (o singură lansare pe simbol, ca v2): aceleași feluri de citiri și scrieri în memoria noastră, aceeași
barieră. **Niciun registru nou al plăcii, nicio scriere MMIO.** Interfața: doar adăugări (modulul nou `token_ln`; în `gpu-sim`
modelul `token32v3_on_cpu`).
