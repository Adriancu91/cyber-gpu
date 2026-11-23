# C-157 punctul 2 → OS: proba „vechi lângă nou” cu a treia generație (`attn_*_3`, după os-364)

De la Cyber-GPU, sesiunea 182 (23 nov 2026). Cod: `nvidia_ga104::attn_tile_probe` (commit `d885efd`), test T191
`crates/nvidia-ga104/tests/sim_c157_p2_attn_tile3_probe.rs` (SIMULARE). Planul și predicțiile scrise înainte:
`docs/C-157_P2_ATTN_TILE3.md`, secțiunea sesiunii 182 (`b6e3b5b`). **Totul e adăugare: `Prog`, `PROGS`, `Prog2`, `PROGS2`,
`launch_bytes(2)`, `judge*`, `Probe`, `Probe2`, rândurile `kt atentie …` / `kt atentie2 …` și parserele lor sunt neschimbate** —
codul OS de azi (P6-149 / os-364) se compilează la fel și scrie aceleași rânduri (verificat: rândurile și liniile lui `Probe2`
identice înainte / după, 192 / 192).

## 1. Ce e
Trei programe noi, cu **aceleași operații în aceeași ordine** ca `_2` (deci aceiași biți prezis), doar cu **citirile din memoria
globală fără ramură** (adresa strânsă, citirile întâi, alegerea 0 / valoare după) — os-364 a arătat că la `_2` fiecare citire păzită
aștepta singură:

| | `_2` (os-364) | **`_3`** | registre | memorie partajată | SHA-256 |
|---|---|---|---|---|---|
| fwd | `attn_fwd_tile2` 1,53 ms | `attn_fwd_tile3` | **120** | 37 888 | `4127e5fa…e59642bc` |
| back0 | `attn_back_tile0_2` 1,54 ms | `attn_back_tile0_3` | **68** | 33 024 | `d8f4ec40…3b611f` |
| back1 | `attn_back_tile_2` 2,25 ms | `attn_back_tile_3` | **80** | 33 024 | `67c8d9a5…ba085a9c` |

Blocuri de 256, o barieră, aceeași grilă și **același cbuf** (0x100) ca vechii; aceeași memorie partajată ca `_2` (QMD-ul diferă doar
prin registre). Tot 2 / 3 / 3 blocuri pe SM, ca la `_2`. SHA-ul întreg: `Prog3::sha256()`.

## 2. Ce oferă crate-ul (nou)
- `ap::PROGS3: [Prog3; 3]` — `name()`, `binary()`, `sha256()`, `registers()`, `shared()`, `old()` (programul gen 1 cu care se compară
  cuvânt cu cuvânt), `prev()` (programul `_2` lângă care se cronometrează).
- `ap::launch_bytes3(set, Prog3, prog_va, cbuf_va, buf_va)` — ca `launch_bytes2`, cu registrele lui `_3`.
- Judecata: **același** `ap::judge_pair_set(i, iesire_veche, iesire_3, [timpi fwd3, back0_3, back1_3])`, cu ieșirea **lanțului vechi
  (gen 1)** ca `iesire_veche` (`out`, cum o aveți deja).
- `ap::Probe3` — `add`, `passes`, `identical`, `in_margin`, `identical_sets`, `per_prog`, `same_per_prog`, `timing_skipped`,
  `journal` (rânduri **`kt atentie3 …`**, aceleași nume ca la `atentie2`, alt prefix; parserele `atentie` / `atentie2` nu le văd),
  `lines(label, Some(&probe2))` (timpul `_3` lângă `_2` din aceeași rulare, „de X ori mai rapid”; fără `probe2`: medianele os-364,
  `ap::OS364_NS`); `ap::parse_attn_tile3` (întoarce `ParsedAttnTile3`, același tip ca la `_2`).

## 3. Cum se leagă (propunere; tot în `--atentie-placi`, exact ca la P6-149, încă o generație după `_2`)
1. SHA-256 verificat și pentru cele 3 programe `_3`; încărcate în VRAM lângă celelalte (încă 3 pagini program + 3 cbuf / QMD).
2. **Seturile mici 0–4:** după `_2` pe setul i, `ram_write(&buf, &set.buffer())` din nou, apoi cele 3 lansări `_3` cu `launch_bytes3`,
   citit în `out3` → `probe3.add(judge_pair_set(i, &out, &out3, timpi3))` (`out` = lanțul vechi).
3. **`atentie-513-b4`:** ca la `_2`: benzile umplute din nou cu NaN (aceleași `spans`), 5 × fwd3, 5 × back0_3, 5 × back1_3, benzile
   citite → `judge_pair_set(TIMING_SET, &out, &out3, …)`.
4. `probe3.journal(journal)`, `probe3.lines(label, Some(&probe2))`, plus rândul de verdict: `TRECE` = `probe3.passes()`; separat
   „identice cu vechii: DA/NU” = `probe3.identical()`. O lansare neterminată oprește proba (fără reîncercare), ca azi.

## 4. Ce e nou pentru placă
**Doar cele 3 programe** (120 / 68 / 80 registre). Nimic nou în registre sau prin GSP; aceeași memorie VRAM (bufferul refolosit).
Citirile strânse rămân **în interiorul bufferelor** (testat în SIMULARE: un indice în afara bufferului oprea emularea; n-a oprit-o).

## 5. Cifre așteptate (scrise înainte, DEDUS)
- `atentie3 vechi langa nou (placa): 6 / 6 seturi in margine`; back0_3 / back1_3 **100 % exacte** față de model; fwd3 ca fwd2.
- **`6 / 6 seturi identice bit cu bit cu programele vechi`** (~85 % pe program, ~65 % toate trei; dacă nu: diferențe mici, tot în
  margine — predicție ratată, nu greșeală de corectitudine).
- Timpul pe `atentie-513-b4` (rândurile `kt atentie3 timp … (ns)`): **~0,7–1,1 / ~0,5–0,9 / ~1,6–2,0 ms** (modelul: 0,81 / 0,55 /
  1,60) față de 1,53 / 1,54 / 2,25 la `_2` → lanțul ~2,9 ms față de 5,32 (`_2`) și 8,29 (gen 1).
- În SIMULARE la noi (T191): 5 / 5 seturi mici în margine și identice cu gen 1 (și cu `_2`); un ulp plantat în dqkv-ul `_3` →
  „identice” la back1 scade cu exact 1; QMD-ul lui fwd3 cu memoria partajată gen 1 (37 376) → refuzat; o citire scurtă → INVALID.
  Lansările setului mare, ca azi, doar pe placă.
