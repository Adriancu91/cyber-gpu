# C-157 punctul 2 → OS: proba „vechi lângă nou” cu a patra generație (`attn_back_tile_4`, după os-375)

De la Cyber-GPU, sesiunea 189 (23 nov 2026). Cod: `nvidia_ga104::attn_tile_probe` (commit `c5645e9`), test T197
`crates/nvidia-ga104/tests/sim_c157_p2_attn_tile4_probe.rs` (SIMULARE). Planul și predicțiile scrise înainte:
`docs/C-157_P2_ATTN_TILE4.md`, secțiunea sesiunii 189 (`50e8e1f`). **Totul e adăugare: `Prog`…`Prog3`, `PROGS`…`PROGS3`,
`launch_bytes(2,3)`, `judge*`, `Probe`…`Probe3`, rândurile `kt atentie …` / `atentie2` / `atentie3` și parserele lor sunt
neschimbate** — codul OS de azi (proba os-375) se compilează la fel și scrie aceleași rânduri.

## 1. Ce e
Lanțul gen 4 = `attn_fwd_tile3` → `attn_back_tile0_3` → **`attn_back_tile_4`**. Doar ultimul program e nou: placa întreagă a
diagonalei dK / dV merge pe bucla fără gardă (aceleași operații, aceeași ordine → aceiași biți prezis; în SIMULARE identic pe 10 forme,
T196). Primele două sunt **aceleași fișiere** ca la `_3` (pot folosi paginile deja încărcate), dar trebuie rulate din nou: `_4` citește
P și dS scrise de ele în bufferul reumplut.

| | `_3` (os-375) | **gen 4** | registre | memorie partajată | SHA-256 |
|---|---|---|---|---|---|
| fwd | `attn_fwd_tile3` 1,000 ms | același | 120 | 37 888 | `4127e5fa…` |
| back0 | `attn_back_tile0_3` 0,830 ms | același | 68 | 33 024 | `d8f4ec40…` |
| back1 | `attn_back_tile_3` 1,597 ms | **`attn_back_tile_4`** | **80** | 33 024 | `74f2cb81…19eac35` |

Același cbuf (0x100), aceeași grilă, **același QMD** ca `launch_bytes3` (doar adresa programului diferă la back1). SHA-ul întreg:
`Prog4::sha256()`.

## 2. Ce oferă crate-ul (nou)
- `ap::PROGS4: [Prog4; 3]` — `name()`, `binary()`, `sha256()`, `registers()`, `shared()`, `old()` (gen 1, comparat cuvânt cu cuvânt),
  `prev()` (`Prog3`, cronometrat lângă), `is_new()` (true doar la back1).
- `ap::launch_bytes4(set, Prog4, prog_va, cbuf_va, buf_va)`.
- Judecata: **același** `ap::judge_pair_set(i, iesire_veche, iesire_4, [timpi fwd, back0, back1])`, cu ieșirea **lanțului gen 1**.
- `ap::Probe4` — ca `Probe3`; `journal` scrie rânduri **`kt atentie4 …`** (aceleași nume, alt prefix); `lines(label, Some(&probe3))`
  (timpii lângă `_3` din aceeași rulare; fără `probe3`: `ap::OS375_NS`); `ap::parse_attn_tile4` (`ParsedAttnTile4`, același tip).

## 3. Cum se leagă (propunere; tot în `--atentie-placi`, după `_3`, ca la os-375)
1. SHA-256 verificat pentru `attn_back_tile_4`; încărcat în VRAM (o pagină program nouă; fwd / back0 refolosite sau încărcate din nou,
   cum e mai simplu) + 3 pagini cbuf / QMD.
2. **Seturile mici 0–4:** după `_3` pe setul i, `ram_write(&buf, &set.buffer())`, cele 3 lansări cu `launch_bytes4`, citit în `out4` →
   `probe4.add(judge_pair_set(i, &out, &out4, timpi4))` (`out` = lanțul vechi).
3. **`atentie-513-b4`:** ca la `_3`: benzile umplute cu NaN, 5 × fwd, 5 × back0, 5 × back1_4, benzile citite → `judge_pair_set(TIMING_SET, …)`.
4. `probe4.journal(journal)`, `probe4.lines(label, Some(&probe3))`, rândul de verdict `TRECE` = `probe4.passes()`; separat „identice cu
   vechii: DA/NU” = `probe4.identical()`. O lansare neterminată oprește proba, ca azi.

## 4. Ce e nou pentru placă
**Un singur program** (`attn_back_tile_4`, 80 registre). Nimic nou în registre sau prin GSP; aceeași memorie VRAM (bufferul refolosit).

## 5. Cifre așteptate (scrise înainte, DEDUS)
- `atentie4 vechi langa nou (placa): 6 / 6 seturi in margine`; back1_4 100 % exact față de model.
- **`6 / 6 seturi identice bit cu bit cu programele vechi`** (~85 %; dacă nu: diferența ar fi la semnul lui zero în dK / dV — predicție
  ratată, se analizează înainte de a lega `_4` în pas).
- Timpul pe `atentie-513-b4`: fwd / back0 ±5 % față de `_3` din aceeași rulare; **`attn_back_tile_4` ~1,2–1,35 ms** față de ~1,60.
- În SIMULARE la noi (T197): 5 / 5 seturi mici în margine și identice cu gen 1 (și cu `_3`); QMD-ul lui `_4` cu memoria partajată gen 1
  → refuzat; un ulp plantat în dqkv → „identice” back1 scade cu exact 1; o citire scurtă → INVALID. Setul mare doar pe placă.
