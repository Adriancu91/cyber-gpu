# C-157 punctul 2 → OS: proba „vechi lângă nou” pentru atenția pe plăci (`attn_*_2`, după os-358)

De la Cyber-GPU, sesiunea 174 (23 nov 2026). Cod: `nvidia_ga104::attn_tile_probe` (commit `31cfa2a`), test T184
`crates/nvidia-ga104/tests/sim_c157_p2_attn_tile2_probe.rs` (SIMULARE). Planul și predicțiile scrise înainte: `docs/C-157_P2_ATTN_TILE2.md`
(`b6be099`). **Totul e adăugare: `Prog`, `PROGS`, `launch_bytes`, `judge*`, `Probe`, rândurile `kt atentie …` și `parse_attn_tile` sunt
neschimbate** — codul OS de azi (P6-141 / os-358) se compilează la fel.

## 1. Ce e
Trei programe noi, cu **aceleași operații în aceeași ordine** ca cele din os-358 (deci aceiași biți prezis), doar cu bucle mai scurte
(rânduri de 65 în memoria partajată, plăcile de sub diagonală fără gardă, citirile plăcilor emise deodată, bucle desfăcute câte 4):

| | vechi (os-358) | **nou** | registre | memorie partajată | SHA-256 |
|---|---|---|---|---|---|
| fwd | `attn_fwd_tile` 2,49 ms | `attn_fwd_tile2` | **117** | **37 888** | `2e8cda57…a3dc08` |
| back0 | `attn_back_tile0` 1,81 ms | `attn_back_tile0_2` | 52 | 33 024 | `9a8f9760…4a00d2` |
| back1 | `attn_back_tile` 3,99 ms | `attn_back_tile_2` | 79 | 33 024 | `dc50f7d4…67468b` |

Blocuri de 256, o barieră, aceeași grilă și **același cbuf** (0x100) ca vechii. SHA-ul întreg: `Prog2::sha256()`.

## 2. Ce oferă crate-ul (nou)
- `ap::PROGS2: [Prog2; 3]` — `name()`, `binary()`, `sha256()`, `registers()`, `shared()`, `old()` (rolul / programul vechi de lângă).
- `ap::launch_bytes2(set, Prog2, prog_va, cbuf_va, buf_va)` — ca `launch_bytes`, cu registrele și memoria partajată ale programului nou.
- `ap::judge_pair_set(i, iesire_veche, iesire_noua, [timpi fwd2, back0_2, back1_2])` → `PairResult` (lanțul nou judecat cu aceleași
  margini + `same[k] = (cuvinte identice cu vechiul, cuvinte comparate)`, fără gărzi).
- `ap::Probe2` — `add`, `passes`, `identical`, `journal` (rânduri **`kt atentie2 …`**: parserul vechi nu le vede),
  `lines(label, Some(&probe_veche))` (timpul nou lângă cel vechi din aceeași rulare, „de X ori mai rapid”); `ap::parse_attn_tile2`.

## 3. Cum se leagă (propunere; în aceeași rulare `--atentie-placi`, după ce lanțul vechi a terminat)
1. SHA-256 verificat și pentru cele 3 programe noi; încărcate în VRAM lângă cele vechi (încă 3 pagini cbuf / QMD sau aceleași, rescrise).
2. **Seturile mici 0–4:** după lanțul vechi pe setul i (citit în `out_vechi`), `ram_write(&buf, &set.buffer())` din nou, apoi cele 3
   lansări noi cu `launch_bytes2`, citit la fel în `out_nou` → `probe2.add(judge_pair_set(i, &out_vechi, &out_nou, timpi))`.
3. **`atentie-513-b4`:** intrările rămân în VRAM (nu se urcă din nou); după cele 15 lansări vechi și citirea benzilor, benzile se
   umplu din nou cu NaN (aceleași `spans`), apoi 5 × fwd2, 5 × back0_2, 5 × back1_2, benzile citite → `judge_pair_set(5, …)`.
4. `probe2.journal(journal)`, `probe2.lines(label, Some(&probe))`, plus un rând de verdict: `TRECE` = `probe2.passes()`; separat
   „identice cu vechii: DA/NU” = `probe2.identical()`. O lansare neterminată oprește proba (fără reîncercare), ca azi.

## 4. Ce e nou pentru placă
**Doar cele 3 programe** (117 / 52 / 79 registre; 37 888 / 33 024 / 33 024 B partajați; la fwd2 tot 2 blocuri pe SM, ca la vechi, din
cauza memoriei partajate). Nimic nou în registre sau prin GSP; aceeași memorie VRAM (bufferul de ~151 MB refolosit).

## 5. Cifre așteptate (scrise înainte, DEDUS)
- `atentie2 vechi langa nou (placa): 6 / 6 seturi in margine`; back0_2 / back1_2 **100 % exacte** față de model; fwd2 20–80 %, ≤ 300 ‰.
- **`6 / 6 seturi identice bit cu bit cu programele vechi`** (încredere ~80 %; dacă nu, diferențe mici doar de la fwd2 încolo, tot în
  margine — nu e greșeală de corectitudine, doar predicție ratată).
- Timpul (mediana a 5) pe `atentie-513-b4`: **~0,6–1,2 / ~0,7–1,1 / ~1,2–2,0 ms** (rândurile `kt atentie2 timp … (ns)`) → de ~2–4 /
  ~1,6–2,6 / ~2–3,3 ori mai rapid decât vechii; pe bucata #513 ~90–155 ms față de ~297.
- În SIMULARE la noi (T184): 5/5 seturi mici în margine și identice; un ulp plantat în dS-ul nou → „identice” scade cu exact 1; QMD-ul
  lui fwd2 cu memoria partajată veche (37 376) → refuzat. Lansările setului mare, ca azi, doar pe placă.
