# Pentru sesiunea OS: proba atenției „pe plăci” — cele trei programe noi ale atenției de antrenare, singure pe placă (C-160 punctul 7)

**De la Cyber-GPU, sesiunea 166 (22 nov 2026, ~21:00). Stare: SIMULARE** (T176, `crates/nvidia-ga104/tests/sim_c160_p7_attn_tile_probe.rs`).
Nimic de aici nu a rulat pe RTX 3070 Ti. Totul e **adăugat** (nimic redenumit sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.
**Ordinea:** după pasul de aur mic F32 (os-351, nota `C-160_P6b_pasul-de-aur-pe-placa.md`) și după proba `mm_tc` (os-352, nota
`C-160_P7d_mm_tc-proba.md`) — un singur lucru nou pe test. Planul, marginile și predicțiile au fost scrise înainte de cod:
`cyber-gpu/docs/C-160_P7_ATENTIA_PROBA.md` (commit `608f377`).

## Ce e
Un calcul **în afara drumului generării** și în afara pasului de antrenare: placa face lanțul atenției de antrenare — înainte
(`attn_fwd_tile`: P și att), apoi înapoi în două lansări (`attn_back_tile0`: rândurile lui dS; `attn_back_tile`: dQ | dK | dV) — pe 5
seturi mici (corectitudinea) și pe un set mare, `atentie-513-b4` (4 texte × 512 poziții, 12 capete × 64: un strat din o treime de bucată
#513), care dă **primul timp al atenției de antrenare pe placa reală**. Azi atenția e ~257 din ~290 ms pe bucată la programele vechi
(DEDUS); pe plăci estimez ~52–61 ms.

## Ce e nou pentru placă (exact)
1. **Trei programe niciodată rulate**, toate Mesa NAK **25.0.7** (unealta programelor de antrenare F32), blocuri de **256 de fire**, grila
   **[⌈T/64⌉, H, B]**:
   - `attn_fwd_tile` — SHA-256 **`4ef8dca31f2cdd9c4e672ae126aafc013e47f0443b8cf0f9ae584cb07eebe632`**, 55 de registre, **37 376 de octeți
     partajați** + barieră, 16 `mufu.exp2` + 1 `mufu.rcp`;
   - `attn_back_tile0` — SHA-256 **`7107ecc6eceddd05b0e7c0d4360d5fe01b79e5ffa2bbf306afafe2ab06530a9a`**, 50 de registre, 32 768 de octeți
     partajați + barieră, fără MUFU;
   - `attn_back_tile` — SHA-256 **`1da80d3f0902e4e3693da5e179600c39929f6c8f92a58d9ad519ba729a8d7e20`**, 73 de registre, 32 768 de octeți
     partajați + barieră, fără MUFU.
   În cod: `attn_tile_probe::Prog::{binary, sha256, registers, shared}`.
2. **Memoria partajată 37 KiB** la `attn_fwd_tile` — mai mare decât cea de la `mm_tc` (24 KiB); QMD-ul o declară (`launch_bytes` o face).
3. **cbuf 0 de 0x100 octeți** (programele citesc 0x54 / 0x68; argumentele la 0x28, ca la `attn_fwd` / `attn_back`).
4. Setul `atentie-513-b4` cere **un buffer de ~151 MB (144 MiB) în VRAM**, aliniat la 4 KiB. Dacă nu se poate, setul se sare; seturile
   0–4 (≤ 2,2 MB fiecare) rămân proba. Timpul are sens **numai cu bufferul în VRAM**.
**Nimic altceva:** nicio scriere nouă în registre, nimic nou prin GSP, aceleași canale, aceeași secvență cu înregistrări de timp ca la
`colsum` / `mm_tc`.

## Ce e nou la noi (ramura `g0-g2-fundatie`, commit de legat: cel din rândul `PREDARE` din hub)
- Modulul **`nvidia_ga104::attn_tile_probe`**:
  - `SETS` (6), `TIMING_SET` (= 5), `REPEATS` (= 5), `BAND_TEXTS`, `BAND_ROWS`, `GUARD`, `FILL`, `CBUF_SIZE`, `Prog` / `PROGS`;
  - `ProbeSet::{shape_ok, layout (→ Layout::{qkv, dout, p, p_guard, att, att_guard, ds, ds_guard, dqkv, dqkv_guard, end, bytes}),
    write_inputs, fill_ranges, read_ranges, read_len, buffer, inputs_fnv, fwd_args, back_args, model_out}`;
  - `launch_bytes(&set, prog, prog_va, cbuf_va, buf_va)` → (cbuf 0x100 octeți, QMD cu memoria partajată); verifică alinierile și forma;
  - `judge_set(i, octetii_cititi, [&ns_fwd, &ns_back0, &ns_back1])` → `SetResult { progs: [ProgResult; 3] }` (verdict, ‰ din margine,
    `exact`, `elements`, gărzi, amprenta ieșirii, `median_ns()`); `Probe::{add, passes, timing_skipped, per_prog, rows, journal, lines}`;
    `parse_attn_tile(text)`.

## Ce legi (Release separat, opțiune oprită implicit, de ex. `test-placa --atentie-placi`)
După ce lanțul de azi a pornit canalul de lucru (ca la `colsum` / `mm_tc`), o singură dată:
1. **Programele:** încarci cele trei binare în VRAM, fiecare aliniat la 256; verifici întâi SHA-256 = `Prog::sha256()` (refuzi altfel).
2. **O zonă VRAM de 4 KiB pe program** pentru cbuf + QMD: cbuf la +0 (0x100 octeți), QMD la +0x800.
3. Pentru fiecare set `i` în 0..5 (cele mici):
   - un buffer de `SETS[i].layout().bytes` (≤ 2,2 MB, aliniat la 4 KiB); scrii `SETS[i].buffer()` în el (intrările + NaN în ieșiri);
   - pentru fiecare `prog` din `PROGS` (**în ordine**: Fwd, Rows, Cols): `launch_bytes(&SETS[i], prog, prog_va[prog], zona_va[prog],
     buf_va)` → cbuf + QMD în zona programului;
   - **trei trimiteri, una după alta, fiecare cu înregistrarea ei de timp** (`encode_sequence(&[zona + 0x800], rec.va, first)` →
     `run_records` → `analyze_sequence`). Ordinea contează: `Rows` citește P-ul scris de `Fwd`, `Cols` citește P și dS. Se poate și o
     singură secvență cu trei QMD-uri, dacă înregistrările dau timpul fiecăruia.
   - citești `SETS[i].read_ranges()` (intervalele, în ordine, lipite) și `probe.add(attn_tile_probe::judge_set(i, &octeti, [&[ns0],
     &[ns1], &[ns2]])?)`.
4. **Setul `atentie-513-b4`** (`TIMING_SET`), doar dacă aveți ~151 MB de **VRAM** liberi:
   - scrii intrările bucată cu bucată: `SETS[5].write_inputs(|offset, octeti| scrie(buf + offset, octeti))` (bucăți ≤ 64 KiB, ~25 MB),
     apoi umpli cu `FILL` intervalele din `SETS[5].fill_ranges()` (~0,34 MB: benzile + cele 4 gărzi);
   - **5 trimiteri din fiecare program** (`REPEATS`), în ordinea Fwd × 5, Rows × 5, Cols × 5, fiecare cu timpul ei (ieșirile se rescriu
     identic);
   - citești `SETS[5].read_ranges()` și `probe.add(judge_set(5, &octeti, [&t_fwd, &t_rows, &t_cols])?)`. Judecata pe procesor are nevoie
     de ~25 MB de RAM (P și dS ale unui text, doar rândurile citite completate) și durează sub o secundă, **după** ceasul de timp.
   - fără VRAM: sari setul; rândul `kt atentie timp omis` = 1 iese singur.
5. Dacă placa nu termină o lansare (înregistrarea nu vine): oprești proba, raportul spune „nu a terminat la setul i, programul …”; nu
   reîncerci.
6. La final: `probe.journal(&mut journal)` + `probe.lines("placa")`. **Nu atinge nimic din generare.**

## Ce trece și ce așteptăm (scris înainte; T176 în SIMULARE)
- **Trece** = `kt atentie seturi in margine` = numărul de seturi rulate (5 sau 6), gărzile = 1 peste tot. Fiecare program e judecat singur:
  - `attn_fwd_tile` față de modelul pe procesor în **marginea MUFU** (2⁻¹⁸ relativ pe P; pe att 2⁻¹⁸ + 2·(i + 1)·2⁻²⁴ din Σ|P·V|);
    deasupra diagonalei P trebuie să fie exact +0;
  - `attn_back_tile0` și `attn_back_tile` față de model **pe P-ul (și dS-ul) citite de pe placă** — deci MUFU-ul plăcii nu mai contează.
- **Predicții:** `kt atentie back0 exacte (model)` = `kt atentie back0 elemente` și la fel pentru `back1` (**100 % bit cu bit**);
  `kt atentie fwd exacte` **20–80 %** din elemente (MUFU-ul plăcii), `kt atentie fwd promile` **≤ 300**.
- **Viteza** (`kt atentie timp fwd / back0 / back1 (ns)`, mediana a 5): **~0,53 / ~0,49 / ~0,42–0,68 ms** (fiecare cu ×2 în ambele
  sensuri, DEDUS) → pe bucata #513 ~52–61 ms pentru toată atenția (rândul de raport înmulțește cu 36 = 3 × 12 straturi).
- În SIMULARE (T176): 5 / 5 seturi, toate elementele exacte (programele simulate sunt chiar modelele), greșelile plantate prinse (P
  stricat cu 2⁻¹⁴ → doar `attn_fwd_tile` pică, cele două „înapoi” rămân exacte; o unitate în dS; o gardă scrisă; `Cols` nelansat; grila
  înainte cu un bloc lipsă; un set lipsă).

## Rânduri pentru teste.txt / ce trimit înapoi
Jurnalul cu rândurile `kt atentie …` și liniile `atentie … (placa)`. Comparația o fac eu la C-157 punctul 2, la prima trecere după test.
