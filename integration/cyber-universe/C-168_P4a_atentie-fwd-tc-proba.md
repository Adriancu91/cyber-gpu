# PREDARE C-168 p.4a → cyber-universe: proba `--atentie-fwd-tc` (atenția înainte pe circuitele speciale, lângă `attn_fwd_tile3`)

Sesiunea Cyber-GPU 285, 28 nov 2026, în cloud, FĂRĂ placă. Planul și predicțiile: `docs/C-168_ATTN_FWD_TC_PLAN.md` §4 (H4, H5) și
§10 (scrise înainte de orice rulare pe placă). Testul nostru: T301 (`crates/nvidia-ga104/tests/sim_c168_attn_fwd_tc_probe.rs`),
PASS — **SIMULARE**. Nimic de aici n-a rulat pe RTX 3070 Ti.

## Ce e
`attn_fwd_tc` = atenția înainte (P și att) pe circuitele speciale, în două treceri, cu P scris o singură dată. **Aceeași treabă și
aceleași argumente ca `attn_fwd_tile3`** (qkv, p, att, T, C, H, d, scale), dar argumentele stau la **0x700** într-un cbuf de **0x800**,
grila e **în perechi** [⌈T/64⌉ / 2, H, B] și programul ia **doar formele aliniate** (d = 64, T multiplu de 64, adrese multiplu de 16 B).
Aritmetica e f16 (ca atenția înapoi pe circuite), deci verdictul e o **margine față de `attn_fwd_tile3`**, nu aceiași biți.

| | `attn_fwd_tc` |
|---|---|
| binar | `tools/nak/out2/attn_fwd_tc.sm86.sass` (Mesa NAK 26.3.1), 19 952 B |
| SHA-256 | `ce2b2f22ee58c6e50091f532e7520b4493bf04555a2e2b2f8acfc1ae7e929257` (`attn_fwd_tc::ATTN_FWD_TC_SHA256`) |
| registre / fire / memorie comună | 120 / 256 / 27 648 B, 1 barieră |
| treapta memoriei comune | **64 KiB** (2 blocuri pe SM); QMD-ul îl dă gata `attn_fwd_tc_probe::launch_bytes` |

**P după program:** scris în toate plăcile până la placa diagonală inclusiv (0 exact deasupra diagonalei în ea); **plăcile de deasupra
plăcii diagonale NU se scriu** (`attn_fwd_tile3` scria 0 acolo; înapoi nu le citește — verificat în SIMULARE, T299). În probă ele trebuie
să rămână NaN.

## Ce vă dăm în Rust (doar adăugări: `nvidia_ga104::attn_fwd_tc_probe`)
- `SETS` (5): patru seturi mici **aliniate** citite întregi (`tc-t64`, `tc-t192-b2`, `tc-t256-h2`, `tc-t320-ascutit`, fiecare buffer
  ≤ 2,1 MB, ca cele ale probei atenției) + `atentie-513-b4` (același set ca în proba atenției, citit pe benzi). **Din cele 5 seturi mici ale probei atenției niciunul
  nu e aliniat** (`t130`, `d40`, `d16`, `t200`, `t96`), de aceea seturile noi. Același tip `ProbeSet`, aceeași așezare
  (`layout`, `buffer`, `write_inputs`); se folosesc doar qkv, P și att.
- `fits(&set)`; `fwd_ranges(&set)` / `fwd_read_len(&set)` = ce umpleți cu NaN (`FILL`) **înainte de fiecare program** și ce citiți
  după (rândurile lui P, garda, rândurile lui att, garda).
- `launch_bytes(&set, prog_va, cbuf_va, buf_va)` → (cbuf 0x800, QMD la treapta 64 KiB) pentru `attn_fwd_tc`.
  `attn_fwd_tile3` se lansează ca în proba atenției: `attn_tile_probe::launch_bytes3(&set, Prog3(Prog::Fwd), …)`.
- `judge(&set, citire_tile3, citire_tc, ns_tile3, ns_tc)` → `SetResult` (O, P, zerourile, plăcile nescrise, gărzile, pe seturile mici
  și câte cuvinte = modelul pe procesor); `Probe` cu `lines("placa")` (rândurile de raport, gata scrise) și `parse` (le citim noi).

## Ce vă rugăm (proba „vechi lângă nou”)
Opțiunea `--atentie-fwd-tc` (sub opțiune, oprită implicit). Pe fiecare set din `attn_fwd_tc_probe::SETS`:
1. bufferul setului (inputurile ca în proba atenției; la `atentie-513-b4` se poate refolosi bufferul ei din VRAM);
2. `fwd_ranges` umplute cu NaN → `attn_fwd_tile3` (la `atentie-513-b4`: 5 lansări, mediana) → citirea;
3. `fwd_ranges` umplute iar cu NaN → `attn_fwd_tc` (la fel, 5 lansări la `atentie-513-b4`) → citirea;
4. `judge(...)`, apoi rândurile din `Probe::lines("placa")` în raport. SHA-256 verificat înainte de orice trimitere; o lansare
   neterminată oprește proba (fără reîncercare).

Locul în rând e alegerea voastră (ca `--atentie-tc`, după `--atentie3`); de exemplu:
`test-placa --tablou-viteza --atentie-placi --atentie2 --atentie3 --atentie-fwd-tc`

## Ce așteptăm (scris înainte de placă; DEDUS din SIMULARE + socoteli)
- **H4 (80 %):** 5 / 5 seturi `TRECE`: O ≤ 1·10⁻³ și P ≤ 2·10⁻³ față de `attn_fwd_tile3`, zerourile da, nescrise da, gărzi neatinse.
  În SIMULARE (aceeași aritmetică, `ex2` exact): O 3,7–6,1·10⁻⁴ și P 2,3–5,4·10⁻⁴ pe seturile mici (cel mai aproape de prag:
  `tc-t320-ascutit`, O 6,1·10⁻⁴), la `atentie-513-b4` O 1,9·10⁻⁴, P 5,2·10⁻⁵. Pe placă cifrele pot ieși puțin altfel (`ex2` / `rcp`
  aproximative și la `tile3`); „exacte față de model” va fi **sub** total (nu e cerință).
- **H5 (45 %):** la `atentie-513-b4`, `attn_fwd_tc` **0,37–0,62 ms** pe lansare față de `attn_fwd_tile3` ~1,00 ms (os-364…375) →
  ×1,6–2,7. (Aceeași fereastră ca H5 la lot 6, 550–930 µs pe 3 valuri, trecută la 2 valuri.)
- **Prag scris:** dacă H4 trece și timpul e peste 0,62 ms, frâna e în trecerea 2 (S prin memoria comună, 5 bariere), nu în produse —
  pasul următor la noi e o probă de aranjare a acumulatorilor, nu legarea în pas.

## Ce e nou pentru placă
Doar programul `attn_fwd_tc` (SHA-256 de mai sus), la treapta 64 KiB (aceleași câmpuri pe care le folosiți deja pentru `--treapta-pas`).
Nicio scriere în registre, niciun firmware. **Nicio schimbare de interfață** (doar adăugări: modulul `attn_fwd_tc_probe`).
Legarea în pas (`TcPick` pentru atenția înainte, pasul 4) o pregătim noi abia după rezultatul acestei probe.
