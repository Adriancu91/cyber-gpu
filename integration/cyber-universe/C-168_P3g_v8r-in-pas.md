# C-168 p.3 P3g — `mm_tc_v8r` legat în pas unde regula alege v8 (gata, testat doar în SIMULARE)

Sesiunea Cyber-GPU 292, 28 nov 2026, în cloud, fără placă. De ce și ce prezicem: `docs/C-168_V8R.md` §6 (scris înainte de cod).
Testul: T309 (`crates/nvidia-ga104/tests/sim_c168_p3g_v8r_in_pas.rs`, SIMULARE).

## Ce e
os-505 a arătat că `mm_tc_v8r` dă aceiași biți ca `mm_tc` și e ×1,09–1,11 mai rapid ca `mm_tc_v8` pe aceleași forme. Acum:
**orice produs pe care regula valurilor îl dă lui v8 merge pe v8r**. Regula și vitezele (`RATES_OS500`) nu se schimbă, deci
împărțirea v7 / v8 / v8n rămâne aceeași; `mm_tc_v8n` și variantele GELU' nu sunt atinse.

Rust (doar adăugări): `train_step::TcPick::V8r` (numele `mm_tc_v8r`), `train_step::pack_mixed_fast_attn11` și
`batch_ram::pack_mixed_fast_via_ram_attn11` = variantele `attn10` cu încă un argument, **`v8r_va`** (pus după `gelu_fast`):
adresa la care ați încărcat `mm_tc_v8r` (aliniată la 256; **0 = neîncărcat = exact `attn10`, octet cu octet**). QMD-ul unei
lansări V8r: programul v8r, **120 de registre** (`mm_tc_v8r::REGISTERS`), blocul de 256, grila și memoria comună ale lui v8
(`mm_tc_v8r::grid` / `BLOCK` / `SHARED`), același cbuf. Refuzat înainte de trimitere: adresă nealiniată; `v8r_va ≠ 0` fără `v8`.

## Ce cerem OS-ului (când are placa)
1. O opțiune nouă, de ex. `--pas-v8r` (cu `--pas-v8`): încărcați `mm_tc_v8r` (SHA-256 `6e28d531…274bf4`, același binar ca în
   os-505) și împachetați cu `pack_mixed_fast_attn11` / `pack_mixed_fast_via_ram_attn11` cu `v8r_va` = adresa lui.
   `fast_picks_text` / rândul alegerilor să numere și `mm_tc_v8r`.
2. **Întâi lot 2 cu oracolul** (acolo sunt cele 24 de lansări v8), apoi lot 6 (acolo nu se schimbă nimic: 0 lansări v8).
   Rândul, lângă cel din os-507: `… --pas-v8 --pas-v8-garda --pas-atentie-fwd-tc --pas-v8r` (lot 2), apoi cu `--lot-513 6`.
3. Se poate pune împreună cu P5a / P5b (notele `C-168_P5a…`, `C-168_P5b…`): cu P5b la lot 2 trec pe v8r doar 12 (celelalte 12
   sunt `mm_tc_v8_gelu`, rămân așa).

## Așteptat (scris înainte)
- **K5:** lot 2: `mm_tc_v8r` **5,2–5,4 ms** pe 24 de lansări (os-507: `mm_tc_v8` 5,81 ms) → pasul −0,4…−0,7 ms; oracolul lot 2
  aceleași numere.
- **K6:** lot 6 neschimbat (±0,5 ms), alegerile identice (0 v8).

## Ce e nou pentru placă
Doar folosirea lui `mm_tc_v8r` în pas (programul a rulat deja în proba os-505, 7 / 7 aceiași biți). Nicio scriere în registre,
niciun firmware, nicio schimbare la programele vechi. Nicio schimbare incompatibilă (doar adăugări; `attn10` neschimbat).
**Atenție la QMD:** 120 de registre (nu 128 ca v8).
