# C-157 — `attn_back_tile_tc16_v4` / `_v4pf` legate în pasul mixt rapid (alegerea pe formă)

Sesiunea 253, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**.
Nimic de aici nu a rulat pe RTX 3070 Ti; tot ce iese pe procesor e **SIMULARE**, timpii sunt **DEDUS**.

## 1. De ce

Sesiunea 248 a scris două variante ale programului al doilea al atenției înapoi pe circuitele speciale (dQ, dK, dV, f16):
`attn_back_tile_tc16_v4` (citiri de 128 de biți, 72 registre) și `attn_back_tile_tc16_v4pf` (plus placa următoare citită în
avans, 104 registre). Ambele dau aceiași biți ca `attn_back_tile_tc16` în SIMULARE (T258) și merg **doar pe forme aliniate**.
Proba P2zq (os-465, `--atentie-tc16-v4`) va spune pe placă care e mai rapidă. Pasul mixt rapid (`pack_mixed_fast_attn3`) știe azi
o singură adresă pentru acest program (`AttnTcProgram::Tc` sau `Tc16`). Legătura se face acum în SIMULARE, pentru ambele
variante, ca OS-ul să încarce doar câștigătoarea după P2zq, fără să mai aștepte o sesiune Cyber-GPU. Aceeași regulă ca la
`mm_tc_v4` și `attn_back_tile0_tc_v4` (P2zu): varianta rapidă unde forma e aliniată, altfel programul vechi.

## 2. Ce va fi (plan, doar adăugări)

- `train_step::AttnTcFast` — `V4(adresa)` sau `V4pf(adresa)`: varianta rapidă încărcată de OS (doar una, câștigătoarea P2zq).
- `TcPick::AttnTc16V4` și `TcPick::AttnTc16V4pf` (numele programelor).
- `pack_mixed_fast_attn4(…, attn, attn_fast: Option<AttnTcFast>, attn0, …)`: o lansare `MixedLaunch::AttnTc` merge la varianta
  rapidă dacă e încărcată și `attn_back_tile_tc16_v4::fits_cbuf` spune da, altfel la `attn` (refuz dacă lipsește). QMD-ul
  rapid: registrele variantei (72 / 104), memoria partajată a lui tc16 (16 KiB, o barieră), blocuri de 256, aceeași grilă, același
  cbuf. `pack_mixed_fast_attn3` devine `pack_mixed_fast_attn4(…, None, …)` (aceiași octeți).
- `batch_ram::pack_mixed_fast_via_ram_attn4` (pentru `--lot-prin-ram`), la fel.
- Testul nou `crates/nvidia-ga104/tests/sim_c157_attn_tc16_v4_mixed.rs` (T263).

## 3. Predicțiile (scrise ÎNAINTE de cod)

- **W1 — fără varianta rapidă nimic nu se schimbă.** `pack_mixed_fast_attn4` cu `None` = `pack_mixed_fast_attn3` octet cu octet
  (mini și #513 lot 2).
- **W2 — lista mini, tc16 + v4.** Cele 2 lansări `AttnTc` aleg `AttnTc16V4`; zona are aceeași lungime; diferă doar octeții
  adresei programului și ai numărului de registre din cele 2 QMD-uri (≤ 2 × 10 octeți). Cu v4pf la fel, registrele 104.
- **W3 — antrenarea mini (5 pași) cu v4 = cu v4pf = cu tc16, bit cu bit** (pierderi, gradientul pasului 1, greutăți), cu ambele
  programe „înapoi” ale atenției pe circuite (dS pe `attn_back_tile0_tc`).
- **W4 — lista #513 (lot 2).** 12 `AttnTc16V4` (sau 12 `AttnTc16V4pf`) + 12 `Attn0TcV4` + 147 `mm_tc_v4`; zona cât la attn3;
  QMD-ul: adresa variantei, 72 / 104 registre, 16 384 B partajată, grila lui tc16.
- **W5 — forma nealiniată cade pe tc16.** Adresa dS + 4 → `AttnTc16`; refuz dacă doar varianta rapidă e încărcată; adresă
  nealiniată la 256 → refuz; un QMD mutat de mână pe v4 la o formă nealiniată → înlocuitorul din `gpu-sim` refuză.
- **W6 — prin RAM.** `pack_mixed_fast_via_ram_attn4` cu `None` = attn3 octet cu octet; cu v4 aceleași alegeri ca fără RAM.
- **W7 — timpul (DEDUS, doar pe placă).** Cât arată P2zq pentru câștigătoare; estimarea veche ×1,3–2,0 pe acest program la
  `atentie-513-b4`, încredere mică. NOT TESTED.

## 4. Rezultatul (sesiunea 253, ~16:55; T263 pe procesor = SIMULARE; W7 DEDUS)

| # | prezis | ieșit | verdict |
|---|---|---|---|
| W1 | attn4 fără varianta rapidă = attn3 octet cu octet | exact așa, pe mini și pe #513 lot 2; nicio alegere rapidă | CONFIRMATĂ |
| W2 | mini: 2 × varianta rapidă; aceeași lungime; doar adresa + registrele diferă (≤ 20 octeți) | 2 × `AttnTc16V4` (la fel 2 × `AttnTc16V4pf`), zona 94 464 B, **4 octeți** diferiți, toți în cele 2 QMD-uri (adresa + registrele 72 / 104; tc16 avea 48) | CONFIRMATĂ |
| W3 | antrenarea mini 5 pași, v4 = v4pf = tc16 bit cu bit | **2 301 760 de valori** aceiași biți, cu v4 și cu v4pf, și cu tc16 neîncărcat; pierderea 9,7128 → 8,8200 | CONFIRMATĂ |
| W4 | #513 lot 2: 12 rapide + 12 dS + 147 `mm_tc_v4`; zona cât attn3; QMD: adresa, 72 / 104 registre, grila tc16 | așa, cu o abatere de montaj: testul a încărcat doar `attn_back_tile0_tc` pentru dS, deci 12 `Attn0Tc` (nu `Attn0TcV4`; alegerea dS e cea din T262, neatinsă); zona 554 240 B; grila [8, 12, 2]; fiecare QMD rapid diferă de cel tc16 în ≤ 10 octeți; toate cbuf-urile trec `fits_cbuf` | CONFIRMATĂ |
| W5 | nealiniat → tc16; doar rapidă → refuz; adresă nealiniată la 256 sau 0 → refuz; QMD mutat de mână pe v4 → înlocuitorul refuză | exact așa (refuzul: „forma nealiniata pentru attn_back_tile_tc16_v4 si attn_back_tile_tc nu e incarcat”; în pas: „SIM: attn_back_tile_tc16_v4: forma nealiniata”); pasul nealiniat merge pe tc16 până la capăt | CONFIRMATĂ |
| W6 | prin RAM: fără rapidă = attn3 octet cu octet; cu rapidă aceleași alegeri | exact așa (v4 și v4pf) | CONFIRMATĂ |
| W7 | cât arată P2zq | — (doar pe placă) | NOT TESTED |

Cod (doar adăugări): `nvidia_ga104::train_step::{AttnTcFast, TcPick::AttnTc16V4, TcPick::AttnTc16V4pf, pack_mixed_fast_attn4}`,
`nvidia_ga104::batch_ram::pack_mixed_fast_via_ram_attn4`; `pack_mixed_fast_attn3` / `pack_mixed_fast_via_ram_attn3` cheamă acum
varianta 4 cu `None` (aceiași octeți, W1 / W6). Test: `crates/nvidia-ga104/tests/sim_c157_attn_tc16_v4_mixed.rs`.
Nota pentru OS: `integration/cyber-universe/C-157_P2zv_attn-tc16-v4-legat.md`.

### Ce înseamnă (pe scurt)

1. Ambele programe „înapoi” ale atenției au acum varianta rapidă legată în pasul mixt: OS-ul încarcă doar câștigătoarea probei
   și pasul o alege singur unde formele permit — fără vreo schimbare în rezultat (aceiași biți, în SIMULARE).
2. Pe placă nu e nimic nou față de P2zq; legarea se folosește doar după ce P2zq arată aceleași cuvinte și un câștig de timp.
