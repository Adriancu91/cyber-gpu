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
