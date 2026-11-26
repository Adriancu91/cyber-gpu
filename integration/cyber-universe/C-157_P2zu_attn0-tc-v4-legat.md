# C-157 P2zu — `attn_back_tile0_tc_v4` legat în pasul mixt rapid: v4 ales singur pe formele aliniate (sesiunea Cyber-GPU 252, 26 nov 2026)

**PREDARE.** Cod de legătură, **nerulat pe placă**, testat **doar în SIMULARE** (T262). Nota întreagă:
`cyber-gpu/docs/C-157_ATTN_BACK0_TC_V4_LEGAT.md` (planul și predicțiile V1–V7 urcate înainte de cod, `140df52`).

## Ce e

Până acum pasul mixt rapid (`pack_mixed_fast_attn2`, P2zs) știa o singură adresă pentru programul dS pe circuite.
Acum `nvidia_ga104::train_step::pack_mixed_fast_attn3(…, attn, Attn0Programs { tc, v4 }, …)` primește ambele adrese și
alege pe fiecare lansare dS: **`attn_back_tile0_tc_v4` dacă e încărcat și forma e aliniată** (`fits_cbuf`), altfel
`attn_back_tile0_tc` (refuz, fără nimic trimis, dacă cel ales nu e încărcat). QMD-ul e același; diferă doar adresa programului.
Alegerea iese ca `TcPick::Attn0TcV4` (nume „attn_back_tile0_tc_v4”) în lista de alegeri, pentru rândurile din raport.

## Ce e nou la noi (doar adăugări)

- `train_step::{Attn0Programs, TcPick::Attn0TcV4, pack_mixed_fast_attn3}`; `pack_mixed_fast_attn2(…, attn0_va)` =
  `pack_mixed_fast_attn3(…, { tc: attn0_va, v4: 0 })`, aceiași octeți (verificat pe mini și #513).
- `batch_ram::pack_mixed_fast_via_ram_attn3` (pentru `--lot-prin-ram`), la fel.
- `TcPick` are o variantă nouă: strică doar un `match` complet; la voi `fast_picks_text` folosește o listă explicită
  (`[TcPick::V4, …, TcPick::AttnTc16]`) — `Attn0TcV4` (ca și `Attn0Tc`) nu apare în rândul de alegeri până nu-l adăugați acolo.

## Ce ar face OS-ul (propunere; decide sesiunea OS și centrul de comandă)

**Doar după ce proba P2zt trece pe placă** (aceleași cuvinte ca tc, și cel puțin ×1,15 la `atentie-513-b4`): încărcați
`attn_back_tile0_tc_v4` lângă `attn_back_tile0_tc` și treceți la `pack_mixed_fast_attn3` cu ambele adrese (opțiunea care
azi pune `attn0_va`). Rândul propus: `kt pas-mixt attn0: v4 N / tc M`. Pe #513 aștept 12 v4 / 0 tc.
**Aștept (scris înainte):** pierderile și gradientul **identice bit cu bit** cu pasul pe `attn_back_tile0_tc` (aceeași aritmetică);
timpul programului dS ×1,2–1,8 mai mic (V7, DEDUS, încredere mică). Dacă P2zt nu trece, nu schimbați nimic: attn2 rămâne cum e.

**Nou pentru placă:** nimic în plus față de P2zt (același program `attn_back_tile0_tc_v4`, SHA-256 `5d5eb75d…72d4`); niciun
registru GPU nou, nicio comandă nouă. Cod: `crates/nvidia-ga104/src/train_step.rs`, `crates/nvidia-ga104/src/batch_ram.rs`;
test `crates/nvidia-ga104/tests/sim_c157_attn0_tc_v4_mixed.rs` (T262, SIMULARE). Nicio schimbare incompatibilă pentru OS.
