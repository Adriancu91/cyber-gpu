# C-168 p.4 — P4b: `attn_fwd_tc` legat în pasul #513 (cererea din os-504; testat doar în SIMULARE)

Sesiunea Cyber-GPU 289, 28 nov 2026, în cloud, fără placă. Planul și predicțiile (scrise înainte de cod):
`docs/C-168_ATTN_FWD_TC_PLAN.md` §11; rezultatul: §12. Testul: T305 (`crates/nvidia-ga104/tests/sim_c168_attn_fwd_tc_mixed.rs`).

## Ce e nou (doar adăugări)
- `MixedLaunch::AttnFwdTc(Launch)` și `TcPick::AttnFwdTc` (numele în raport: `attn_fwd_tc`);
- `train_step::attn_fwd_tc_list(list)` — mută pe `attn_fwd_tc` fiecare lansare `attn_fwd_tile3` aliniată (d = 64, T multiplu de 64,
  adresele la 16 B); cele nealiniate rămân pe tile3, fără eroare. Se pune lângă `attn_tc_list` / `attn0_tc_list`, în orice ordine;
- `train_step::pack_mixed_fast_attn9(launches, prog_va, progs, v5_va, v6_va, v7_va, v8, long_k, attn_fwd_va, attn, attn_fast,
  attn0, step_bytes, zone_va)` — ca `pack_mixed_fast_attn8k`, cu `attn_fwd_va` (adresa unde ați încărcat `attn_fwd_tc`) pus după
  `long_k`; `batch_ram::pack_mixed_fast_via_ram_attn9(…, long_k, attn_fwd_va, …, copy_va, zone_va, v)` la fel, prin RAM.

Cu `attn_fwd_va = 0` și lista nemutată ies exact octeții de azi (`…attn8k` le cheamă acum pe acestea cu 0). O listă mutată cu
`attn_fwd_va = 0` (sau dată unui packer vechi) e refuzată înainte de trimitere. Lansarea mutată ocupă `SLOT_TC` (0x900) în zona de
control, nu `SLOT` (0x200): zona crește cu 12 × 0x700 = 21 504 B la #513 (`MixedLaunch::slot` dă deja mărimea).

## Ce cerem OS-ului
1. Opțiunea nouă (de ex. `--pas-atentie-fwd-tc`, după `--pas-v8`, și după `--pas-v8-garda` dacă o faceți pentru P3e): încărcați
   `nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC` (SHA-256 `ce2b2f22…7e929257` verificat întâi, ca la proba os-504), apoi
   `ml = ts::attn_fwd_tc_list(ml)?` după `attn_tc_list` / `attn0_tc_list`, și `pack_mixed_fast_via_ram_attn9` /
   `pack_mixed_fast_attn9` cu `attn_fwd_va` = adresa programului și `long_k` = `0` (sau `mm_tc_v8::LONG_K_OS503` cu garda).
   Treapta pasului (64 KiB) rămâne: `attn_fwd_tc` cere 27 648 B, 2 blocuri pe SM.
2. **Întâi lot 2 cu oracolul** (ca os-503), apoi lot 6:
   `test-placa --i9 c135 --tablou-viteza --antrenare-513 --mixt-513 --mixt-513-rapid --lot-prin-ram --doua-locuri --atentie-tc16 --atentie-v4 --atentie-ds --pas-v5 --pas-v6 --pas-v7 --pas-v8 --pas-atentie-fwd-tc --lot-513 2`
   și același rând cu `--lot-513 6`.
3. **Așteptat (scris înainte de placă, §11, DEDUS):**
   - alegerile: 12 × `attn_fwd_tc`, iar `attn_fwd_tile3` nu mai apare; restul alegerilor neschimbat (lot 6 fără gardă v7 12 / v8n 135,
     cu gardă 13 / 134; lot 2 v7 73 / v8 24 / v8n 50);
   - H7: oracolul lot 2 **ACELEAȘI NUMERE** (în toleranțe; nu aceiași biți ca os-503: produsele atenției înainte sunt f16);
   - H6c: atenția înainte în profil **2,6–4,0 ms** pe pas la lot 6 (azi 16,1 ms, eticheta `attn_fwd_tile3`);
   - H6d: pasul lot 6 **126–130 ms = 23 600–24 400 simb/s** (os-504: 140,5 / 21 859); cu garda P3e împreună **121–126 ms**.
   - Dacă oracolul lot 2 nu trece: rândul lot 6 nu se rulează; ne trimiteți cifrele oracolului (care tensor, cât).

## Ce arată SIMULAREA (T305) și ce nu
- S1–S3, S5: aceiași octeți fără `attn_fwd_tc`; la #513 exact 12 lansări mutate, QMD-ul lui `attn_fwd_tc` (120 registre, 27 648 B,
  256 fire, grila [4, 12, lot]); refuzurile; prin RAM aceeași listă.
- S4 (model mic aliniat, 5 pași, prin pushbuffer + QMD, față de același pas cu tile3): pierderile (4,8·10⁻⁶), scăderea (1,7·10⁻⁵) și
  gradientul pasului 1 (6,3·10⁻⁴) **în toleranțe**; greutățile după 5 pași **peste 1·10⁻³** pe biasuri (cel mai rău: un singur
  element din `ln2.bias` al stratului 1, cu gradientul aproape 0, pe care Adam îl împinge diferit) — și schimbarea deja trecută pe
  placă (atenția înapoi pe circuite față de F32) e peste același prag în același model (2,5·10⁻³). Verdictul pe greutăți = oracolul
  pe placă (H7).
- Nimic din toate acestea n-a rulat pe placă în pas.

## Ce e nou pentru placă
Doar folosirea în pas a programului `attn_fwd_tc` (același binar ca în proba os-504). Nicio scriere în registre, niciun firmware.
Nicio schimbare incompatibilă (enum-urile au primit câte o variantă la coadă, ca la `Attn0Tc` / `V8n`).
