# C-168 p.5 — P5b: GELU' lipit în coada înmulțirilor rapide (v7 / v8 / v8n) în pasul #513 (testat doar în SIMULARE)

Sesiunea Cyber-GPU 291, 28 nov 2026, în cloud, fără placă. Planul și predicțiile J1–J7 (scrise înainte de cod):
`docs/C-168_P5_LIPIRE_PLAN.md` §7; rezultatul: §8. Testul: T307 (`crates/nvidia-ga104/tests/sim_c168_p5b_gelu_fast.rs`).
Se pune peste P5a (`with_fused_back`).

## Ce e nou (doar adăugări)
- Trei programe noi, din aceleași surse cu `-DGELU` (Mesa NAK 26.3.1, SM 8.6), în `nvidia_ga104::mm_tc_fast_gelu`:

  | Program | Constanta | SHA-256 | Registre | Bloc / memorie partajată |
  |---|---|---|---|---|
  | `mm_tc_v7_gelu` | `MM_TC_V7_GELU` | `ed8082045e2255d96dbcca4f42bb44bb4e199a91e57e1dc03e95d57c756a10c2` | 128 | ca `mm_tc_v7` (128 fire, 14 336 B) |
  | `mm_tc_v8_gelu` | `MM_TC_V8_GELU` | `d0d061ce7711fdf9f1a9e56ea52162fc84752f23f53987e78907be694af05417` | 128 | ca `mm_tc_v8` (256 fire, 24 576 B) |
  | `mm_tc_v8n_gelu` | `MM_TC_V8N_GELU` | `81e6e4ab50145e08dfa04627a0fc46afb17ba331087f547ddb7bd3c3fa259714` | 144 | ca `mm_tc_v8n` (128 fire, 18 432 B) |

  `mm_tc_fast_gelu::check()` verifică cele trei sume. Bucla K e a programului fără GELU (aceiași octeți ai sursei până la scriere);
  scrierea e a lui `mm_tc_gelu` (`v · GELU'(u)`, `u` din locul biasului). Programele vechi (v7 / v8 / v8n / v8r / `mm_tc_gelu`)
  au ieșit **cu aceleași SHA-256** la recompilare.
- `train_step::GeluFast { v7, v8, v8n }` (unde le-ați încărcat; 0 = neîncărcat), `TcPick::V7Gelu` / `V8Gelu` / `V8nGelu`.
- `train_step::pack_mixed_fast_attn10(…, attn_fwd_va, gelu_fast, attn, …)` și `batch_ram::pack_mixed_fast_via_ram_attn10(…)`:
  ca `attn9` cu un argument nou `gelu_fast` după `attn_fwd_va`. Cu `None` = `attn9` **octet cu octet**. Cu `Some`, fiecare
  `MixedLaunch::TcGelu` aliniat ia varianta GELU a programului pe care l-ar alege `pick_long_k` pentru același produs fără GELU.
  **Dacă varianta aleasă nu e încărcată, lista e refuzată înainte de orice trimis** (nu cade tăcut pe `mm_tc_gelu`, cel lent).

## Ce cerem OS-ului
1. O opțiune nouă (de ex. `--pas-gelu-lipit`, peste `--pas-rezidual-lipit` din P5a): planul real și cel de probă cu
   `.with_fused_back().with_fused_gelu()`; încărcați cele trei programe (SHA-256 de mai sus), dați `Some(GeluFast { … })` packerului
   `attn10` (prin RAM: `via_ram_attn10`). `mm_tc_gelu` (`TcPrograms::gelu`) rămâne nevoie doar pentru produse nealiniate (în #513 nu
   sunt). `Plan::fused_programs()` cu ambele lipiri dă `[ln_back_res, mm_train_gelu]`; în pasul mixt `mm_train_gelu` nu se lansează
   (se transformă în `TcGelu`), deci încărcarea lui e inutilă, dar nu strică.
2. **Întâi lot 2 cu oracolul**, apoi lot 6.
3. **Așteptat (scris înainte de placă):**
   - J4: **12 lansări mai puțin pe pas** față de P5a (467 → 455), `gelu_tr` 24 → 12 (rămân cele 12 înainte); „proj-dX” devine
     „proj-dX+gelu”: **lot 6 12 × `mm_tc_v8n_gelu`, lot 2 12 × `mm_tc_v8_gelu`**; restul alegerilor neschimbat;
   - J7: oracolul lot 2 **ACELEAȘI NUMERE** ca rândul P5a (în SIMULARE sunt aceiași biți);
   - J6 (DEDUS): `gelu_tr` în profil ~4,25 → ~1,7 ms; „proj-dX+gelu” mai lent decât „proj-dX” cu ~70–110 µs pe strat (citește `u`);
     câștig net **1,0–1,8 ms la lot 6**. Sub 0,5 ms = RATAT (trimiteți liniile `gelu_tr` / `proj-dX+gelu` din profil).

## Ce arată SIMULAREA (T307) și ce nu
- S0 / J1–J2 (CALCUL, pe listările NAK): registrele = ale programelor fără GELU; `tools/nak/compara_epilog.py --float` găsește în
  fiecare din cele 32 / 64 / 64 de cozi aceleași 21 de operații GELU' ca în `mm_tc_gelu`.
- S1: modelul mic aliniat, 5 pași rapizi: P5a fără lipire = `mm_tc_gelu` = variantele rapide, **aceiași biți**; 90 → 88 lansări.
- S2: #513 lot 2 și 6: exact −12 lansări, restul identic și în ordine; QMD-urile au programul, registrele, blocul și grila lor.
- S3 / S5: refuzurile; `None` = `attn9`; prin RAM aceeași listă.
- S4: același produs, forțat pe fiecare din cele trei programe = biții lui `mm_tc_gelu`.
- Nimic din toate acestea n-a rulat pe placă.

## Ce e nou pentru placă
Trei programe noi (`mm_tc_v7_gelu`, `mm_tc_v8_gelu`, `mm_tc_v8n_gelu`), doar pe produse aliniate, aceleași argumente ca
`mm_tc_gelu`. Nicio scriere în registre, niciun firmware, nicio schimbare incompatibilă.
