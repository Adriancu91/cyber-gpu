# C-168 p.5 — P5c: GELU înainte lipit în coada lui „fc” (v7 / v8 / v8n) în pasul #513 (testat doar în SIMULARE)

Sesiunea Cyber-GPU 293, 28 nov 2026, în cloud, fără placă. Planul și predicțiile L1–L7 (scrise înainte de cod, sesiunea 292):
`docs/C-168_P5_LIPIRE_PLAN.md` §9; rezultatul: §11. Testul: T310 (`crates/nvidia-ga104/tests/sim_c168_p5c_gelu_fwd.rs`).
Se pune peste P5a (`with_fused_back`) și P5b (`with_fused_gelu`); merge și cu v8r (P3g).

## Ce e nou (doar adăugări)
- Trei programe noi, din aceleași surse cu `-DGELU_FWD` (Mesa NAK 26.3.1, SM 8.6), în `nvidia_ga104::mm_tc_fast_gelu_fwd`:

  | Program | Constanta | SHA-256 | Registre | Bloc / memorie partajată | Mărime |
  |---|---|---|---|---|---|
  | `mm_tc_v7_geluf` | `MM_TC_V7_GELUF` | `8e869b7ff19cd77c142e9edba57123c55281c78a3bd984bb83eedb44ce9c3fe1` | 128 | ca `mm_tc_v7` (128 fire, 14 336 B) | 43 488 B |
  | `mm_tc_v8_geluf` | `MM_TC_V8_GELUF` | `82aceab9f48d727083f7dff91f7f271663e83dc8887630ea3b49561c5dd4bfbf` | 128 | ca `mm_tc_v8` (256 fire, 24 576 B) | **70 032 B** |
  | `mm_tc_v8n_geluf` | `MM_TC_V8N_GELUF` | `8e8890a4276f248fd7c36925706558955310b51a006a0716f4dc98ffbbea5a07` | 144 | ca `mm_tc_v8n` (128 fire, 18 432 B) | **68 944 B** |

  `mm_tc_fast_gelu_fwd::check()` verifică cele trei sume. Totul până la scrierea lui C e al programului simplu (C = u = produsul
  + bias, aceiași biți); apoi fiecare element scrie în plus `act = u · Φ(u)` — exact operațiile lui `gelu_tr` mod 0 de pe placă.
  Cele două mai mari trec de 64 KiB: alocați-le după mărime (`program_map_bytes`, cum faceți deja), nu într-un loc fix de 64 KiB.
  Programele vechi (v7 / v8 / v8n / v8r / variantele GELU' / `mm_tc_gelu`) au ieșit **cu aceleași SHA-256** la recompilare.
- `mm_tc_fast_gelu_fwd::Args { tc: mm_tc::Args, act }` — cbuf-ul lui `mm_tc` cu `act` la 0x730 (argumentele cresc de la 0x30 la
  0x38 octeți; cbuf-ul rămâne 0x800). Refuză `act` = 0, nealiniat la 4 sau suprapus peste A, B, C, bias.
- `Plan::with_fused_gelu_fwd()` / `fused_gelu_fwd()`: în `step_mixed`, fiecare „fc” (`MixedLaunch::Tc`, eticheta `inainte:fc`)
  plus `gelu_tr`-ul înainte de după el devin **o singură** lansare `MixedLaunch::TcGeluFwd` (eticheta `inainte:fc+gelu`; la
  straturile recalculate `recalcul:fc+gelu`). `launches_per_step_mixed` scade în consecință. Pasul F32 (`step`) nu se schimbă.
- `MixedLaunch::TcGeluFwd` e o variantă nouă: codul vostru folosește doar `matches!`, deci se compilează la fel; **atenție** la
  numărătorile `matches!(x, MixedLaunch::Tc { .. })` (gpu_i14.rs, `tc_per_step`): cele 12 „fc+gelu” nu mai sunt `Tc`.
- `train_step::GeluFwdFast { v7, v8, v8n }` (unde le-ați încărcat; 0 = neîncărcat), `TcPick::V7GeluF` / `V8GeluF` / `V8nGeluF`.
- `train_step::pack_mixed_fast_attn12(…, gelu_fast, v8r_va, gelu_fwd, attn, …)` și `batch_ram::pack_mixed_fast_via_ram_attn12(…)`:
  ca `attn11` cu un argument nou `gelu_fwd` după `v8r_va`. Cu `None` = `attn11` **octet cu octet** (și o listă cu „fc+gelu” e
  refuzată). Cu `Some`, fiecare „fc+gelu” ia varianta `geluf` a programului pe care l-ar alege același „fc” simplu (pe valuri;
  unde valurile aleg v8, ia `v8_geluf` chiar dacă v8r e încărcat — v8r nu are variantă `geluf`). **Varianta neîncărcată sau un
  produs nealiniat = lista refuzată înainte de orice trimis** (nu există un `geluf` lent).

## Ce cerem OS-ului
1. O opțiune nouă (de ex. `--pas-gelu-inainte-lipit`, peste `--pas-gelu-lipit` din P5b): planul real și cel de probă cu
   `.with_fused_back().with_fused_gelu().with_fused_gelu_fwd()`; încărcați cele trei programe (SHA-256 de mai sus), dați
   `Some(GeluFwdFast { … })` packerului `attn12` (prin RAM: `via_ram_attn12`), restul argumentelor ca la rândul P5b.
2. **Întâi lot 2 cu oracolul**, apoi lot 6.
3. **Așteptat (scris înainte de placă):**
   - L5: **12 lansări mai puțin pe pas** față de P5b (pe placă 455 → 443 cu copierile lotului); `gelu_tr` 12 → 0; „fc” devine
     „fc+gelu”: **lot 6 12 × `mm_tc_v8n_geluf`, lot 2 12 × `mm_tc_v8_geluf`**; restul alegerilor neschimbat;
   - L7: oracolul lot 2 **ACELEAȘI NUMERE** ca rândul P5b (în SIMULARE sunt aceiași biți);
   - L6 (DEDUS): `gelu_tr` dispare din profil (~1,7 ms la lot 6); „fc+gelu” mai lent decât „fc” cu ~70 µs pe strat (scrie `act`);
     câștig net **0,6–1,1 ms la lot 6**. Sub 0,4 ms = RATAT (trimiteți liniile `fc` / `fc+gelu` / `gelu_tr` din profil).
     La lot 2 câștigul poate fi mai mic: acolo „fc” trece de pe v8r pe v8_geluf (v8 e ~9 % mai lent ca v8r la qkv, os-505).

## Ce arată SIMULAREA (T310) și ce nu
- L1–L3 (CALCUL, pe listările NAK): registrele = ale programelor simple; `tools/nak/compara_gelu_fwd.py` găsește în fiecare din cele
  32 / 64 / 64 de cozi arborele lui `gelu_tr` mod 0 (58 de operații) din programul de pe placă (Mesa 25.0.7), pornit din valoarea
  scrisă în C și scris la `act`; restul programului = programul simplu. Trei greșeli plantate sunt prinse.
- S1: modelul mic aliniat, 5 pași rapizi (P5a + P5b): cu și fără P5c **aceiași biți**; 88 → 86 lansări; și cu v8 + v8r.
- S2: #513 lot 2 și 6: exact −12 lansări (448 → 436 fără copieri), restul alegerilor și cbuf-urilor identice și în ordine; QMD-urile
  au programul, registrele, blocul și grila variantei.
- S3 / S5: refuzurile; `None` = `attn11`; prin RAM aceeași listă.
- S4: același produs (cu bias), forțat pe fiecare din cele trei programe = C și act ale perechii „fc” + `gelu_tr`, bit cu bit.
- Nimic din toate acestea n-a rulat pe placă.

## Ce e nou pentru placă
Trei programe noi (`mm_tc_v7_geluf`, `mm_tc_v8_geluf`, `mm_tc_v8n_geluf`), doar pe produse aliniate, cu o scriere în plus pe element
(la `act`). Nicio scriere în registre, niciun firmware, nicio schimbare incompatibilă.
