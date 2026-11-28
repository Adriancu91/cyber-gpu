# C-168 p.5 — P5d: reziduurile înainte lipite în coada lui „out” și „proj” (v7 / v8 / v8n / v8r) în pasul #513 (testat doar în SIMULARE)

Sesiunea Cyber-GPU 294, 28 nov 2026, în cloud, fără placă. Planul și predicțiile M1–M7 (scrise înainte de cod, aceeași sesiune, commit
`2b60215`): `docs/C-168_P5_LIPIRE_PLAN.md` §12; rezultatul: §13. Testul: T311 (`crates/nvidia-ga104/tests/sim_c168_p5d_res_fwd.rs`).
Se pune peste P5a (`with_fused_back`), P5b (`with_fused_gelu`) și P5c (`with_fused_gelu_fwd`); merge și cu v8r (P3g).

## Ce e nou (doar adăugări)
- Patru programe noi, din aceleași surse cu `-DRES` (Mesa NAK 26.3.1, SM 8.6), în `nvidia_ga104::mm_tc_fast_res`:

  | Program | Constanta | SHA-256 | Registre | Bloc / memorie partajată | Mărime |
  |---|---|---|---|---|---|
  | `mm_tc_v7_res` | `MM_TC_V7_RES` | `b1bf01ef80c7aec52124a778791d0ba1f7099fb47cde63266af052c4e96b7c62` | 128 | ca `mm_tc_v7` (128 fire, 14 336 B) | 33 760 B |
  | `mm_tc_v8_res` | `MM_TC_V8_RES` | `4346e09afccb2dd49b34e5a0f730f4605d842e4858075fd08e8f1aea7da702ca` | 128 | ca `mm_tc_v8` (256 fire, 24 576 B) | 50 576 B |
  | `mm_tc_v8n_res` | `MM_TC_V8N_RES` | `088ead60c2815d2b4dfe7d79e18f97c3f7ef1af3911f2d87c580ffebccb946ab` | 144 | ca `mm_tc_v8n` (128 fire, 18 432 B) | 49 488 B |
  | `mm_tc_v8r_res` | `MM_TC_V8R_RES` | `650480bb8f223515c898519f2651136ef855f8c79951afa22f3a4e7140303a43` | 120 | ca `mm_tc_v8r` (256 fire, 24 576 B) | 49 344 B |

  `mm_tc_fast_res::check()` verifică cele patru sume. Toate sub 64 KiB. Totul până la valoarea scrisă azi e al programului simplu
  (produsul + bias, aceiași biți); apoi fiecare element adună `res[rând, col]` cu o singură `fadd.ftz` (adunarea lui `vadd`) și scrie
  rezultatul în C. Programele vechi (v7 / v8 / v8n / v8r / variantele GELU' / `geluf` / `mm_tc_gelu`) au ieșit **cu aceleași SHA-256**.
- `mm_tc_fast_res::Args { tc: mm_tc::Args, res }` — cbuf-ul lui `mm_tc` cu `res` la 0x730 (același loc ca `act` la P5c). Refuză `res`
  = 0, nealiniat la 4 sau suprapus peste A, B, C, bias.
- `Plan::with_fused_res_fwd()` / `fused_res_fwd()`: în `step_mixed`, fiecare „out” / „proj” (`MixedLaunch::Tc`, etichetele
  `inainte:out` / `recalcul:out` / `inainte:proj`) plus `vadd`-ul de după el devin **o singură** lansare `MixedLaunch::TcRes`
  (etichetele `inainte:out+rez` / `recalcul:out+rez` / `inainte:proj+rez`): C = ieșirea vechiului `vadd` (`xmid` / `xout`), `res` =
  prima lui intrare (`xin` / `xmid`); `tmp` nu mai e scris acolo. `launches_per_step_mixed` scade cu 2 pe strat (1 pe strat recalculat).
  Pasul F32 (`step`) nu se schimbă.
- `MixedLaunch::TcRes` e o variantă nouă: **atenție** la numărătorile `matches!(x, MixedLaunch::Tc { .. })` (`tc_per_step`): cele
  24 „out+rez” / „proj+rez” nu mai sunt `Tc` (ca „fc+gelu” la P5c).
- `train_step::ResFast { v7, v8, v8n, v8r }` (unde le-ați încărcat; 0 = neîncărcat), `TcPick::V7Res` / `V8Res` / `V8nRes` / `V8rRes`.
- `train_step::pack_mixed_fast_attn13(…, v8r_va, gelu_fwd, res_fwd, attn, …)` și `batch_ram::pack_mixed_fast_via_ram_attn13(…)`:
  ca `attn12` cu un argument nou `res_fwd` după `gelu_fwd`. Cu `None` = `attn12` **octet cu octet** (și o listă cu „+rez” e refuzată).
  Cu `Some`, fiecare „+rez” ia varianta `_res` a programului pe care l-ar alege același produs simplu (pe valuri; unde valurile aleg v8
  și v8r e încărcat, `v8r_res`). **Varianta neîncărcată sau un produs nealiniat = lista refuzată înainte de orice trimis.**

## Ce cerem OS-ului
1. O opțiune nouă (de ex. `--pas-rezidual-inainte-lipit`, peste opțiunea P5c): planul real și cel de probă cu
   `.with_fused_back().with_fused_gelu().with_fused_gelu_fwd().with_fused_res_fwd()`; încărcați cele patru programe (SHA-256 de mai
   sus), dați `Some(ResFast { … })` packerului `attn13` (prin RAM: `via_ram_attn13`), restul argumentelor ca la rândul P5c.
   Dacă P5c nu e încă pe placă, P5d merge și fără el: doar `with_fused_res_fwd` peste planul vostru, `gelu_fwd = None`.
2. **Întâi lot 2 cu oracolul**, apoi lot 6.
3. **Așteptat (scris înainte de placă):**
   - M5: **24 de lansări mai puțin pe pas** față de rândul de dinainte; `vadd` înainte 24 → 0 (cele 24 `vadd` înapoi au dispărut deja
     la P5a); „out” / „proj” devin „+rez”: **lot 6 24 × `mm_tc_v8n_res`, lot 2 24 × `mm_tc_v7_res`** (SIMULARE, vitezele os-500 și
     garda os-503); restul alegerilor neschimbat;
   - M7: oracolul lot 2 **ACELEAȘI NUMERE** ca rândul fără P5d (în SIMULARE sunt aceiași biți);
   - M6 (DEDUS): `vadd` dispare din profil (~1,24 ms la lot 6); „out” / „proj” mai lente cu ~18 µs fiecare (citesc `res`); câștig net
     **0,5–1,1 ms la lot 6, 0,3–0,9 ms la lot 2**. Sub 0,3 ms la lot 6 = RATAT (trimiteți liniile `out`, `proj`, `vadd` din profil).

> **Actualizare (sesiunea 295, după os-512):** M6 de mai sus e probabil prea optimist. La GELU' (P5b) placa a arătat că o coadă care
> citește câte o valoare pe rând așteaptă memoria de 64 de ori pe fir (+190 µs pe lansare la lot 6); coada `_res` are aceeași formă.
> Socoteala nouă (scrisă înainte de rândul vostru): **−0,1…−0,6 ms la lot 6, 0…−0,4 ms la lot 2** (`docs/C-168_P5_LIPIRE_PLAN.md` §14).
> Leacul e P5e (`C-168_P5e_citiri-coada-inainte.md`): aceleași locuri, programele `_res_pf`. Dacă legați P5d acum, puteți încărca direct
> `_res_pf` în locurile `_res` (aceiași biți, același tot restul) — dar un rând cu `_res` simplu ne spune cât a costat așteptarea.

## Ce arată SIMULAREA (T311) și ce nu
- M1–M3 (CALCUL, pe listările NAK): programele vechi ies cu aceleași sume; registrele = ale programelor simple, 0 „spills” noi;
  `tools/nak/compara_res.py` găsește în fiecare din cele 32 / 64 / 64 / 64 de cozi exact o citire de la adresa din 0x730 (cu indicele
  lui C) și o `fadd.ftz` cu ea, chiar înainte de scriere; fără ele, restul programului = programul simplu, în ordine. Trei greșeli
  plantate sunt prinse.
- S1: modelul mic aliniat, 5 pași rapizi (P5a + P5b + P5c): cu și fără P5d **aceiași biți**, −2 lansări pe strat; pe v7, v8, v8 + v8r, v8n.
- S2: #513 lot 2 și 6: exact −24 lansări (436 → 412 fără copieri), restul alegerilor și cbuf-urilor identice și în ordine; QMD-urile au
  programul, registrele, blocul și grila variantei.
- S3 / S5: refuzurile; `None` = `attn12`; prin RAM aceeași listă.
- S4: același produs (cu bias), forțat pe fiecare din cele patru programe = „produs + `vadd`”, bit cu bit.
- Nimic din toate acestea n-a rulat pe placă.

## Ce e nou pentru placă
Patru programe noi (`mm_tc_v7_res`, `mm_tc_v8_res`, `mm_tc_v8n_res`, `mm_tc_v8r_res`), doar pe produse aliniate, cu o citire și o
adunare în plus pe element. Nicio scriere în registre, niciun firmware, nicio schimbare incompatibilă.
