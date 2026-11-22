# C-160 punctul 7 — programul `attn_p`: doar P (fără P·V), pentru refacerea lui P la întoarcere

Sesiunea 168, 22 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE** și prin compararea codului compilat; costul e **DEDUS** (nemăsurat).

## De ce
Sesiunea 167 (`docs/C-160_P7_P_RECALCULAT.md`) a făcut pasul „fără P păstrat”: la întoarcere, stratul l relansează `attn_fwd`
întreg ca să-și refacă P. Dar `attn_fwd` face și P·V (ieșirea `att`), care la întoarcere nu trebuie (se rescrie peste el însuși
cu aceiași biți). Un program care face **doar P** ar costa cam jumătate.

## Ideea (scrisă înainte)
`tools/nak/attn_p.comp` = `attn_fwd.comp` fără ultima buclă (P·V). Aceeași structură a argumentelor (qkv / p / att la
0x28 / 0x30 / 0x38, T 0x40, C 0x44, H 0x48, d 0x4c, scale 0x50) → același cbuf ca `attn_fwd`, `att` nefolosit (nu se scrie
nimic acolo). Aceleași blocuri de 32, aceeași grilă [⌈T/32⌉, H, B]. Fiecare operație care duce la P e aceeași, în aceeași ordine.

Doar adăugări: modulul nou `nvidia_ga104::attn_p` (cu `Args = attn_fwd::Args`), modelul `gpu_sim::cmdproc::attn_p_on_cpu`.
**Nu** intră încă în `train_step::Plan`: enumerarea `train_step::Prog` e potrivită exhaustiv de OS (`gpu_i10.rs`), deci o
variantă nouă acolo ar strica construirea OS-ului — legarea în pas se face doar după ce OS-ul e anunțat (în hub, „Ce oferă”).

## Predicții (scrise acum)
1. **Compilarea** (Mesa NAK 25.0.7): ≤ 25 de registre (cât `attn_fwd`), mai puține instrucțiuni decât `attn_fwd`; 0 memorie
   partajată / locală, fără barieră.
2. **Codul compilat:** șirul instrucțiunilor de calcul cu virgulă mobilă care duc la P (`ffma` Q·K, `fmul` cu scale, `fmnmx`,
   `fadd` / `fmul` / `mufu.ex2` pentru exp, `fadd` suma, `mufu.rcp`, `fmul` cu 1/sum) e **același** (aceleași operații, aceleași
   modificatoare `.ftz`) ca în `attn_fwd` — deci pe placă P iese bit cu bit la fel. Verificat comparând listele de operații.
3. **SIMULARE:** P scris de `attn_p` = P scris de `attn_fwd` **bit cu bit** pe ≥ 6 forme (inclusiv T nemultiplu de 32, H 1 și 4,
   d 1 / 8 / 64), iar bufferul `att` (umplut cu NaN) rămâne neatins.
4. **Greșeli plantate prinse:** scale greșit, grila fără ultimul bloc de rânduri, P al altui strat (qkv greșit).
5. **Costul (DEDUS din codul compilat, nemăsurat):** pe un rând i, `attn_fwd` face (i+1)·d `ffma` pentru Q·K și încă (i+1)·d pentru
   P·V, iar bucla P·V recitește P de d ori → `attn_p` ≈ **0,40–0,55** din timpul lui `attn_fwd` pe formele #513. Deci relansările
   din `Plan::new_p_recalc` (L − 1 pe pas) ar costa cam jumătate.

## Ce NU face
Nu schimbă niciun program existent, nici `Plan::new` / `Plan::new_p_recalc`, nici `train_step::PROGRAMS`. Nimic nou predat pe placă.
Varianta „doar P” a programului pe plăci (`attn_fwd_tile`) rămâne pasul următor, odată cu legarea programelor pe plăci în pas.
