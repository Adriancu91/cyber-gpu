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

## Rezultatul (după cod, sesiunea 168)
| Predicția | Ieșit | |
|---|---|---|
| 1. ≤ 25 registre, mai puține instrucțiuni, fără partajată / barieră | **22 de registre**, **173** de instrucțiuni (`attn_fwd`: 221), 2 768 de octeți; nicio memorie partajată / locală, nicio barieră; SHA-256 `786ae995…148d386e` (Mesa NAK 25.0.7 reconstruit în cloud; `attn_fwd` a ieșit identic, `fe97ab49…`) | PASS |
| 2. aceleași operații cu virgulă mobilă pentru P | `attn_p`: 1 `ffma.ftz`, 3 `fmul.ftz`, 3 `fadd.ftz`, 1 `fmnmx.ftz`, `mufu.exp2`, `mufu.rcp`; `attn_fwd` = exact acestea + un singur `ffma.ftz` (lanțul P·V); `att` (0x38) nu apare deloc în codul lui `attn_p` | PASS |
| 3. P bit cu bit = `attn_fwd`, `att` neatins | 7 forme (pasul mic 4×32 C32 H4, T 45, H1 d40, d 1, d 64, softmax ascuțit, 3 texte T 96): **bit cu bit**; `att` rămâne NaN; nimic scris după P; refacerea unui P suprascris dă exact P-ul și `att`-ul lăsate de `attn_fwd` | PASS (SIMULARE) |
| 4. greșeli plantate | scale greșit, grila fără ultimul bloc de rânduri, `qkv`-ul altui strat: toate prinse | PASS (SIMULARE) |
| 5. costul ≈ 0,40–0,55 din `attn_fwd` | din codul compilat, ciclurile scrise de NAK pe o pereche (j, e): `attn_fwd` 81 (Q·K) + 110 (P·V) = 191, `attn_p` 98 → **≈ 0,51** (DEDUS) | PASS (DEDUS), nemăsurat |

Testele: T178 în `crates/nvidia-ga104/tests/sim_attn_p.rs` (6 teste). Modelul: `gpu_sim::cmdproc::attn_p_on_cpu` →
`gpu_compute::layer::attn_p_row_gpu` (partea P a lui `attn_fwd_row_gpu`, acum folosită și de acesta — aceeași funcție).

**Două observații din codul compilat (DEDUS):**
- bucla Q·K din `attn_p` are 4 `mov` în plus față de cea din `attn_fwd` (alt fel de alocare a registrelor: 98 față de 81 de cicluri
  scrise pe pereche) — nu schimbă biții (aceeași `ffma.ftz` pe aceleași numere), doar ~20 % din bucla Q·K;
- în `attn_fwd`, bucla P·V citește P din memorie cu `ld.global…strong.sys` la fiecare pereche (bufferul P nu e „doar citire”):
  e partea cea mai scumpă (110 cicluri scrise, plus așteptarea memoriei). `attn_fwd_tile` are și el 112 citiri `strong.sys` în cod —
  de văzut, la varianta „doar P” pe plăci, dacă marcarea bufferelor de intrare „doar citire” le schimbă (idee, nefăcut).

**Ce nu e dovedit:** pe placă, „aceiași biți” se sprijină pe aceleași instrucțiuni pe aceleași numere (verificat static mai sus) —
de măsurat o dată pe placă, odată cu proba atenției. Timpul (≈ 0,51) e nemăsurat.

**Legarea în pas (următorul pas, nefăcut):** `train_step::Prog` primește o variantă nouă (`AttnP`), iar OS-ul o potrivește exhaustiv
(`kernel/src/drivers/gpu_i10.rs`, `match prog`) → OS-ul ar avea nevoie de **un rând** în plus (`T::AttnP => cp.register_program(code,
c::attn_p_on_cpu)`). Anunțat în hub, în `stare/cyber-gpu.md` („Ce oferă”), înainte de legare.
