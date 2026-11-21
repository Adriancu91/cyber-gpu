# Pentru sesiunea OS: LayerNorm nou (C-155 punctul 2) — gata de legat, după P3

**De la Cyber-GPU, sesiunea 132 (21 nov 2026). Stare: SIMULARE** (T143, `crates/nvidia-ga104/tests/sim_c155_ln.rs`). Nimic din ce
scrie aici nu a rulat pe RTX 3070 Ti. Totul e **adăugat** (nimic redenumit sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.

## De ce
În os-322 LayerNorm a luat **543 µs din 2 631 µs** pe simbol (21 %): ~21,8 µs pe lansare pentru 768 de numere. Programul vechi are
un singur grup de 32 de fire și citește `x` de trei ori din memorie, în bucle cu citiri care se așteaptă una pe alta.
Programul nou (`layernorm256`): **256 de fire**, `x`, `g`, `b` citite **o singură dată** (12 citiri pornite una după alta, fără
salt între ele, o singură așteptare), sumele prin `shfl.bfly` în fiecare warp + 8 sume prin 32 de octeți de memorie partajată și o
barieră. Aceleași argumente, **aceiași octeți de cbuf** — se schimbă doar programul și mărimea grupului.

## Ce e nou la noi (ramura `g0-g2-fundatie`)
- `nvidia_ga104::layernorm256`: `LAYERNORM256` (4 576 octeți), `LAYERNORM256_SHA256`, `REGISTERS` = 25, `BLOCK` = 256,
  `SHARED` (32 B, **o barieră**), `MAX_N` = 1024, `widen`.
- `kronos_layer::{Prog::LayerNorm256, PROGRAMS_FUSED_LN, token_launches_fused_ln}`; `kronos_model::token_launches_fused_ln`.
  `Launch::qmd` pune singur 256 de fire, 25 de registre, memoria partajată și `BARRIER_COUNT` = 1.
- `kronos_diag::TokenList::FusedLn` (`name()` = `lipita-ln`): lista lipită (88 de lansări la #513) cu toate cele 25 de LayerNorm
  (2 pe strat + cel final) pe programul nou. `programs()` = cele 8 de la P3 + `LayerNorm256` (9).
- `gpu_sim::cmdproc::{layernorm256_on_cpu, LAYERNORM256_NEEDS}`; `gpu_compute::layer::{layernorm_gpu256, LN256_MAX_N}` (oracolul
  în ordinea exactă a codului NAK).

## Ce legi (după Release-ul cu P3, separat; opțiune oprită implicit, de ex. `--ln-nou`)
1. Unde ai legat P3 (`list.programs()` și `kd::send_token_list(self, list, ...)`): cu opțiunea pornită, `list = TokenList::FusedLn`
   în loc de `TokenList::Fused`. Atât — programele, QMD-urile, tăieturile (`submissions_of`), pushbuffer-ul și zonele sunt ca la P3.
2. Felurile `kp`: `kp::token_kinds_fused(...)` ca la P3 (aceleași 88 de lansări, aceleași feluri); rândul `kt lista: lipita-ln`.
3. Dieta (C-154) și viteza maximă: neschimbate.

## Proba pe placă (ce trebuie să iasă)
- **Oracolul pe procesor** (`km::judge`, deja în OS): aceleași simboluri greedy și fiecare logit în `1e-3 + 1e-3·|procesor|`.
  Cele 16 simboluri `[44, 10, 73, 2524, 10766, 1438, 1505, 1428, 1765, 99, 1547, 3644, 44, 10, 15879, 1505]`, ca în os-322.
- **Față de lista lipită cu LayerNorm vechi: NU bit cu bit** (altă ordine a adunărilor). Pragul scris: aceleași simboluri și
  abaterea logit-urilor sub **1e-4** (în SIMULARE: 3,1e-7). Rândul propus: `kt ln nou fata de vechi: max |d| = …, simboluri identice: da/nu`.
- Viteza (DEDUS): ~3–5 µs pe lansare în loc de ~21,8 → **~0,45 ms mai puțin pe simbol** la #513 (25 de lansări); `kp` va arăta
  ln1 / ln2 / ln_final mult mai mici.

## Ce e nou pentru placă
**Un program nou** (`layernorm256`, SHA-256 `a407dd68…334d`, Mesa NAK 25.0.7), **primul din lista lui Kronos cu 256 de fire, `shfl` și
barieră (`bar.sync`)**; aceleași
tipuri de citiri (`ld.global.constant`) și scrieri ca programele de azi. Nicio scriere nouă în registre, nimic nou prin GSP.
Dacă proba nu trece dar P3 trece: e programul nou; `TokenList::Fused` rămâne la îndemână.
