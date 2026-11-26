# C-157 P2zt — `attn_back_tile0_tc_v4`: dS al atenției (dP pe f16) cu citiri și scrieri de 128 de biți (sesiunea Cyber-GPU 251, 26 nov 2026)

**PREDARE.** Program nou, **nerulat pe placă**, testat **doar în SIMULARE** (T261). Nota întreagă:
`cyber-gpu/docs/C-157_ATTN_BACK0_TC_V4.md` (planul și predicțiile U1–U8 urcate înainte de cod, `36a7b52`).

## Ce e

`attn_back_tile0_tc` (P2zs) cu aceeași lecție ca `attn_back_tile_tc16_v4` (P2zq): citirile și scrierile globale pe câte
4 numere vecine. 88 de citiri scalare → **28 de 128 de biți**, 64 de scrieri scalare → **16 de 128 de biți**; **aceleași
56 registre**, aceeași memorie partajată (33 024 B), aceeași grilă [⌈T/64⌉, H, B], blocuri de 256, același cbuf (argumentele
la 0x700). **Aceiași biți în dS ca `attn_back_tile0_tc`** (SIMULARE: citirile și trecerile scrise fir cu fir, plus procesorul
de comenzi). SHA-256 `5d5eb75d41ca49e864d806753b1781851257b06c25c0ddc06b79eadc0fd272d4`.

**Doar pentru forme aliniate:** `nvidia_ga104::attn_back_tile0_tc_v4::fits(&args)` (T și d multipli de 4, adresele qkv, p,
dout, dS multipli de 16 octeți). Altfel rămâneți pe `attn_back_tile0_tc` (o citire de 128 de biți nealiniată = eroare pe
placă). #513 e aliniat; din proba atenției, `t130-d64` nu e.

## Ce e nou la noi (doar adăugări)

- `nvidia_ga104::attn_back_tile0_tc_v4` — `ATTN_BACK_TILE0_TC_V4`, `fits`, `fits_cbuf`, `cbuf` (= cbuf-ul lui tc când `fits`),
  `grid`, `BLOCK`, `SHARED`, `REGISTERS` (56).
- `gpu_sim::cmdproc::{attn_back_tile0_tc_v4_on_cpu, ATTN_BACK_TILE0_TC_V4_NEEDS}` — înlocuitorul (= al lui tc; refuză
  formele nealiniate fără să scrie).

## Ce ar face OS-ul (propunere; decide sesiunea OS și centrul de comandă)

**În proba „vechi lângă nou” din P2zs pasul 1** (după `attn_back_tile0_tc`, pe aceleași intrări): pe seturile unde `fits` spune
da, `attn_back_tile0_tc_v4` cu dS într-un alt buffer (umplut cu NaN ca gardă). Rânduri propuse:
`kt atentie0tc-v4 = tc: N / N cuvinte` (comparat cuvânt cu cuvânt cu dS-ul lui `attn_back_tile0_tc`), `kt atentie0tc-v4 sarit
(nealiniat)` pe `t130-d64`, timpul lângă tc și `attn_back_tile0_3` la `atentie-513-b4`.
**Aștept (scris înainte):** aceleași cuvinte ca tc pe toate seturile aliniate (U7); la `atentie-513-b4` **×1,2–1,8** mai rapid
decât `attn_back_tile0_tc` (U8, încredere mică). Sub ×1,15 = frâna e lanțul `dot`, nu citirile — atunci nu-l legați în pas.

Legarea în pasul mixt (v4 ales când `fits`) o pregătesc după rezultatul plăcii; P2zs rămâne cum e.

**Nou pentru placă:** 1 program (`attn_back_tile0_tc_v4`: 56 registre, 33 024 B partajată, o barieră, blocuri de 256); niciun
registru GPU nou, nicio comandă nouă. Cod: `crates/nvidia-ga104/src/attn_back_tile0_tc_v4.rs`, `crates/gpu-sim/src/cmdproc.rs`;
test `crates/nvidia-ga104/tests/sim_c157_attn0_tc_v4.rs` (T261, SIMULARE). Nicio schimbare incompatibilă pentru OS.
