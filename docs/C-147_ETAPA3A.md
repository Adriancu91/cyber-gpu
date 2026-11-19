# C-147 punctul 2 — etapa 3a: calculele mărunte lipite (sesiunea 120, 20 nov 2026, cloud, FĂRĂ placă) — SIMULARE

## Ce s-a făcut
Un program nou pentru placă, `tools/nak/q8_fused.comp` → `tools/nak/out/q8_fused.sm86.sass` (Mesa NAK 25.0.7, SM 8.6, reconstruit în cloud;
`q8_matvec` a ieșit **identic**, aceeași sumă, deci unealta e aceeași). E `q8_matvec` cu operațiile mărunte de după el lipite în aceeași lansare:

| Lansarea lipită | Înlocuiește (din cele 16 ale stratului) |
|---|---|
| QKV: `W·x + b`, iar treimea cheii și a valorii scrise și în cache | qkv, qkv+b, k, v (4) |
| out: `x + (W·x + b)` | out, out+b, rez1 (3) |
| fc: `gelu(W·x + b)` | fc, fc+b, gelu (3) |
| proj: `x1 + (W·x + b)` | proj, proj+b, rez2 (3) |

Stratul: **16 → 7 lansări**; #513 (12 straturi): **195 → 87** pe simbol, **196 → 88** cu copierea logit-urilor în RAM.

## De ce trebuie să dea aceiași biți
Listarea NAK (`q8_fused.sm86.txt`), comparată cu `q8_matvec.sm86.txt` și `gelu.sm86.txt`:
- produsul scalar: **aceleași 488 de instrucțiuni** ca `q8_matvec`, doar alte poziții în bufferul constant (0x60/0x64 în loc de 0x40/0x44);
- termenul liber: `fadd.ftz rezultat, b` (ca `vadd`: a = rezultatul, b = termenul); rezidualul: `fadd.ftz x, v` (ca `vadd(x, ob)`);
- GELU: aceleași 21 de instrucțiuni, aceleași constante, în aceeași ordine ca `gelu.sm86.txt` (inclusiv `MUFU.RCP` / `MUFU.EX2`);
- citirile: tot `ld.global.constant`, ca la programele vechi (deci aceeași regulă de golire C-139, `EachLaunch`); 19 registre; fără `BAR.SYNC`;
  128 B memorie partajată (ca `q8_matvec`). 578 instrucțiuni, 9 248 octeți, SHA-256 `d86e53e2f117e8769026a907fe53a9c4a33350f27807e5c0674c1dea3344dd08`.

Cheia și valoarea scrise în cache sunt aceleași valori f32 pe care `copy_b128` le copia din `qkvb`.

## Cod (doar adăugări)
`nvidia_ga104::q8_fused` (programul, `cbuf`, `Tail`), `kronos_layer::{Prog::Q8Fused, PROGRAMS_FUSED, LAUNCHES_PER_TOKEN_FUSED,
token_launches_fused, Prog::registers}`, `kronos_model::{token_launches_fused, launches_per_token_fused}`,
`kronos_perf::{LAYER_KINDS_FUSED, token_kinds_fused}`, `gpu_sim::cmdproc::{q8_fused_on_cpu, Q8_FUSED_NEEDS}`.
`PROGRAMS`, `token_launches` și tot ce primește placa la `kronos --gpu` de azi: **neschimbate** (QMD-urile vechi au aceiași octeți: 19 registre).
`token.comp` (C-87) nu are operația lipită: `kronos_token::work_list` o refuză.

## Teste (SIMULARE) — `tests/sim_c147_fused.rs`, T136
- stratul lipit față de cele 16 lansări, pe două plăci simulate: `x`, cache-ul de chei și de valori **identice bit cu bit**, la E 64 (6 poziții)
  și E 768 / 12 capete (2 poziții); în toleranța I13 față de motorul de pe procesor;
- un model mic (3 straturi, 256 simboluri) trimis ca de OS: logit-urile și cache-urile identice bit cu bit la 4 poziții; felurile `kp` ale listei
  lipite = lansările trimise, unu la unu;
- greșeli plantate (fără rezidual, fără GELU, cheile/valorile nescrise, termenul liber greșit): niciuna nu trece.

## Ce NU e dovedit
Nimic pe RTX 3070 Ti. Câștigul de viteză (estimarea planului: până la ~100 simb/s dacă frâna e costul fiecărei lansări) se vede doar pe placă,
cu proba obișnuită: aceleași 16 simboluri ca procesorul. Așteptarea noastră: **aceleași logit-uri bit cu bit ca lista de 196** (aceleași instrucțiuni),
nu doar aceleași simboluri.
