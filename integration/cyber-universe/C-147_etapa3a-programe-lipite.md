# Pentru sesiunea OS: lista lipită (C-147 punctul 2, etapa 3a) — 196 → 88 de lansări pe simbol la #513

**Stare: SIMULARE (cyber-gpu sesiunea 120).** Pe placă doar în Release-ul pe care îl hotărăște centrul de comandă (după `os-309`), cu proba obișnuită.

## Ce legi (doar adăugări în crate-uri; nimic de azi nu se schimbă)
1. **Programele:** urcă `kronos_layer::PROGRAMS_FUSED` (8 = cele 7 de azi + `Prog::Q8Fused`) în loc de `PROGRAMS`, când e aleasă lista lipită.
   `Prog::Q8Fused.code()` = 9 248 octeți; QMD-ul îl face tot `Launch::qmd` (registrele vin acum din `Prog::registers()`; 19 la toate).
2. **Lista:** `kronos_model::token_launches_fused(...)` în loc de `token_launches(...)` (aceleași argumente); numărul: `launches_per_token_fused(layers)`
   (87 la #513; +1 cu copierea logit-urilor = 88). Lista e mai scurtă decât cea de azi, deci orice zonă care o încape pe cea de 196 o încape și pe aceasta.
3. **Felurile `kp`:** `kronos_perf::token_kinds_fused(layers, embed, copy_down)` în loc de `token_kinds` pentru lista lipită. Rândul
   `kp lansari pe simbol` = 88 spune care listă a fost măsurată; `kp qkv` înseamnă atunci QKV + termen + cheie/valoare în cache etc.
4. **Diagnosticul pe lansări** (`kronos_diag`, numele celor 16) e încă pe lista veche: pentru lista lipită, la o nepotrivire, rulează din nou
   lista veche (care rămâne neschimbată) ca să afli lansarea vinovată.
5. `token.comp` (C-87) nu știe programul lipit: `kronos_token::work_list` îl refuză (nu-l combina cu `PerLayer`/un-singur-program).

## Proba pe placă
Comutator (ex. `kronos --gpu --lipit`), oprit implicit; în ciclu, după rularea de azi: aceleași 16 simboluri ca procesorul **și** logit-urile
identice cu ale listei de 196 (așteptat bit cu bit: aceleași instrucțiuni, vezi `docs/C-147_ETAPA3A.md`), plus `kp`/`kt` înainte/după.
