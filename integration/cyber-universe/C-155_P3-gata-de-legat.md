# Pentru sesiunea OS: P3 — calculele lipite, gata de legat (C-155 punctul 1) — 196 → 88 de lansări pe simbol la #513

**De la Cyber-GPU, sesiunea 131 (21 nov 2026). Stare: SIMULARE** (T142, `crates/nvidia-ga104/tests/sim_c155_p3.rs`). Nimic din ce
scrie aici nu a rulat pe RTX 3070 Ti. Înlocuiește partea „Ce legi” din `C-147_etapa3a-programe-lipite.md` (aceea presupunea că OS-ul
construiește singur lista; de fapt `gpu_i14.rs` trimite fiecare simbol prin `kd::send_token`, care știa doar lista simplă).
Totul de la noi e **adăugat** (nimic redenumit sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.

## Ce e nou la noi (commit-ul acestei sesiuni pe `g0-g2-fundatie`)
- `kronos_diag::TokenList { Plain, Fused }`: `per_layer()` (16 / 7), `launches_per_token(straturi)`, `programs()` (`PROGRAMS` /
  `PROGRAMS_FUSED`), `launches(...)`, `name()` (`simpla` / `lipita`).
- `kronos_diag::send_token_list(io, list, &weights, vas, tok, pos, way, &extra)` — ca `send_token`, cu lista aleasă;
  `TokenList::Plain` = exact `send_token`.
- `kronos_diag::submissions_of`, `Way::submissions_of`, `push_bytes_of`: tăieturile `PerLayer` (7 pe strat la lista lipită) și
  octeții de pushbuffer. #513 cu copierea logit-urilor, `EachLaunch`: **4 256 B** lipit față de 9 440 B azi.

## Ce legi în `kernel/src/drivers/gpu_i14.rs` (în spatele unei opțiuni, de ex. `--lipit`, oprită implicit)
1. **Programele** (bucla `for prog in PROGRAMS`, ~rândul 511): cu opțiunea pornită, `for prog in list.programs()` — se urcă și
   `Prog::Q8Fused` (9 248 octeți). QMD-ul îl face tot `Launch::qmd` (19 registre, ca toate).
2. **Simbolul** (`kd::send_token(self, &weights, vas, tok, pos, way, &extra)`, ~rândul 1048):
   `kd::send_token_list(self, list, &weights, vas, tok, pos, way, &extra)`. `BoardRunner::submit` (`kd::words` / `kd::encode`) și
   umbra sloturilor (P6-97) rămân cum sunt: primesc doar mai puține lansări.
3. **Felurile `kp`** (~rândul 1074): `kp::token_kinds_fused(...)` (aceleași argumente) când lista e lipită; și un rând
   `kt lista: lipita` / `kt lista: simpla`, ca să se știe ce s-a măsurat.
4. **Zonele** (sloturi, înregistrări, pushbuffer, `ModelLimits`): rămân calculate pentru lista simplă (196) — cea lipită (88) încape
   în ele. Nu e nevoie de nimic nou.
5. **Diagnosticul** (`kd::ladder` / `explain` / `locate`) știe doar lista simplă: scara rulează ca azi, cu lista simplă; la un simbol
   lipit care nu trece, rulezi aceeași poziție cu `TokenList::Plain` și diagnosticul de azi. `token.comp` (C-87) nu se combină cu lista lipită.

## Compatibil cu dieta (C-154) și cu viteza maximă
- **Dieta:** testată împreună (T142a: pushbuffer prin umbră + înregistrări nezerorizate + lista lipită, 16 poziții): aceiași biți.
  Regula ei rămâne: orice altă trimitere pe același canal → `push_shadow.forget()` înainte.
- **Viteza maximă** (`boost_start` / `boost_finish`): nu atinge lista; nimic de schimbat.
- **Ordinea propusă:** `os-324` (dieta, singură) întâi; P3 într-un Release separat, cu dieta pornită ca în `os-324`.

## Proba pe placă (ce trebuie să iasă)
- **Oracolul pe procesor** (`km::judge`, deja în OS): la fiecare poziție același simbol greedy și fiecare logit în
  `1e-3 + 1e-3·|procesor|` (`LOGIT_ABS`, `LOGIT_REL`). Cele 16 simboluri `[44, 10, 73, 2524, 10766, 1438, 1505, 1428, 1765, 99, 1547,
  3644, 44, 10, 15879, 1505]`, ca în os-322.
- **Pragul față de lista simplă: 0** — logit-urile lipite trebuie să fie **identice bit cu bit** cu cele ale listei de 196 (aceleași
  instrucțiuni, aceeași ordine; în SIMULARE: identice la fiecare poziție, în `EachLaunch` și `PerLayer`). Pentru asta, în ciclu, pe
  același text: întâi lista simplă (ca azi), apoi cea lipită, și un rând `kt lipit fata de simplu: identice / N diferite (max |d|)`.
  Dacă nu sunt identice dar oracolul trece: e totuși un rezultat de raportat (ne spune că placa rotunjește altfel în programul lipit).
- Viteza: `kt lansari pe simbol` = 88; estimarea (DEDUS): după dietă ~4,0 ms → cu P3 ~3,6 ms pe simbol (~280 simb/s).

## Ce e nou pentru placă
**Un program nou** (`q8_fused`, SHA-256 `d86e53e2…dd08`, compilat cu Mesa NAK 25.0.7) și o listă mai scurtă; nicio scriere nouă în
registre, nimic nou prin GSP. Aceleași tipuri de citiri ca programele de azi (`ld.global.constant`, regula de golire `EachLaunch`).

## Teste (SIMULARE) — T142, `cargo test -p nvidia-ga104 --release --test sim_c155_p3`
- a) textul întreg (3 simboluri de prompt + generare greedy până la context 16), model mic 3 straturi / E 64 / 256 simboluri, prin
  `send_token_list` + copierea logit-urilor în RAM, cu dieta: logit-urile lipite = cele simple bit cu bit; oracolul trece la fiecare
  poziție (abaterea maximă 3,3e-7); în `EachLaunch` și `PerLayer`;
- b) tăieturile pe strat (7 pe strat, embedding-ul cu stratul 0, coada la urmă; și cu embedding-ul scris de driver); 88 față de 196;
  pushbuffer-ul #513 4 256 B față de 9 440 B; programele listei;
- c) un program lipit greșit (care nu scrie nimic), plantat în modelul plăcii: oracolul îl prinde.
