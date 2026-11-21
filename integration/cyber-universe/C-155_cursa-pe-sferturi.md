# Pentru sesiunea OS: cursa înmulțirilor, în lista de simbol (C-155 punctul 4) — `TokenList::Fused32`

**Stare: SIMULARE (cyber-gpu sesiunea 135).** Nimic dovedit pe RTX 3070 Ti. Pe placă doar printr-un Release al OS, opțiune oprită
implicit, în ciclul centrului de comandă. Comparația cu estimarea: `docs/C-157_P2_COMPARATII.md` (os-328).

## De ce acum
În os-328 (lista lipită) înmulțirile iau **60 % din placă**: fc 486 + proj 406 + qkv 381 + logit 186 = 1 459 µs din 2 421 pe simbol.
Socotite pe octeții citiți, merg cu **~60–75 GB/s** (≈ 12 % din 608 GB/s), iar copia VRAM din același test a mers cu 282 GB/s (DEDUS
din rândurile `kp` și `kc`). Programul de azi dă unui fir un bloc întreg de 36 de octeți: la 768 de intrări lucrează doar 24 din 32 de
fire, iar cele 32 de fire citesc cuvinte la 36 de octeți distanță. Varianta „pe sferturi” (C-147 punctul 3, `q8_matvec32`) dă unui fir un
sfert de bloc: lucrează toate 32, citesc 288 de octeți la rând, x vine cu citiri de 16 octeți.

Cursa separată (`q8_race`, nota `C-147_etapa4-cursa-q8.md`) n-a ajuns pe placă, iar între timp lista de azi e cea lipită. De aceea cursa
se mută **în lista de simbol**, exact ca proba `--lipit` din os-328: același text de două ori, lista lipită, apoi lista lipită pe sferturi.

## Ce legi (doar adăugări în crate-uri; ce primește placa azi nu se schimbă)
- `kronos_diag::TokenList::Fused32` (numele `lipita-32`): aceleași 88 de lansări, **aceleași cbuf-uri octet cu octet**; doar programul
  se schimbă: cele 4 înmulțiri lipite pe strat merg pe `q8_fused32`, logit-urile pe `q8_matvec32`. O lansare al cărei x nu e aliniat la 16
  își păstrează programul vechi (nu se face niciodată o citire de 128 de biți nealiniată). În #513 toate sunt aliniate (blob-ul are toate
  părțile aliniate la 16).
- Programele de urcat: `TokenList::Fused32.programs()` = `kronos_layer::PROGRAMS_FUSED32` (cele 8 de azi + `Q8Fused32` + `Q8Matvec32`).
- Rândurile pe fel de lansare: `kronos_perf::token_kinds_fused` (aceeași ordine a lansărilor ca la `Fused`) — în `gpu_i14.rs` ramura `_ =>`
  o prinde deja.
- `kronos_token::work_list` le refuză (token.comp nu le are; OS-ul nu-l folosește).
- Module noi: `nvidia_ga104::q8_fused32` (`Q8_FUSED32`, `Q8_FUSED32_SHA256`, `REGISTERS`, `BLOCK`, `X_ALIGN`, `SHARED`, `quarter`, `narrow`,
  `count`); `kronos_layer::{Prog::Q8Fused32, Prog::Q8Matvec32, PROGRAMS_FUSED32}`; `kronos_model::token_launches_fused32`;
  `gpu_sim::cmdproc::{q8_fused32_on_cpu, Q8_FUSED32_NEEDS}`.

## Ce e nou pentru placă
Două programe: **`q8_fused32`** (SHA-256 `f26faedd…bcb2`, 4 384 B, 22 de registre, 32 de fire, 128 B partajați, fără barieră) și **`q8_matvec32`**
(`86ae584b…c303`, compilat din sesiunea 121, nerulat încă pe placă). Nicio scriere nouă în registre, aceleași cbuf-uri.

## Proba propusă (un rând nou de test, de ex. `test-placa --sferturi`)
1. Același text de două ori: `TokenList::Fused`, apoi `TokenList::Fused32` (ca `--lipit`).
2. **Corect:** 16 simboluri identice cu procesorul în ambele; oracolul pe procesor trece la fiecare poziție. Logit-urile **nu** mai sunt
   bit cu bit ca la `Fused` (altă ordine a adunărilor) — de aceea amprenta se schimbă; pragul față de `Fused`: abaterea maximă sub 1e-4
   (în SIMULARE: vezi T146). Un rând `kt sferturi fata de lipit: abaterea maxima …` ar ajuta (și rândul `kt lipit fata de simplu` din P3).
3. **Mai rapid:** rândurile `kp` pe fel pentru ambele liste (fc, proj, qkv, out, logit). Regula, scrisă înainte de rulare (ca la `q8_race`):
   un fel trece pe sferturi doar dacă e corect și mai rapid cu cel puțin **50 ‰**; altfel rămâne programul de azi. Dacă unele feluri câștigă și
   altele nu, facem lista mixtă (pasul nostru următor, după cifre).

## Ce așteptăm (DEDUS, nu promis)
Dacă frâna e felul în care citesc firele (24 din 32 de fire, cuvinte risipite), pe sferturi câștigăm mult pe fc/qkv/out/logit (cel mai puțin
pe proj, unde toate firele lucrau deja). Dacă frâna e altundeva (de ex. un singur grup de 32 de fire pe rând = prea puține citiri în zbor),
câștigul e mic și pasul următor e altul (mai multe rânduri pe grup sau greutățile rearanjate la urcare). Cursa ne spune care din două.
