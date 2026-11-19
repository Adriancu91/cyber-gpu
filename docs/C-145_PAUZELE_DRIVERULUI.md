# C-145 — Pauzele driverului la `kronos --gpu` (de la 3,1 spre 44 simboluri/s)

Sesiunea 116, 19 noiembrie 2026, în cloud, **fără placă**. Sursa cifrelor: primul ciclu automat `os-307`
(hub `rezultate/2026-11-19-workstation-os307-automat/`, jurnalele `rezultate/os-automat/2026-11-19/gpu-jurnal-011.txt`
și `ciclu-auto-20261119-0852.txt`). Codul OS citit (doar citit): `cyber-universe` `kernel/src/drivers/gpu_i6.rs`
(jurnalul pe stick), `gpu_i14.rs` (`BoardRunner`, `CpuCheck`). Tot ce e nou aici e testat **doar în SIMULARE**.

## 1. Ce include fiecare cifră din raportul `os-307`

| Cifra | Ce măsoară | Include verificarea pe procesor? |
|---|---|---|
| **43,93 simboluri/s** „doar lansările” | ceasul plăcii: suma duratelor celor 196 de lansări ale unui simbol (median 22,9 ms) | nu; nici driverul, nici jurnalul |
| **3,10 simboluri/s** „cu driverul, jurnalul și ecranul” | ceasul OS-ului de la prima lansare scrisă până la logit-urile citite (median 294,8 ms) | **nu** |
| **0,24 simboluri/s** „ceasul plăcii, cu pauzele driverului” | ceasul plăcii de la prima înregistrare a primei poziții la ultima a ultimei, împărțit la poziții (18) | **da**: ~3,87 s pe poziție, motorul de referință din nucleu pe 1 nucleu, rulat **între** două simboluri |

Deci 0,24 nu contrazice 3,10: diferența e verificarea pe procesor. Verificat pe jurnal: între două sonerii trec
4,16 s, din care 3,89 s fără niciun rând (procesorul calculează poziția), iar 0,27 s sunt cele 15 rânduri ale simbolului.
(Rândul `kv motorul procesorului median pe pozitie` = 3 870 293 µs.) Constanta `kronos_fast::FIGURES` are aceste explicații.

## 2. Unde se duce timpul unui simbol (din jurnalul `gpu-jurnal-011`, cifre măsurate de OS)

Un simbol în timpul generării = **15 rânduri** în jurnal, mereu aceleași: nota permisului, 4 mutări ale ferestrei
VRAM (`0x1700`), nota, 3 scrieri de invalidare TLB (`0xb830a0/a4/b0`), nota soneriei, soneria (`0xbb0090`), nota,
3 mutări ale ferestrei.

| Parte | Cât (median, aproximativ) | De unde știm |
|---|---|---|
| **Jurnalul pe stick** | **~246 ms** = 15 rânduri × ~16,4 ms | fiecare rând e scris **și sincronizat** pe stick înainte de următorul acces (`gpu_i6::sink`: `fs::write` + `fs::flush`); pasul tipic dintre rânduri e 16,4 ms; raportul ciclului: scrieri USB în medie 9 ms |
| **Scrierea lansărilor prin fereastra VRAM** | **~41 ms** | după mutarea ferestrei la `0x1b2d…` urmează 57 ms în loc de 16 ms: cele 196 × 512 B (cbuf + QMD) sunt rescrise cuvânt cu cuvânt prin fereastră, fiecare cuvânt **recitit** (~25 000 de scrieri + citiri MMIO) |
| Placa (lansările) | 22,9 ms | ceasul plăcii; în mare parte se suprapune cu scrierea rândurilor de după sonerie |
| Restul (așteptarea înregistrărilor, citirea a 64 KiB de logit-uri din RAM) | ~5–10 ms | diferența |
| **Total** | **294,8 ms** | rândul `kv median procesor` |
| Verificarea pe procesor (în afara celor 294,8 ms) | **~3 870 ms** | rândul `kv motorul procesorului` |

Urcarea greutăților (7,27 s pe ceasul procesorului): ~330 de mutări ale ferestrei, fiecare cu rândul ei → ~5,4 s
doar jurnal. Ecranul: niciun rând de ecran în timpul simbolului (textul se afișează la sfârșit) — ~0.

**Concluzie:** nu calculul și nici mutările ferestrei în sine costă (o scriere de registru durează microsecunde), ci
(1) sincronizarea stick-ului la fiecare rând, (2) rescrierea celor 100 KiB de lansări cu recitire la fiecare simbol,
(3) verificarea cu motorul de referință, pe 1 nucleu, între simboluri.

## 3. Ce s-a adăugat (doar adăugări, totul oprit implicit)

Nimic din ce primește placa nu se schimbă: aceleași comenzi, aceleași lansări, aceleași valori în registre, aceeași
ordine a acceselor. De aceea regula „trei surse” (pentru schimbarea felului de trimitere) nu se aplică aici.

1. **Jurnal cu rândurile de rutină ținute în memorie** (`gpu_core::diag`): `Journal::with_live_sink` +
   `set_routine(Some(kronos_fast::gen_routine(&[note…])))`. În timpul generării, rândurile de rutină — scrierile
   la cele 4 registre ale ferestrei/invalidării și la sonerie, plus notele repetate pe care le dă OS-ul — ajung la
   stick cu `Flush::Later`: OS-ul le ține în memorie și le scrie la următorul rând obișnuit, la `flush_live()` (OS-ul
   îl cheamă după fiecare simbol) sau după cel mult 64 de rânduri (`GEN_MAX_PENDING`).
   **Ce nu slăbește:** fiecare acces e tot în jurnal, cu valorile înainte/după, **în aceeași ordine** (testul T134c
   compară textul rând cu rând); pașii (`pas …`), valorile (`kg`/`kv`/`kt`), citirile, configurația PCI și orice altă
   scriere merg pe stick imediat, ca acum; gărzile de scriere nu sunt atinse (sunt în altă parte: `GuardedBar0`,
   `VramWindow`, gardul soneriei). Singurul preț: dacă placa se blochează în mijlocul unui simbol, pe stick pot lipsi
   cel mult ultimele rânduri de rutină ale acelui simbol (≤ 64) — rândul pasului și rândurile dinainte rămân.
2. **Umbra scrierilor** (`kronos_fast::WriteShadow`): ține minte ce s-a scris ultima dată în zona lansărilor; la
   simbolul următor se scriu (și se recitesc) doar cuvintele care s-au schimbat. În SIMULARE: 36 de lansări × 8
   poziții = 147 456 B scriși întregi față de 18 640 B cu umbra (primul simbol întreg, apoi ~30 B pe simbol);
   tokenurile și logit-urile identice bit cu bit. Doar pentru zone pe care **numai procesorul** le scrie (sloturile
   lansărilor) — niciodată pentru înregistrări, logit-uri sau cache-uri. După orice eroare: `forget()`.
3. **Timpul pe simbol, pe bucăți** (`kronos_fast::TimeSplit`): placă / driver / jurnal / ecran / verificare, cu
   rândurile `kt …` în jurnal și cititorul `parse_split` (cerut și de C-146 pentru raportul ciclului).
4. **Verificarea după placă** (`kronos_gen::Generation::run_check_after`): toate pozițiile pe placă, apoi
   procesorul judecă fiecare poziție, în ordine, la fel de strict; la prima poziție refuzată generarea se oprește
   acolo și simbolurile de după ea se aruncă (testat: aceeași oprire ca verificarea între simboluri). Memorie:
   logit-urile tuturor pozițiilor (`deferred_bytes`; pentru 18 poziții ~1,2 MB, pentru 512 ~32 MB).

## 4. Viteza așteptată pe placă (ESTIMARE, nemăsurată)

| Varianta (OS-ul alege) | ms pe simbol (estimat) | simboluri/s |
|---|---|---|
| azi (`os-307`) | 294,8 (+3 870 verificare) | 3,1 (0,24 cu verificarea) |
| jurnal ținut + o golire pe simbol + umbra | ~23 + ~17 + ~1 + ~5 ≈ **45** | **~20–22** |
| idem, golire la 16 simboluri sau la final | ~23 + ~1 + ~1 + ~5 ≈ **30** | **~30–34** |
| + verificarea cu motorul optimizat al OS-ului (72,7 simb/s ≈ 14 ms) între simboluri | +14 | ~16–23 |
| + verificarea după placă (`run_check_after`) | +0 la generare (verificarea separat) | ca rândul 2 sau 3 |
| plafonul: doar lansările | 22,9 | 43,9 |

Urcarea greutăților: de la 7,3 s la ~2 s (estimat). Siguranța estimării: ~70 % că `os-308` dă peste 15
simboluri/s fără verificarea de referință între simboluri; dacă verificarea de referință (3,87 s) rămâne între
simboluri, viteza „cu verificare” rămâne ~0,25 orice am face în driver.

## 5. Ce n-am făcut (și de ce)

- **Fereastra VRAM fixă** (o singură poziție, fără mutări): după ce rândurile nu mai sunt sincronizate pe loc, o
  mutare costă microsecunde; câștigul ar fi sub 1 ms pe simbol, iar codul OS-ului (`VramWindow::new`/`restore` pe
  fiecare citire) ar trebui schimbat. Rămâne o idee pentru după `os-308`, dacă cifrele `kt` o cer.
- **Mai puține sincronizări între lansări / fără invalidarea dinaintea fiecărei lansări:** ar schimba felul de
  trimitere care a trecut I14 pe placă (C-139, `os-306`); cere regula „trei surse” și o probă separată. Nu acum.
- **Fără recitirea cuvintelor scrise în VRAM:** ar slăbi verificarea; umbra dă același câștig fără să renunțe la ea.
- **Nimic pe placa reală:** testul îl face ciclul automat după ce OS-ul leagă (C-146 → `os-308`).

## 6. Riscuri

- Umbra presupune că placa nu scrie înapoi în sloturile lansărilor (QMD/cbuf). Nu știm de niciun câmp pe care placa să-l
  rescrie acolo, dar nu e verificat (nici în surse, nici pe placă): OS-ul poate reciti la final toată zona și compara cu umbra
  (`changes(pa, citit)` gol = identic); dacă diferă, oprește umbra și o spune în jurnal.
- Golirea mai rară a jurnalului: la o blocare lipsesc ultimele ≤ 64 de rânduri de rutină de pe stick (sunt în
  memoria jurnalului). Pentru probele noi, cu pași necunoscuți, OS-ul nu pornește rutina (implicit e oprită).
