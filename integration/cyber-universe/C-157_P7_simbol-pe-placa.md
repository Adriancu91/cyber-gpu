# Pentru OS: simbolul ales rămâne pe placă — mai multe poziții pe o singură sonerie (C-157 punctul 7, etapa 5 / P9)

Cyber-GPU, sesiunea 170, 23 nov 2026. Commit de legat: `0d22357` (ramura `g0-g2-fundatie`, CI verde; codul: `40a82b0`). **Testat doar în SIMULARE**
(`gpu-sim`, T180, 864 PASS); **nimic din asta n-a rulat pe RTX 3070 Ti.** Planul și predicțiile: `docs/C-157_P7_SIMBOL_PE_PLACA.md`.

## Ce e, pe scurt
Azi fiecare poziție = o trimitere: lista (~89 de lansări) → soneria → așteptarea → citirea celor 16 octeți ai lui argmax →
simbolul intră în `cbuf`-ul lansării `Embed` a poziției următoare. Singurul lucru care depinde de simbolul de dinainte e
**rândul de încorporare**. Programul nou `embed_dev` citește simbolul direct din rezultatul argmax al poziției de dinainte, în
memorie, deci **tot textul (sau un lot de K poziții) intră într-o singură trimitere**, iar OS-ul citește `16 × poziții` octeți o dată.

## Ce e nou pentru placă (exact)
1. **Un program nou:** `embed_dev` — `nvidia_ga104::kronos_batch::EMBED_DEV` (`tools/nak/out/embed_dev.sm86.sass`, 1 088 de
   octeți, Mesa NAK 25.0.7, SM 8.6), SHA-256 `5eb883c68fb91fd8e0919d51357d171b62455f20041eab731da46b3cbaddfce0`; 19 registre,
   blocuri de 256 (`Prog::EmbedDev.block()`), fără memorie partajată, fără barieră (`Prog::EmbedDev.shared() == None`).
   Calculul valorii e identic cu `embed` (în codul compilat: aceleași `i2f`, `fmul.ftz` pe rândul poziției, `ffma.ftz` pe rândul
   simbolului) → x iese bit cu bit ca la `embed` cu același simbol.
2. **O trimitere cu mai multe poziții** (până acum: una pe trimitere). Mărimi: `kronos_batch::launches_per_position(L)` lansări pe
   poziție (#513: 87 + 1 = **88**), deci zona de lansări și pushbuffer-ul cresc de K ori pentru un lot de K poziții; înregistrările
   la fel (`n + 1`).
3. **Bufferul de rezultate:** 16 octeți pe poziție, `rezultate + 16 · poziție` (în RAM-ul procesorului, ca azi rezultatul argmax);
   pentru context 512: 8 KiB. Umplut cu `0xEE` înainte (ca azi), ca un loc nescris să nu treacă drept rezultat.

Nimic altceva nou: listele de azi (`lipita-ln-32`), argmax-ul, QMD-urile, invalidarea — neschimbate.

## Cum se leagă (în OS)
```rust
use nvidia_ga104::kronos_batch::{self as kb, Input};
// la încărcare: încă un program în VRAM, Prog::EmbedDev (lângă PROGRAMS_ARGMAX + Q8Fused32 + Q8Matvec32)
while let Some((p0, inputs)) = gen.next_batch(K) {                    // K = cât încape în zona de lansări
    let l = kb::batch_launches(kb::TokenList::FusedLn32, &shape, vocab, ctx, &vas, rezultate_va, p0, &inputs)?;
    // o singură trimitere, felul EachLaunch (invalidare înaintea fiecărei lansări), apoi:
    let buf = citește(rezultate, 16 * ctx);                            // o singură citire
    let res = kb::read_results(&buf, p0, inputs.len())?;
    // TopOutput pe poziție: start = ora înregistrării j·88, sfârșit = ora înregistrării (j+1)·88
    gen.accept_batch(Ok(outs));                                         // aceleași verificări ca accept_top
}
```
`next_batch` pune `Input::Cpu(simbol)` pentru textul dat (și pentru simbolurile deja citite dintr-un lot de dinainte) și
`Input::Board` pentru pozițiile al căror simbol vine din argmax-ul poziției de dinainte, din același lot.

**OBLIGATORIU `SeqMode::EachLaunch`.** T180f: cu o singură invalidare pe trimitere (`Once`) și modelul de citire veche al plăcii
pornit, lotul dă **alte simboluri fără nicio eroare** (citește locuri vechi). Citirile lui `embed_dev` (rezultatul, tabela) sunt
`ld.global.constant` (prin memoria rapidă de constante), ca la `embed`.

## Proba pe placă (propunere: un singur lucru nou pe test)
- Rândul pentru `teste.txt`: ceva ca `test-placa --lot` (numele îl alegeți voi), pe #513, `lipita-ln-32`, `--argmax-fara-copie
  --dieta2`, același text ca azi.
- **Aceleași 16 simboluri** (și aceleași 4 cuvinte argmax pe fiecare poziție) ca rularea „o poziție pe trimitere” din același ciclu;
  la prima diferență: înapoi la felul de azi, raport.
- Întâi loturi de **4** poziții, apoi **tot textul** într-o trimitere (dacă încape în zonă).
- Rânduri `kt` cerute: `kt lot pozitii K`, `kt lot sonerii`, timpul pe poziție (din înregistrări), driverul pe lot (scrierea listelor,
  separat de așteptare), simb/s.

## Ce aștept (DEDUS, de verificat)
Placa ~1 075 µs pe poziție (argmax rămâne ~16 µs, `embed_dev` ≈ `embed`). (a) Lot scris tot înainte, o sonerie: driverul nu mai
așteaptă / citește între poziții, jurnalul la sfârșit → **~1 250–1 400 µs pe poziție = ~710–800 simb/s**. (b) Cu scrierea lotului
următor cât placa lucrează la cel de acum (două zone de lansări pe rând) → **~1 100–1 150 µs = ~870–910 simb/s**. Față de
635,90 (os-353): ×1,1–1,4. Partea „scriere” din cei 390 µs ai driverului e NECUNOSCUTĂ pe bucăți — de aici marja mare.

## Cinste / siguranță
- Un simbol pe care argmax nu l-a găsit (`0xffffffff`) sau orice indice ≥ vocabular: `embed_dev` **nu citește nimic din tabelă** și
  scrie NaN în x → tot ce urmează iese NaN → argmax „niciun simbol” → `accept_batch` se oprește la primul loc rău
  (`Eroare` / `LogitiNefiniti`), restul lotului se aruncă. Niciun simbol inventat (T180e).
- Simbolul de sfârșit (dacă e dat) în mijlocul lotului: generarea se oprește acolo, pozițiile de după s-au calculat degeaba (pierdere de
  timp, nu de corectitudine).
- Doar adăugări în crate-uri: `kronos_layer::Prog` primește varianta `EmbedDev` (OS-ul nu potrivește exhaustiv `Prog` — verificat cu
  grep în `kernel/src/drivers/`), `kronos_gen::Generation` primește `next_batch` / `accept_batch`, `gpu_sim::cmdproc::embed_dev_on_cpu`.
