# C-157 punctul 7 — generarea fără drum la procesor, primul pas: simbolul ales rămâne pe placă

Sesiunea 170, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE** (`gpu-sim`); viteza e **DEDUS** (nemăsurată). Nimic nu e predat încă pentru placă.

## De ce
Azi (os-353, cel mai bun rezultat: 635,90 simb/s), fiecare simbol e o trimitere separată: procesorul scrie lista celor ~89 de
lansări, sună soneria, **așteaptă** ultima înregistrare, citește cei 16 octeți ai lui argmax, află simbolul și abia apoi scrie
lista simbolului următor (simbolul intră în `cbuf`-ul lansării de încorporare `Embed`: adresa rândului `emb_tok + simbol·rând`).
Placa stă degeaba cât lucrează driverul: placa 1 075 µs + driverul 390 µs + jurnalul 63 µs pe simbol.

Etapa 5 din planul de performanță („generare fără drum la procesor”) cere: simbolul ales pe placă, N simboluri pe o singură
sonerie, procesorul citește 16 octeți. Lanțul P7 (QMD-uri legate între ele) **nu** e necesar pentru asta: singurul lucru din lista
unui simbol care depinde de simbolul de dinainte este rândul de încorporare. Pozițiile (adresele din cache-uri, lungimea atenției)
le știe procesorul dinainte.

## Ideea (scrisă înainte)
1. **Program nou `embed_dev`** (`tools/nak/embed_dev.comp`, Mesa NAK 25.0.7): ca `embed.comp`, dar **citește indicele simbolului din
   memorie** (cuvântul 0 al rezultatului argmax al poziției de dinainte), nu primește adresa rândului de la procesor.
   Argumente: tabela de simboluri 0x28, rezultatul argmax 0x30, rândul poziției 0x38, x 0x40, n 0x48, vocabularul 0x4c,
   lățimea unui rând în octeți 0x50. Dacă indicele nu e în vocabular (argmax n-a găsit simbol: `0xffffffff`, sau orice valoare
   greșită) programul **nu citește nimic din tabelă** și scrie **NaN** în tot x: calculul de mai departe iese NaN, argmax-ul
   următor spune „niciun simbol” → generarea se oprește cinstit (`LogitiNefiniti` / `Eroare`), niciodată un simbol inventat.
   Calculul unei valori e același ca în `embed.comp` (aceeași expresie) → biți identici.
2. **Modulul `nvidia_ga104::kronos_batch`** (doar adăugări): lansarea `embed_dev`, lista unui lot de poziții (lista de azi
   `lipita-ln-32` pentru fiecare poziție, cu `Embed` înlocuit de `embed_dev` acolo unde simbolul vine de pe placă, + argmax-ul
   fiecărei poziții în **locul ei** din bufferul de rezultate: `rezultate + 16·poziție`), citirea rezultatelor.
   `Prog::EmbedDev` se adaugă la `kronos_layer::Prog` (OS-ul nu potrivește exhaustiv această enumerare — verificat cu grep
   în `cyber-universe`).
3. **`Generation`** primește două funcții noi: `next_batch(max)` (ce poziții intră în lotul următor și de unde vine simbolul
   fiecăreia: de la procesor — textul dat — sau de pe placă) și `accept_batch(rezultate)` (aceleași verificări ca `accept_top`,
   poziție cu poziție; după oprire, restul lotului se aruncă).
4. **Modelul în `gpu-sim`:** `embed_dev_on_cpu`.

## Predicții (scrise acum)
1. **Compilarea:** ≤ 25 de registre, 0 memorie partajată / locală, fără barieră; calculul valorii = aceleași operații ca `embed`
   (`i2f`, `fmul.ftz`, `ffma.ftz`) → modelul din SIMULARE folosește aceeași funcție `embed_gpu`.
2. **SIMULARE, aceleași simboluri:** textul `[5, 200, 17]` + 12 simboluri, modelul de probă (3 straturi, e 64, vocabular 256):
   generarea „un simbol pe trimitere” (azi) și generarea „tot textul într-o singură trimitere” dau **aceleași simboluri**,
   **aceleași 4 cuvinte argmax** la fiecare poziție și **aceleași cache-uri K/V bit cu bit**; la fel cu loturi de 4.
3. **O singură sonerie** pentru tot textul (15 poziții), față de 15 azi; procesorul citește 16·15 octeți o singură dată.
4. **Cu modelul de citire veche al plăcii pornit** (`stale_line`) și invalidarea înaintea fiecărei lansări (`EachLaunch`, felul
   care a trecut I14): același rezultat.
5. **Greșeli plantate prinse:** (a) `embed_dev` care citește locul greșit (poziția proprie în loc de cea de dinainte) → alte
   simboluri, prins de comparație; (b) argmax care scrie un indice în afara vocabularului → `embed_dev` scrie NaN, generarea se
   oprește la acea poziție cu `Eroare`, niciun simbol după ea; (c) argmax „niciun simbol” → oprire `LogitiNefiniti`.
6. **Viteza (DEDUS, de verificat pe placă):** placa ~1 075 µs pe simbol + ~16 µs argmax (rămâne) + `embed_dev` ≈ `embed`.
   (a) Lot scris tot înainte, o sonerie: driverul scrie listele (din cei 390 µs, partea „scriere” — umbra, QMD-urile — e
   NECUNOSCUTĂ pe bucăți; presupun ~150–300 µs pe simbol) dar nu mai așteaptă și nu mai citește între simboluri, jurnalul se scrie
   la sfârșit → **~1 250–1 400 µs pe simbol = ~710–800 simb/s**. (b) Cu scrierea lotului următor cât placa lucrează la cel de acum
   (două zone de lansări, pe rând): **~1 100–1 150 µs = ~870–910 simb/s**. Față de 635,90: ×1,1–1,4.

## Ce NU face
Nu schimbă niciun program existent, nici listele de azi, nici `Generation::run` / `run_top`. Nu atinge lanțul P7, nici
`SM_GLOBAL_CACHING_ENABLE`. Nimic nou trimis la placă; predarea către OS vine după, într-o notă separată, ca un singur lucru nou
pe test (programul `embed_dev` + o trimitere cu mai multe poziții).

## Rezultatul (după cod)
_(se completează după teste)_
