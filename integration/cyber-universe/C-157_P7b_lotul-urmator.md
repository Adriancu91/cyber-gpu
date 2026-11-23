# Pentru OS: lotul următor scris cât placa lucrează — două zone (C-157 punctul 7, partea a doua)

Cyber-GPU, sesiunea 185, 23 nov 2026. Commit de legat: `aba9aeb` (ramura `g0-g2-fundatie`). **Testat doar în SIMULARE** (`gpu-sim`,
T194, 931 PASS); **nimic din asta n-a rulat pe RTX 3070 Ti.** Planul, predicțiile și rezultatul: `docs/C-157_P7_LOT_URMATOR.md`.
Se leagă **după** proba „lot pe o sonerie” (`C-157_P7_simbol-pe-placa.md`, `embed_dev`) — întâi aceea pe placă.

## Ce e, pe scurt
Cu `embed_dev`, lotul k+1 nu depinde de rezultatele lotului k (simbolurile le citește placa). Deci OS-ul poate scrie lotul k+1
**cât placa lucrează la k**: două zone (lansări + înregistrări), lotul k în zona `k mod 2`, cel mult 2 loturi la placă. Bucla e în
bibliotecă (`kronos_batch::run_pipelined`); OS-ul scrie doar două funcții.

## Ce e nou pentru placă (exact)
1. **Două trimiteri la placă în același timp** (azi: niciodată mai mult de una). Aceleași liste, același `SeqMode::EachLaunch`
   (obligatoriu, ca la lotul simplu), același `embed_dev`. Nimic în registre, nimic prin GSP, niciun program nou.
2. Zona de lansări și înregistrările de **2 ×** mărimea unui lot: `2 · K · launches_per_position(L)` locuri de 512 B
   (#513, K = 4: 704 locuri = 352 KiB) și `2 · (K · 88 + 1)` înregistrări de 16 B. Pushbuffer-ul: loc pentru 2 loturi
   (`WorkQueue` nu suprascrie o bucată încă la placă).

## Cum se leagă (în OS)
```rust
use nvidia_ga104::kronos_batch::{self as kb, BatchIo, Input, TwoZones};
struct Os { /* canalul, zona de lansări, înregistrările, rezultatele */ }
impl BatchIo for Os {
    // scrie batch_launches(p0, inputs) în locurile zones.first_slot(zona).., înregistrările zonei, sună — FĂRĂ așteptare
    fn send(&mut self, zona: usize, p0: usize, inputs: &[Input]) -> GpuResult<()> { ... }
    // așteaptă gardul zonei (ultima înregistrare), citește 16·n octeți de la rezultate + 16·p0, TopOutput pe poziție
    fn wait(&mut self, zona: usize, p0: usize, n: usize) -> GpuResult<Vec<TopOutput>> { ... }
}
let per = kb::launches_per_position(layers);
let mut zones = TwoZones::new(per * K)?;
let run = kb::run_pipelined(&mut gen, &mut os, &mut zones, per, K);   // run.stop, run.sent, run.overlapped, run.drain_error
```
Reguli pe care le ține bucla (testate): zona unui lot se rescrie doar după ce gardul lotului de dinainte din ea a fost citit;
după o oprire (EOS, eroare, „niciun simbol”) nu mai trimite nimic, dar **așteaptă** lotul deja trimis și îi aruncă rezultatele;
o eroare la acea așteptare vine în `run.drain_error` (de scris în jurnal). Un `send` refuzat nu trebuie să sune.

## Proba pe placă (propunere: un singur lucru nou pe test)
- Același text ca proba „lot”, loturi de 4: **aceleași simboluri și aceleași 4 cuvinte argmax** ca rularea „o poziție pe trimitere”
  din același ciclu; la prima diferență: înapoi la lotul simplu, raport.
- Rânduri `kt` cerute: `kt lot2 sonerii`, `kt lot2 suprapuse` (`run.overlapped`), `kt lot2 risipite` (`run.drained_positions`),
  timpul pe poziție din înregistrări, driverul pe lot (scrierea, separat de așteptare), simb/s.

## Ce aștept (DEDUS, de verificat)
Scrierea lotului se ascunde după calculul lotului de dinainte → **~1 100–1 150 µs pe poziție = ~870–910 simb/s** (față de ~710–800
cu lotul simplu și 635,90 azi). Prețul: la oprire, până la un lot de poziții calculate degeaba (K = 4: ~4,4 ms, o dată pe generare).
