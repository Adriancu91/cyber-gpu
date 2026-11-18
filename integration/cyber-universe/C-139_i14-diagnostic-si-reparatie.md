# C-139 — I14 după `os-305`: reparația, scara și diagnosticul din jurnal (pentru sesiunea OS, `os-306`)

**Doar SIMULARE.** Nimic de aici nu a rulat pe RTX 3070 Ti. Toate schimbările din crate-uri sunt **adăugări** (nimic redenumit sau șters); OS-ul fixat
pe `f732365` compilează și merge neschimbat. Analiza: `docs/os-305_ANALIZA_I14.md`. Modelul de urmat, linie cu linie: funcția `step` din
`crates/nvidia-ga104/tests/sim_i14_diag.rs` (testul `the_step_as_the_note_describes_it` o rulează pe toate cazurile de mai jos).

## Pe scurt

La `os-305` simbolul a ieșit altul și jurnalul nu spune unde. Ipoteza principală (nedovedită): placa a citit valori vechi, fiindcă golim cache-urile
o singură dată pe trimitere. Pasul `i14-simbol` primește trei lucruri:
1. **scara** (`kronos_diag::ladder`), o dată, înaintea textelor: primul simbol trimis ca la `os-305` (probă de control), apoi cu reparația, apoi —
   doar dacă trebuie — în felurile de rezervă; alege felul în care rulează textele și scrie în jurnal unde e prima etapă greșită;
2. **textele trimise în felul ales** (`kronos_diag::send_token`);
3. **explicația** (`kronos_diag::explain`) la primul simbol al unui text care nu trece: argmax placă / vectori / procesor, primele 8 logit-uri, abaterea
   maximă, etapa — exact rândurile cerute de C-139.

## Ce e nou în crate-uri (fixați `cyber-gpu` pe commit-ul din `stare/cyber-gpu.md`)

- `nvidia_ga104::transfer::encode_sequence_fresh` + `sequence_words_fresh(n)` (= `8 + 12·n` cuvinte): secvența veche plus
  `INVALIDATE_SHADER_CACHES_NO_WFI` înaintea fiecărei lansări. `encode_sequence` rămâne cum era.
- `nvidia_ga104::kronos_diag` (modul nou), mai jos `kd`:
  - `SeqMode { Once, EachLaunch, PerLayer }`, `EmbedBy { Program, Driver }`, `Way { mode, embed }` (`Way::OS_305`, `Way::REPAIR`, `name()`);
  - `encode(mode, pb, qmds, records_va, first)`, `words(mode, n)`, `push_bytes(mode, straturi, lansari)`;
  - `trait TokenIo { submit, read, write, journal }` — cele 4 lucruri pe care le dați voi (mai jos);
  - `ladder(io, &ctx) -> Ladder { probes, chosen: Option<Way> }`, `send_token(io, &w, &vas, simbol, poz, way, &[lansari_in_plus])`,
    `explain(io, &ctx, &logituri_citite) -> Explained { probe, vram_logits, vram_pass }`, `locate(io, &ctx, way.embed) -> Vec<Probe>`;
  - `ProbeCtx { w, vas, prev, token, pos, cpu_logits, want }`; `Probe::lines()` / `Ladder::lines()` pentru ecran și raport;
  - `parse_journal(text)` (folosit de unelte).
- `nvidia_ga104::kronos_layer::copy_launch(src, dst, octeti)`: copierea ca `Launch` (aceiași octeți ca `transfer::copy_cbuf` + `copy_qmd`).
- `gpu_compute::layer::{LayerTrace, trace_gpu, attention_gpu}`, `gpu_compute::model::{TokenTrace, trace_gpu}` (folosite de diagnostic).
- `gpu_sim::cmdproc::CommandProcessor::model_stale_reads(linie)` + `stale_bytes_served()`: ipoteza „date vechi” în modelul de GPU, **oprită implicit**.

## Ce se schimbă în `gpu_i14.rs`

### 1. `TokenIo` pe `BoardRunner` (tot ce atinge placa rămâne codul vostru de azi)

```rust
impl<V: Mmio + ?Sized, K: Clock + ?Sized> kd::TokenIo for BoardRunner<'_, '_, V, K> {
    /// O trimitere: cbuf + QMD ale lansării i scrise în locul `first_slot + i`, înregistrările șterse, secvența
    /// codată cu `kd::encode(mode, ..)` (NU `transfer::encode_sequence`), `run_records`, `analyze_sequence`.
    fn submit(&mut self, launches: &[kl::Launch], first_slot: usize, mode: kd::SeqMode) -> GpuResult<()>;
    /// Cuvinte din VRAM citite prin fereastră (cum citește I13 x-ul): pentru fiecare (VA, cuvinte).
    fn read(&mut self, ranges: &[(u64, usize)]) -> GpuResult<Vec<Vec<u32>>>;
    /// Cuvinte scrise în VRAM (`put_bytes` în colector; ajung pe placă la următorul `kick_and_wait`).
    fn write(&mut self, va: u64, words: &[u32]) -> GpuResult<()>;
    /// Jurnalul pasului (cel cu scriere pe stick).
    fn journal(&mut self) -> &mut Journal;
}
```

- `submit` e bucla de azi din `run_token`, cu trei diferențe: locurile sunt `first_slot + i` (nu `j`), codarea e `kd::encode` / `kd::words`, iar
  valoarea de început a înregistrărilor crește **pe trimitere** (`TOKEN_FIRST + STRIDE · trimiteri_facute`), nu pe simbol — un simbol poate avea
  mai multe trimiteri. `STRIDE` trebuie să rămână ≥ lansările unei trimiteri + 1. O înregistrare care nu vine = `Err(Timeout)`, ca azi.
- Zona lansărilor rămâne de `launches_per_token(straturi) + 1` locuri. Copierea logit-urilor nu mai are loc scris dinainte (`down_cb_va`): e ultima
  lansare a listei (`kl::copy_launch(vas.logits, host_logits.va, logits_bytes)`) și `submit` o scrie ca pe oricare alta.
- `read` / `write` au nevoie de VA → PA pentru blob (zona de lucru, vectorul final, logit-urile) și pentru cele 2 cache-uri ale fiecărui strat.
- Duratele pentru `TokenOutput`: adunate în `submit` (prima înregistrare a primei trimiteri, ultima a ultimei, suma din `analyze_sequence`).

### 2. Scara, înaintea textelor (doar pentru modelul de control)

```rust
let tok0 = runs[0].prompt[0];
let cpu0 = cm::step_cpu(&weights, &mut Caches::new(layers, e), tok0, 0)?;
let prev = Caches::new(layers, e);
let ctx = kd::ProbeCtx { w: &weights, vas: &vas, prev: &prev, token: tok0, pos: 0,
                         cpu_logits: &cpu0, want: runs[0].argmax.map(|a| a[0]) };
let ladder = kd::ladder(&mut runner, &ctx);          // scrie singură rândurile în jurnal
lines.extend(ladder.lines().into_iter().map(|l| format!("{p}: {l}")));
let Some(way) = ladder.chosen else {
    return Err(bad("I14: niciun fel de a trimite simbolul nu trece (diagnosticul e in jurnal)"));
};
```

Scara trimite cel puțin 2 simboluri întregi (control + reparație; fără copierea în RAM — citește logit-urile din VRAM) și cel mult 13 trimiteri la modelul de control.
O trimitere care nu se termină o oprește pe loc (`chosen = None`). `kronos --gpu` (alt model) folosește `way`-ul ales aici, fără altă scară.

### 3. Un simbol al unui text

```rust
ram_write(&self.host_logits, &vec![0xee; bytes]);
let down = kl::copy_launch(vas.logits, self.host_logits.va, self.logits_bytes)?;
kd::send_token(self, &weights, vas, tok, pos, way, &[down])?;   // embedding-ul driverului (dacă e cazul) + trimiterile
// logit-urile din RAM, ca azi
```

Verificarea zonei de comenzi: `kd::push_bytes(SeqMode::EachLaunch, straturi, n) ≤ push.size / 2` (în loc de `transfer::sequence_words(n)`).
Modelul de control: 1 760 B — zona de 16 KiB ajunge. **#513: 9 440 B → `PUSH_BYTES` trebuie 32 KiB** (64 KiB pentru 24 de straturi); altfel
`kronos --gpu` pe #513 e refuzat de această verificare (nu de placă).

### 4. Ordinea verificărilor și explicația

- Scoateți din `run_token` oprirea „argmax diferit de vectorii lui Kronos”: lăsați `Generation` să judece întâi față de procesor (spune și cât de
  departe sunt logit-urile). Comparația cu `vectori.json` se face după ce textul s-a terminat curat (țineți argmax-ul fiecărei poziții).
- La `Stop::NuTrece { pos, .. }` sau `Stop::LogitiNefiniti { pos }`:

```rust
let tok = generation.seq[pos];
let cpu = check.cpu_logits(tok, pos)?;      // aceleași rânduri de cache, aceleași logit-uri
let ctx = kd::ProbeCtx { w: &weights, vas: &vas, prev: &check.caches, token: tok, pos,
                         cpu_logits: &cpu, want: run.argmax.and_then(|a| a.get(pos).copied()) };
let ex = kd::explain(&mut runner, &ctx, &logituri_citite_de_pas);   // nu trimite nimic; scrie în jurnal
lines.extend(ex.probe.lines().into_iter().map(|l| format!("{p}: {l}")));
if ex.vram_pass && citesc_din_ram {
    // doar copierea în RAM e greșită: textul o ia de la capăt cu logit-urile citite din VRAM (fără lansarea de copiere)
} else {
    if matches!(ex.probe.verdict, kd::Verdict::Mostenit(_)) {
        kd::locate(&mut runner, &ctx, way.embed);      // prefixe pe straturi la acea poziție; numește lansarea
    }
    return Err(..);
}
```

`prev` = cache-urile motorului de pe procesor așa cum sunt după pasul lui la `pos` (diagnosticul citește doar rândurile dinaintea lui `pos`).

### 5. Ce raportează pasul (regula de cinste)

| Cum s-au terminat textele | `i14-simbol` |
|---|---|
| toate trec, `way.embed == Program`, logit-urile din RAM | **ok** — rândul de rezultat spune felul: `way.name()` |
| trec doar cu embedding-ul scris de driver (`way.embed == Driver`) | **ESEC**: „I14: programul embed e gresit pe placa; cu embedding-ul scris de driver textele trec” |
| trec doar cu logit-urile citite din VRAM | **ESEC**: „I14: copierea logit-urilor in RAM e gresita; cu logit-urile din VRAM textele trec” |
| altfel | **ESEC** cu motivul; diagnosticul e în jurnal |

Un fel de a **trimite** aceleași lansări e reparație (pasul poate fi ok). O **ocolire** a unei lansări greșite nu e: textele rulează până la capăt doar
ca să aflăm în aceeași pornire dacă restul e bun, iar `kronos --gpu` nu pornește.

## Ce NU schimbați la `os-306`

Așezarea blobului (`pack`), bucățile de urcare, paginile DMA, ordinea pașilor dinainte. Proba de control trebuie să repete `os-305` cât mai exact.

## Scrieri noi pe placă

Niciun registru nou și nicio metodă nouă. `INVALIDATE_SHADER_CACHES_NO_WFI` (aceeași valoare, `0x1011`) apare mai des în zona de comenzi. Mai multe
sonerii ale canalului de lucru și mai multe mutări ale ferestrei VRAM în `i14-simbol` (verificatorul lanțului le permite). Doar pe rezervă: E cuvinte
scrise la x prin colector (ca la I13).

## Ce apare în jurnal

Rânduri `valoare i14d … = 0x…` și `text i14d … = "…"` (numele în `kd::rows`; numerele reale ca biți). Pe ipoteză: vreo 150 de rânduri pentru scară
(control „date vechi” + reparația care trece), cam 2,5 s. Nu ating verificatorul lanțului (`chain_log`). Se citesc cu:

```
cargo run --release -p nvidia-ga104 --example verifica_jurnal_lant -- <gpu-jurnal.txt> <adresa_GspFwWprMeta> [apeluri_rm]
cargo run --release -p nvidia-ga104 --example analiza_i14 -- <gpu-jurnal.txt>      # cu KGR1_VECTORI, compară și cu semnăturile ipotezelor
```

## Teste în QEMU (SIMULARE)

- Cu modelul de GPU neschimbat: scara = 2 probe care trec, `chosen = Way::REPAIR`, textele trec ca înainte.
- Cu `CommandProcessor::model_stale_reads(128)` (ipoteza): proba de control = „DATE VECHI”, etapa „normalizare 1 (stratul 1, lansarea 0)”, cu
  valoarea veche a lui x; reparația trece; textele trec. Așa arată „ca pe placă, dacă ipoteza e adevărată”. (Pe placă verdictul poate spune și
  „o parte din ieșire” — cache-urile sunt pe fiecare procesor al plăcii; modelul are unul singur.)
- Testele noastre plantează și: `embed` greșit, o lansare greșită în primul / al doilea strat, o lansare greșită doar de la poziția 1, copierea în
  RAM care nu ajunge, o trimitere care nu se termină — le puteți relua cu aceleași funcții greșite (`tests/sim_i14_diag.rs`).

## Întrebări pentru OS

1. Vă e de ajuns `TokenIo` așa (4 operații)? Dacă `read` prin fereastră e incomod pentru 16 384 de cuvinte, spuneți — pot cere logit-urile separat.
2. Mărirea zonei de comenzi la 32 KiB o faceți la `os-306` sau doar când rulează #513?
