# Pentru sesiunea OS — os-295: motivul pasului eșuat ajunge pe stick (cyber-gpu `45dfc77` sau mai nou)

Sesiunea Cyber-GPU 93, 12 noiembrie 2026. Analiza jurnalului `gpu-jurnal-010.txt`: `docs/os-295_ANALIZA_I9.md`. Pe scurt: **I8 a trecut** (varianta 1),
golden a trecut, **I9 s-a oprit la al 11-lea mesaj RM (`PROMOTE_CTX`, jumătatea virtuală) și motivul nu era în jurnal** — `run_chain_ready` scria doar
`pas i9-canal-lucru: ESEC`, textul erorii mergea doar în `lines` (ecranul), iar ecranul înghețase de la pornirea GSP-RM.

## 1. Fără nicio schimbare în OS (doar legarea)

`kernel/Cargo.toml`: cele 5 crate-uri la `rev = "45dfc77"` (sau mai nou). De atunci `nvidia_ga104::chain::run_chain_ready` scrie singur, după fiecare
`pas X: ESEC`:

```
text motivul esecului = "eroare hardware 0x00000057"
```

(sau, pentru un text mai lung de 32 de octeți, bucăți: prima `motivul esecului`, următoarele `motivul esecului (urmare)`). La fel pentru pașii opriți de o
poartă („blocat in plan …”, „refuzat: pas care scrie, iar modul e doar citire …”). Interfața publică e neschimbată (doar adăugări:
`chain::{STEP_FAIL_REASON, STEP_FAIL_REASON_MORE}`, `gpu_core::diag::Journal::push_text`, `gsp::golden::{QueueTransportBellKeep, SKIPPED_KEEP_MAX,
journal_skipped, SKIPPED_TOTAL, SKIPPED_AT}`, `variant_log::VariantRun::{fail_reasons, fail_reason}`).

Sinkul OS-ului (C-51) primește evenimentele `Event::Text` ca pe oricare altele — rândul apare pe stick ca `text motivul esecului = "..."`.

## 2. Opțional, o schimbare mică în `gpu_golden.rs` (`OsRm::call`): și mesajele GSP-RM sărite în I9

Când GSP-RM refuză o comandă, de obicei a trimis și un mesaj separat (NOCAT, RC) care spune de ce. Azi `QueueTransportBell` păstrează doar numerele
funcțiilor (`events`), pe care `OsRm` le aruncă. Cu `QueueTransportBellKeep` mesajele rămân întregi și se pot jurnaliza după apel (jurnalul nu mai e
împrumutat de `GuardedBar0` în acel moment):

```rust
use nvidia_ga104::gsp::golden::{self, QueueTransportBellKeep, RM_REPLY_US};

fn call(&mut self, function: u32, payload: &[u8]) -> GpuResult<Vec<u8>> {
    flush(&mut *self.vbar, &mut *self.journal, self.permit, self.zone, self.pending, &mut *self.words)?;
    let (r, rung, skipped) = {
        let mut b = GuardedBar0::new(&mut *self.bar, &mut *self.journal, self.permit)?;
        let poll = &mut self.poll;
        let prep = &mut *self.prep;
        let mut t = QueueTransportBellKeep {
            q: &mut prep.queues,
            mem: &mut prep.shared,
            clock: &mut *self.clock,
            timeout_us: RM_REPLY_US,
            poll: |m: &mut dyn SharedMem| poll(m),
            events: Vec::new(),
            bell: || falcon::ring_doorbell(&mut b),
            rung: 0,
            skipped: Vec::new(),
        };
        let r = t.call(function, payload);
        (r, t.rung, t.skipped)
    };
    *self.calls += rung;
    // os-295: ce a spus GSP-RM pe drum (NOCAT decodat, LibOS print, altele numite); nimic daca n-a spus nimic.
    golden::journal_skipped(&mut *self.journal, "apel RM", &skipped);
    r
}
```

Rândurile scrise: `valoare mesaje GSP-RM sarite in asteptarea raspunsului RM: total = N`, apoi pentru fiecare `text mesaj GSP-RM sarit: la pasul = "apel RM"`
+ conținutul (ca rândurile NOCAT din I8). Nicio scriere nouă în placă; aceeași sonerie, același protocol de cozi.

## 3. Ecranul înghețat (pentru informare; decizia e a sesiunii OS)

Ipoteza noastră: GSP-RM, la inițializare, a preluat motorul de afișare (NOCAT „Display Engine”, `NV_ERR_INVALID_DMA_SPECIFIER`) → framebuffer-ul UEFI nu
mai e afișat; OS-ul a mers mai departe (jurnal complet, Bus Master oprit, tur după). Nu avem o setare publică, cu sursă, care să țină GSP-RM departe de afișaj
— nu propunem valori inventate. Propunere pentru OS: după `pas i8-gsp-init: ok` ecranul se consideră pierdut; raportul e jurnalul de pe stick + `trimite` la
pornirea următoare (a mers azi). Dacă OS-ul vrea un semn vizibil la final, trebuie să vină din altceva decât ecranul (de ex. portul serial, dacă există).

## 4. Testul următor pe placă

Un singur test: Release OS cu `cyber-gpu` ≥ `45dfc77`, `gpu lant --varianta 1`, `trimite` după repornire. În jurnal așteptăm `text motivul esecului = "…"`
imediat după `pas i9-canal-lucru: ESEC` (dacă I9 mai eșuează). `compara_variante 0x9c66000 gpu-jurnal-010.txt <jurnalul nou>` arată motivul în tabel.
