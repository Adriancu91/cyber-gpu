# C-147 etapa 0 → C-148 punctul 3: măsurătorile pentru `os-309` (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 119 (19 nov 2026). Detaliile și sursele: `docs/C-147_ETAPA0.md`. Totul e **adăugat** (nimic redenumit
sau șters; OS-ul fixat pe un commit mai vechi compilează neschimbat), testat **doar în SIMULARE** (`tests/sim_c147.rs`, T135).
Nimic de aici nu schimbă calculele sau felul de trimitere al lui `kronos --gpu`. Commit-uri: `9c5e8e8`, `c2c228b` (+ docs).

## (a) Timpul pe fel de lansare → rânduri `kp`
```rust
use nvidia_ga104::kronos_perf::{self as kp, KindProfile};
// o dată, înaintea generării:
let kinds = kp::token_kinds(m.layers.len(), way.embed == EmbedBy::Program, !self.from_vram); // 196 la #513
let mut prof = KindProfile::new();
// în BoardRunner::run_token: adună `t` (transfer::analyze_sequence) din FIECARE trimitere a simbolului, în ordine,
// într-un Vec<u64> al simbolului (azi se adună doar suma în `self.timing`); la sfârșitul simbolului:
prof.add_token(&kinds, &durate_simbol)?;   // lungime diferită = eroare, nu o cifră ghicită
// la final, o singură dată, după rândurile kg/kv/kt:
prof.journal(journal);                      // 44 de rânduri `kp …` la #513
```
În raportul ciclului, tabelul „timp pe fel de lansare”: `kp::parse_journal(text).kinds` dă `(fel, µs pe simbol, ns pe lansare)`.

## (b) Costul lansării goale → rânduri `kl`
După `launch_cost::analyze(...)` din `i10-cost-lansare`: `kp::journal_launch_cost(&cost, journal)` (5 rânduri). Ecranul rămâne ca azi.

## (c) Starea de performanță → rânduri `ks` (doar citire, la SFÂRȘITUL lanțului)
După ce `kg`/`kv`/`kt`/`kp` sunt pe stick, pe clientul și subdevice-ul canalului de lucru:
```rust
use nvidia_ga104::gsp::perf::{self as gp, When};
let sent = gp::encode_get_pstate(h_client, h_subdevice);
let reply = /* același drum GSP_RM_CONTROL ca celelalte comenzi RM */;
gp::journal_pstate(journal, When::End, gp::read_pstate(&sent, &reply)?); // refuz = rând 8000xxxx, NU eroare a lanțului
journal.push(Event::Note(gp::rows::NO_LEVELS)); // opțional: că nivelurile și temperatura nu se citesc (fără cale documentată)
```
Doar o eroare de transport sau un răspuns pentru altă comandă e eroare (`?`).

## (d) Copierea VRAM → VRAM → rânduri `kc`
```rust
use nvidia_ga104::kronos_perf::VramCopy;
let bytes = VramCopy::size_for(blob_bytes);              // 64 MiB la #513 (greutățile au ~106 MiB)
let c = VramCopy::new(blob_va, scratch64_va, bytes, kp::VRAM_COPY_REPEATS)?; // scratch: 64 MiB liberi în VRAM
// programul copy_b128 e deja pe placă (I11); scrie c.cbuf() și c.qmd(prog_va, cbuf_va) într-un slot,
// c.encode(&mut pb, qmd_va, rec_va, first) → o sonerie → înregistrările (c.record_bytes()):
let s = c.analyze(&recs, first)?;
let got = c.spots().map(|o| vw.read_u32(scratch64_pa + o)); // 3 cuvinte prin fereastră
let wrong = c.wrong_spots(&blob_din_ram, &got);
c.journal(&s, Some(wrong), journal);                     // 11 rânduri `kc`
```
Locul propus: după `kronos --gpu`, înainte de citirea `ks` (copierea nu atinge greutățile, doar le citește).

## Viteza maximă (C-147 punctul 4) — NU pentru `os-309`
`gsp::perf::BoostPlan` e **oprit implicit** (`encode_boost` → `Blocked`). Se leagă într-un Release separat doar după ce centrul de comandă
confirmă în hub că `ks`/`kc` arată o stare joasă (C-148 completare). Forma implicită e cea a driverului NVIDIA
(`INTERNAL_PERF_BOOST_SET_2X`), 600 s, ștergerea la final în aceeași formă; ordinea în `docs/C-147_ETAPA0.md` §3.

## Kronos #651 (C-150 / C-148 completare 2)
`kronos_speed::ModelLimits::k651()`: 388 de lansări pe simbol, zona de comenzi 64 KiB, **pasul valorilor pe sonerie 389 → OS-ul are nevoie
de `STRIDE = 0x200` pentru #651** (azi `0x100`, cât ajunge la #513). `check(...)` numește orice limită neîndeplinită înainte de urcare.
