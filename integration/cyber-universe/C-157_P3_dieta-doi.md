# C-157 punctul 3 → „dieta 2”: driverul de la 759 µs spre ~350 µs pe simbol (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 136 (21 nov 2026). Detalii și cifre: `docs/C-157_P3_DIETA_DOI.md`. Totul de la noi e **adăugat** (nimic
redenumit sau șters), testat **doar în SIMULARE** (T147, `crates/nvidia-ga104/tests/sim_c157_diet2.rs`).

**Ce e nou pentru placă: nimic.** Aceleași programe, aceleași lansări, aceiași octeți în pushbuffer, aceleași scrieri în registre;
se schimbă doar ce face procesorul între două simboluri: citește mai puțin, așteaptă mai fin, calculează mai ieftin. Poate merge în
orice Release (de ex. împreună cu P6-112), în spatele unei opțiuni (propunere: `--dieta2`, oprită implicit până la proba pe placă).

## Ce am găsit (os-328, `gpu-jurnal-026.txt`, driverul = 759 µs pe simbol)
~166 µs citirea celor 89 de înregistrări prin fereastră (356 de citiri) · ~50 µs așteptarea la 100 µs · ~100–150 µs amprenta FNV-1a
în ceasul simbolului · 85 µs argmax pe numere într-un nucleu **soft-float** · restul: lansările, umbrele, ferestrele.

## Ce legi (patru bucăți independente; fiecare se poate lega singură)

### A. Două înregistrări în loc de toate (`gpu_i11.rs`, `run_records_inner`, în `wait`)
```rust
use nvidia_ga104::kronos_diet2 as d2;
// parametru nou: `plan: d2::RecordPlan` (d2::RecordPlan::ALL = ca azi) și `pos`
let b = if plan.reads_all(pos) {
    /* ca azi: toate RECORD * n_rec octeții */
} else {
    let mut b = alloc::vec![0u8; (RECORD * n_rec) as usize];   // aceeași formă; doar capetele citite
    for o in (0..RECORD).step_by(4) {
        b[o as usize..o as usize + 4].copy_from_slice(&rd.read_u32(rec.pa + o)?.to_le_bytes());
        let l = RECORD * (n_rec - 1) + o;
        b[l as usize..l as usize + 4].copy_from_slice(&rd.read_u32(rec.pa + l)?.to_le_bytes());
    }
    b
};
```
În `BoardRunner::submit`: pe pozițiile complete, ca azi (`analyze_sequence`, `launch_ns`, profilul `kp`); pe celelalte,
`let span = d2::ends_span(&b[..16], &b[16 * n..16 * (n + 1)], n, first)?;` și `busy = span`, `timing` cu `time(0)`, `time(n)` (aceleași
cifre, T147), **fără** `launch_ns` pentru poziția asta (profilul o sare, nu o ghicește).

### B. Așteptarea planificată (tot în `wait`)
```rust
// un `d2::WaitPlan` ținut în BoardRunner pe durata generării; după fiecare simbol: plan.observe(span / 1000);
let (_, ws) = d2::poll_planned(clock, WAIT_US, &plan, || {
    last = rd.read_u32(last_pa).ok()?;
    (last == last_payload).then_some(())
})?;   // Timeout → ca azi (Ok(Err(last))); orice eroare sau o trimitere care nu e simbol → plan.forget()
```
Prima citire e imediată; somnul e în bucăți de cel mult 100 µs, cu tasta Esc întrebată între ele (ca `poll_until`).

### C. Amprenta în afara ceasului (`run_token`)
Mută `self.print = fnv1a(self.print, &logits);` **după** linia `let host = …` (ceasul simbolului oprit). Aceeași funcție, aceeași
amprentă — comparabilă bit cu bit cu os-324/os-328 (`0xda9ae084`).

### D. argmax pe biți (`run_token`)
În locul lui `cm::argmax(&logits)`: `gpu_compute::model::argmax_bits(&bits)`, unde `bits` sunt cuvintele citite din RAM (înainte de
conversia în `f32`). Aceeași alegere (cea mai mare valoare, cel mai mic indice la egalitate, −0 = +0, `None` la NaN/infinit), doar
comparații de întregi.

### Rânduri în jurnal (ca să vedem efectul)
`d2::Diet2Stats` — după fiecare simbol `stats.add(cuvinte_de_inregistrari_citite, ws.polls)`; la final `stats.journal(on, journal)`:
`kt dieta 2 (0/1)`, `kt cuvinte de inregistrari citite pe simbol (median)`, `kt citiri ale asteptarii pe simbol (median)`.
Cititorul: `d2::parse_diet2`.

## Proba pe placă
- aceleași 16 simboluri ca procesorul, **amprenta identică cu os-328** (`0xda9ae084`); `kt median placa` neschimbat (~2 421 µs);
- `kt median driver` de la 759 spre **~350–400 µs** (DEDUS); `kt cuvinte de inregistrari citite pe simbol (median)` = 8;
- fără `--dieta2` = exact ca azi.
