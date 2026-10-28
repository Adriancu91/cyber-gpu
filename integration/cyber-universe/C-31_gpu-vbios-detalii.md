# C-31 — `gpu vbios --detalii` (pentru sesiunea OS)

**De la:** Cyber-GPU, sesiunea 13 (28 oct 2026), după C-30. **Doar citire.** Testat doar în SIMULARE (imagini VBIOS sintetice).

## Ce s-a schimbat în `cyber-gpu` (doar adăugări față de `dac2e32`)
- `nvidia_ga104::vbios_diag::diagnose(bar0, journal, clock, budget_us) -> GpuResult<VbiosDiag>` — aceeași fereastră de 4 MiB
  (doar citire) ca `vbios::read_layout`, același buget (`vbios::VBIOS_TIMEOUT_US`).
- `VbiosDiag::lines() -> Vec<String>` — lista gata de afișat (ASCII, română fără diacritice, cel mult câteva zeci de rânduri):
  pentru fiecare imagine offsetul, semnătura (`aa55`/`4e56` „NV”/`bb77`), câmpurile PCIR (vendor, device, clasă, tip cod, lungime,
  blocuri, indicatorul „ultima”), NPDE (dacă există); la fiecare ruptură primii 16 octeți + unde s-a reluat; semnăturile găsite după
  ultima imagine; tabela BIT; ultimul rând = rezumatul cu „criteriul I3” și **„0 scrieri”**.
- `VbiosDiag::writes` = scrierile din jurnal în timpul diagnosticului (mereu 0; cititorul nu are cale de scriere).
- `vbios::read_layout` acceptă acum și imaginile cu semnătura „NV” (`0x4e56`) — exact ce a oprit I3 pe placa reală la `0x24600`.
  Interfața (tipuri, câmpuri, funcții) e neschimbată.
- Pentru `--sim`: `gpu_sim::rom::asus_like_chain()` (lanț sintetic cu imaginile „NV” la `0x24600`, ca pe placă) și
  `gpu_sim::rom::ga10x_like_chain()` (cel vechi).

## Cum se folosește (schiță; codul îl scrie sesiunea OS)
```rust
let d = nvidia_ga104::vbios_diag::diagnose(&mut win, &mut journal, &mut clock, nvidia_ga104::vbios::VBIOS_TIMEOUT_US);
match d {
    Ok(d) => for l in d.lines() { /* ecran + raport, cu prefixul plăcii */ },
    Err(e) => { /* „VBIOS diagnostic: {e}” — doar fereastră prea mică / placă absentă */ }
}
```
În modelul `--sim`: `bar0.load_bytes(regs::PROM_BASE, &rom::asus_like_chain())`, apoi aceeași chemare; verificare în CI:
rândul `imagine 2 la 0x24600: semnatura 0x4e56 (NV)` și rezumatul cu `criteriul I3: indeplinit` și `0 scrieri`.

## Fixarea
- Commit nou: cel din `stare/cyber-gpu.md` din hub (ramura `g0-g2-fundatie`). Față de `dac2e32`, `nvidia-ga104` depinde și de
  `gpu-memory` și `nvidia-rm570` (același repo, fără dependențe externe).
- Ce așteptăm la următoarea fereastră pe workstation: `gpu vbios` (acum trebuie să treacă de `0x24600`) **și** `gpu vbios --detalii`
  (poză cu toate rândurile). Dacă `gpu vbios` tot pică, rândurile `--detalii` ne spun exact de ce.
