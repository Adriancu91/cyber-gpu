# C-147 punctul 7 → „dieta driverului”: de la 3,1 ms spre ~1,3 ms pe simbol (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 125 (20 nov 2026). Detalii și cifre: `docs/C-147_PUNCTUL7_ESTIMARE.md`.
Totul de la noi e **adăugat** (nimic redenumit sau șters), testat **doar în SIMULARE** (T141, `tests/sim_c147_diet.rs`).
**Nu se trimite nimic nou la placă:** aceleași programe, aceleași lansări, aceiași octeți în pushbuffer; doar nu mai rescriem cuvintele
pe care VRAM-ul le are deja și nu mai zerorizăm înregistrările. Nicio scriere nouă în registre. Dacă e nevoie de „da”, decide centrul de comandă.

## Ce am găsit în jurnalul os-322 (`gpu-jurnal-017.txt`, ceasul rândurilor)
Un simbol constant (median 5 788 µs), împărțit după fereastra VRAM deschisă (unealta `timpul_driverului`):
`0x1b2f` 2 905 µs (placa + citirea înregistrărilor) · **`0x10e6` 1 517 µs** · **`0x1b2d` 529 µs** · `0x101f` 16 µs · procesorul 840 µs.
Citind codul vostru (`gpu_i14.rs` `BoardRunner::submit`, `gpu_i11.rs` `run_records`, `gpu_golden.rs` `flush`), credem (DEDUS, ~85 %):
- `0x10e6` = **pushbuffer-ul**, rescris întreg la fiecare simbol: `kd::words(EachLaunch, 197)` = 2 363 de cuvinte, fiecare scris și recitit.
  `WorkQueue::place` îl pune mereu la începutul zonei (simbolul de dinainte e retras), iar între două simboluri se schimbă doar valorile
  înregistrărilor (`first + k`) — ~200 de cuvinte.
- `0x1b2d` = cuvintele schimbate din lansări (umbra P6-97 le-a redus deja) + **zerorizarea celor 197 de înregistrări** (788 de cuvinte).

## Ce legi (în spatele unei opțiuni, de ex. `--dieta`, oprită implicit până la proba pe placă)
1. **Pushbuffer-ul prin umbră** (`nvidia_ga104::kronos_fast::{WriteShadow, ShadowedMemory}`):
   ```rust
   // la începutul generării, lângă umbra lansărilor (P6-97):
   let mut push_shadow = kf::WriteShadow::new(l.push.pa, l.push.size)?;   // l.push = zona pushbuffer-ului canalului de lucru
   // run_records primește un parametru nou `push_shadow: Option<&mut kf::WriteShadow>` (None = ca azi);
   // în locul lui `l.wq.submit_fenced(st.mem.writer_mut(), words, last_va, last_payload)?`:
   let bell = match push_shadow {
       Some(sh) => {
           let mut m = kf::ShadowedMemory::new(st.mem.writer_mut(), sh);
           let r = l.wq.submit_fenced(&mut m, words, last_va, u64::from(last_payload));
           if r.is_err() { sh.forget(); }
           r?
       }
       None => l.wq.submit_fenced(st.mem.writer_mut(), words, last_va, u64::from(last_payload))?,
   };
   ```
   **Regula:** umbra e valabilă doar dacă **toate** scrierile în zona pushbuffer-ului trec prin ea. Orice altă trimitere pe același canal
   care nu trece prin umbră (urcarea greutăților, `kc`, cererea de viteză, diagnosticul) → `push_shadow.forget()` înainte. Orice eroare
   (scriere, recitire, `flush`) → `forget()`. Primele 1–2 simboluri se scriu întregi (locul se stabilizează), apoi ~200 de cuvinte.
2. **Fără zerorizarea înregistrărilor** în `BoardRunner::submit` (rândul `put_bytes(w, self.rec.pa, &vec![0u8; 16 * (n + 1)])`) cât timp
   opțiunea e pornită. De ce e sigur: valorile sunt `TOKEN_FIRST + STRIDE * subs + k`, unice pe trimitere; așteptarea cere ultima
   înregistrare egală cu valoarea nouă, iar `transfer::analyze_sequence` verifică fiecare înregistrare — una rămasă de la simbolul
   dinainte e prinsă (T141b; `records_seen` numără corect și după o oprire la jumătate).
3. **Rânduri în jurnal**, ca să vedem efectul: cuvintele VRAM scrise pe simbol (aveți deja `vram_words`): median pe simbol, de ex.
   `kt cuvinte VRAM pe simbol (median)`; și dacă dieta a fost pornită (`kt dieta (0/1)`).
4. **Două mărci de timp noi** în `run_token` (măsurare, nimic schimbat): după citirea logit-urilor din RAM și după argmax → rânduri
   `kt median citirea logit-urilor (us)`, `kt median argmax (us)`. Cei ~840 µs ai procesorului sunt azi necunoscuți; dacă citirea celor
   64 KiB din RAM e partea mare, următorul pas e un argmax pe placă.

## Proba pe placă (după ce e legată)
- aceleași 16 simboluri ca procesorul **și logit-urile identice bit cu bit cu os-322** (aceleași programe, aceiași octeți);
- `kp` (timpul plăcii) neschimbat; `kt median driver` de la ~3 081 spre **~1 300 µs** (DEDUS); fără dietă = exact ca os-322.
- Poate merge în același Release cu P3 (calculele lipite) doar dacă centrul de comandă vrea: atunci comparația cu os-322 nu mai e
  bit cu bit pentru logit-uri la nivelul listei (P3 are propria probă). Mai curat: întâi dieta singură, apoi P3.

## Estimarea (DEDUS, #513, viteza maximă, median pe simbol)
azi 5,8 ms (172 simb/s) → dieta ~4,0 ms (~250) → + P3 ~3,6 ms (~280) → + LayerNorm nou ~3,1 ms (~320) → + etapa 4 ~2,1 ms (~470).
