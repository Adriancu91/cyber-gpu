# C-142 — #513 pe placă: limitele, contoarele de viteză și propunerea FLR (pentru sesiunea OS, C-141 / C-143)

**Doar SIMULARE.** Nimic de aici nu a rulat pe RTX 3070 Ti. Toate schimbările din crate-uri sunt **adăugări** (nimic redenumit sau șters); OS-ul fixat
pe `db34429` compilează și merge neschimbat. Analiza: `docs/C-142_K513_VITEZA_FLR.md`. Testul care face pașii de mai jos pe forma lui #513:
`crates/nvidia-ga104/tests/sim_k513.rs`.

## Pe scurt

- **Pe partea driverului nu trebuie nimic nou pentru #513.** Codul care a trecut I14 la `os-306` merge și pe forma lui #513 (în SIMULARE: 14 bucăți de
  urcare, 196 de lansări pe sonerie, 4 poziții identice cu procesorul), cu **zona de comenzi de 32 KiB** (deja în C-141).
- `STRIDE = 0x100` (≥ 197 trebuie), `UP_FIRST`/`TOKEN_FIRST`, arena I11 de 2 × 8 MiB și rezerva de 4 MiB ajung așa cum sunt.
- Nou, de folosit în `kronos --gpu`: `kronos_speed::ModelLimits` (refuz înainte de urcare, cu motivul numit) și `kronos_speed::Speed` (contoarele de viteză,
  rânduri `kv` în gpu-jurnal).

## Ce e nou în crate-uri (fixați `cyber-gpu` pe commit-ul din `stare/cyber-gpu.md`)

`nvidia_ga104::kronos_speed` (mai jos `ks`):
- `ks::ModelLimits::of(&shape, straturi, vocabular, context, bucata)` / `ks::ModelLimits::k513()` → `launches`, `push_once/each/layer`, `push_area`,
  `stride`, `vram` (`VramBudget`), `pieces`; `lines()` pentru ecran; `check(push.size, STRIDE, liber, cea_mai_lunga, VRAM_RESERVE) -> Vec<ks::Limit>`
  (gol = încape; altfel fiecare limită cu `line()`).
- `ks::Speed::new(ks::Source::Placa)`; `add_piece(octeti, ns_placa, Some(us_procesor))` la fiecare bucată urcată; `add_token(poz, &TokenOutput,
  Some(us_procesor))` la fiecare poziție; `lines()` (3 rânduri pentru ecran și raport); `journal(&mut journal, per_token)` (rândurile `kv`);
  `summary()`. Cititorul pentru unelte: `ks::parse_journal(text)`.
- `ks::K513_SHAPE`, `K513_LAYERS`, `K513_VOCAB`, `K513_CONTEXT`.

## Ce se schimbă în `gpu_i14.rs` (propunere; codul OS e al vostru)

1. În `run_model`, înaintea urcării, pe lângă `VramBudget`:
   ```rust
   let lim = ks::ModelLimits::of(&m.shape, m.layers.len(), m.vocab, m.context, I11_BOARD_BYTES)?;
   lines.extend(lim.lines().into_iter().map(|l| format!("{}: {}", p, l)));
   let (free, largest) = st.mem.vram_free();
   let refuz = lim.check(l.push.size, STRIDE, free, largest, VRAM_RESERVE);
   if !refuz.is_empty() {
       for r in &refuz { lines.push(format!("{}: I14: {}", p, r.line())); }
       return Err(bad("I14: modelul nu incape (motivul e pe ecran): nimic urcat"));
   }
   ```
   (Verificările voastre de azi — `push_need > l.push.size / 2` și `budget.check` — spun același lucru; `check` le adună și adaugă `STRIDE`.)
2. La urcare: `speed.add_piece(chunk.len() as u64, ns_bucata, Some(us))`, unde `ns_bucata` e ce dă azi `analyze_sequence(..)[0]` și `us` = ceasul OS-ului
   de la `ram_write` până după înregistrarea citită.
3. La fiecare poziție (`run_token`): `speed.add_token(pos, &out, Some(us))`, `us` de la scrierea primei lansări până după citirea logit-urilor din RAM —
   **fără** verificarea pe procesor (`step_cpu`), ca să se vadă cât e placa + driver și cât e verificarea (ea se poate măsura separat, dacă vreți).
4. La final, pentru `kronos --gpu`: `lines.extend(speed.lines())` și `speed.journal(journal, false)` (14 rânduri, oricât de lung e textul; `true` = încă 4
   pe poziție). Un rând de jurnal costă ~17–23 ms pe stick la `os-306`, de aceea implicit fără rândurile pe poziție.

## Ce am văzut în jurnalul `os-306` (pentru așteptări)

În faza textelor, un simbol al modelului de control = **~0,27 s** între sonerii, cu **15 rânduri de jurnal** pe simbol (~18 ms fiecare): timpul e aproape
tot jurnalul. Pentru #513 estimăm (nemăsurat) **~0,3 s pe simbol + timpul plăcii + verificarea pe procesor**, iar urcarea ~4–5 s + citirea fișierului de pe
stick. Contoarele de mai sus vor da cifrele adevărate.

## Pentru C-143 (ciclul fără Adrian): FLR

Placa anunță FLR (K-2). Din cele trei surse (RM 570.144, tinygrad, nouveau/nova — doc §4): **tinygrad face exact asta când găsește WPR2 ridicat** (Bus Master
oprit → reset PCI al funcției → 100 ms → așteaptă GFW boot → merge mai departe); RM folosește FLR ca „reset complet al cipului” și după el așteaptă ca
VBIOS să refacă inițializarea. Propunerea noastră, **de probat întâi cu Adrian de față**: FLR la începutul lanțului, doar dacă WPR2 e ridicat (salvați
configurația PCI, Device Control bit 15, 100 ms, rescrieți configurația, I2 GFW boot, WPR2 recitit = 0). Dacă WPR2 rămâne ridicat → oprire curată
„trebuie pornire rece (Adrian)”, ca în C-143 punctul 3. Nu e cod de-al nostru încă: așteaptă „da”-ul lui Adrian.
