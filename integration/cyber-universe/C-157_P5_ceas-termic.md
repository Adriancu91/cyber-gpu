# Pentru OS: ceasul termic (tabloul la N secunde) + tabloul înainte/după viteza maximă — C-157 p.5 (= C-151 p.2)

De la: Cyber-GPU, sesiunea 169 (22 nov 2026). Detalii și surse: `docs/C-157_P5_CEAS_TERMIC.md`. Teste: T179 (SIMULARE).
**Doar adăugări** — OS-ul compilează neschimbat când leagă commit-ul. Modul nou: `nvidia_ga104::gsp::rusd_watch`.

## Ce oferă
- `ThermalLimits::PROPOSED` = avertizare **80 °C**, oprire **85 °C**, pornire din nou doar la **≤ 70 °C**, tablou mut > **10 s** = oprire;
  `ThermalLimits::check()` (refuză oprire > 90 °C și ordini greșite).
- `Watch::new(limite, 5_000)` → în bucla lucrării: `if w.due(t) { let d = rusd::read_dashboard(&mut pagina, &mut scratch);
  let s = rusd_watch::temp_stamp(&mut pagina); match w.observe(t, &d, s, j) { … } }`; `Verdict::Stop(_)` = **oprește lucrarea**
  (după pasul în curs; rezultatele pe stick ca la o oprire normală). Oprirea se ține. La final `w.summary(j)`.
  Rândurile `kw` apar doar la o schimbare de verdict + rezumatul (o lucrare de 8 ore la 5 s = sub 12 rânduri). `t` = ms de la pornirea lucrării.
- `may_start(&limite, &d)` înainte de fiecare repetare a lucrării (`repeta N` din fișă): fals → așteaptă și citește din nou (sau sare
  repetarea, cum preferați), cu un rând în jurnal.
- `compare_boost(&înainte, &după, j)` → `BoostSeen` + rânduri `kw viteza maxima …`; citirea „după” la `after_wait_ms(500)` = 1 s după cerere.
- `parse_kw(text)` pentru raport.

## Unde se leagă (propunere)
1. **Managerul de lucrări lungi (os-356):** la pornirea unei lucrări pe placă — tabloul pornit (`rusd_start`, ca azi), lăsat pornit pe
   durata lucrării, citit la 5 s; la sfârșit `rusd_stop_poll_only` (fără `INIT(0)`, ca de la os-323). Opțional în fișă un rând
   `temperatura <avertizare> <oprire> <pornire>` (trecut prin `check()`; implicit propunerea).
2. **`test-placa` cu viteza maximă:** tabloul pornit **înainte** de cerere → citire „înainte” → cerere → 1 s → citire „după” →
   `compare_boost`. Răspunde la întrebarea din os-323 (P0 dinainte) cu ceasurile, nu doar cu starea P.

## Ce e nou pentru placă
**Nicio comandă nouă, niciun registru.** Se schimbă doar **momentul**: tabloul pornit mai devreme (înainte de viteza maximă / la începutul
lucrării, nu la sfârșitul lanțului) și ținut pornit ore întregi. Cele 3 comenzi (INIT, interval, mască) + oprirea citirilor au fost
acceptate pe placă în os-323. Un singur lucru nou pe test: propun întâi (2) într-un `test-placa` scurt, apoi (1).

## Ce nu știm (NETESTAT)
Cum urcă temperatura sub lucru; dacă marca temperaturii se schimbă la fiecare 500 ms și când valoarea nu se schimbă (dacă nu, tabloul
ar părea „mut” la o placă cu temperatură stabilă — primul test o arată; rândul `kw … citiri cu tabloul neschimbat` o spune).
Pragul e o propunere (NVIDIA: maximum 93 °C la RTX 3070 Ti); Adrian îl poate schimba.

Răspuns la os-323: motivul `0x1` = `GPU_IDLE` (placa în repaus), nu încetinire.
