# C-71 — fereastra VRAM în jurnalul `gpu lant` (sesiunea Cyber-GPU 50, 5 nov 2026)

**Doar informare, nimic de schimbat acum în OS.** Testat doar în SIMULARE.

`verifica_jurnal_lant` (cyber-gpu ≥ `a392789`) verifică acum și registrul ferestrei VRAM (`0x1700`, `NV_PBUS_BAR0_WINDOW`),
pe tot jurnalul lanțului:
- fiecare mutare a ferestrei trebuie recitită exact cu valoarea scrisă;
- până la sfârșitul jurnalului, fereastra trebuie pusă înapoi la valoarea dinaintea primei mutări (ca Bus Master).

Pentru OS, când I9/I10 vor scrie tabelele de pagini prin `VramWindow` (după „da”-ul lui Adrian): chemați
`VramWindow::restore()` și pe drumul de eroare (de exemplu cu un pas de curățare), altfel un lanț oprit la I9 iese
„DE ANALIZAT” în loc de „OPRIT CURAT”. Interfața publică: doar o adăugare (constanta `chain_log::VRAM_WINDOW`); nimic redenumit sau șters.
