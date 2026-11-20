# Pentru sesiunea OS: cursa înmulțirilor Q8 (C-147 punctul 3, etapa 4) — `q8_matvec` față de `q8_matvec32`

**Stare: SIMULARE (cyber-gpu sesiunea 121, commit `c11225d`).** Pe placă doar în Release-ul pe care îl hotărăște centrul de comandă (după `os-309` și
lista lipită), oprită implicit; sunt programe noi care calculează același lucru — proba obișnuită (C-147 punctul 3: „în SIMULARE, apoi pe placă”).
Detalii: `docs/C-147_ETAPA4.md`.

## Ce legi (doar adăugări în crate-uri; nimic de azi nu se schimbă)
Comanda propusă: `gpu cursa-q8` (sau un pas în ciclu, după `kronos --gpu`, când rezultatele sunt deja pe stick). Pentru fiecare bandă din
`q8_race::K513_LANES` (5):
1. W sintetic: `q8_matvec::test_matrix(lane.rows, lane.cols, seed)` (ca la I12, fără fișier) — sau greutățile reale ale lui #513 deja urcate,
   dacă e mai simplu (orice W bun; logits = 16 384 × 768 ≈ 13,5 MB). **x aliniat la 16** (cerința lui `q8_matvec32`).
2. Ambele programe urcate în VRAM: `Entrant::code()` (`q8_matvec` 8 048 octeți, `q8_matvec32` 3 184 octeți).
3. Pentru fiecare serie din `q8_race::heats(5, RACE_LANES)` (A B B A pe bandă): bufferul constant `heat.entrant.cbuf(...)`, QMD-ul
   `heat.entrant.qmd(...)`, y umplut cu 0xff, înregistrările golite; pushbuffer-ul `q8_race::encode_heat` (= `launch_cost::encode`: marcaj de start,
   apoi n × lansare + marcă de timp); o sonerie; așteptarea ca la `i10-cost-lansare`.
4. După serie: `launch_cost::analyze(records, n, first)`; y citit; `heat.entrant.judge(w, rows, cols, x, y)`; `Standing::add(5, &HeatResult{..})`.
   O marcă lipsă = rând de eroare și seria se oprește (nu se ghicește un număr).
5. La final: `q8_race::journal(&standing, &K513_LANES, &mut journal)` → rânduri `kr …`; raportul ciclului poate folosi `q8_race::parse_journal`
   (mediana pe bandă și program, câștigătorul, suma pe simbol azi / cu câștigătorii).

Timp estimat pe placă: 5 benzi × 4 serii × 16 lansări = 320 de lansări, plus 20 de sonerii — sub o secundă dacă o lansare durează ~100 µs.

## Ce înseamnă rezultatele
- `kr <bandă> q8_matvec32 trece = 1` și `… castiga = 1`: programul nou e corect pe placă și mai rapid cu ≥ 50 ‰ pe acea bandă.
- `… trece = 0`: programul nou **nu** dă rezultatul așteptat pe placă → rămâne `q8_matvec`; jurnalul merge la cyber-gpu.
- Egalitate sau câștig sub prag: rămâne `q8_matvec`.
Folosirea câștigătorului în lista de simbol (și un `q8_fused` cu aceeași împărțire a muncii) e pasul nostru următor, după cifre.
