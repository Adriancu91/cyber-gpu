# C-87 pașii 1 și 4 — generare continuă și bugetul VRAM (pentru sesiunea OS)

**Doar SIMULARE.** Nimic de aici nu a rulat pe RTX 3070 Ti. Toate schimbările din crate-uri sunt **adăugări** (nimic redenumit sau șters).

## Ce e nou în `nvidia-ga104`

- `kronos_gen::Generation` — bucla „N simboluri la rând”: textul dat, apoi `n` simboluri greedy; o poziție = o trimitere
  (`kronos_model::token_launches` + copierea logit-urilor, `WorkQueue::submit_fenced`). OS-ul dă un `TokenRunner`
  (exact ce face azi `run_model` pentru o poziție: scrie lansările, trimite, sună, așteaptă ultima înregistrare, citește
  înregistrările + logit-urile) și, opțional, un `TokenCheck` (motorul de pe procesor). `Generation::run` se oprește
  **la prima eroare** (`Stop::Eroare { pos, err }`), fără nicio trimitere în plus; simbolurile făcute până acolo rămân în `seq`.
  Alte opriri curate: `Gata`, `Sfarsit` (simbol de sfârșit, opțional `plan.eos`), `ContextPlin`. Opriri cu eroare:
  `LogitiNefiniti`, `NuTrece` (față de procesor), `CeasInapoi`.
- `kronos_gen::GenStats` (`Generation::stats()`): viteza **doar din ceasul plăcii** (marcajele de timp din înregistrări):
  `per_s_x100()` = poziții/s × 100 pe toată durata (cu pauzele driverului — ce vede omul), `busy_per_s_x100()` = doar lansările,
  median și cel mai lent. `line("placa")` dă un rând de jurnal gata făcut.
- `TokenOutput { logits, start_ns, end_ns, busy_ns }`: `start_ns` = marcajul înregistrării 0, `end_ns` = al ultimei,
  `busy_ns` = suma din `transfer::analyze_sequence`.
- `kronos_gen::VramBudget::of_model(&m)` + `.check(free, largest_free, reserve)` — memoria VRAM calculată din forme,
  **înainte de urcare**; `GpuMemory::vram_free()` dă `(liber total, cea mai lungă bucată liberă)`. Blob-ul (greutăți + spațiu
  de lucru + logit-uri) trebuie să fie dintr-o bucată.
- `kronos_gen::push_bytes_with_copy(layers)` — octeții de pushbuffer ai unei poziții, cu copierea logit-urilor (ca la voi).

## Cifre (calculate, nu măsurate)

| Model | Lansări/poziție | Pushbuffer/poziție | VRAM total |
|---|---|---|---|
| #513 (12 straturi, E 768) | 195 (+1 copiere) | 7 872 B | ≈ 142 MiB |
| 183M (24 straturi, E 768) | 387 (+1 copiere) | 15 552 B | ≈ 270 MiB (greutăți+lucru 197, cache-uri 72) |

**Atenție pentru 24 de straturi:** zona de pushbuffer de 16 KiB ține o singură poziție, iar verificarea voastră din
`run_model` (`4 · cuvinte > push.size / 2`) o refuză. Pentru modelul mare zona trebuie să fie **32 KiB** (două poziții în zbor).

## Testat (SIMULARE, `gpu-sim`)

- 40 de simboluri la rând pe modelul sintetic mic (2 straturi): aceleași 43 de simboluri ca procesorul, logit-uri ≤ 3,2e-7;
  zona de pushbuffer liberă după fiecare poziție.
- Defect plantat la poziția 5 (programul GELU șters din VRAM): oprire la poziția 5, 6 trimiteri în total, cele 3 simboluri
  de dinainte păstrate și identice cu procesorul. Bucata trimisă rămâne „în zbor” — curățenia canalului e pasul 3 din C-87.
- Context plin: oprire curată la 16 poziții.
- 24 de straturi (E 128, ca să fie rapid): 387 lansări/poziție, 4 poziții, toate ca procesorul.
- Bugetul acoperă ce a urcat cu adevărat banca de test.

## Ce ar trebui făcut în OS (P6-66, decizia voastră)

1. `run_model`: o poziție → `impl TokenRunner`; bucla → `Generation::run`; rândul de viteză → `stats().line("placa")`.
2. Înainte de urcare: `VramBudget::of_model(&m)?.check(free, largest, rezerva)` cu `st.mem.vram_free()`; refuz cinstit dacă nu încape.
3. Pentru 24 de straturi: zona de pushbuffer 32 KiB.

# C-87 pasul 3 — recuperare fără repornirea calculatorului

**Doar SIMULARE.** Adăugări: `recovery` (modul nou), `user_space::stop_compute_group`, `transfer::records_seen`.

Pericolul după o blocare nu e placa, ci RAM-ul calculatorului: paginile RAM date plăcii (I11, copierea logit-urilor) ar putea fi
scrise mai târziu de un canal încă viu, după ce OS-ul le-a dat altcuiva. De aceea:

1. după `Stop::Eroare` nu se mai trimite nimic;
2. `gsp_ping` — un control RM doar de citire (ex. `query_gr_info`), cu limita voastră de timp;
3. dacă GSP răspunde: `stop_compute_group` (control RM `GPFIFO_SCHEDULE` cu `bEnable = 0`, **niciun registru scris de noi**);
4. citire `GPGet`/`GPPut` din USERD;
5. **Bus Master oprit, întotdeauna** (aceeași scriere + recitire ca pasul de curățenie al lanțului);
6. `recovery::recover(&mut ops)` dă verdictul: paginile RAM **eliberate** doar dacă grupul a fost oprit prin RM ȘI Bus Master e
   oprit; altfel **carantină** (OS-ul nu le refolosește până la repornire) și placa rămâne oprită până la următoarea pornire —
   calculatorul merge mai departe, Kronos pe procesor. `Recovered::lines` = rândurile de jurnal; `recovery::classify` +
   `transfer::records_seen` spun în jurnal dacă placa n-a luat comanda sau s-a oprit la o lansare anume.

Testat: pașii în ordine și fiecare încercat chiar dacă cel dinainte a picat; GSP tăcut → carantină, Bus Master tot oprit;
cu `FakeRm` + spațiu de configurare simulat, după defectul de la poziția 5: grupul scos de pe listă, Bus Master oprit, RAM eliberat.

Ce ar trebui în OS (decizia voastră): în `gpu lant`, la orice eroare după I10, `recover` înainte de pasul de curățenie;
paginile din `Carantina` scoase din alocator până la repornire; un `gpu lant` nou refuzat după `DoarDupaRepornire`.

# C-87 pasul 2 — un singur program pe simbol (pregătit, nu pentru fereastra de luni)

**Doar SIMULARE + compilat cu NAK.** Adăugări: `kronos_token` (`TOKEN`, `work_list`, `token_cbuf`, `token_qmd`, `groups_for`), în `gpu-sim`
`token_on_cpu` + `TOKEN_NEEDS`. O poziție = o singură lansare (18 cuvinte de pushbuffer în loc de 1 968 pentru #513); logit-urile ies **bit cu bit**
ca la lansările separate (testat pe 12 poziții, E 64, și la lățimea lui Kronos, E 768). Pe placă: abia după ce lansările separate trec, cu
recuperarea (pasul 3) gata — dacă `G` ar fi prea mare, bariera s-ar putea bloca. Driverul alege `G = groups_for(gr_info.shape, ...)` (48 pe 3070 Ti).
Contorul barierei (un u32) trebuie golit înainte de fiecare lansare. Propunerea noastră: **luni NU**, întâi lansările separate.
