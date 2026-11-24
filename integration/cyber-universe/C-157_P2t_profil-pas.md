# Pentru OS: profilul pasului de antrenare pe programe (C-157 p.2, după os-392)

**Ce e nou pentru placă: nimic.** Doar o împărțire a timpilor pe care placa îi scrie deja (o înregistrare după fiecare lansare) și pe
care OS-ul îi citește deja întregi la 3 pași din 10 (`RecordPlan::DIET`, „citite toate la 3 pași”). Proba: cyber-gpu T210 (SIMULARE),
planul și predicțiile `docs/C-157_P2_PROFIL_PAS.md`, modulul `nvidia_ga104::train_profile` (doar adăugări).

## De ce (os-392, MĂSURAT)
Placa face pasul lui `antrenare mini` în 3 602 µs (69 de lansări). Lipirea a scutit 26 µs pentru 6 lansări (~4,3 µs pe lansare, cât
costul măsurat la os-331) → lansările în sine sunt ~0,3 ms. **Restul de ~3,3 ms e calcul și nu știm pe ce program.** Fără cifra asta
orice tăietură pe placă e ghicită.

## Ce (în `train_mini_resident`, `gpu_i14.rs`, ambele drumuri; rulările `--pas-lipit` au câte un profil pe listă)
1. La primul pas, după `plan.step(...)`: `let mut prof = train_profile::StepProfile::from_launches(&list)?;`
   (cu lotul prin RAM: `.with_copies(v.copies_in.len(), v.copies_out.len())?`).
2. La pașii la care se citesc toate înregistrările (`RecordPlan::DIET.reads_all(k)`): `prof.add_records(&recs, first)?;` — aceleași
   octeți și același `first` cu care OS-ul face deja `analyze_sequence`. O eroare de acolo = același refuz ca acum (nu se ghicește nimic).
3. La sfârșit, o dată: `for l in prof.lines(8) { jurnal / ecran }` — ~30 de rânduri: totalul, cele 5 etape, cele mai lungi 8 etichete,
   programele. ASCII, ca restul jurnalului. Rândurile intră în jurnalul de rutină (nu forțează scrieri pe stick).

## Ce așteptăm pe placă (DEDUS din forma listei; de verificat — scopul profilului)
- Suma rândurilor = „placa” din raport (± câteva µs: mediana sumei, nu suma medianelor).
- **Primul: `inapoi:cap-dX`, 0,3–1,0 ms** — înmulțirea cu doar 16 blocuri (placa are 48 de multiprocesoare) și 16 384 de termeni pe
  element, făcuți în 1 024 de ture una după alta. Apoi `eroarea` (50–200 µs) și `opt:pas` (60–120 µs); `inainte:logit` și
  `inapoi:cap-dW` sub 60 µs fiecare; cele 13 `colsum` (un singur bloc fiecare) + 6 `attn_*` împreună ~0,3–0,5 ms.
- Dacă `cap-dX` iese primul, tăietura următoare (la Cyber-GPU, în SIMULARE, apoi nota) e împărțirea sumei pe mai multe blocuri.
