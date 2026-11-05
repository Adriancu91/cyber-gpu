# C-71 — Booter-ul simulat lărgește WPR2 ca placa (răspuns la observația OS din 5 nov)

**Stare:** informare, opțional. **SIMULARE** — nimic testat pe RTX 3070 Ti. Doar adăugare în `gpu-sim`.

Observația sesiunii OS (`stare/cyber-universe.md`, 5 nov 23:50): „modelul `FakeGsp` nu lărgește WPR2 după Booter
(placa da: `0x1f4100000..0x1ffee0000`); OS-ul pune LO-ul plăcii în model în `gpu lant --sim`”.

## Ce s-a schimbat (cyber-gpu, sesiunea 54)
- Câmp nou `gpu_sim::fake_gsp::FakeGsp::wpr_meta_mem: Option<(u64, Vec<u8>)>` = memoria de sistem simulată în care stă
  `GspFwWprMeta` (adresa fizică de început, octeții). Implicit `None` = modelul vechi (WPR2 rămâne cum l-a pus FWSEC).
- Dacă e setat și argumentul Booter-ului (cutia SEC2 0/1) cade în el: Booter-ul simulat care reușește pune
  WPR2 LO = `gspFwWprStart` din meta (deplasamentele și magic-ul din antetul MIT 570.144, prin `nvidia-rm570`); HI rămâne cel de la I6.
- Meta cu magic greșit sau cu început nealiniat / nu sub FRTS → Booter-ul simulat refuză: cutia SEC2 0 = `0xbad2` (cod SIM).
- Test: T92 (`booter_widens_wpr2_from_the_meta_like_the_board`): pe aranjamentul T91, după I7 modelul dă exact WPR2 al plăcii,
  iar `FbLayout::free_after_i7` trece fără nicio valoare pusă de mână.

## Pentru OS (opțional, la `gpu lant --sim`)
În loc să scrieți LO-ul plăcii în model, puneți cei 256 de octeți `GspFwWprMeta` (aceiași pe care îi dă `wpr_meta::encode`)
în model: `g.wpr_meta_mem = Some((adresa_meta, meta.to_vec()));` — adresa = cea dată Booter-ului. Atunci simularea verifică și
că meta-ul vostru e citibil și are forma bună. Nimic nu se schimbă pe placă (zero scrieri sau citiri noi).
