# C-157 punctul 2 — după fiecare test pe placă: ce am estimat, ce a ieșit, ce tai mai departe

Fiecare test nou pe placă primește aici 5–10 rânduri (cel mai nou sus). Sursele: `Cyber-hub/rezultate/…/REZULTAT.md` și jurnalele brute.

## os-328 `--lipit` (P3, 21 nov 2026, 12:56) — sesiunea 135
- **Estimat** (`C-155_P3-gata-de-legat.md`, DEDUS): ~4,0 → **~3,6 ms** pe simbol (~280 simb/s); 88 de lansări; logit-uri identice bit cu bit cu lista simplă; ~130 de cuvinte VRAM pe simbol.
- **Ieșit:** **3 284 µs** (304,5 simb/s) — cu ~320 µs mai bine decât estimarea; 88 de lansări, 130 de cuvinte; amprenta `0xda9ae084` = cea din os-324 (lista simplă) → identic bit cu bit, 16/16 simboluri.
- **Unde a fost estimarea prea prudentă:** am pus tot câștigul pe placă. De fapt **driverul** a câștigat cel mai mult: 1 347 → **759 µs** (−588; 108 lansări mai puțin = mai puține QMD-uri și cuvinte de scris), iar **placa** doar 2 624 → 2 421 µs (−203).
- **Pe placă, pe fel (os-324 → os-328, median pe simbol):** cele 4 feluri mici (adunări/bias, ~107 µs, 48 de lansări) au dispărut; programele lipite au plătit coada: fc 471 → 486, qkv 368 → 381, proj 400 → 406, out 144 → 148 (+38 µs). Restul ~134 µs = pauzele dintre lansări (~1,2 µs pe lansare scoasă). ln1/ln2/atenție/logit neschimbate (±3 µs).
- **Prima rulare (lista simplă) la 17,57 simb/s** (de lămurit de OS, punctul 2 din REZULTAT): în `gpu-jurnal-026.txt`, după cele 14 sonerii ale urcării, singura pauză mare din prima generare e de **0,90 s între soneria 30 și 31 = înainte de poziția 16** — exact locul scrierii jurnalului pe stick „la 16 simboluri” (aceeași poziție ca cea lentă din lista lipită, 77 ms). E mai lungă pentru că prima golire duce și rândurile urcării greutăților (DEDUS din ora rândurilor; ora e a scrierii în memorie, nu a plăcii). Încă o pauză de 0,30 s după soneria 32 = sfârșitul primei generări. Concluzie: tot scrierea pe stick (P6-112 o repară); placa și driverul nu au nimic.
- **Ce tai mai departe:** înmulțirile (fc + proj + qkv + logit = **1 459 µs = 60 %** din placă) merg la **~60–75 GB/s**, adică ~12 % din 608 GB/s (copia VRAM: 282 GB/s în același test). Acolo e următorul câștig mare → C-155 punctul 4 (cursa înmulțirilor). Apoi ln1 + ln2 (522 µs) → LayerNorm nou (gata de legat). Driverul (759 µs) e acum sub placă; ținta 0,5 ms rămâne C-157 punctul 3.
- **De lămurit de OS (nu de noi):** rândul `kt lipit fata de simplu` (propus în nota P3) nu e în jurnal; amprenta ajunge pentru proba de azi.

## os-323 `--flr-proba` (21 nov 2026, 08:32) — sesiunea 132
- **Estimat** (`C-151_reset-flr.md`, sesiunea 122): placa răspunde după FLR în 100 ms + câteva citiri (limita 1 s); WPR2 coborât; al doilea lanț trece fără repornire („varianta 1”).
- **Ieșit:** placa răspunde după **131 ms**, programul de pornire al plăcii gata la **206 ms**, WPR2 = 0, al doilea lanț trece **tot** (I6…I14, turul de citire). Varianta 1 dovedită; rezerva (Booter Unload) nu mai e nevoie.
- **Diferit:** doar cele ~31 ms peste cele 100 ms de așteptare (citiri până răspunde); limita de 1 s are marjă de ×7. Nimic de schimbat la noi.
- **De lămurit de OS (nu de noi):** cele „11 scrieri refuzate în afara gardului” la al doilea lanț — gardul nostru pentru rescrierea configurației cere doar 0x04–0x3c, Device/Link Control și MSI; dacă cele 11 sunt în afara lor, sunt ale adaptoarelor OS, nu ale `gpu_pci::flr`.
- **Ce tai:** nimic. Deschide calea „mai multe teste într-o pornire” (C-158 punctul 5).

## os-323 `--tablou` (21 nov 2026, 08:21) — sesiunea 132
- **Estimat** (`C-151_tablou-rusd.md`): cele 3 comenzi acceptate, `sections` = 0x7f, 30–50 °C în repaus, P8. **Ieșit:** exact așa (35,8 °C, P8, 225/405 MHz, 25,2 W, limita 290 W, 0x7f).
- **Motivele încetinirii = 0x1:** e bitul `RUSD_CLK_THROTTLE_REASON_GPU_IDLE` din `cl00de.h` (570.144) — „placa e în repaus”, nu o încetinire. Deja în `gsp::rusd::reason::GPU_IDLE`.
- **Diferit: oprirea zonei comune refuzată** (`INIT(0)` → `0x8000001f` = `NV_ERR_INVALID_ARGUMENT`, plus un ASSERT intern). Cauza (DEDUS, două surse): driverul NVIDIA 570.144 **nu trimite niciodată** `INIT(0)` — `gpuDestroyRusdMemory` doar eliberează memoria, iar singurul care trimite `INIT` e `_gpushareddataInitGsp`, mereu cu o adresă reală. Comentariul „0 to de-initialize” din antet nu e urmat de GSP-RM 570.144.
- **Ce tai:** oprirea fără `INIT(0)` — `gsp::rusd::rusd_stop_poll_only` și `dashboard_at_end_keep_page` (doar adăugări; T140h în SIMULARE: 4 comenzi în loc de 5, niciun `INIT(0)`). Pagina rămâne rezervată până la FLR sau repornire (așa era oricum după refuz). Pe placă: **o comandă mai puțin, nimic nou**; un ASSERT mai puțin pe rulare (așteptat).
- **P0 înainte de viteza maximă (ieri P5):** NECUNOSCUT. Ipoteză (DEDUS): placa nu coboară încă din P0 după lucrul abia terminat al lanțului (I10–I14 rulează chiar înainte); tabloul arată P8 la 1,7 s după. Proba ieftină: aceeași citire a tabloului și **înainte** de lanț (C-151 punctul 2, în coadă).
- **Viteza (testul obișnuit):** 94,44 simb/s, driverul 3 087 µs/simbol — ca os-322 (94,84). Estimarea pentru dietă rămâne ~1,3 ms driver; se vede în `os-324`.
