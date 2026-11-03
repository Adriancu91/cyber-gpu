# C-58 — verificarea I6: capătul WPR2 (pentru sesiunea OS)

**Ce s-a întâmplat pe placă (3 nov, os-214):** `gpu i6 --scrie` a dat „verificare esuata la 0x1fa824: asteptat 0x1ffe00, citit 0x1ffe00”.
Începutul WPR2 (LO) era exact cel cerut; diferea capătul (HI), dar mesajul numea registrul LO.

**Ce s-a schimbat în cyber-gpu** (doar adăugări, compatibil; `af02bea` sau mai nou):
- `gsp::falcon::check_frts` (același nume, aceeași semnătură): start greșit → eroare la `0x1fa824`; capăt greșit → eroare la `0x1fa828`
  cu valorile așteptat/citit; capătul e acceptat dacă HI = sfârșitul exclusiv **sau** ultima pagină de 4 KiB a regiunii.
  Sursele publice (RM 570.144, nova-core) verifică doar LO; pe placă HI nu a fost sfârșitul exclusiv → cel mai probabil ultima pagină.
- nou: `gsp::falcon::{compare_frts, check_frts_detail, FrtsCheck, Wpr2End}` (valorile brute LO/HI + forma capătului).
- nou: `i6_log::wpr2_after_i6(events, frts)` — citește LO/HI din jurnalul de pe stick și aplică aceeași regulă.
- `gpu_sim::fake_gsp::FakeGsp::wpr2_hi_last_page` (implicit `true`, SIMULARE).

**Ce ar trebui să facă OS-ul** (propunere, doar cu acordul lui Adrian / centrului de comandă):
1. Mută fixarea cyber-gpu pe `af02bea` (sau vârful ramurii `g0-g2-fundatie` de după el).
2. Opțional, în `gpu_i6.rs::i6_falcon`, la succes: afișează și HI brut, de ex. prin `falcon::check_frts_detail` după `fwsec_frts_dma`
   (doar citiri, 3 registre) — „WPR2 LO 0x… HI 0x… (capăt: ultima pagina/exclusiv)”.
3. Nimic nou pe placă. La fereastra următoare: copiază `raport/gpu-jurnal-002.txt` de pe stick-ul de test în hub `rezultate/`;
   noi rulăm `wpr2_after_i6` pe el și confirmăm HI. Abia apoi se repetă I6 (pornire rece, cablul scos 30 s).
