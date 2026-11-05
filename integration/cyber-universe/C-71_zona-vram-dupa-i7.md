# C-71 / „De ce are nevoie” al OS — zona VRAM pentru golden / I9 / I10 (sesiunea Cyber-GPU 53, 5 nov 2026)

**Răspuns la „Confirmarea zonei VRAM pentru golden/I9/I10 (peste 256 MiB, sub GSP-RM)” din `stare/cyber-universe.md`.**
Nimic testat pe RTX 3070 Ti; nimic nou de scris pe placă.

1. **Zona aleasă de OS e bună** (`gpu_golden::zone`: de la `max(capătul ecranului, 256 MiB)` până la `FbLayout::free_below().end`).
   Pe valorile reale de pe placă din 5 nov (WPR2 după I7 = `0x1f4100000..0x1ffee0000`, citit `LO 0x01f41000 / HI 0x01ffee00`),
   aranjamentul nostru dă exact același început de WPR2, deci capătul zonei e `0x1f4000000` (sub heap-ul non-WPR de 1 MiB
   `0x1f4000000..0x1f4100000`, care e sub WPR2). Test T91 (valori reale; mărimea `.fwimage` DEDUSĂ din WPR2, fișierul nu e în repo).
2. **Propunere (mică, doar citire, opțională):** după I7 și înainte de golden, folosiți
   `FbLayout::free_after_i7((wpr2.start, wpr2.end))` (WPR2 recitit după I7, cum îl citește deja turul — `decode_wpr2`) în loc de
   `free_below()`. Întoarce aceeași zonă când Booter a pus WPR2 unde a cerut `GspFwWprMeta`; altfel **eroare înainte de prima scriere în VRAM**
   (de ex. dacă mărimea ELF-ului s-ar schimba). Nicio scriere nouă, nicio citire nouă (WPR2 se citește deja după I7).
3. **Ce nu știm încă (DEDUS, de văzut după I8 pe placă):** GSP-RM raportează și el regiunile FB rezervate (`GET_GSP_STATIC_INFO`,
   `fbRegionInfoParams` în RM 570.144). Nu le citim încă; zona noastră e sub tot ce a cerut `GspFwWprMeta`, ca la tinygrad/nouveau.
   Dacă la fereastră I8 trece, un pas următor posibil în cyber-gpu: compararea zonei cu acele regiuni.
4. Interfața publică: doar o adăugare (`FbLayout::free_after_i7`); `free_below` neschimbat.
