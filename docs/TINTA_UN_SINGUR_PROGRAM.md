# Țintă de plan: Kronos pe placă ca un singur program pe simbol

- **Sursa:** hub `idei/2026-11-03_cyber-gpu-un-singur-bloc.md` — aprobată de Adrian („ok”, 3 nov 2026 13:25) **ca direcție de plan, nu sarcină de acum**.
- **Stare:** doar plan. Nimic din acest document nu are cod nou și nimic nu e testat pe RTX 3070 Ti. Ordinea rămâne: I6 → I7 … I10 → primul calcul pe placă → treptele matmul → abia apoi asta.

## Ideea, pe scurt

Pe procesor, Kronos pierde ~63 % din timp așteptând între cele ~63 de calcule ale unui simbol (C-59). Pe placă, echivalentul „clopoțelului” este
**lansarea fiecărui calcul de către procesor**. Ținta: un singur program GPU pornit o dată pe simbol (sau pe mai multe), care face toate calculele
fără să se întoarcă la procesor. Blocurile de lucru își iau bucăți dintr-un contor comun în memoria plăcii, așteaptă doar datele calculului de
dinainte (un „caiet” de contoare în VRAM), iar cine termină ajută.

## Ce avem deja (SIMULARE, `gpu-sim`)

| Piesă | Unde | Ce permite |
|---|---|---|
| Semafor la sfârșitul lucrului (`release0` din QMD) | `nvidia-ga104::qmd` | placa anunță „gata” scriind în VRAM, fără întrerupere |
| Așteptare/eliberare de semafor în coada de comenzi | `gpu-sim::cmdproc` (metodele host de semafor) | două calcule înlănțuite fără procesor între ele — azi doar între lansări |
| Inel GPFIFO + `GP_PUT`/`GP_GET` | `gpu-command::ring`, `gpu-sim::cmdproc` | procesorul pune multe lansări deodată și nu așteaptă după fiecare |

## Ce lipsește (de proiectat când ajungem acolo)

1. **Contoare atomice în programul GPU** (compilat cu Mesa NAK): luarea bucății următoare și „caietul” de dependențe. Necesită operații atomice
   globale în shader — de verificat ce emite NAK pentru `atomicAdd` pe memorie globală și cum se modelează în `gpu-sim`.
2. **Toate blocurile rezidente deodată.** Un bloc care așteaptă după altul care încă n-a pornit = blocaj pe veci. Grila trebuie limitată la câte
   blocuri încap simultan (GA104 pe 3070 Ti: 48 SM × blocuri pe SM, după registre și memorie partajată) — calculat din parametrii QMD, nu ghicit.
3. **Ceas de siguranță și oprire curată.** Un program lung blocat ține placa ocupată. Programul citește un „steag de oprire” din VRAM la fiecare
   bucată; procesorul are un termen-limită, ca la `gpu i6`. Fără scrieri în registre necunoscute pentru oprire forțată.
4. **Format de comenzi:** o singură lansare cu un tabel de calcule (adrese, mărimi, dependențe) în VRAM, în loc de 63 de QMD-uri.

## Cum se măsoară (în ordinea asta)

1. **Costul unei lansări** pe placa reală: N lansări, după fiecare o eliberare de semafor cu marcă de timp scrisă de placă. Fără asta nu știm dacă merită.
   **Pregătit (C-69, sesiunea 47, SIMULARE):** `nvidia-ga104::launch_cost` (`encode` + `analyze`), test T75; pe placă în fereastra 4 din `docs/FEREASTRA_C-69.md`.
2. **„Câte un program pe calcul”** (varianta simplă, cu semafoare între lansări) pe un strat Kronos.
3. **„Un singur program”** pe același strat; rezultat identic bit cu bit cu varianta 2 și cu referința de pe procesor (`gpu-compute::reference`).
   Câștigă măsurătoarea; dacă diferența e mică, rămâne varianta simplă.

## Reguli care nu se schimbă

- Nicio scriere pe placă fără Adrian de față; orice cod care scrie stă în spatele opțiunii `hw-write`, oprită implicit.
- Testele pe `gpu-sim` se etichetează SIMULARE. Clean-room neschimbat (`docs/PROVENANCE.md`).
- Lecțiile de pe procesor din C-59 (bloc fără clopoțel, ajutor la cel lent, un fir liber pentru întreruperi) se aplică și firului care comandă placa
  — asta e partea sesiunii OS.
