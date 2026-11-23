# C-157, punctul nou (a): proba vitezei RAM ↔ VRAM cu motorul de copiere al plăcii

Cerut de Adrian (23 nov 08:24, prin centrul de comandă): **viteza maximă RAM ↔ VRAM în ambele sensuri**, cu motorul de
copiere al plăcii, blocuri mari, mai multe în paralel. Azi vedem ~6,2 GB/s la urcarea greutăților — dar acolo copiază
**programul de calcul** `copy_b128` (C-83 I11), nu motorul de copiere. PCIe 3.0 x16 are plafon teoretic ~15,8 GB/s.

Planul și predicțiile sunt scrise **înainte** de cod (sesiunea 183). Totul se testează doar în SIMULARE.

## Ce e nou pentru placă (un singur lucru)

**Obiectul de copiere `AMPERE_DMA_COPY_B` (clasa `0xC7B5`) folosit pe canalul de lucru.** Obiectul există deja: imaginea
„golden” îl alocă pe canal de la I8e (`gsp::golden`, pasul 6), dar nimeni nu i-a trimis până acum vreo metodă. Acum:

- `SET_OBJECT` cu clasa `0xC7B5` pe **subcanalul 4** (calculul rămâne pe subcanalul 1);
- pe copie: `OFFSET_IN_UPPER/LOWER`, `OFFSET_OUT_UPPER/LOWER`, `LINE_LENGTH_IN` (copie într-o singură linie, ca tinygrad),
  apoi `LAUNCH_DMA` cu: sursă și destinație „pitch”, adrese virtuale, `FLUSH_ENABLE` când copia scrie o înregistrare;
- înregistrarea de timp a copiei: semaforul motorului de copiere (`SET_SEMAPHORE_A/B/PAYLOAD` + `LAUNCH_DMA`
  `SEMAPHORE_TYPE = RELEASE_FOUR_WORD_SEMAPHORE`), 16 octeți: valoarea la +0, ceasul plăcii la +8 — **aceeași formă** ca
  înregistrările canalului pe care placa le scrie deja (DEDUS: forma celor 4 cuvinte din antet, de confirmat pe placă).

Toate numerele vin din antetul NVIDIA `clc7b5.h` (open-gpu-doc, MIT), generat acum în `nvidia-rm570::regs::clc7b5` (doar
adăugare). Niciun registru nou. Memoria: aceleași PTE spre RAM ca la I11 (`mmu::alloc_sys_mapped`).

**Corectură aflată la cod (sesiunea 183, înainte de orice test):** canalul de lucru (`open_work_channel`) alocă doar
obiectul de calcul; cel de copiere e alocat doar pe canalul „golden”. Deci mai trebuie **un mesaj RM**: alocarea aceleiași
clase `AMPERE_DMA_COPY_B` (fără parametri) pe canalul de lucru — `gsp::user_space::alloc_copy_object` (adăugare). E aceeași
alocare pe care placa a acceptat-o deja la imaginea golden, pe alt canal. Fără ea, `SET_OBJECT` pe subcanalul 4 ar fi
refuzat de placă (modelul nostru nu verifică asta — scris și în notă).

## Proba (o singură trimitere)

Mărimea maximă `M` o alege OS-ul (implicit **64 MiB**; în QEMU mai mică). Bufferele:

| Buffer | Unde | Mărime | Rol |
|---|---|---|---|
| `A` | RAM | `M` | sursa, umplută de OS cu modelul `transfer::pattern_word` |
| `B` | RAM | `2M` | destinația, umplută cu `0xEE`: prima jumătate pentru copiile simple, a doua pentru cele în paralel |
| `V1`, `V2` | VRAM | `M` fiecare | V1 pentru copiile simple, V2 doar pentru cele în paralel |
| înregistrări | VRAM (ca la I11) | 31 × 16 B | ceasul |

Ordinea, după înregistrarea 0 (a canalului, ca până acum):

1. **sus** (RAM → VRAM, `A → V1`): mărimile `M/64`, `M/16`, `M/4`, `M`, fiecare de 3 ori, câte o înregistrare după fiecare;
2. **jos** (VRAM → RAM, `V1 → B[0..M]`): aceleași mărimi, de 3 ori;
3. **sus în paralel** (`A → V2`): 4 copii de `M/4`, prima „neînlănțuită” (așteaptă ce era înainte), celelalte trei
   `PIPELINED` (pot porni înainte să se termine cea dinainte), o singură înregistrare după a patra; de 3 ori;
4. **jos în paralel** (`V2 → B[M..2M]`): la fel.

Verificarea, după ultima înregistrare: tot `B` (ambele jumătăți) = modelul, cuvânt cu cuvânt; `A` neatins. V2 e scris
**doar** de copiile în paralel, iar a doua jumătate a lui `B` **doar** din V2, deci o copie în paralel greșită se vede.
Copiile mici sunt prefixe ale celor mari (aceleași metode, alte lungimi): rezultatul lor e acoperit de copia de `M`;
contează timpul lor.

Rândurile pentru jurnal: `kt ramvram …` (mărime, sens, cele 3 timpi, mediana, MB/s), verdictul, primul cuvânt greșit.

## Predicții (scrise înainte; pe placă, workstation, PCIe 3.0 x16)

| Ce | Predicție | De ce |
|---|---|---|
| Placa primește metodele (nicio eroare de canal) | DA, cu riscul de mai jos | obiectul e deja alocat pe canal de RM |
| sus, `M` = 64 MiB | **11–13 GB/s** | PCIe 3.0 x16 cu memorie fixată dă în practică ~12–13 GB/s (pachete de 256 B, antete) |
| jos, 64 MiB | **11–13,5 GB/s** | la fel; adesea puțin peste sus |
| sus / jos, 1 MiB | 6–11 GB/s | costul fix al unei copii (~5–15 µs) e mare lângă ~85 µs |
| în paralel 4 × 16 MiB față de o copie de 64 MiB | **±5 %** | un singur motor umple deja legătura; nu e de așteptat câștig |
| față de `copy_b128` (6,2 GB/s) | **~1,8–2,1 ×** mai repede la sus | |
| Toți octeții corecți | DA | |

**Riscuri.** (1) Motorul de copiere pe canalul de calcul (runlist-ul GR, deci un motor „GRCE”) — tinygrad folosește un
canal separat pentru copii. Dacă placa refuză, se vede ca înregistrare lipsă → oprire curată; atunci varianta a doua e un
canal separat pe runlist-ul unui motor de copiere (obiect RM nou, cu „da”). (2) Înregistrarea în 4 cuvinte poate avea altă
formă (de pildă valoarea pe 64 de biți) — analiza citește doar cuvântul de la +0 și ceasul de la +8, ca la canal.
(3) PTE-urile spre RAM sunt `SYSTEM_COHERENT` (cu snoop): poate fi mai lent decât varianta necoerentă; ar ieși sub 11 GB/s.

## Surse

- `clc7b5.h` (NVIDIA open-gpu-doc, MIT, commit fixat în `tools/gen-rm570/generate.sh`) — metodele și câmpurile;
- tinygrad `ops_nv.py` (MIT), citit ca specificație: copia într-o linie (`OFFSET_IN/OUT`, `LINE_LENGTH_IN`, `LAUNCH_DMA` =
  neînlănțuit + pitch + pitch) — codul nostru e scris de noi.

## Rezultat în SIMULARE (sesiunea 183)

- Cod: `nvidia_ga104::ram_vram` (codarea, analiza, judecata octeților, rândurile `kt ramvram …`, `parse_ramvram`),
  `gsp::user_space::alloc_copy_object`, modelul motorului de copiere în `gpu_sim::cmdproc` (metodele de mai sus; orice
  bit necunoscut din `LAUNCH_DMA` refuzat; semaforul fără `FLUSH_ENABLE` refuzat; copiile `PIPELINED` **nu** se
  suprapun în model — timpul lor în SIMULARE nu spune nimic despre paralelism).
- T192 (`tests/sim_c157_ram_vram.rs`, 4 teste) + 4 teste în modul: obiectul alocat pe canalul de lucru; proba cu
  M = 256 KiB și 1 MiB trece bit cu bit, 31 de înregistrări, 48 de copii (18 înlănțuite), timpii = exact ceasul
  modelului; greșeli plantate prinse (ultimul sfert „jos în paralel” mutat cu 16 octeți → văzut în a doua jumătate a lui
  B; trimitere tăiată → înregistrarea 30 lipsește, oprire curată; un bit necunoscut în `LAUNCH_DMA` → refuzat).
- `tools/check.sh`: 918 teste, 0 eșecuri. **Pe placă: NOT TESTED.**
