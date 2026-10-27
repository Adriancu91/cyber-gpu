# Observație pentru sesiunea OS: unde e imaginea ecranului (doar afișare, pentru fereastra B0)

*27 oct 2026, sesiunea Cyber-GPU 10. Doar o observație/propunere: OS-ul decide (o poate prelua doar cu „da”-ul lui Adrian).
Nimic de aici nu e aplicat în `cyber-universe`.*

## De ce
Driverul GPU face acum (I9a, doar SIMULARE) propriile tabele de pagini și își alege memoria video dintr-o zonă dată ca parametru
(`nvidia_ga104::mmu::GpuMemory::new(start, end, ...)`). Pe workstation, RTX 3070 Ti ține și ecranul: imaginea pe care Limine o
primește de la UEFI (GOP) stă în memoria plăcii. Ca driverul să nu scrie peste ea, trebuie să știm **unde** e: adresa fizică a
framebuffer-ului Limine minus baza BAR1 a plăcii = poziția în VRAM (DEDUS: GOP mapează VRAM-ul liniar de la începutul BAR1;
de confirmat cu cifrele).

## Ce ar fi util în OS (doar citire, nicio scriere în placă)
O linie în plus în `hw` (sau în `gpu identifica`):
```
framebuffer: adresa fizica 0x<...>, <latime>x<inaltime>, pitch <...> B, 32 bpp
```
Adresa fizică vine din răspunsul Limine (cererea de framebuffer pe care kernelul o face deja), nu din placă. Împreună cu BAR1 din
`lspci`/`gpu identifica`, Adrian o fotografiază la pasul 9 din `docs/TEST_MATRIX.md` (B0).

## Ce NU face
Nu atinge placa video, nu schimbă nimic din grafică. Dacă OS-ul nu o preia, B0 merge oricum; atunci zona se alege conservator
(de exemplu peste primii 256 MiB) până la proba P.
