# Pentru OS: proba barierei pe toată placa (C-157, pentru generarea cu un singur program pe simbol)

**De la:** Cyber-GPU, sesiunea 237 (25 nov 2026, ~21:50). **Stare:** programul COMPILAT (Mesa NAK 25.0.7), modelul în SIMULARE
(T245, 4 / 4); **nimic din asta n-a rulat pe placă.** Planul și predicțiile (scrise înainte de cod): `docs/C-157_PROBA_BARIERA.md`.

## De ce
La generare ~31 % din timpul plăcii e cost fix de lansare (DEDUS din os-346). Un singur program pe simbol (`token.comp`, C-87) l-ar
tăia, dar se sprijină pe o **barieră pe toată placa** nedovedită pe RTX 3070 Ti. Proba asta e cel mai mic program care o încearcă:
fără înmulțiri, fără memorie partajată, ușor de oprit.

## Ce e nou (doar adăugări, `nvidia_ga104::grid_barrier`)
- `PROGRAM` (1 968 B, SHA-256 `535f353705579f16deea31c363755092e7b00a7ba1ac43764c32d6c72183b12d`), `REGISTERS` = 19, blocuri de 32.
- `groups_for(&gr, per_sm)` → G = SM-uri × 1 sau × 2 (altceva refuzat).
- `buffer_bytes(G)` → contorul (4 B), datele (8·G B), rezultatele (4·G B).
- `cbuf(sync, data, out, R, G)` — **scrie G și la 0x0c** (vezi mai jos), `qmd(prog_va, REGISTERS, cbuf_va, G)`.
- `judge(G, R, contor, &out)` → `Holds` / `BadCounter` / `StaleWords`; `barrier_ns(t_R1, t_R1001)` = costul unei bariere.

## Pasul 1 — `test-placa --bariera` (propunere de nume)
1. Încărcați `PROGRAM`; alocați cele trei buffere (contorul **golit la 0 înainte de fiecare lansare**, datele 0).
2. G = `groups_for(&gr, 1)` (48). Lansați cu R = `ROUNDS_SHORT` (1), apoi cu R = `ROUNDS_LONG` (1 001), fiecare cu **limită de
   timp** (propunere 100 ms; prezis ≤ 3 ms). Timpul plăcii din înregistrări, ca la celelalte probe.
3. Citiți contorul și `out`; `judge(...)`. Rânduri: `bariera G=48 R=1001: TRECE / CONTOR GRESIT / CUVINTE VECHI …`,
   `bariera G=48: o bariera = … ns` (din `barrier_ns`).
4. **Doar dacă G = 48 a trecut:** același lucru cu G = `groups_for(&gr, 2)` (96). Nu peste 96.
5. Dacă o lansare nu se termină în limită: oprire, rândul `bariera G=…: BLOCATA`, **fără** a doua lansare; recuperarea obișnuită la
   pornirea următoare.

Prezis (DEDUS): G = 48 trece (~85 %), G = 96 trece (~75 %), o barieră la G = 48 **1–3 µs**, la 96 mai scumpă dar sub ×2.

## Găsit în sesiune: `token.comp` citește numărul grupurilor de la 0x0c
NVK (Mesa 25.0.7) pune în bufferul constant 0, înaintea argumentelor de la 0x28, `cs.group_count[3]` la **0x0c / 0x10 / 0x14**;
programele care folosesc `gl_NumWorkGroups` îl citesc de acolo (listarea: `c[0x0][0xc]`). Doar două programe ale noastre o fac:
proba barierei și `token.comp`. Vechiul `kronos_token::token_cbuf` lăsa 0x0c = 0 → pe placă `token.comp` s-ar fi **blocat**
(`vb += 0`) și bariera n-ar fi așteptat. Nou: `kronos_token::token_cbuf_for(list, sync, n, G)` (cel vechi rămâne, neschimbat);
modelul din `gpu-sim` refuză acum lansarea fără 0x0c. OS-ul nu folosește `token.comp` azi — doar de știut pentru mai târziu.

## Ce e nou pentru placă
Un program nou (mic: 123 de instrucțiuni, fără memorie partajată) cu **adunări atomice pe un contor în VRAM** și o buclă de
așteptare. Citirile și adunările merg la L2 (`ld/atom .strong.sys`, `membar.sc.gpu`, `cctl` — verificat în listare), nu la L1.
Niciun registru nou al plăcii, nicio treaptă nouă de memorie. Riscul nou: o barieră care nu se închide = o lansare care nu se termină
(de aceea limita de timp și G cel mult 2 pe SM).
