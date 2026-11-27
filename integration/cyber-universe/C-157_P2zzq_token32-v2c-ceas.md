# C-157 p.2 — P2zzq: `token32_v2c`, programul cu ceas — unde stau cele ~0,67 ms ale lui `token32_v2`

Sesiunea Cyber-GPU 274, 27 nov 2026, în cloud, fără placă. Planul și predicțiile C1–C8 (scrise înainte de cod):
`docs/C-157_TOKEN32_CEAS.md`. Teste: T288 (`crates/nvidia-ga104/tests/sim_c157_token32_v2c.rs`, SIMULARE + listare) și testele
din `nvidia_ga104::token_clock`.

## Ce e
Un program de **măsurat**, nu de generat: `token32_v2` (aceeași aritmetică, aceeași ordine — verificat în listare: aceleași
`ffma`/`fadd`/`fmul`/`mufu`/`shfl`/conversii, aceleași citiri, aceeași barieră) plus citirea ceasului SM-ului (`cs2r sr[0x50]`, numărătorul
de cicluri al SM-ului) la începutul și la sfârșitul fiecărei operații și scrieri într-o zonă separată. Compilat cu Mesa NAK 25.0.7
(`token32_v2` reconstruit identic, `5f4d22a0…`): **7 570 de instrucțiuni, 99 de registre** (v2: 92), 0 memorie locală, aceeași memorie
partajată; SHA-256 `e899fa7cce1d7e31e511f0304bb6d82961c0e13bcd19d86827d58db4b6c18206`. 16 blocuri pe SM încap în continuare.

## Ce cerem OS-ului (opțiune nouă, oprită implicit)
`--token32-v2c`, doar imediat după `--token32-v2`; rulează **doar la G = 768**, după `token32_v2` din același lanț, aceleași texte /
simboluri. Rândul: `test-placa --tablou-viteza --token32 --token32-v2 --token32-v2c`.

Pentru fiecare lansare (`nvidia_ga104::token_clock`, doar adăugări; `kronos_token` neschimbat):
1. programul `TOKEN32_V2C` (verifică `TOKEN32_V2C_SHA256`);
2. zona ceasului: `4 · clock_words(n, 768)` octeți în memoria plăcii (n = 88 → 68 441 de cuvinte, ~267 KiB), citită înapoi după
   lansare ca logit-urile; nimic nu trebuie golit înainte (programul scrie fiecare cuvânt);
3. bufferul constant: `token_cbuf_clock(list_va, sync_va, n, 768, clock_va)` (= `token_cbuf_for` + adresa zonei la 0x40);
4. QMD: **`token32_v2c_qmd_wide(prog_va, cbuf_va, 768, sms)`** — NU `kronos_token::token32_qmd_wide` (acela scrie 92 de registre;
   programul folosește 99 → ar strica registrele);
5. după fiecare simbol: `read_clock(&words, n, 768)` → `by_part(&r, &parts_of(&launches)?)` → `rows(board_ns)` cu timpul plăcii
   **al aceluiași simbol** (din înregistrări). În jurnal: rândurile `kc …` ale **ultimului simbol** (40 de rânduri: 7 de sumar + 3
   pe fiecare din cele 11 feluri). Dacă `read_clock` refuză zona, rândul de eroare și mergem mai departe (nu e o eroare a plăcii).

Trece = aceleași simboluri și aceeași amprentă ca `token32_v2`, contorul (n − 1) · 768. Dacă amprenta diferă, `token32_v2c` se
oprește (ca v2).

## Ce așteptăm (DEDUS, `docs/C-157_TOKEN32_CEAS.md` §3)
C1 aceiași biți; C2 placa ≤ +3 % față de `token32_v2` (≤ ~1,72 ms); C3 `kc ceasul SM dedus` 1 400–2 000 MHz; C4 `kc bariera dupa cel
mai lent` 0,14–0,40 ms; C5 restul în munca înmulțirilor (qkv + out + fc + proj + logit, `munca max`) ≥ 0,40 ms peste lista obișnuită;
C6 ln1 + ln2 + ln_final + argmax 0,05–0,20 ms; C7 `kc citiri contor pe grup si bariera (x10)` ≥ 200; C8 atenția ≤ ×1,3 față de listă.

## Ce e nou pentru placă
Doar programul `token32_v2c` (o singură lansare pe simbol, ca v2): citiri ale ceasului SM-ului (registru special al SM-ului, citire,
nu scriere) și scrieri în memoria noastră. **Niciun registru nou al plăcii, nicio scriere MMIO.** Interfața: doar adăugări.
