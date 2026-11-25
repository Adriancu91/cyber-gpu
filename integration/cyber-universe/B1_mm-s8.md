# B1 — `mm_s8`: înmulțirea pe 8 biți pe plăci 128×128, gata de legat în SIMULARE

**Pentru:** sesiunea OS. **De la:** Cyber-GPU, sesiunea 225 (25 nov 2026). **Nu e urgent și NU acum pe placă:**
abia după ce proba `mma16_s8` (nota `B1_mma16-s8.md`) a trecut pe placă — care la rândul ei vine după `mm_tc`
(os-352 / os-419) și cursa cu 5 variante (os-421). Ordinea o alege centrul.

## Ce e
- `C[M,N] = op(A)·op(B) (+ bias)` cu A, B întregi cu semn pe 8 biți și C, bias întregi pe 32 de biți, pe tensor cores
  (`imma.m16n8k32.i8.i8`, 16 în bucla principală).
- **Aceleași constante ca `mm_tc` / `mm_tc128`**: `nvidia_ga104::mm_s8::cbuf(&mm_tc::Args { .. })` (= `Args::cbuf`,
  a/b/c/bias la 0x700…, m/n/k/flags la 0x720…; opțiunile `TA`, `TB` = B ținută [N,K] ca greutățile Q8_0, `BIAS`).
  `a`, `b` arată spre octeți, `c`, `bias` spre întregi pe 32 de biți.
- **Aceeași lansare ca `mm_tc128`**: `mm_s8::BLOCK` = 256×1×1, `mm_s8::grid(&args)` = (⌈n/128⌉, ⌈m/128⌉, 1),
  `mm_s8::SHARED` (16 KiB, o barieră) în `to_bytes_with_shared`, `mm_s8::REGISTERS` = 104.
- Programul: `mm_s8::MM_S8`, SHA-256 `76ce12d512ebf563657b2d4b16356f07405d42995c75743a503eec296dca92c6`
  (57 568 octeți, Mesa NAK 26.3.1); mapat cu pagina de pază (`qmd::program_map_bytes`), ca toate.
- Model pentru teste: `gpu_sim::cmdproc::{mm_s8_on_cpu, MM_S8_NEEDS}`; oracolul `gpu_compute::mma_s8::{ProductS8, mm_s8_ref}`.

## Verdictul pe placă (scris înainte de orice rulare)
La întregi nu se rotunjește nimic: **C trebuie să fie bit cu bit oracolul** pe procesor. Propunere de rând de test (când
va fi cerut): seturile T231b (6 forme × 8 opțiuni) + valorile extreme T231c + o formă #513 întreagă (qkv: 1024×2304×768,
TB|BIAS) pentru timp; C umplut înainte cu `0x7fff_fff7`; rând `mm_s8 set N: ACEIASI BITI` / `DIFERA la (i, j)`,
apoi `proba mm_s8: TRECE: x / y` și `kt mm_s8 qkv: µs median, TOPS`. **Nu se trimit date care depășesc i32**
(oracolul le refuză; cu |a|, |b| ≤ 128 și K ≤ 3 072 nu se întâmplă).

## Ce e nou pentru placă
Doar programul `mm_s8` (cel de mai sus). Nicio scriere nouă în registre, nicio comandă nouă — calea de lansare a lui
`mm_tc128`. **Doar adăugări** în crate-uri; nimic redenumit.

## Viteza (DEDUS, nemăsurat)
Whitepaper-ul GA102 dă int8 ≈ ×4 față de f16 cu acumulare f32 (calea `mm_tc`). Încărcarea felilor e încă octet cu
octet — poate fi frâna; o vedem doar pe placă. Pentru antrenare, int8 cere și scara pe bloc (Q8_0) — pasul următor,
după întrebarea pentru Kronos dacă antrenarea pe 8 biți ține nota (riscul pistei B1).

Teste: T231 (`crates/nvidia-ga104/tests/sim_b1_mm_s8.rs`, 7/7, SIMULARE; T231g = codul GLSL transcris literal pe
procesor, ca să prindă greșelile de index din program, nu doar din model). Documentul: `docs/B1_MM_S8.md`.
