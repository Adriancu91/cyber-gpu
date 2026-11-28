# C-168 p.5 — P5e: coada lui `_gelu` / `_res` cu citirile pornite înainte (aceiași biți; testat doar în SIMULARE)

Sesiunea Cyber-GPU 295, 28 nov 2026, în cloud, fără placă. De ce: `docs/C-168_P5_LIPIRE_PLAN.md` §14 (os-512: GELU' lipit a câștigat
~0 la lot 6, pentru că fiecare din cele 64 de citiri ale lui `u` dintr-un fir aștepta memoria singură, una după alta: +190 µs pe
lansare). Planul și predicțiile N1–N7 (scrise înainte de cod, commit `89a30de`): §15; rezultatul: §16. Testul: T312
(`crates/nvidia-ga104/tests/sim_c168_p5e_pf.rs`).

## Ce e nou (doar adăugări)
- Șapte programe noi, din aceleași surse cu `-DPF` (Mesa NAK 26.3.1, SM 8.6), în modulul nou `nvidia_ga104::mm_tc_fast_pf`:

  | Program | Constanta | SHA-256 | Registre | Înlocuiește | Mărime |
  |---|---|---|---|---|---|
  | `mm_tc_v7_gelu_pf` | `MM_TC_V7_GELU_PF` | `4c5a3e96d27f4133c9eafb0118d3e4d798cca73b1633cd1d7af3aac01613e713` | 128 | `mm_tc_v7_gelu` | 41 536 B |
  | `mm_tc_v8_gelu_pf` | `MM_TC_V8_GELU_PF` | `7f8a0e9478e2708653bed3ec6d7ce637b62260295422a023df735594f0ac5023` | 128 | `mm_tc_v8_gelu` | 65 488 B |
  | `mm_tc_v8n_gelu_pf` | `MM_TC_V8N_GELU_PF` | `f90955d272158714aae0ba269b048a44cc9c0f2441514b2bfb2717dbd83417a2` | 144 | `mm_tc_v8n_gelu` | 64 400 B |
  | `mm_tc_v7_res_pf` | `MM_TC_V7_RES_PF` | `f4948df06a49d301061346e3b26c3e1113980bd3bee070180ef6177764d3fc0b` | 128 | `mm_tc_v7_res` | 35 392 B |
  | `mm_tc_v8_res_pf` | `MM_TC_V8_RES_PF` | `e8adabc871bc54c6342acf879081c5511c99459f761ac33690a27e6ade3048b5` | 128 | `mm_tc_v8_res` | 53 200 B |
  | `mm_tc_v8n_res_pf` | `MM_TC_V8N_RES_PF` | `5c32a4e31bc262863448ed2372d475612b2cd2ec79afbf33fe82709a523cb7c6` | 144 | `mm_tc_v8n_res` | 52 112 B |
  | `mm_tc_v8r_res_pf` | `MM_TC_V8R_RES_PF` | `5ed7ad98fcea454b83d1239f1ebe0e28569f7f7179437d938390c717357a20bd` | 120 | `mm_tc_v8r_res` | 51 968 B |

  `mm_tc_fast_pf::check()` verifică cele șapte sume. Registrele, blocul, grila, memoria partajată și cbuf-ul sunt **exact** ale
  programului pe care îl înlocuiește; aritmetica fiecărui element e aceeași, literă cu literă, pe aceeași valoare. Diferența: citirile
  lui `u` / `res` ale bucății următoare pornesc înainte să se lucreze bucata curentă (8 citiri în așteptare deodată, la v7 16; azi 1).
  Toate cele 15 programe de azi au ieșit **cu aceleași SHA-256** la recompilare.
- Nimic altceva nu se schimbă în bibliotecă: un program `_pf` intră **în același loc** ca programul pe care îl înlocuiește
  (`GeluFast { v7, v8, v8n }` pentru `_gelu`, `ResFast { v7, v8, v8n, v8r }` pentru `_res`). Packerul scrie aceleași QMD-uri.

## Ce cerem OS-ului
1. O opțiune nouă (de ex. `--pas-coada-pf`), peste rândul P5c (`--pas-gelu-inainte-lipit`) și, dacă e deja legat, peste P5d: încărcați
   în locurile lui `mm_tc_v7_gelu` / `mm_tc_v8_gelu` / `mm_tc_v8n_gelu` octeții `MM_TC_V7_GELU_PF` / `MM_TC_V8_GELU_PF` /
   `MM_TC_V8N_GELU_PF` (și, cu P5d, în locurile `_res` cele patru `_RES_PF`), cu sumele de mai sus. Restul rândului neschimbat.
2. **Întâi lot 2 cu oracolul**, apoi lot 6 (și, dacă vreți, lot 6 pe datele reale). Ordinea care ne spune cel mai mult: (a) rândul
   os-514 + `--pas-coada-pf` doar cu cele trei `_gelu_pf` (măsoară N5 curat); (b) P5d cu `_res` simplu; (c) P5d cu `_res_pf` (N6).
   Dacă fereastra pe placă e scurtă, (a) și (c) ajung.
3. **Așteptat (scris înainte de placă):**
   - aceleași lansări și aceleași alegeri ca rândul de dinainte (numele programelor din profil rămân; doar sumele SHA-256 sunt altele);
   - N7: oracolul lot 2 **ACELEAȘI NUMERE** (în SIMULARE sunt aceiași biți);
   - N5 (DEDUS): lot 6 „inapoi:proj-dX+gelu” **9,75 → 7,6–8,2 ms** (−1,5…−2,1 ms pe pas); lot 2 `mm_tc_v8_gelu` 3,39 → 2,85–3,05 ms.
     Sub −0,8 ms la lot 6 = RATAT (trimiteți liniile `proj-dX+gelu`, `mm_tc_v8n_gelu` din profil);
   - N6 (DEDUS, doar cu P5d): „out+rez” / „proj+rez” cu încă −0,5…−0,9 ms la lot 6 față de `_res`.

## Ce arată SIMULAREA (T312) și CALCULUL, și ce nu
- N1–N3 (CALCUL, pe listările NAK): programele de azi ies cu aceleași sume; registrele = ale programelor înlocuite, 0 „spills” în
  memorie; `tools/nak/compara_pf.py`: aceleași operații în virgulă mobilă în aceeași ordine (1 248 / 1 256 / 640 la GELU', 160 / 168 /
  160 / 96 la reziduu), același număr de scrieri, citirile în așteptare deodată 1 → 8 (v7: 1 → 16); greșeli plantate prinse.
- S1: modelul mic, 5 pași rapizi cu toate lipirile (P5a–P5d), programele `_pf` în locurile `_gelu` / `_res`: **aceiași biți**, aceleași
  alegeri, pe vitezele os-500, v8, v8 + v8r, v8n. (Simulatorul folosește același model de calcul pentru `_pf` ca pentru programul
  înlocuit — că programele compilate fac aceeași aritmetică arată N3, nu S1.)
- S2: programele `_pf` încap în locurile lor; în QMD-urile lui #513 lot 2 și 6 adresa și registrele sunt cele bune pentru ele.
- Nimic din toate acestea n-a rulat pe placă.
