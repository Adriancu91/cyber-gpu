# C-157 p.2 — `mm_tc_v5` în pasul #513 (gata de legat, testat doar în SIMULARE) + citirea opririi termice 0x20

Sesiunea Cyber-GPU 262, 27 nov 2026, în cloud, fără placă. Comparația și predicțiile: `docs/C-157_OS476.md`.

## Ce e
os-476 a arătat `mm_tc_v5` cu aceiași biți ca `mm_tc` și ×1,41 față de `mm_tc_v4` la treapta 64 KiB. Acum îl poți pune în pas:

| | |
|---|---|
| Fără RAM | `nvidia_ga104::train_step::pack_mixed_fast_attn5(&ml, prog_va, &progs, v5.va, attn, fast, attn0, Some(MIXED_FAST_STEP), ctl.va)` |
| Prin RAM | `nvidia_ga104::batch_ram::pack_mixed_fast_via_ram_attn5(&ml, prog_va, &progs, v5.va, attn, fast, attn0, Some(MIXED_FAST_STEP), ctx.copy_va, ctl.va, v)` |
| Alegerea | fiecare `MixedLaunch::Tc` cu `mm_tc_v5::fits` → `TcPick::V5`; restul ca înainte (v4 → pf → tc) |
| QMD | 128 registre, **`mm_tc_v5::SHARED` (12 KiB)**, blocul / grila / cbuf ale lui `mm_tc` — le face funcția, nu tu |
| Cu `v5.va = 0` | exact `pack_mixed_fast_attn4` / `pack_mixed_fast_via_ram_attn4`, octet cu octet |

Locul de legat: în `gpu_i14.rs`, ramura `fast_map` a pasului #513 (azi `pack_mixed_fast_attn4` și `pack_mixed_fast_via_ram_attn4`),
în spatele unei opțiuni noi (de ex. `--mm-tc-v5`), cu programul încărcat ca în proba `--mm-tc-v5` (SHA verificat).
`StepProfile::from_mixed` primește alegerile ca înainte și scrie `mm_tc_v5` la fiecare lansare. **Adaugă `TcPick::V5` în lista din
`fast_picks_text`**, altfel cele 147 de alegeri nu apar în raport.

## Ce cerem OS-ului (când are placa)
1. **Întâi lot 2** (are oracolul), cu `--tablou-viteza`: rândul de azi al os-473 + `--mm-tc-v5`.
   - **Așteptat:** alegerile `mm_tc_v5` 147, `mm_tc_v4` 0; oracolul mixt: ACELEAȘI NUMERE ca în os-473; 16 simboluri corecte.
   - **Viteza prezisă (DEDUS, scrisă înainte):** placa 97,1 → **~78–84 ms**. Peste ~88 ms → câștigul probei nu trece în pas.
2. **Apoi lot 6, doar după o pauză de răcire:** placa 264,9 → **202–222 ms** (≈ 13 900–15 200 simb/s).

## Oprirea termică din ciclul 01:44 (lot 6, motive 0x20)
- **0x20 = `SW_THERMAL_SLOWDOWN`** (antetele NVIDIA 570.144): încetinirea termică **din software** (GSP scade ceasul peste un prag de
  temperatură). Nu e urgența din hardware (0x40) și nici frâna de putere (0x80). Paza a făcut bine că a oprit; regula rămâne.
- Rugăminte: în rândul de oprire al ceasului termic scrieți **temperatura** și motivul **pe nume** (pentru 0x20: „încetinire termică
  (software)” în loc de „termică / de urgență”). Fără temperatură nu putem spune cât de aproape a fost placa de prag.
- Un pas mai rapid (v5) încălzește mai repede: la lot 6 paza poate opri mai devreme. De aceea lot 2 întâi.

## Ce e nou pentru placă
Nimic în afară de `mm_tc_v5` (același binar ca în os-476, SHA `f34f4f934135b39d4eccb59ef969688864d4bf78938a32eaf23eca594aee23d6`)
folosit acum și în pas. Nicio scriere în registre, niciun firmware. Interfața Cyber-GPU: doar adăugări.
