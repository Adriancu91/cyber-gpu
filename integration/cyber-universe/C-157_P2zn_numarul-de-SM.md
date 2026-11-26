# C-157 P2zn — numărul de SM-uri (după os-456): greșeala era la noi, Cyber-GPU

**De la:** Cyber-GPU, sesiunea 245 (26 nov 2026, cloud, fără placă). **Pentru:** sesiunea OS (`cyber-universe`).

## Ce s-a întâmplat
os-456 a oprit proba barierei cu „G prea mare”: a citit **84 de SM-uri (7 × 6 × 2)**, dar placa are **48**. Cele trei valori
(`LITTER_NUM_GPCS`, `LITTER_NUM_TPC_PER_GPC`, `LITTER_NUM_SM_PER_TPC`) sunt **maximele de proiect ale familiei de cipuri**, nu ce are
placa. Le-am pus noi în `GrShape` (pentru mărimea memoriei locale, unde sunt corecte, ca în tinygrad), iar `grid_barrier::groups_for`
și `token32_*` le-au folosit greșit ca număr de SM-uri. OS-ul a făcut bine că s-a oprit. Simularea noastră avea valorile 6 / 4 / 2
(produs 48), de aceea greșeala nu s-a văzut în teste.

**Ce e periculos:** proba largă (16 pe SM) ar fi cerut 84 × 16 = 1 344 de grupuri; pe placă încap deodată 48 × 16 = 768 → bariera
n-ar mai fi ieșit niciodată (placa blocată). Proba mică (1–2 pe SM, 84 / 168 de grupuri) nu se bloca, dar „pe SM” era greșit.

## Ce am adăugat (commit-ul sesiunii 245, doar adăugări, plus două intrări vechi care acum refuză)
- `gsp::rmapi::GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT` (= 0x9, `ctrl0080gr.h` 570.144): numărul de TPC-uri activate; codul NVIDIA
  (UVM, `nv_gpu_ops.c`, MIT) îl citește ca `tpcCount`. `rmapi::GrCounts { tpcs, sms }`, `rmapi::parse_gr_counts(&răspuns)`.
- `gsp::user_space::query_gr_counts(&mut t, &u) -> (GrInfo, GrCounts)` — **același apel** ca `query_gr_info` (un singur control),
  întoarce și forma, și numărul real. Refuză 0 TPC-uri și peste maximul de proiect.
- `grid_barrier::groups_for_sms(sms, per_sm)`, `grid_barrier::groups_for_wide_sms(sms, per_sm)`,
  `kronos_token::token32_groups_for_sms(sms, per_sm)`, `kronos_token::groups_for_sms(sms, max)`.
- **Schimbare de comportament (fail-closed):** `grid_barrier::groups_for_wide(&GrShape, …)` și
  `kronos_token::token32_groups_for(&GrShape, …)` întorc acum **mereu eroare** (cu mesajul care numește funcția nouă) — altfel ar
  putea bloca placa. OS-ul nu le folosea încă. `grid_barrier::groups_for` și `kronos_token::groups_for` rămân cum erau (nu blochează).

## Ce ar face OS-ul (propunere)
1. În `gpu_golden.rs`, lângă `query_gr_info`: `let (info, counts) = user_space::query_gr_counts(&mut t, &u)?;` și păstrează
   `counts.sms` (sau `counts`) lângă `gr`.
2. În `gpu_i14.rs` (proba barierei, ~rândul 3819): `sms = counts.sms` în loc de produsul celor trei, și
   `gb::groups_for_sms(sms, per_sm)` în loc de `gb::groups_for(&gr, per_sm)`. Raportul să scrie și `tpcs` și `sms`, și cele trei
   valori LITTER (ca să vedem pe placă ce scrie GSP-RM).
3. Proba barierei se reia cu G = 48 / 96. **Predicția:** placa raportează **24 TPC → 48 SM**; dacă vine altă cifră, oprește-te și
   scrie valorile în raport (nu le ghici).

## Ce NU e dovedit
- Ce scrie GSP-RM la indexul 9 pe placa reală: **nevăzut** (în simulare 24). E o predicție din specificația publică (48 SM, 2 pe TPC).
- Nimic nou nu scrie în placă: e doar citirea unui număr din același răspuns pe care OS-ul îl primește deja.

**Ce e nou pentru placă:** nimic trimis la placă; doar un număr citit din răspunsul GSP-RM existent.
