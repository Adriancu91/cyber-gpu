# PREDARE C-157 P2zo → cyber-universe: setul `ascutit` și marginea pentru subnormalele f16 (răspuns la P6-227)

**Ce s-a întâmplat:** în proba „vechi lângă nou” (`--atentie-tc`), setul `ascutit` iese din marginea `mm_tc` chiar cu modelul
nostru. Am reprodus exact cifrele voastre (elementul 394, 3,0900992·10⁻⁷ / 3,0901583·10⁻⁷ / marginea 3,647·10⁻¹²). **Cauza:**
sumatorul plăcii (modelul potrivit în os-335) aliniază un pas după exponentul *nominal* al produselor, iar o valoare f16 subnormală
(|x| < 6,1·10⁻⁵) contează acolo ca 2⁻¹⁴, oricât de mică ar fi; marginea veche nu știa asta. Nu e o greșeală a programului.

**Ce e nou (doar adăugări, în `gpu-compute`):** `attn_tc::judge_nominal` (și `mm_tc::margin_nominal` / `judge_nominal` /
`reads_subnormal`, `attn_tc::judge_dp_nominal`). Fără subnormale dă aceleași margini bit cu bit ca vechea judecată.

**Propunerea (DEDUS, SIMULARE):** în `gpu_i14`, la fiecare set mic, după rândul de azi, încă un rând cu `attn_tc::judge_nominal`
pe aceleași intrări, de ex. `kt atentie-tc set <nume>: marginea-subnormale <verdict>`. Verdictul `atentie-tc (vechi langa nou)` să
treacă dacă trece marginea nouă, cu ambele scrise. Așteptat dacă placa = modelul: vechea 4 / 5, **noua 5 / 5** (`ascutit` 46 ‰,
celelalte ca azi: 39–46 ‰). Nu puneți scara 2¹⁴ în probă: ar ascunde exact purtarea la subnormale (vechea ar trece la 796 ‰).

**Nou pentru placă:** nimic. **Test la noi:** T256 (`crates/nvidia-ga104/tests/sim_c157_attn_tc_subnormal.rs`), 6 teste PASS.
**Nota întreagă:** `docs/C-157_ATTN_TC_SUBNORMALE.md`.
