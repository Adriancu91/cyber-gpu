# C-157 p.2 — P2zzn: după `token32_v2` pe placă (ciclul 27 nov 10:01)

Sesiunea Cyber-GPU 271, 27 nov 2026, în cloud, fără placă. Comparația întreagă: `docs/C-157_TOKEN32_V2_PLACA.md` (CALCUL T285).

## Ce a ieșit (MĂSURAT de OS)
- `token32_v2` dă aceiași biți ca lista obișnuită la G = 384 și 768 (16 simboluri, amprenta `0x6c8501f0`, contorul corect).
- Viteza: la **G = 768 placa 1,668 ms, cu driverul 2,281 ms** pe simbol (lista: 1,067 / 2,083). Încă cu ~10 % mai lent decât lista,
  dar partea driverului (0,61 ms) e acum mai mică decât a listei (1,02 ms). La G = 384 e mai lent (1,969 / 2,578).

## Ce propunem OS-ului (nimic obligatoriu)
1. **Rândurile viitoare:** doar `--token32-v2` (fără `--token32`: v1 e mai lent peste tot) și, dacă se poate, doar G = 768.
2. **Rândul `cuvinte scrise in lista pe simbol`:** azi e media cu primul simbol (scris întreg) inclus → 348. Mai clar: primul
   separat și mediana celorlalte (pe placă au fost 37–38). Doar textul jurnalului.
3. `token32_v2` **nu** se folosește încă la generare: e cu ~10 % mai lent decât lista. Pragul: placa sub ~1,47 ms.

## Ce e nou pentru placă
Nimic: aceleași programe, aceleași opțiuni. Interfața neschimbată (doar un test nou la noi, T285).
