# C-157 p.2 — P2zzs: după `token32_v2c` pe placă (os-495, ciclul 27 nov 18:06)

Sesiunea Cyber-GPU 276, 27 nov 2026, în cloud, fără placă. Comparația întreagă: `docs/C-157_TOKEN32_V2C_PLACA.md` (CALCUL T291).

## Ce a ieșit (MĂSURAT de OS)
- `token32_v2c` dă aceiași biți (16 simboluri, amprenta `0x6c8501f0`, contorul corect); ceasul costă doar +0,6 %.
- Cele ~0,68 ms în plus față de lista obișnuită stau în: **LayerNorm făcut de un singur grup** (+0,36 ms), **atenția pe 12 grupuri**
  din 768 (+0,15 ms) și **bariera** (0,17 ms, 2 µs fiecare). Înmulțirile costă cât în listă.

## Ce propunem OS-ului (nimic obligatoriu)
1. **Rândurile viitoare:** fără `--token32-v2c` (răspunsul l-am primit; e doar program de măsurat). `--token32-v2` rămâne proba de
   bază când vine următorul program `token32`.
2. Nimic de legat acum. Următorul program (LayerNorm calculat de fiecare grup al înmulțirii următoare) vine cu o PREDARE nouă.

## Ce e nou pentru placă
Nimic: aceleași programe, aceleași opțiuni. Interfața neschimbată (doar un test nou la noi, T291).
