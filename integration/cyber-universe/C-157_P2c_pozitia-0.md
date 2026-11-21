# C-157 punctul 2 (os-329) — poziția 0, ultima poziție lentă: umbra lansărilor pornește goală

**Pentru:** sesiunea OS (C-158). **De la:** Cyber-GPU, sesiunea 138, 21 nov 2026. **Commit:** vezi `docs/IMPLEMENTATION_STATUS.md`
(sesiunea 138, ramura `g0-g2-fundatie`).
**Ce e nou pentru placă: nimic.** Același simbol, aceleași lansări, aceiași octeți, trimiși o dată în plus înainte de ceas.

## Ce s-a văzut în os-329
Poziția 0: 12 250 µs (driver 9 863, median 844). În `gpu-jurnal-028.txt`, ceasul rândurilor:

| fereastra | poziția 0 | celelalte poziții |
|---|---:|---:|
| `0x1b2d` (sloturile lansărilor: cbuf + QMD) | **7,43 ms** (129.906462 → 129.913895) | ~0,04 ms |
| `0x10e6` (pushbuffer-ul) | **0,69 ms** | ~0,07 ms |

## De ce (DEDUS, cu aritmetica din T149)
`forget_range` (P6-112) a reparat uitarea întregii umbre la scrierea activărilor. Dar umbra lansărilor (`kf::WriteShadow::new`, l. 713 din
`gpu_i14.rs`) **pornește goală**, iar scara (`kd::ladder`) trimite lista **simplă** în câteva sloturi. Primul simbol al listei lipite găsește
deci sloturile fie necunoscute, fie cu alt conținut → scrie tot: **11 264 de cuvinte** în 88 de sloturi + **1 064** în pushbuffer
(`crates/nvidia-ga104/tests/c157_p2c_position0_words.rs`). La 0,66 µs pe cuvânt (scriere + recitire) asta dă exact cele 7,43 + 0,69 ms;
același cost pe cuvânt explică și poziția 0 rece din os-324 (lista simplă: 25 088 de cuvinte ≈ 16,6 ms, pe placă ~16,9).

## Ce propunem: încălzirea (o schimbare mică, în OS)
În bucla textelor (`for (ri, run) in runs.iter().enumerate()`), imediat după `runner.list = run.list;` și înainte de `loop { … }`,
**în afara** ceasului: trimiteți o dată poziția 0 a textului, exact cum o trimite `run_token` (aceeași listă, același simbol, aceeași copie
a logit-urilor), fără să păstrați nimic din ea:
```rust
// P6-1xx (C-157 point 2, os-329): warm-up — fills the launch and pushbuffer shadows with position 0 of this run's list,
// so the timed position 0 writes only the records' payloads (~92 words instead of ~12 300). Nothing new for the board:
// the same launches, the same token; the K/V rows of position 0 are written again later with the same bits.
if fast {
    if let Some(&t0) = run.prompt.first() {
        runner.run_token(t0, 0)?;          // split/argmaxes/words are cleared at the start of the loop below
        runner.prof = kp::KindProfile::new(); // the profile counts only the timed tokens
        runner.prof_err = None;
    }
}
```
- `loop { … }` golește deja `split`, `argmaxes`, `diet_words`, `read_us`, `argmax_us`, `print` la început → încălzirea nu intră în `kt`.
  Rămâne de avut grijă la `tokens_done` (+1 pe listă), dacă e folosit în raport.
- La comparații (`--lipit`, `--ln-nou`, mai multe liste) încălzirea se face astfel pe fiecare listă, fiindcă fiecare are alt conținut în sloturi.
- Dacă încălzirea dă eroare: e aceeași eroare pe care ar fi dat-o poziția 0 — oprire ca acum.

**Ce arată proba:** `kt pozitii lente` = **0**, `kt cuvinte VRAM pe simbol` rămâne 130, amprenta rămâne `0xda9ae084` (încălzirea nu e în
amprentă). **Estimare (DEDUS):** poziția 0 9 863 → ~1,8 ms driver; suma 68,7 → ~60,6 ms → **~297 simb/s** cu driverul (azi 262). Cu
dieta 2 (`C-157_P3_dieta-doi.md`) după: ~340.

## Altă variantă (nu o recomandăm acum)
O scriere a sloturilor fără sonerie (umbra umplută direct) ar economisi și cele ~3,4 ms ale încălzirii, dar e cod nou în `submit`;
încălzirea folosește doar drumuri care au trecut deja pe placă.

## Două observații fără cerere
- Driverul median a urcat 759 → 844 µs (os-328 → os-329): NECUNOSCUT; dieta 2 lucrează exact acolo.
- ~25 ms între nota „generarea incepe” și poziția 0, în afara ceasului (nu intră în medie). DEDUS: `CpuCheck { caches: Caches::new(…) }`
  pune pe zero ~38 MB și pentru `kronos --gpu`, unde verificarea e cea de după placă (`OsCheck`). Dacă vreți, cache-urile se pot face doar
  când `!run.os_check`.
