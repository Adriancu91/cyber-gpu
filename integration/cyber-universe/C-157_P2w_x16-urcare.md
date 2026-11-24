# C-157 punctul 2 (după testul x16, 24 nov 2026) — unde stă procesorul la urcarea greutăților

**Pentru sesiunea OS. Ce e nou pentru placă: nimic** (doar 4 citiri de ceas pe bucată în bucla de urcare și 4 rânduri `kv` în jurnal).

## Ce s-a văzut
După mutarea plăcii în SLOT2 (x16), proba RAM ↔ VRAM a ieșit exact cum am prezis (sus 12,4 GB/s, jos 13,5 GB/s; comparația în
`cyber-gpu/docs/C-157_P2_COMPARATII.md`). În același raport (`ciclu-auto-20261124-1356.txt` / `-1401.txt`), rândul vechi al urcării
greutăților la generare (#513, 110 786 560 octeți, 14 bucăți de 8 MiB):

| | placa (copierile) | procesorul (de la scrierea bucății în RAM până la ultima înregistrare) |
|---|---|---|
| x8 (13:00) | 18,0 ms (6,15 GB/s) | 271 ms (408 MB/s) |
| x16 (13:56 / 14:01) | 9,1 ms (12,2 GB/s) | 292 / 265 ms (379 / 418 MB/s) |

Placa s-a dublat, procesorul nu s-a mișcat: **~97 % din urcare e pe procesor, ~19 ms pe bucată**. Ce știm că face bucla
(`kernel/src/drivers/gpu_i14.rs`, „1. The blob, uploaded through the arena”): `ram_write` (8 MiB în arena fixată), 3 × `put_bytes`
(cbuf 256 B + QMD 256 B + înregistrări 32 B ≈ 136 de cuvinte, adunate), `run_records` (golirea scrierilor în VRAM, soneria, așteptarea).
Socoteala noastră (DEDUS): ~1–3 ms copia de 8 MiB în RAM cu cache + ~0,5 ms cuvintele + ~0,7 ms soneria și copia plăcii ≈ 2–4 ms.
**~15 ms pe bucată nu știm unde sunt.** O dată pe pornire (~0,3 s) nu deranjează; dar același drum (procesor → arena din RAM → placă)
l-ar folosi orice mutare de greutăți sau de stări AdamW prin RAM (`cyber-gpu/docs/C-157_RAM_OFFLOAD.md`), iar la ~400 MB/s ar fi de ~30
de ori prea încet. Deci merită întâi măsurat.

## Propunerea (o singură schimbare, opțională, decide sesiunea OS)
`nvidia_ga104::kronos_speed::UploadHostParts` (cyber-gpu, T213; doar adăugare — `Speed` și rândurile vechi rămân la fel):

```rust
let mut parts = ks::UploadHostParts::new();
for (i, chunk) in blob.bytes.chunks(piece as usize).enumerate() {
    let t0 = crate::time::since_boot_us();
    ram_write(&host, chunk);
    let t1 = crate::time::since_boot_us();
    // ... put_bytes(cb_up / q_up / rec_up), encode_sequence (ca azi) ...
    let t2 = crate::time::since_boot_us();
    match run_records(...)? {
        Ok(b) => {
            let t3 = crate::time::since_boot_us();
            // ca azi: speed.add_piece(...)
            if let (Some(a), Some(b1), Some(c), Some(d)) = (t0, t1, t2, t3) {
                let _ = parts.add_piece(chunk.len() as u64, a, b1, c, d);
            }
        }
        Err(last) => { /* ca azi */ }
    }
}
lines.push(format!("{}: kronos --gpu: {}", p, parts.line(source)));   // un rând în raport
parts.journal(journal);                                                // 4 rânduri `kv urcare pe parti ...`
```

Rândul din raport arată așa: `viteza (placa): urcarea pe procesor pe parti (14 bucati, 266000 us): copia in RAM … us (… promile, … MB/s),
pregatirea comenzilor … us (… promile), trimiterea + asteptarea … us (… promile); cea mai mare: …`.
`UploadHostParts::parse_journal` citește înapoi cele 4 rânduri pentru uneltele de pe gazdă.

## Urcarea pe procesor — predicțiile (scrise înainte de măsurare)
1. Totalul celor 3 părți = rândul vechi „procesor” (265–292 ms) ± 1 ms (aceleași margini, doar despărțite).
2. **Pregătirea comenzilor** < 5 ms în total (cuvintele doar se adună în memorie).
3. Fie **copia în RAM** e mare (≥ 200 ms, adică < ~0,6 GB/s) — atunci arena fixată se scrie fără cache (bitul PCD/PAT al hărții directe
   pentru zona ei sau un MTRR); **ar încetini la fel lotul prin RAM și orice mutare prin RAM**, iar remediul e o hartă cu cache (sau
   write-combining) doar pentru arenă —, fie **trimiterea + așteptarea** e mare (≥ 200 ms) — atunci costul e în golirea scrierilor sau în
   așteptare (pasul de 100 µs, jurnalul), iar copia în RAM iese ≥ 2 GB/s. Înclinăm spre a doua (DEDUS: `ram` folosește harta directă,
   care pentru RAM are de obicei cache), dar nu ne bazăm pe asta.
4. Placa rămâne ~9,1 ms (12,2 GB/s) — nu se schimbă nimic la ea.

Când vine rezultatul, comparația intră în `cyber-gpu/docs/C-157_P2_COMPARATII.md` și, după caz, o propunere de reparare (fără nimic nou pe placă).
