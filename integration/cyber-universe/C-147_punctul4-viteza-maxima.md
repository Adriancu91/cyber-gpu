# C-147 punctul 4 → viteza maximă gata de legat (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 123 (20 nov 2026). Cerută de centrul de comandă (C-147, nota din 20 nov 09:20): `os-313` a arătat
`ks` = **P8** și `kc` = 9 165 MB/s (3 % din 608 GB/s) → condiția (a) a lui Adrian e îndeplinită. Detalii și surse:
`docs/C-147_PUNCTUL4.md`. Totul e **adăugat** (nimic redenumit sau șters), testat **doar în SIMULARE** (T139, `tests/sim_c147_boost.rs`).

## Ce legi (un Release separat, oprit implicit, pornit DOAR în modul ciclu)
```rust
use nvidia_ga104::gsp::perf::{self as gp, BoostPlan};
// planul: oprit implicit; îl pornești doar în modul ciclu (condiția lui Adrian), nimic altceva nu se schimbă
let plan = BoostPlan { enabled: mod_ciclu, ..BoostPlan::default() };   // forma NVIDIA, 600 s
// transportul: același drum GSP_RM_CONTROL ca în `pstate_step` (gpu_golden.rs)
let mut rm = |sent: &[u8]| t.call(nvidia_ga104::gsp::rpc::function::GSP_RM_CONTROL, sent);

// 1) O SINGURĂ DATĂ, imediat după ce canalul de lucru e deschis (după I9), înainte de i14-simbol / kronos --gpu:
let start = gp::boost_start(&plan, u.client, u.subdevice, &mut rm, journal)?;
//    rânduri: ks inainte → ks cerere (0 sau 8000xxxx) → ks dupa cerere (doar dacă e acceptată) → ks … verdict

// 2) lanțul ca azi: i14-simbol, kronos --gpu (#513, kg/kv/kt/kp), kc (copierea VRAM → VRAM) — NESCHIMBATE

// 3) La sfârșit, ÎN LOCUL lui `pstate_step` (citirea `ks la final` e inclusă):
let end = gp::boost_finish(&start, &plan, u.client, u.subdevice, &mut rm, journal)?;
//    dacă cererea a fost acceptată: ks inainte de stergere → ks stergerea → apoi mereu ks la final
```
- **Pe căile de eroare** (după `boost_start` acceptat, dacă lanțul pică mai târziu dar canalul încă răspunde): cheamă tot
  `boost_finish`, ca să ștergi cererea. Dacă nici canalul nu mai răspunde, cererea se stinge singură după 600 s (durată finită).
- Un **refuz** al plăcii (la citire sau la cerere) e un rând (`8000xxxx`), nu o eroare: lanțul merge mai departe ca azi și nu se
  trimite nicio ștergere. Doar o eroare de transport sau un răspuns pentru altă comandă întoarce `Err` (ca la `pstate_step`).
- Cu planul oprit (`enabled = false`) nu se trimite nicio cerere: doar citirile `ks inainte` + `ks la final` și nota „oprita”.

## Ce intră în raportul ciclului
- `gp::parse_ks(text)`: `before`, `boost`, `after`, `verdict`, `before_clear`, `clear`, `end` (valorile: 0..15 = P0..P15;
  `8000xxxx` = refuz). Verdictul: **0** oprită, **1** urcată (P mai mic după cerere), **2** acceptată fără efect, **3** refuzată.
- Comparația cu `os-313` (fără cerere): `kronos_perf::PerfGain::compare(&parse_journal(jurnal_os313), &parse_journal(jurnal_nou))`
  → timpul plăcii pe simbol (`kp toate`), `kc` MB/s și `ln1` (un program mic, legat de ceasuri): înainte, după, după ÷ înainte (‰).
  Cifrele `os-313` de comparat: `kp toate` ≈ 22 968 µs pe simbol, `kc` 9 165 MB/s, `ln1` 2 260 µs pe simbol.
- Proba rămâne cea de azi: **aceleași 16 simboluri ca procesorul**. Viteza nu schimbă nimic din ce calculează placa.

## Ce NU face
- Nu citește temperatura (fără cale documentată în 570.144; tabloul de bord RUSD e C-151 punctul 1, separat, cu „da”-ul lui).
- Nu trimite forma tinygrad (`PERF_BOOST` public) — doar dacă forma NVIDIA e refuzată **și** centrul de comandă cere.
- Nu e legată de proba de reset (C-152 punctul 1): Release separat, cum a cerut centrul de comandă.

## Ce așteptăm pe placă (NETESTAT)
Dacă e acceptată: `ks dupa cerere` mai mic decât 8 (ideal 0), `kc` și `kp` mult mai bune decât în `os-313`. Placa poate accepta
fără să urce imediat („alți factori pot limita starea”, `ctrl2080perf.h`): atunci verdict 2, iar `ks inainte de stergere` spune
dacă a urcat între timp. Un refuz (verdict 3) înseamnă că mergem pe forma B doar cu acordul centrului de comandă.
