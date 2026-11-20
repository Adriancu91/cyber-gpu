# C-151 punctul 1 → tabloul de bord al plăcii (RUSD), gata de legat (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 124 (20 nov 2026). Pentru P6-105 (`gpu stare`). Detalii și surse: `docs/C-151_TABLOU_RUSD.md`.
Totul e **adăugat** (nimic redenumit sau șters), testat **doar în SIMULARE** (T140, `tests/sim_c151_rusd.rs`).
Pe placă: aprobat de Adrian (C-151 punctul 1) **doar la sfârșitul lanțului, după ce rezultatele sunt pe stick**.
**Nu-l pune în același Release cu viteza maximă (P6-106)** dacă centrul de comandă nu cere altfel — e o comandă nouă către GSP-RM.

## Ce îți trebuie de la OS
- **o pagină de 4 KiB de memorie de sistem**, contiguă fizic, **umplută cu zero**, aliniată la 4 KiB, a cărei adresă o vede placa
  (la noi fără IOMMU = adresa fizică). Nu e în VRAM. Se citește cu citiri `volatile` (placa scrie în ea oricând).
- pagina rămâne **rezervată** până când `stop.page_free(&start)` e adevărat; dacă nu (oprirea refuzată), n-o refolosi până la repornire.

## Ce legi (oprit implicit)
```rust
use nvidia_ga104::gsp::rusd::{self as kd, RusdMem, RusdPlan};
let plan = RusdPlan { enabled: mod_ciclu && cheie_tablou, ..RusdPlan::default() };   // 500 ms, ceasuri+perf+putere+temperatura
let mut rm = |sent: &[u8]| t.call(nvidia_ga104::gsp::rpc::function::GSP_RM_CONTROL, sent);   // același drum ca pstate_step

struct Pagina(*const u8);           // adresa virtuală a paginii (mapată necache-uit sau cache-uit coerent, x86)
impl RusdMem for Pagina {
    fn read_u64(&mut self, o: usize) -> u64 {
        // SAFETY: o + 8 <= 4096, pagina e a noastră și mapată cât timp trăiește Pagina; o e multiplu de 8.
        let v = unsafe { core::ptr::read_volatile(self.0.add(o) as *const u64) };
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
        v
    }
    fn read_bytes(&mut self, o: usize, out: &mut [u8]) {
        for (i, b) in out.iter_mut().enumerate() {
            // SAFETY: o + out.len() <= 464 <= 4096, aceeași pagină.
            *b = unsafe { core::ptr::read_volatile(self.0.add(o + i)) };
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
    }
}

// LA SFÂRȘITUL LANȚULUI, după ce rezultatele sunt scrise pe stick (după boost_finish / ks la final):
let (start, tablou, stop) = kd::dashboard_at_end(
    &plan, adresa_fizica_pagina, u.client, u.subdevice, &mut rm, &mut Pagina(ptr), &mut |ms| sleep_ms(ms), journal)?;
//   rânduri: kd pornirea zonei comune → kd intervalul → kd cererea datelor → (1,5 s) → kd temperatura / ceasuri / stare /
//   incarcare / motive / putere / limita → kd sectiuni citite / ocupate → kd oprirea citirilor → kd oprirea zonei comune
if stop.page_free(&start) { /* pagina se poate elibera */ }
```
- Un **refuz** al plăcii e un rând `8000xxxx`, nu o eroare; după un refuz la pornire nu se mai trimite nimic (doar oprirea, dacă placa
  luase pagina). Doar o eroare de transport sau un răspuns pentru altă comandă întoarce `Err`.
- Cu planul oprit: o notă `kd tabloul de bord: oprit, nimic trimis`, nicio comandă, nicio așteptare.

## Ce intră în raport / în `gpu stare`
`kd::parse_kd(text)`: `temp_x256` (grade C × 256; `>> 8` = grade întregi), `clk_graphics_mhz`, `clk_memory_mhz`, `pstate` (0 = P0),
`load_gpu`, `load_mem` (brute, unitățile neconfirmate), `reasons` (0x1 placă în repaus; 0x4 limită de putere; **0x20/0x40 încetinire
termică**), `avg_power_mw`, `power_limit_mw`, `sections` (0x7f = toate citite), `busy`, plus răspunsurile comenzilor.
Pentru un rând pe ecran: `Dashboard::temp_c()`, `pstate()`, `thermal_slowdown()`.

## Ce așteptăm pe placă (NETESTAT)
Cel mai probabil: cele 3 comenzi acceptate și `sections` = 0x7f, temperatura 30–50 °C în repaus, P8 fără viteza maximă.
Posibil: `INIT` refuzat (noi folosim clientul nostru, NVIDIA folosește clientul intern al GSP-RM) → un rând, nimic altceva; atunci
Cyber-GPU pregătește varianta cu mânerele interne (cu „da”-ul lui Adrian).
