# Pentru sesiunea OS — C-108: `gpu variante` și `gpu lant --varianta N` (cyber-gpu `20830f5` sau mai nou)

Sesiunea Cyber-GPU 88, 11 noiembrie 2026. Nimic testat pe placă. Ce e nou în `cyber-gpu`: modulul **`nvidia_ga104::gsp::variants`**
(doar adăugări; funcțiile vechi `sysinfo::encode`, `registry::boot_entries`, `wpr_meta::encode`, `libos::encode_gsp_args` rămân neschimbate).
Analiza: `docs/os-283_ANALIZA_ORDINE_RPC.md` — pe scurt: **os-283 trimitea `SetSystemInfo` + `SetRegistry` la I8, după ce I7 pornise GSP-RM;
toate cele patru drivere publice le pun în coadă ÎNAINTE de Booter**, iar NVIDIA scrie că GSP-RM „may skip these RPCs if not sent in time”.
Varianta 1 e exact această schimbare.

## Ce dă biblioteca

```rust
use nvidia_ga104::gsp::variants::{self, Settings, Variant};

variants::list() -> &'static [Variant]           // 10 variante: nr (1..=10), name, changes (ro), source, settings
variants::get(nr: u8) -> Option<&'static Variant> // None pentru 0 sau necunoscut
variants::settings(nr: Option<u8>) -> GpuResult<Settings> // None = variants::OS_283 (comportamentul de acum)

impl Settings {
    fn encode_sysinfo(&self, &SystemInfoInput) -> GpuResult<[u8; 928]>       // in loc de sysinfo::encode
    fn encode_registry(&self) -> GpuResult<Vec<u8>>                          // in loc de registry::encode(&boot_entries())
    fn encode_wpr_meta(&self, &FbLayout, &SysmemInputs) -> GpuResult<[u8; 256]> // in loc de wpr_meta::encode
    fn encode_gsp_args(&self, &QueueLayout, u64) -> GpuResult<[u8; 72]>      // in loc de libos::encode_gsp_args
    fn queue_init_rpcs(&self, &mut CpuQueues, &mut dyn SharedMem, &SystemInfoInput) -> GpuResult<[u32; 2]>
    fn init_rpcs_ready_for_i8(&self, &CpuQueues) -> GpuResult<bool>          // true = deja in coada (I7 le-a pus); false = I8 le pune; Err = ordinea gresita
    fn journal(&self, nr: Option<u8>, &mut Journal)                          // randurile "valoare varianta I8 ..." / "text varianta I8: numele"
    pub init_rpcs_before_booter: bool                                        // campul pe care OS-ul il citeste pentru ordine
}
```

## Schimbările în OS (propunere; codul OS e al sesiunii OS)

1. **`gpu variante`**: pentru fiecare `v` din `variants::list()`: `"{v.nr}. {v.name} — {v.changes} (sursa: {v.source})"`. Fără placă, fără scrieri.
2. **`gpu lant --varianta N`** (și `--sim`): `let s = variants::settings(nr)?;` — număr greșit → mesaj + nimic scris. Fără `--varianta` → `OS_283`.
   Înainte de „DA”: ecranul arată `varianta N: nume — ce schimbă`. Jurnalul: la începutul I7 (sau al lanțului) `s.journal(nr, journal)` — rândurile ajung
   în `gpu-jurnal-NNN.txt` ca `valoare varianta I8 (C-108): numarul = 0x0000000N` etc.; raportul poartă `N` + numele.
3. **`gpu_i7::prepare`** primește `&Settings`: `s.encode_wpr_meta(&fb, &inputs)` în loc de `wpr_meta::encode`, `s.encode_gsp_args(&ql, at(pl.queues))`
   în loc de `libos::encode_gsp_args`. Apoi, după `CpuQueues::init`, **dacă `s.init_rpcs_before_booter`**: `s.queue_init_rpcs(&mut queues, &mut shared, &sys)?`
   (RAM, fără permis; `sys` = `gpu_i8::system_info(d)?`, calculat înainte). Păstrați `prep.queues`/`prep.shared` așa cum sunt.
4. **`gpu_i7::booter`**: dacă `s.init_rpcs_before_booter`, **înainte** de `booter_load_dma_libos`: `falcon::ring_doorbell(&mut b)?` (aceeași fereastră
   GSP, același registru `NV_PGSP_QUEUE_HEAD(0)` ca la I8; ordinea RM: sonerie → reset GSP în RISC-V → cutii LibOS → Booter). `i6_log::check_i7_libos`
   (deci `verifica_jurnal_lant` și verificarea din OS) acceptă soneria ca prima scriere a I7.
5. **`gpu_i8::init`**: în loc de cele două `send_rpc`:
   ```rust
   if !s.init_rpcs_ready_for_i8(&prep.queues)? {
       prep.queues.send_rpc(&mut prep.shared, function::GSP_SET_SYSTEM_INFO, &s.encode_sysinfo(sys)?)?;
       prep.queues.send_rpc(&mut prep.shared, function::SET_REGISTRY, &s.encode_registry()?)?;
   }
   ```
   (sau `s.queue_init_rpcs(...)`), apoi soneria și așteptarea ca până acum. Rândul „I8: in coada: …” poate spune „(puse la I7)” când `ready == true`.
6. `SimPeer` din OS (QEMU): consumă cele două comenzi oricând, deci scenariile vechi trec; pentru varianta 1 verificați că `seen` are ambele funcții
   **înainte** de Booter (sau lăsați verificarea la Cyber-GPU: T122).
7. Bucla automată / hub-ul: `gpu lant` rămâne refuzat, cu sau fără variantă (C-105 e oprit).

## Fereastra de joi (procedura din C-108)

Fiecare variantă = o pornire rece: oprire → cablu 30 s → F8 stick de test → `gpu lant --varianta N` → `DA` → `trimite` → `N+1`. Începeți cu **1**.
Reușita: `valoare GSP_INIT_DONE: rpc_result … = 0x00000000` și lanțul trece la `i8e-golden`. Se oprește la prima variantă care trece de I8.

Ordinea și valorile: numai din surse publice (`docs/PROVENANCE.md`); nicio scriere într-un registru pe care os-283 nu-l scria deja.

## După jurnalele de joi (sesiunea Cyber-GPU 89, 11 nov 19:45) — pentru sesiunea Cyber-GPU, nimic de făcut în OS

Jurnalele variantelor (`trimite` → hub `rezultate/os-automat/<data>/gpu-jurnal-NNN.txt`) se compară toate deodată cu referința os-283:

```
cargo run --release -p nvidia-ga104 --example compara_variante -- 0x9c66000 <os-283: gpu-jurnal-009.txt> <jurnalele de joi>...
```

Pentru fiecare jurnal: varianta (numărul + numele, din rândurile `valoare varianta I8 (C-108): numarul` / `text varianta I8: numele` pe care OS-ul le scrie
la începutul jurnalului), commit-ul din antet, rezultatul I8 (`I8 TRECUT` doar cu `pas i8-gsp-init: ok`; altfel `rpc_result` + numele), NOCAT primite / adrese de
assert, setările pe scurt și **rândurile care diferă de referință** (fără prefixul `[ timp]`, antet, marcaje de timp NOCAT, `asteptat …` și rândurile variantei).
Concluzia: varianta care a trecut sau „nicio varianta nu a trecut de I8”. Codul de ieșire: 0 = una a trecut, 3 = niciuna, 2 = un jurnal încalcă o regulă a lanțului.
Nu înlocuiește `verifica_jurnal_lant` (verdictul pas cu pas e tipărit și aici, la final); nu spune nimic ce nu e în jurnale.
