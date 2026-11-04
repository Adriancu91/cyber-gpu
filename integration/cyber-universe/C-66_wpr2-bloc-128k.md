# C-66 — I6: capătul WPR2 = ultimul bloc de 128 KiB + citirea de după I6 fără permis (pentru sesiunea OS)

**Ce s-a întâmplat pe placă (4 nov, os-222, cyber-gpu `3286a3b`):** `gpu i6 --scrie` a dat „verificare esuata la 0x1fa828: asteptat 0x1fff00,
citit 0x1ffee0”. LO = `0x1ffe00` (exact începutul FRTS), HI = `0x1ffee0` = începutul ultimului bloc de 128 KiB al zonei de 1 MiB.
FWSEC-FRTS a reușit; regula noastră nu cunoștea această formă. Al doilea defect: „WPR2 citit dupa I6: nepermis de politica de siguranta”
— `GuardedBar0::new(..., None)` pe placa reală cere permis chiar și pentru citiri.

**Ce s-a schimbat în cyber-gpu** (doar adăugări, compatibil; commit-ul `9ac89b8` sau mai nou pe `g0-g2-fundatie`):
- `gsp::falcon::check_frts` (același nume și semnătură, folosit de `fwsec_frts_dma`): acceptă acum și HI = ultimul bloc de 128 KiB
  → pe aceleași valori ca pe 4 nov, `gpu i6` ar da TRECE.
- nou: `gsp::falcon::{compare_frts_full, check_frts_full, FrtsCheckFull, Wpr2EndForm, WPR2_BLOCK_128K}`; `Wpr2EndForm` are 3 forme
  (exclusiv / ultima pagină de 4 KiB / ultimul bloc de 128 KiB), e `#[non_exhaustive]` și are `label()` (text în română).
- nou: `gsp::falcon::read_frts_after_i6(mmio, journal, frts)` + `StatusReadBar0`: **doar citiri** (3 registre de stare: SCRATCH 0xe,
  WPR2 HI, WPR2 LO), fără permis, fără nicio funcție de scriere; fiecare citire intră în jurnal (deci și pe stick).
- neschimbate, pentru compatibilitate: `check_frts_detail`, `compare_frts`, `FrtsCheck`, `Wpr2End` (2 forme) — ele **refuză** în continuare
  forma de 128 KiB; nu le mai folosiți pentru afișare.
- `gpu_sim::fake_gsp::FakeGsp::wpr2_hi_back` (nou, implicit `Some(128 KiB)` = ca placa). Testele de simulare ale OS-ului care
  afișează linia „WPR2 citit dupa I6” prin `check_frts_detail` vor arăta acum eroare la HI (nu am găsit în OS teste care o verifică).

**Ce ar trebui să facă OS-ul** (C-66, partea cyber-universe):
1. Mută fixarea cyber-gpu pe `9ac89b8` (sau vârful ramurii după el).
2. În `kernel/src/drivers/gpu_i6.rs::wpr2_detail`: în loc de `GuardedBar0::new(rw, journal, None).and_then(check_frts_detail)`:
   ```rust
   match falcon::read_frts_after_i6(rw, journal, frts) {
       Ok(c) => format!("{}: WPR2 citit dupa I6: LO {:#010x} HI {:#010x} (capat: {}) = {:#x}..{:#x}",
                        p, c.lo_raw, c.hi_raw, c.form.label(), c.wpr2.start, c.wpr2.end),
       Err(e) => format!("{}: WPR2 citit dupa I6: {}", p, e),
   }
   ```
3. Release nou cu stick-ul de test. La fereastra următoare, cu Adrian: pornire rece (cablul scos 30 s), `gpu pregatire-i6`, `gpu i6 --scrie`
   o singură dată → așteptat „TRECE” și linia „capat: ultimul bloc de 128 KiB”. Apoi copiați jurnalul de pe stick în hub `rezultate/`;
   noi rulăm `verifica_jurnal_i6` pe el. Abia după asta, plan separat pentru I7 (Booter), cu comandă nouă.
