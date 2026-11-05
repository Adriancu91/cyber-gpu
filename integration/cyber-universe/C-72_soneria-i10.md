# C-72 / întrebarea 3 a OS — soneria canalului de lucru la I10 (sesiunea Cyber-GPU 52, 5 nov 2026)

**Răspuns la întrebarea 3 din `stare/cyber-universe.md`.** Nimic testat pe RTX 3070 Ti; nimic nou de scris pe placă.

1. **Adresa `0xBB0090` nu mai e doar DEDUSĂ:** e în antetul MIT 570.144 (`tu102/dev_vm.h`): `NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090`
   la baza `NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET = 0xB80000`. RM 570.144 o scrie exact așa pe GA104 când RM rulează pe procesor
   (`kfifoUpdateUsermodeDoorbell_GA100` → `_TU102` → `GPU_VREG_WR32(NV_VIRTUAL_FUNCTION_DOORBELL, token)`).
   În cyber-gpu (≥ commit-ul sesiunii 52): constanta nouă `nvidia_ga104::push::DOORBELL` (aceeași valoare ca `DOORBELL_INFERRED`, verificat
   la compilare și în testul T90). Stare: **DOCUMENTAT, încă NEOBSERVAT pe placă.**
2. **Lista permisă:** soneria intră pe lista scrierilor permise **doar în pasul I10 al `gpu lant`** (aprobarea lui Adrian „da i7-i10”, C-72),
   cu valoarea = tokenul primit de la GSP-RM (`rmapi::doorbell_token`), după ce I8 + golden + I9 au TRECUT în același lanț. Nu în `gpu tur`,
   nu în bucla automată, nu în afara lanțului.
3. **Citirile VRAM ale I10** (semaforul, rezultatul `add64`): prin `VramWindow::read_u32`; **invalidarea TLB** prin `VramWindow::invalidate_tlb`
   o dată, după ce tabelele de pagini sunt scrise și înainte de prima sonerie; `VramWindow::restore()` și pe drumul de eroare (nota C-71).
4. Interfața publică: doar o adăugare (`push::DOORBELL`); `DOORBELL_INFERRED`, `chain_log::WORK_DOORBELL` neschimbate.
