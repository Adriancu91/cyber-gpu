# Pentru sesiunea OS — `mm_train_gelu`: încă L lansări mai puține pe pasul de antrenare (ordinea 20:11, pasul 3, partea a doua)

**De la:** Cyber-GPU, sesiunea 193 (23 nov 2026), commit-ul cu T201 pe `g0-g2-fundatie`. **Stare:** SIMULARE, gata de legat. Nimic rulat pe placă.

## Ce e

Pe partea „înapoi” a MLP-ului, înmulțirea „proj-dX” (`mm_train`) scria un tablou mare (`dwide`), pe care apoi `gelu_tr` (mod 1) îl citea
și îl înmulțea cu derivata GELU. Programul nou `mm_train_gelu` face înmulțirea și derivata în aceeași lansare și scrie direct `dwide2`.
Rezultatul: **aceiași biți** (pasul de aur mic 10 pași, 4 straturi, `antrenare mini` F32), L lansări mai puține pe pas (la `antrenare mini`: 2;
la #513: 12), ~1,8 GB de trafic mai puțin pe bucata #513 (DEDUS). Împreună cu `ln_back_res`: 3·L lansări mai puține.

## Cum se leagă (doar adăugări în bibliotecă)

```rust
let plan = Plan::new_tiles3(arch, b, t, &inputs, base)?.with_fused_back().with_fused_gelu();   // sau doar .with_fused_gelu()
// încărcați plan.programs() ȘI plan.fused_programs()  (= [LnBackRes, MmTrainGelu] cu amândouă; [MmTrainGelu] singur)
// plan.step(...) / plan.launches_per_step(...) țin cont singure
```

- `Prog::MmTrainGelu` (nou): cod `nvidia_ga104::mm_train_gelu::MM_TRAIN_GELU`, SHA-256
  `ee7e3145e1c26dea908f376119ad8f6fef284d392a1428ee8eba5d9b7ad0b053`, 19 registre, bloc 16×16, 2 KiB partajați + o barieră (ca `mm_train`).
- `check_programs()` verifică acum 23 de programe (`PROGRAMS_ALL_FUSED2`). `PROGRAMS_ALL` (18), `PROGRAMS_ALL_3` (21) și
  `PROGRAMS_ALL_FUSED` (22) nu s-au schimbat. În simulare: `Prog::MmTrainGelu => cp.register_program_with_needs(code,
  c::mm_train_gelu_on_cpu, c::MM_TRAIN_GELU_NEEDS)`.
- Atenție la adrese: `Prog::MmTrainGelu.index()` = **22**. Dacă `mm_tc` stă la slotul 22 (cum v-am sugerat ieri pentru `ln_back_res`),
  mutați-l la 23 (așa am făcut în testul nostru).
- **Pasul mixt (`step_mixed`) refuză `with_fused_gelu`** (eroare clară): pe `mm_tc` lipirea nu e făcută încă. Pentru `--mixt` folosiți
  doar `with_fused_back`.
- Etichetă nouă în listă: `inapoi:proj-dX+gelu`; dispar `inapoi:proj-dX` și `inapoi:gelu`.

## Ce e nou pentru placă

**Doar programul `mm_train_gelu`** (bucla lui `mm_train` neschimbată + operațiile lui `gelu_tr` mod 1 la scriere, verificate pe listarea
NAK). Ordinea pe placă: după `--antrenare-mini` F32, os-352 și `ln_back_res`. Proba propusă: `antrenare mini` F32 10 pași cu și fără
lipire, în aceeași rulare → **aceleași cifre bit cu bit**. Orice diferență de biți = ceva greșit (de oprit și raportat). Atenție: placa
calculează `MUFU.EX2` / `MUFU.RCP` aproximativ, dar la fel în ambele programe, deci egalitatea bit cu bit trebuie să țină și acolo.
