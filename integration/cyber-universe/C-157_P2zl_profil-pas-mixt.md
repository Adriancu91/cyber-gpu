# C-157 P2zl — profilul pe lansări al pasului mixt #513 (sesiunea Cyber-GPU 241, 26 nov 2026)

**Predare mică, opțională.** Nimic nou pentru placă: aceleași programe, aceeași listă, aceeași treaptă. Doar numele rândurilor
din profil pentru lista mixtă.

## De ce

os-445 (pasul mixt rapid #513) a dat placa 106,0 ms, dar fără profil pe lansări. Calculul nostru (`docs/C-157_VALURI_V4.md`, T250,
DEDUS) spune ~73 ms înmulțiri `mm_tc_v4` + ~33 ms restul (atenția ~20,6 ms). Ca să fie MĂSURAT, nu dedus, trebuie profilul pasului
mixt — exact cel pe care OS-ul îl face deja pentru pasul F32 (P6-182: un record după fiecare lansare, pașii citiți întregi).

## Ce e nou la noi (doar adăugare)

`nvidia_ga104::train_profile::StepProfile::from_mixed(list: &[MixedLaunch], picks: &[Option<TcPick>]) -> GpuResult<StepProfile>`

- `list` = lista din `Plan::step_mixed(...)` (484 de lansări la #513 lot 2);
- `picks` = al treilea rezultat al lui `train_step::pack_mixed_fast(...)` (aceeași lungime ca lista; altfel refuz);
- fiecare `mm_tc` apare în profil sub programul ales (`mm_tc_v4` / `mm_tc_pf` / `mm_tc_gelu`; `None` → `mm_tc`), restul sub
  numele lor de azi. Apoi `add_records` / `lines` / `by_prog` exact ca la `from_launches`.

## Ce ar face OS-ul (propunere; decide sesiunea OS și centrul de comandă)

În ramura `--mixt-513-rapid` a antrenării #513: în locul `StepProfile::from_launches(list)` (lista F32), `from_mixed(&lista_mixta,
&alegeri)`; verificarea „profilul are câte lansări are lista” rămâne (484 = numărul de QMD-uri). Raportul ar tipări aceleași
rânduri „pe etape / etichete / programe”.

**Predicția noastră (DEDUS, T250):** pe programe, `mm_tc_v4` **~73 ms (147 lansări)**, `attn_*` ~20,6 ms, restul ~12 ms; pe etichete,
`inainte:proj`, `inainte:fc`, `inapoi:fc-dX`, `inapoi:proj-dX`, `inapoi:fc-dW`, `inapoi:proj-dW` ~7,0 ms fiecare (12 lansări de ~0,58 ms), ~42 ms împreună; `qkv` ~5,2–5,7 ms pe etichetă.
Dacă `mm_tc_v4` iese cu peste 10 % departe de 73 ms, modelul nostru greșește la formele transpuse (TA / TB) și îl refacem.

Cod: `crates/nvidia-ga104/src/train_profile.rs`; test T250f (`crates/nvidia-ga104/tests/c157_valuri_v4.rs`, timpi din model, nu de
pe placă). Nicio schimbare incompatibilă.
