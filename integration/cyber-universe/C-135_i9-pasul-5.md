# Pentru sesiunea OS — C-135: `gpu lant --i9 c135`, oprită implicit (cyber-gpu: commit-ul sesiunii 108 sau mai nou)

Sesiunea Cyber-GPU 108, 17 noiembrie 2026. Context: `os-304` (`--i9 c133`) — încercarea B a trecut pe placă de `PROMOTE_CTX` și de obiectul compute;
**`GET_WORK_SUBMIT_TOKEN` a fost refuzat cu `0x40` `NV_ERR_INVALID_STATE`** (un ASSERT nou, `0x0185eb2e`); programarea grupului nu a mai fost trimisă.
Analiza și motivele: `docs/os-304_ANALIZA_I9.md`. **Totul e testat doar în SIMULARE; pe placă doar în fereastra cu Adrian, după DA.**

## 1. Fără nicio schimbare în OS: nimic nu se schimbă

Doar adăugări. `open_work_channel_cleared`, `open_work_channel_with` (`OS_302`, `C125`) și `open_work_channel_c133` trimit exact ce trimiteau
(testele `c135_off_changes_nothing`, `c133_off_changes_nothing`, `c125_off_sends_exactly_what_os302_sent`; constructorul vechi al parametrilor de canal e
octet cu octet același: `c135_channel_params_change_only_what_they_say`). Legarea la commit-ul nou fără alt cod = comportamentul din `os-304`.

## 2. Opțiunea nouă (propunere): `gpu lant --i9 c135`

Nou în cyber-gpu:
- `nvidia_ga104::gsp::user_space::{open_work_channel_c135, C135Report, C135Try, C135Variant, C135Stage, C135Token, journal_rows_c135, C135_BUFFERS,
  C135_GLOBAL_VAS, C135_F_CHID}`;
- `nvidia_ga104::gsp::rmapi::{ErrorNotifier, work_channel_params_with, internal_flags, channel_flags, fixed_chid_flags, chid_of_flags,
  control_reply_status, GSP_RESERVED_CHIDS}`;
- pentru teste: `gpu_sim::fake_rm::{TokenRule, FakeRm::token_rule, ChannelFacts, FakeRm::channel_facts, FakeRm::fixed_chid, GSP_RESERVED_CHIDS,
  SIM_CHANNELS, status::NO_FREE_FIFOS}`. În model, canalul golden primește acum ChID-ul din `flags` (3), nu 0; canalele fără ChID ales primesc aceleași
  id-uri ca înainte (1, 2, …).

`open_work_channel_c135` **înlocuiește perechea** `open_compute_group` + `open_work_channel_with` (ca `c133`: își face singură grupul — unul nou pentru
fiecare încercare) și întoarce `(ComputeGroup, WorkChannel)`. Aceeași semnătură ca `open_work_channel_c133`, cu `C135Report`. În `I9Choice` e o a patra
valoare. Schița pentru `kernel/src/drivers/gpu_golden.rs`, `channel_step`:

```rust
use nvidia_ga104::gsp::user_space::{self, C135Report};

let mut rep135 = C135Report::default();
// … în match-ul pe I9Choice:
I9Choice::C135 => user_space::open_work_channel_c135(&mut t, &mut *mem, &u, gi, &mut rep135)?.1,
// … după ce `t` s-a terminat (jurnalul e liber), pe toate drumurile, ca la c133:
I9Choice::C135 => rep135.journal(journal),
```

Ce trimite (niciun tip de mesaj nou, niciun registru nou — aceleași alocări și controale RM, cu alți parametri la alocarea canalului):

- **Încercarea N:** TSG + context share + canal **fără notificator de erori** (`hObjectError` = 0, `internalFlags` = `0x14`, `errorNotifierMem` zero; USERD
  pus pe zero) → **un** `PROMOTE_CTX` în forma B (MAIN + PATCH cu PA + VA, FECS_EVENT + PRIV_ACCESS_MAP cu VA-ul din golden) → obiectul compute →
  `GET_WORK_SUBMIT_TOKEN` → programarea grupului. Nu mai există încercarea A.
- **Încercarea F** — doar dacă GSP-RM refuză jetonul lui N: TSG + context share + canal **noi**; canalul are în plus `flags` = `0x00202000` (ChID 16 ales
  de noi: pagina 2, index 0) → `PROMOTE_CTX` (forma B) → compute → **fără** `GET_WORK_SUBMIT_TOKEN` (jetonul = runlist `<< 16` | 16, calculat) →
  programarea grupului.
  Grupul lui N rămâne neprogramat (nu rulează nimic).
- Orice alt refuz (grup, context share, canal, promovare, compute, programare) sau o eroare de transport oprește imediat.

Canalul întors are `notifier_pa = 0` (nu mai alocă pagina de notificator) și `ctx_bufs` = MAIN + PATCH (fără PM); restul câmpurilor ca până acum, deci
I10–I14 îl folosesc neschimbat. Memorie: o încercare = canalul de la `c133` B minus pagina de notificator; dacă se ajunge la F, încă un inel + instanță +
buffer de metode + MAIN + PATCH.

Rândurile din jurnal încep toate cu `i9 c135 `: `incercari facute`; pentru fiecare încercare `incercarea (1 N…, 2 F…)`, handle-urile, cele 4 intrări
(`buf id / va hi / va lo / marime / b0 PA+init b1 nonmapped`), `jeton (0 neajuns, 1 de la RM, 2 refuzat de RM, 3 calculat de noi)`, `id de canal`, la un
jeton refuzat `jeton refuzat: stare RM` + `jeton refuzat: campul din raspuns (doar notat)` (sau nota `fara raspuns intreg…`), `ultimul pas acceptat
(1 grup … 5 jeton, 6 programat)`, la refuz `refuz RM la pasul urmator (stare)` + `text i9 c135 refuz = "NV_ERR_…"`; la sfârșit `varianta reusita
(0 niciuna, 1 N, 2 F)`. Raportul are cel mult 61 de rânduri (N refuzat la jeton + F refuzat la programare); 29 dacă N trece.

Pentru scenariul din QEMU: `FakeRm` cu `promote_rule = PromoteRule::GlobalsNeeded` și `token_rule = TokenRule::NotifierNotVram` dă pentru `--i9 c133`
exact rezultatul de pe placă de la `os-304` (A `0x57`, B `0x40` la jeton) și lasă `--i9 c135` să treacă la N; cu `TokenRule::RefusedNoId` trece la F.
Regulile joacă **ipoteze**, nu placa.

## 3. Pe placă — doar în fereastra cu Adrian

O pornire la rece → `gpu lant --i9 c135` → DA → `trimite`. Dacă I9 trece, lanțul merge mai departe singur (forma GR0, apoi I10 add64 — primul calcul).
Ce înseamnă fiecare rezultat: `docs/os-304_ANALIZA_I9.md` §7. De urmărit în plus în jurnal: dacă perechea de ASSERT `0x01ab9bda` / `0x01ab9c56` mai
apare la alocarea canalului (ipoteza spune că dispare).
