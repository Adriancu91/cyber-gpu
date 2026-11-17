# Pentru sesiunea OS — C-133: `gpu lant --i9 c133`, oprită implicit (cyber-gpu: commit-ul sesiunii 107 sau mai nou)

Sesiunea Cyber-GPU 107, 17 noiembrie 2026. Context: `os-303` (`--i9 c125`) — I9 tot `0x57` la `PROMOTE_CTX` virtual; V3 refuzat cu `0x31`.
Analiza și motivele: `docs/os-303_ANALIZA_I9.md`. **Totul e testat doar în SIMULARE; pe placă doar în fereastra cu Adrian, după DA.**

## 1. Fără nicio schimbare în OS: nimic nu se schimbă

Doar adăugări. `open_work_channel_cleared` și `open_work_channel_with` (`I9Options::OS_302`, `C125`) trimit exact ce trimiteau
(testele `c125_off_sends_exactly_what_os302_sent`, `c133_off_changes_nothing`). Legarea la commit-ul nou fără alt cod = comportamentul din `os-303`.

## 2. Opțiunea nouă (propunere): `gpu lant --i9 c133`

Nou în cyber-gpu: `nvidia_ga104::gsp::user_space::{open_work_channel_c133, C133Report, C133Try, C133Variant, C133Stage, journal_rows_c133,
C133_A_BUFFERS, C133_B_BUFFERS, C133_GLOBAL_VAS}`. Pentru teste: `gpu_sim::fake_rm::{PromoteRule, FakeRm::promote_rule}` (implicit `Lenient` = ca înainte).

`open_work_channel_c133` **înlocuiește perechea** `open_compute_group` + `open_work_channel_with` (își face singură grupul de canale — unul nou pentru
fiecare încercare) și întoarce `(ComputeGroup, WorkChannel)`. Nu cere `PrivilegedMemory`. `I9Options` rămâne neschimbat (nu are câmp nou), deci
`c133` e o a treia valoare a opțiunii din linia de comandă, tratată separat. Schița pentru `kernel/src/drivers/gpu_golden.rs`, `channel_step`:

```rust
use nvidia_ga104::gsp::user_space::{self, C133Report};

let mut rep = C133Report::default();
let r = {
    let mut t = OsRm { /* ca acum */ };
    (|| {
        let u = user_space::open_user_space(&mut t, gi.next_handle(), mem.root())?;
        let (_cg, w) = user_space::open_work_channel_c133(&mut t, &mut *mem, &u, gi, &mut rep)?;
        user_space::query_gr_info(&mut t, &u)?.check_sm86()?;
        Ok::<_, GpuError>((u, w))
    })()
}
.and_then(|x| flush(vbar, journal, permit, zone, pending, vram_words).map(|_| x));
// Raportul ajunge pe stick ȘI când I9 pică: ce s-a trimis în fiecare încercare și ce a răspuns GSP-RM.
rep.journal(journal);
```

Ce trimite (niciun tip de mesaj nou, niciun registru nou — aceleași alocări și controale RM ca până acum, în altă ordine):

- **Încercarea A:** TSG + context share + canal (ca acum, USERD pus pe zero) → **un singur** `PROMOTE_CTX` cu PA + VA pentru MAIN, PM, PATCH →
  obiectul compute → token → programarea grupului.
- **Încercarea B** — doar dacă GSP-RM refuză `PROMOTE_CTX` al lui A: TSG + context share + canal **noi** → un `PROMOTE_CTX` cu MAIN și PATCH (PA + VA)
  plus FECS_EVENT și PRIV_ACCESS_MAP doar cu VA-ul din imaginea golden → compute → token → programare. Grupul lui A rămâne neprogramat (nu rulează nimic).
- Orice alt refuz (grup, canal, compute, token, programare) sau o eroare de transport oprește imediat, fără B.

Memorie în plus față de `os-303`: o încercare = cât canalul de acum; dacă se ajunge la B, încă un inel + instanță + buffer de metode + MAIN + PATCH
(câțiva MiB de VRAM, alocate din același `GpuMemory`). Rândurile din jurnal încep toate cu `i9 c133 `: `incercari facute`, pentru fiecare încercare
`incercarea (1 A…, 2 B…)`, handle-urile, fiecare intrare (`buf id / va hi / va lo / marime / b0 PA+init b1 nonmapped`), `ultimul pas acceptat`,
la refuz `refuz RM … (stare)` + `text i9 c133 refuz = "NV_ERR_…"`, la sfârșit `varianta reusita (0 niciuna, 1 A, 2 B)`.

Atenție la capacitatea jurnalului: raportul întreg are cel mult ~55 de rânduri (cazul „A și B refuzate”).

## 3. Pe placă — doar în fereastra cu Adrian

O pornire la rece → `gpu lant --i9 c133` → DA → `trimite`. Ce înseamnă fiecare rezultat: `docs/os-303_ANALIZA_I9.md` §6.
