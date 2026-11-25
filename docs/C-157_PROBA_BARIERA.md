# C-157 — proba barierei pe toată placa (pentru generarea cu un singur program pe simbol)

Sesiunea 237, 25 nov 2026. Vine din `docs/C-157_PROGRAM_PERSISTENT.md` §5 punctul 3: la generare ~31 % din timpul plăcii e cost
fix de lansare; programul persistent `token.comp` (C-87) există, compilat, **nerulat**. Ce lipsește e dovada că **bariera pe toată
placa** (toate grupurile așteaptă unul după altul printr-un contor în memorie) merge pe RTX 3070 Ti: nu se blochează și datele scrise
de un grup înainte de barieră sunt văzute de celelalte după ea. Proba de aici face **doar** asta, fără nicio înmulțire, ca să fie mică
și ușor de oprit dacă se blochează.

## 1. Programul `grid_barrier.comp` (tools/nak)

- `G` grupuri de 32 de fire (un singur „warp” fiecare, ca `token.comp`), grila 1D, `G` ales de driver ≤ numărul de procesoare (48).
- Argumente (constantele de împingere, ca la celelalte programe NAK): `sync` la 0x28 (contorul, un u32 golit de driver înainte),
  `data` la 0x30 (2·G cuvinte, două jumătăți), `out` la 0x38 (G cuvinte), `rounds` la 0x40.
- Pentru fiecare rundă `k = 1..R`: firul 0 al grupului `g` scrie `k` în `data[(k mod 2)·G + g]`; bariera pe toată placa (aceeași
  funcție `grid_sync` ca în `token.comp`: `atomicAdd(sync, 1)` apoi așteaptă până la `k·G`); apoi citește
  `data[(k mod 2)·G + (g + 1) mod G]` (scris de **alt** grup) și numără câte runde n-au dat exact `k`.
  Două jumătăți ca vecinul să nu poată suprascrie valoarea înainte s-o citim (ar trebui să treacă de bariera următoare, care așteaptă
  și după noi).
- La final: `out[g] = numărul de runde greșite` (așteptat 0). Contorul rămâne `R·G`.

## 2. Ce trebuie să măsoare OS-ul (când va fi legat)

1. `R = 1` și `R = 1 001` cu același `G`, timpul plăcii (ceasul ei) pentru fiecare → costul unei bariere = (t₁₀₀₁ − t₁) / 1 000.
2. Verdictul: contorul = `R·G`, toate `out[g] = 0`, lansarea s-a terminat (cu limită de timp; la depășire: recuperarea C-87 pasul 3).
3. `G` = 48 (o dată pe fiecare procesor), apoi `G` = 96 (două pe procesor — încap: 32 de fire, fără memorie partajată).
   **Nu** peste 96 în prima rulare.

## 3. Predicțiile (scrise ÎNAINTE de cod și de compilare)

| # | Predicție | Încredere |
|---|---|---|
| B1 | G = 48, R = 1 001: termină, contorul 48 048, 0 runde greșite | ~85 % |
| B2 | G = 96: termină la fel (două grupuri pe procesor) | ~75 % |
| B3 | costul unei bariere la G = 48: **1–3 µs** (un drum dus-întors prin L2 pentru adunare + așteptarea celui mai lent grup + citirile repetate ale contorului) | ~60 % |
| B4 | la G = 96 bariera costă mai mult decât la 48, dar sub ×2 | ~60 % |
| B5 | dacă B3 iese ≤ 2 µs: `token.comp` pe #513 (195 de operații, 194 de bariere) ar avea ~0,2–0,4 ms doar în bariere, față de ~0,34 ms cost de lansare azi → câștigul la generare vine doar dacă bariera e sub ~1,5 µs **sau** dacă programul persistent lipește operații (mai puține bariere decât lansări) | aritmetică |

Dacă B1 pică (blocare sau runde greșite), `token.comp` nu merge pe placă așa cum e și nu-l cerem OS-ului; motivele posibile (DEDUS):
citirea contorului din L1 (nu din L2), sau grupuri care nu stau toate deodată pe placă.

## 4. Ce se face în sesiune (în ordine)

1. Programul GLSL (`tools/nak/grid_barrier.comp`) — gata în sesiunea 237.
2. Compilarea cu Mesa NAK (`sh tools/nak/build.sh grid_barrier` = `kernel grid_barrier 28`) + verificarea în listare: adunarea atomică și citirea
   contorului merg la L2 (`.strong.gpu` sau `.sys`), nu la L1.
3. Modulul Rust (argumentele, QMD-ul, verdictul pe rezultate) + modelul în `gpu-sim` (contorul și runde fără greșeli, SIMULARE) + test.
4. Nota pentru OS în `integration/cyber-universe/` (ce rulează, ce rânduri tipărește, limita de timp) — abia după 1–3.

**Ce e nou pentru placă (când va fi cerut):** un program nou (mic, fără memorie partajată), adunări atomice pe un contor în VRAM.
Nimic scris în registrele plăcii în afară de lansarea obișnuită.

## 5. Ce s-a făcut (sesiunea 237, ~21:50) — COMPILAT + SIMULARE, nimic pe placă

| Pas | Rezultat |
|---|---|
| 1. GLSL | `tools/nak/grid_barrier.comp` |
| 2. NAK | Mesa 25.0.7 reconstruit în cloud (`add64` ieșit identic, aceeași sumă); `grid_barrier.sm86.sass` **1 968 B** (123 de instrucțiuni), **19 registre**, SHA-256 `535f353705579f16deea31c363755092e7b00a7ba1ac43764c32d6c72183b12d`. Listarea: scrierile `st.global.strong.sys`, adunările `atom.add.strong.sys`, citirea vecinului `ld.global.strong.sys`, între ele `cctl` + `membar.sc.gpu` → totul la L2 / memorie, nu la L1 (cum cere bariera). |
| 3. Rust + model | `nvidia_ga104::grid_barrier` (cbuf, QMD, G, verdict, costul barierei), `gpu_sim::cmdproc::grid_barrier_on_cpu`; T245 4 / 4 în SIMULARE |
| 4. Notă OS | `integration/cyber-universe/C-157_P2zh_proba-barierei.md` |

**Găsit pe drum (important pentru `token.comp`):** programele care folosesc `gl_NumWorkGroups` îl citesc din bufferul constant la
**0x0c** (NVK: `cs.group_count[3]` la 0x0c / 0x10 / 0x14, înaintea argumentelor de la 0x28). `kronos_token::token_cbuf` lăsa 0x0c = 0
→ pe placă `token.comp` s-ar fi blocat (`vb += 0`) și bariera n-ar fi așteptat; modelul din `gpu-sim` nu vedea asta (folosea grila
din QMD). Acum: `token_cbuf_for(…, G)` (adăugat; cel vechi neschimbat), modelul refuză lansarea fără 0x0c, test nou în
`sim_i14_token.rs`. Doar `token.comp` și proba citesc 0x0c (verificat în toate listările din `tools/nak/out*`).

Predicțiile B1–B5 rămân deschise până rulează OS-ul proba.
