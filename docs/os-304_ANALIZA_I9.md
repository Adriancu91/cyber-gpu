# Analiza I9 după `os-304` — pasul 5 (jetonul de lucru, `0x40`) și ce încearcă `--i9 c135` (comanda C-135)

Sesiunea Cyber-GPU 108 (17 nov 2026, cloud, **fără placă**). Tot ce e cod aici e testat **doar în SIMULARE**. Nimic nu a rulat pe RTX 3070 Ti.
Date: hub `rezultate/2026-11-17-workstation-os304/REZULTAT.md` + `rezultate/os-automat/2026-11-17/gpu-jurnal-006.txt` (`gpu lant --i9 c133`, cyber-gpu `b352d50`).
Etichete: **OBSERVAT** = scris în jurnalul de pe placă; **CITIT** = scris în sursele deschise (MIT), citite ca specificație; **DEDUS** = concluzia noastră;
**NECUNOSCUT** = nu se poate afla fără placă (partea „fizică” a GSP-RM e în firmware, pe care nu avem voie să-l analizăm). Continuă `docs/os-303_ANALIZA_I9.md`.

## 0. Pe scurt (pentru Adrian)

- **Ce s-a întâmplat la `os-304`:** blocajul vechi (`0x57`) e depășit — forma B a trecut și de „spune-i plăcii unde sunt bufferele” și de obiectul de
  calcul. Placa a refuzat **mesajul următor**: cel prin care îi cerem „numărul canalului” (jetonul cu care se sună la sonerie). Refuzul e `0x40` („stare
  nepotrivită”), cu un singur semnal intern nou. Mesajul de după el — pornirea grupului de canale — **nu a mai fost trimis**, deci despre el nu știm încă nimic.
- **Cauza cea mai probabilă:** placa a calculat numărul canalului, dar a vrut să-l și **scrie într-un „carnețel de erori”** al canalului. Noi îi declarăm
  un asemenea carnețel, dar îl descriem greșit (adresă 0, „nu e în memoria video”, tip necunoscut) — exact valorile lui tinygrad. Driverul oficial NVIDIA
  și driverul Linux (nouveau) nu trimit niciodată acest mesaj plăcii: își aleg singuri numărul canalului. **Cât de sigur: cam 80 %.**
- **O descoperire care schimbă cum citim tinygrad:** tinygrad **nu se uită la răspunsul „a mers / n-a mers”** al plăcii la aceste mesaje. Deci „tinygrad
  face așa și merge” nu dovedește că placa a acceptat. Foarte probabil primește aceleași refuzuri ca noi și merge mai departe.
- **Ce testează `gpu lant --i9 c135`** (o singură pornire): pornește direct cu forma B (fără încercarea A). **N** = canalul spune limpede „nu am carnețel
  de erori” (ca nouveau), apoi cere numărul și pornește grupul. Doar dacă placa refuză iar numărul: **F** = într-un grup nou, **noi alegem numărul
  canalului** (cum fac NVIDIA și nouveau și cum face deja imaginea golden), fără să mai întrebăm placa.
- **Șansa ca I9 să treacă în această rulare: cam 70–75 %.** Dacă trece, `gpu lant` merge singur mai departe la I10 — primul calcul pe placă.

## 1. Ce s-a observat pe placă (OBSERVAT, `gpu-jurnal-006.txt`)

I0–I8 + golden trec (golden ok la 199,01 s). `i9 c133`: 2 încercări, fiecare în grupul ei.

| Mesaj | A (MAIN + PM + PATCH, un mesaj PA+VA) | B (MAIN + PATCH PA+VA; FECS_EVENT + PRIV_ACCESS_MAP doar VA; fără PM) |
|---|---|---|
| grup + context share | acceptate (`0xcf00000b`) | acceptate (`0xcf00000e`) |
| canal | acceptat (`0xcf00000d`); 2 ASSERT `0x01ab9bda`, `0x01ab9c56` | acceptat (`0xcf000010`); aceleași 2 ASSERT |
| `PROMOTE_CTX` | **refuzat `0x57`**; 2 ASSERT `0x01443f48`, `0x01418a2a` (ca la os-302/303) | **acceptat**, fără ASSERT |
| obiectul compute | — | **acceptat** |
| `GET_WORK_SUBMIT_TOKEN` (pe canal) | — | **refuzat `0x40` `NV_ERR_INVALID_STATE`**; 1 ASSERT nou `0x0185eb2e` |
| `GPFIFO_SCHEDULE` (pe grup) | — | **netrimis** |

**Care mesaj a fost refuzat** (numărând soneriile, câte una pe mesaj, ca în `os-295_ANALIZA_I9.md`): după alocarea canalului B și scrierea tabelelor de pagini
vin exact **3** sonerii (200,875 s; 200,907 s; 200,939 s) = `PROMOTE_CTX`, obiectul compute, **jetonul**; ASSERT-ul sosește la a treia și nu mai urmează a
patra. Deci „pasul 5” din raport înseamnă cererea jetonului, nu programarea grupului. (În cod, amândouă sunt între „4 compute” și „5 programat”.)

Încă două fapte din jurnal: (1) cele 2 ASSERT de la alocarea canalului B au **aceleași** timestamp-uri NOCAT ca la canalul A (`0x19f0f3a0`, `0x19f3d0c0`) —
GSP-RM retrimite înregistrarea păstrată a unui ASSERT care se repetă; (2) alocarea canalului golden dă alte 3 ASSERT (`0x01a9068a`, `0x01a90cf4`,
`0x01394270`) și **niciodată** perechea `0x01ab9bda`/`0x01ab9c56`.

## 2. Ce s-a confirmat din analiza `os-303` și ce nu se știe încă

- **Confirmat (rândul „A `0x57`, B trece” din §6):** refuzul `0x57` ținea de **ce buffere** promovăm — bufferul PM și/sau lipsa adreselor bufferelor
  globale — nu de forma mesajului. A era deja „un singur mesaj, înaintea obiectului” și a primit tot `0x57`, cu aceleași 2 ASSERT ⇒ ipotezele (a) + (d)
  singure sunt **infirmate**.
- **Nu se știe încă** care dintre cele două: PM sau globalele (B le-a schimbat pe amândouă). Nu e nevoie să aflăm ca să mergem mai departe: B e forma
  lui nouveau, iar pentru calcul bufferul PM (măsurători de performanță) nu e necesar. Le-ar despărți o încercare „B + PM” — de făcut doar dacă va conta.
- **Îndreptare la `os-303_ANALIZA_I9.md`.** Acolo scrie că tinygrad rulează 570.190 și că „nu avem nicio dovadă că merge pe 570.144”. A doua parte e
  greșită: commit-ul tinygrad studiat în octombrie (`9cf7829a6`, verificat din nou azi: `gsp-570.144.bin`) rula **aceeași secvență pe 570.144**; abia
  commit-ul de azi (`9ecf871`) e pe 570.190. În plus, fișierele deschise ale RM pentru canale, fifo și GR sunt **identice** între 570.144 și 570.190
  (`kernel_channel.c`, `kernel_channel_group*.c`, `kernel_ctxshare.c`, `kernel_fifo.c`, `kernel_fifo_ga100.c`, `kernel_graphics*.c`,
  `channel_descendant.c` — comparate azi). Ipoteza (f) „diferența de firmware” cade.
- **De ce „merge” atunci tinygrad cu secvența care la noi e refuzată** (CITIT + DEDUS): în `runtime/support/nv/ip.py`, `rpc_rm_alloc` și `rpc_rm_control`
  verifică doar rezultatul transportului RPC (`read_resp`: `hdr.rpc_result`); **câmpul `status` al răspunsului RM nu e citit niciodată**. RM 570.144
  (`rpc.c` `rpcRmApiControl_GSP`: „status = starea transferului; `rpc_params->status` = starea întoarsă de comanda de pe GSP”) și nouveau (`rpc->status`)
  citesc tocmai acel câmp, ca noi. Deci tinygrad primește, cel mai probabil, aceleași `0x57` și `0x40` și merge mai departe; pentru jeton ia numărul
  canalului din parametrii răspunsului. **De acum, „tinygrad trimite X” nu mai e pentru noi dovadă că GSP-RM acceptă X.**

## 3. Pasul 5, mesaj cu mesaj, față de cine merge (C-135 pasul 2)

Surse (toate MIT, citite ca specificație, nimic copiat): NVIDIA open-gpu-kernel-modules **570.144** (`kernel_channel.c`, `kernel_channel_group_api.c`,
`kernel_fifo.c`, `arch/ampere/kernel_fifo_ga100.c`, `arch/maxwell/kernel_channel_gm107.c`, `g_kernel_channel_nvoc.{c,h}`, `alloc/alloc_channel.h`,
`ctrl/ctrlc36f.h`, `ctrl/ctrla06c.h`, `ctrl/ctrla06f/ctrla06fgpfifo.h`, `vgpu/rpc.c`); nouveau (`nvkm/subdev/gsp/rm/r535/fifo.c`, `r570/fifo.c`,
`nvkm/engine/fifo/ga100.c`; `ga102.c`: GA104 ↔ 570.144); tinygrad (`ops_nv.py`, `support/nv/ip.py`; commit `9cf7829a6` cu 570.144 și `9ecf871` cu 570.190).

| După obiectul compute | noi (`os-304`, B) | tinygrad | nouveau (570.144) | NVIDIA 570.144 (RM-ul de pe CPU, client al GSP-RM) |
|---|---|---|---|---|
| **numărul canalului (ChID)** | îl alege GSP-RM (`flags` = 0) | la fel | **îl alege driverul**: în `flags`, `USERD_INDEX_PAGE_FIXED` + pagină/index (ChID = pagină × 8 + index) | **îl alege RM-ul de pe CPU** și îl trimite la fel (`_kchannelSendChannelAllocRpc`: „will make the Physical RMAPI use our ChID”) |
| **jetonul** | `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` (`0xc36f0108`) pe **canal**, `workSubmitToken = 0xffffffff` → **`0x40`** | același mesaj; starea RM necitită; numărul luat din răspuns, runlist-ul adăugat de driver | **nu se trimite**; jeton = (id-ul soneriei runlist-ului `<< 16`) \| ChID | **nu ajunge la GSP-RM**: comanda are flagurile `0x8` (fără `ROUTE_TO_PHYSICAL`), o servește RM-ul de pe CPU: jeton = runlist + ChID. (Varianta internă `0xc36f0301` există, dar nu e chemată nicăieri în sursele deschise.) |
| **notificatorul de erori al canalului** | declarat: `hObjectError` ≠ 0, `errorNotifierMem = {baza 0, 0xecc, addressSpace 0}`, tipuri „necunoscut” (`internalFlags` = 0) | la fel | **fără**: `hObjectError` = 0, `internalFlags` = privilegiu + ambele tipuri `NONE` | cel adevărat al aplicației (tip `MEMORY`/`CTXDMA` + descriere reală) sau `NONE` |
| `BIND` | nu | nu (canal în grup) | `NVA06F_CTRL_CMD_BIND {engineType}` pe **canal**, imediat după alocare | — |
| **programarea** | `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE {bEnable = 1}` pe **grup** — netrimisă încă pe placă | același mesaj, pe grup, după ce a creat canalele | `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE {bEnable = 1}` pe **canal**, imediat după `BIND` — înaintea promovării și a obiectelor | pe grup: RM-ul de pe CPU verifică, apoi trimite aceeași comandă la GSP-RM |

**Unde diferim de cei care verifică răspunsurile** (nouveau, NVIDIA): (1) cerem jetonul de la GSP-RM — ei niciodată; (2) lăsăm GSP-RM să aleagă ChID-ul;
(3) declarăm un notificator de erori pe care nu-l avem, cu tipuri „necunoscut”. Programarea pe grup o face și driverul oficial.

## 4. Unde întoarce RM `NV_ERR_INVALID_STATE` pe aceste drumuri (CITIT, 570.144)

| # | Drum | Condiția | ASSERT? |
|---|---|---|---|
| 1 | jeton (`kchannelCtrlCmdGpfifoGetWorkSubmitToken`) | canal fără grup | da (1) — nu e cazul nostru |
| 2 | jeton, generarea (`kfifoGenerateWorkSubmitToken` → `…Hal_GA100` → partea internă a GSP) | orice eșec în generare | **cel puțin 2** (două `NV_ASSERT_OK_OR_RETURN` suprapuse); pe ramura de CPU: „canalul nu are încă runlist” — pe Ampere runlist-ul se pune la alocare |
| 3 | **jeton, notificarea** (`kchannelNotifyWorkSubmitToken` → `kchannelUpdateNotifierMem`), **după** ce jetonul a fost generat | canalul are un descriptor de notificator și, **pe GSP**, memoria lui **nu e memorie video** (`addressSpace == ADDR_FBMEM`); sau notificatorul nu poate fi mapat | **exact 1** |
| 4 | programare pe grup (`kchangrpapiCtrlCmdGpFifoSchedule`) | un canal „neprogramabil”: spațiu de adrese deținut din afară + motor GR + context „nelegat” de UVM | nu (doar verificare) |
| 5 | programare pe canal (`kchannelCtrlCmdGpFifoSchedule`) | aceeași condiție | da (1); comentariul spune că verificarea se face „pe baremetal, CPU-RM și guest-RM” |
| 6 | alocarea canalului (`kchannelAllocHwID`) | `USERD_INDEX_PAGE_FIXED` împreună cu `USERD_INDEX_FIXED` | da |
| 7 | punerea pe runlist (`kfifoRunlistSetId`) | canal deja pe alt runlist; runlist fără grupuri; grup pe alt runlist | parțial |

Ce am observat — **un singur ASSERT + `0x40`, la jeton** — se potrivește cu rândul 3 și numai cu el.

## 5. Ipotezele pentru `0x40`, în ordinea probabilității

| # | Ipoteza | Ce o susține | Ce o contrazice | Probabilitate |
|---|---|---|---|---|
| (n) | **Notificatorul de erori**: jetonul e generat, apoi GSP-RM vrea să-l scrie în notificatorul declarat de canal, descris de noi ca „nu memorie video” | rândul 3 din §4: exact 1 ASSERT + `INVALID_STATE`, pe drumul acestei comenzi, verificare doar pe GSP; canalul nostru declară notificator (`{0, 0xecc, 0}`, tipuri necunoscute); perechea de ASSERT de la alocare apare la **toate** canalele de lucru (os-302…os-304) și **niciodată** la canalul golden, care își spune tipurile în `internalFlags` — exact cele două `NV_ASSERT` alăturate „tipul ≠ necunoscut” din `kchannelConstruct` ⇒ GSP-RM chiar citește descrierea din aceste câmpuri; nimeni dintre cei care citesc starea nu trimite comanda la GSP-RM; tinygrad (aceiași parametri, 570.144) folosește numărul din răspuns ⇒ numărul e generat **înainte** de refuz, iar după generare comanda nu mai face decât notificarea | adresa ASSERT-ului (`0x0185eb2e`) e departe de cele de la alocare, deși funcțiile sunt în același fișier sursă (nu putem verifica: firmware-ul nu se analizează); când își construiește GSP-RM descriptorul e în partea închisă | **mare (~80 %)** |
| (t) | **Generarea jetonului** eșuează în GSP-RM | adresa în altă zonă | ar da ≥ 2 ASSERT; runlist-ul e pus la alocare; tinygrad primește un număr bun pe același drum | mică (~5–10 %) |
| (r) | **Urma încercării A** (grup refuzat, rămas alocat în aceeași pornire) | B a rulat după A | comanda e per canal; alocarea, promovarea și obiectul lui B nu au fost deranjate | mică (~5 %); `c135` o scoate oricum (nu mai rulează A) |
| (p) | **Forma B** (fără PM / globale doar VA) lasă canalul într-o stare pe care jetonul n-o acceptă | — | drumul jetonului nu atinge contextul GR în sursele deschise | foarte mică |
| (x) | altceva, în partea închisă | — | — | ~5–10 % |

**Mesajul următor (programarea grupului), netrimis încă:** singura capcană vizibilă e rândul 4 din §4 — canalul nostru are spațiu de adrese „deținut din
afară” și motor GR, iar „legarea” o face în driverul oficial doar UVM, pe procesor. Contra: nouveau programează pe 570.144 canale GR cu spațiu extern
**înainte** de orice promovare și merge, iar comentariul din RM spune că verificarea se face pe CPU-RM, nu pe GSP. Risc mic–mediu (~20 %), NECUNOSCUT până la test.
Semnul lui în jurnal ar fi `0x40` la programare **fără** ASSERT.

## 6. Reparația: `--i9 c135` (C-135 pasul 3) — SIMULARE, implicit oprită

`nvidia_ga104::gsp::user_space::open_work_channel_c135` (doar adăugări; `gpu lant` fără opțiune, `--i9 c125` și `--i9 c133` trimit exact ce trimiteau).
Pornește **direct cu forma B** (un `PROMOTE_CTX` înaintea obiectului: MAIN + PATCH cu PA + VA, FECS_EVENT + PRIV_ACCESS_MAP cu VA-ul din golden) — fără A.

| | Încercarea N | Încercarea F (doar dacă GSP-RM refuză jetonul lui N), în grup nou |
|---|---|---|
| ce schimbă față de B din `os-304` | canalul spune explicit **„fără notificator de erori”**: `hObjectError` = 0, `errorNotifierMem` zero, `internalFlags` = `0x14` (ambele tipuri `NONE`) — ca nouveau | ca N **+ ChID ales de noi**: `flags` = `0x00202000` (pagina 2, index 0 ⇒ ChID 16), ca NVIDIA, nouveau și canalul nostru golden (`0x200320` = ChID 3) |
| jetonul | cerut de la GSP-RM, ca până acum | **calculat de noi** (runlist `<< 16` \| 16); GSP-RM nu e întrebat |
| apoi | programarea grupului | programarea grupului |
| ipoteze acoperite | (n), (r) | + (t), (x) la jeton |
| mesaje | grup, context share, canal, `PROMOTE_CTX`, compute, jeton, programare | grup, context share, canal, `PROMOTE_CTX`, compute, programare |

- **De ce N întâi:** e schimbarea cea mai mică (3 câmpuri ale alocării canalului, restul octet cu octet ca B) și lovește cauza cea mai probabilă; dacă
  trece, tot ce am trimis a fost **acceptat** de GSP-RM, iar numărul canalului vine de la el.
- **De ce F ca rezervă:** ocolește cu totul comanda pe care niciun driver care verifică răspunsurile nu o trimite plăcii; mecanismul „noi alegem ChID-ul” e
  cel standard pe GSP și e deja dovedit pe placa noastră de canalul golden. ChID 16 (pagina 2) e liber oricum ar împărți GSP-RM numerele: 0 e al GSP-RM,
  3 e al canalului golden (pagina 0), iar singurul canal cu număr ales de GSP-RM (al lui N) ia primul liber — în pagina 0 sau, dacă GSP-RM dă fiecărui
  client pagina lui, la începutul paginii 1 (ChID 8). Dacă placa îl refuză totuși, lanțul se oprește cu codul ei — nu ghicim alt număr.
- **Refuzul jetonului în N se notează întreg:** starea și **câmpul de jeton din răspunsul refuzat** (sau faptul că refuzul a venit în antetul RPC, fără
  parametri). Câmpul e **doar notat, niciodată folosit** — nu luăm un rezultat dintr-un răspuns refuzat (regula „niciun succes fictiv”); tinygrad o face.
- **Oprire imediată** la orice alt refuz (grup, canal, promovare, compute, programare) sau eroare de transport. Un refuz al promovării în N ar fi o
  noutate (B a trecut pe placă) și nu se „repară” cu F.
- **Sigur în aceeași pornire:** doar alocări și controale RM de felul celor deja trimise, niciun registru nou; grupul lui N, dacă e refuzat, rămâne
  neprogramat (nu rulează nimic), cum a rămas A la `os-304`. Canalul nu mai are pagină de notificator și nici buffer PM.

Ce **nu** face: nu descrie un notificator adevărat în memoria video (ar fi varianta „completă”, cu mai multe necunoscute); nu schimbă felul programării
(pe grup); nu trimite `BIND`.

## 7. Ce spune fiecare rezultat de pe placă (C-135 pasul 4)

| Rezultat în jurnal (`i9 c135 …`) | Ce confirmă / infirmă | Ce urmează |
|---|---|---|
| **N trece** (`varianta reusita = 1`, `jeton = 1 de la RM`) | (n) confirmată. Semn în plus de căutat: perechea de ASSERT `0x01ab9bda`/`0x01ab9c56` **dispare** de la alocarea canalului și `0x0185eb2e` nu mai apare | I9 gata; `gpu lant` continuă singur: forma GR0 de la RM, apoi **I10 (add64)** — primul calcul pe placă |
| **N: jeton refuzat, F trece** (`= 2`) | (n) infirmată în forma „fără notificator = fără refuz”. Câmpul notat spune restul: un număr mic ⇒ jetonul se generează și refuzul vine după; `0xffffffff` ⇒ generarea însăși pică ((t)); „fără răspuns întreg” ⇒ GSP-RM pune refuzul în antetul RPC | I10 cu jetonul calculat de noi (ChID 16); F devine forma de bază |
| **N: jeton refuzat, canalul lui F refuzat** | GSP-RM nu primește ChID-ul 16 de la noi (codul spune de ce) | alt ChID sau notificator adevărat în memoria video — cu „da” nou |
| **programarea refuzată** (în N sau F), `0x40` **fără** ASSERT | verificarea „canal neprogramabil” (§4 rândul 4) se face și pe GSP-RM | programare pe canal ca nouveau (`BIND` + `GPFIFO_SCHEDULE`) sau „legarea” ca UVM — de analizat, cu „da” nou |
| **programarea refuzată** cu ASSERT sau alt cod | ceva din partea fizică (runlist, instanță) | codul + ASSERT-urile noi decid |
| **`PROMOTE_CTX` refuzat în N** | la `os-304`, B a trecut **după** A: reușita lui ar fi depins de urma lui A | se oprește; de analizat (ar fi o noutate) |
| **alocarea canalului refuzată în N** | GSP-RM nu primește „fără notificator” la un canal din grup (nouveau îl trimite la canale fără grup) | notificator adevărat în memoria video — cu „da” nou |
| I9 trece, dar **I10 nu primește răspuns** | canalul e programat, lansarea nu pornește | se compară jetonul cu id-ul soneriei runlist-ului (nouveau îl citește din registrul runlist-ului, noi luăm id-ul runlist-ului de la RM, ca RM-ul oficial) |

## 8. Ce rămâne NECUNOSCUT / ce nu s-a putut face

- Dacă refuzul `0x57` venea de la bufferul PM sau de la lipsa globalelor (§2).
- Dacă GSP-RM pune codul unui control refuzat în câmpul `status` al răspunsului (cum arată sursele) sau în antetul RPC: jurnalele de până acum nu le
  despart; `c135` o notează dacă jetonul e refuzat.
- Programarea grupului pe placă: netrimisă niciodată; tot ce scrie aici despre ea vine din surse.
- Ce e la `0x0185eb2e`: adresă din firmware, neanalizată; rămâne amprentă de comparat între rulări.
- `gpu_sim::fake_rm::{PromoteRule, TokenRule}` joacă **ipoteze**, nu placa: testele arată că `c135` face ce scrie aici în fiecare caz, nu că placa va
  răspunde așa. Regula modelului pentru ChID-urile nealese de driver e inventată (SIM).

## 9. Pe placă — doar în fereastra cu Adrian

După ce OS-ul leagă opțiunea (nota `integration/cyber-universe/C-135_i9-pasul-5.md`): stick nou + `github.token` → **pornire la rece** →
`gpu lant --i9 c135` → DA → `trimite`. O singură rulare ajunge. Dacă I9 trece, aceeași rulare merge mai departe la I10–I14.
