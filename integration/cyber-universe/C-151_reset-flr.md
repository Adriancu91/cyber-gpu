# Pentru sesiunea OS: resetul plăcii din interior (C-151 punctul 3 → C-152 punctul 1)

**Stare: SIMULARE (cyber-gpu sesiunea 122).** Pe placă: proba aprobată în ciclul automat (C-152 punctul 1), într-un Release separat de pachetele de viteză.
Surse și teste: `docs/C-151_RESET_FLR.md`. Varianta 1 din `docs/C-142_K513_VITEZA_FLR.md` §4.

## Ce legi (doar adăugări în crate-uri; nimic de azi nu se schimbă)

1. O implementare a `nvidia_ga104::flr_reset::FlrBoard` peste ce aveți deja: `bar0()` = fereastra BAR0 a plăcii, `cfg()` = configurația PCI a funcției
   `b3:00.0` (ECAM). Nu le folosiți pe amândouă în același timp.
2. Permisul: `gpu_pci::flr::FlrPermit::new("C-152: reset FLR in ciclu (aprobat de Adrian 19 nov)")`; `hw` = permisul BAR0 al lanțului (doar citiri aici).
3. **La începutul lanțului**, înainte de I3: `let r = flr_reset::reset_if_wpr2_up(&mut board, &mut clock, &permit, hw, &mut journal);`
   apoi rândul `r.row()` în jurnal (pe stick imediat, ca pașii) și `r.verdict.meaning()` în raportul ciclului.
4. `r.verdict.chain_may_run()` = `true` → lanțul merge (I3 … I14); `false` → oprire curată, fără a doua încercare.

## Pașii exacți pe care îi face funcția (ca să-i recunoașteți în jurnal)

1. citește `NV_PFB_PRI_MMU_WPR2_ADDR_HI` (BAR0 `0x1fa828`); 0 → rândul `kf nu-era-nevoie …`, **nicio scriere**;
2. salvează configurația (doar citiri: 0x00–0x3c, Device/Link Control 1 și 2, MSI);
3. Bus Master oprit (o scriere în Command, recitită);
4. așteaptă „Transactions Pending” = 0, cel mult 100 ms (dacă nu: notă în jurnal și merge mai departe, ca RM și Linux);
5. **FLR**: Device Control (capabilitatea PCIe + 0x08) | bitul 15, jumătatea de stare scrisă 0;
6. 100 ms, apoi citește Vendor ID până răspunde (nu `0xffffffff`, nu `0x0001`), cel mult 1 s;
7. rescrie configurația salvată (doar ce diferă; BAR-urile cât Command e încă 0; Command la urmă) și o recitește — orice diferență = eroare;
8. așteaptă GFW boot (I2, doar citiri, cel mult 4 s);
9. recitește WPR2.

Atenție la ecran: dacă imaginea OS-ului e pe placa asta (GOP), se pierde la FLR — oricum îngheață după I8. Funcția audio `b3:00.1` nu se atinge.

## Ce înseamnă fiecare rezultat (rândul `kf <cod> wpr2_inainte=… wpr2_dupa=… gata_us=… gfw_us=…`)

| Cod | Ce înseamnă | Ce face OS-ul |
|---|---|---|
| `nu-era-nevoie` | WPR2 era 0 (pornire rece) | lanțul ca azi |
| `wpr2-coborat` | **FLR a coborât WPR2** (varianta 1 merge) | lanțul a doua oară, în aceeași pornire; dacă trece, ciclul poate rula mai multe teste fără Windows (modul îl pornește centrul de comandă) |
| `wpr2-inca-ridicat` | FLR nu coboară WPR2 pe placa asta | oprire curată, rând clar în raport, repornire în Windows; noi pregătim varianta 2 (Booter Unload cu FWSEC-SB) |
| `fara-flr` | placa nu anunță FLR (contrar lui K-2) | nimic scris; ca azi |
| `placa-nu-raspunde` | după FLR placa nu mai răspunde | oprire curată, repornire în Windows (cablul dacă trebuie) |
| `config-nerescrisa` | configurația PCI nu a revenit | placa nu se mai folosește în pornirea asta; repornire |
| `gfw-neterminat` | VBIOS nu și-a terminat pornirea | oprire curată, pornire rece |
| `eroare` | altceva (vezi jurnalul) | oprire curată |

Cititorul: `flr_reset::parse_row(line)` → `(cod, lanțul poate rula)`.

## Proba din C-152 punctul 1, cu funcția asta

lanțul obișnuit trece → `kf` nu e încă necesar; WPR2 citit (așteptat ridicat) → **`reset_if_wpr2_up`** (așteptat `wpr2-coborat`) → lanțul a doua oară.
Trimiteți-ne jurnalul (rândurile `kf`, pașii `C-151 FLR …`, `valoare`/`asteptat`) oricare ar fi rezultatul.
