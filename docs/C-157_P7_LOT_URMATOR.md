# C-157 punctul 7, partea a doua — lotul următor scris cât placa lucrează (două zone de lansări)

Sesiunea 185, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE** (`gpu-sim`); viteza e **DEDUS** (nemăsurată). Prima parte: `C-157_P7_SIMBOL_PE_PLACA.md`.

## De ce
Cu prima parte (`embed_dev`, sesiunea 170), un lot de poziții merge pe o singură sonerie, dar driverul tot face pe rând:
scrie lotul → sună → **așteaptă** → citește → scrie lotul următor. Cât scrie, placa stă (estimat ~150–300 µs pe simbol de scris).
Lotul următor **nu depinde** de rezultatele lotului de acum: simbolurile generate le citește `embed_dev` pe placă, iar pozițiile,
adresele din cache-uri și lungimea atenției le știe driverul dinainte. Deci driverul poate scrie lotul k+1 cât placa lucrează la k.

## Ideea (scrisă înainte)
1. **Două zone** (A, B) pentru tot ce scrie driverul pe lot: cbuf-urile + QMD-urile lansărilor și înregistrările (timpii + gardul).
   Lotul k merge în zona `k mod 2`. Pushbuffer-ul îl împarte deja `WorkQueue` (nu suprascrie o bucată încă la placă).
2. **Regula zonei:** zona lotului k se rescrie abia după ce driverul a citit gardul lotului k−2. Cel mult **2 loturi la placă**.
3. **`Generation::batch_ahead(pos, max)`** (adăugare): lotul care începe la `pos`, fără să aștepte rezultatele de dinainte
   (pozițiile din text → simbolul de la procesor; restul → de pe placă). Primul lot = același ca `next_batch`.
4. **`kronos_batch::run_pipelined`** (adăugare) + trăsătura **`BatchIo`** (`send(zona, poziție, intrări)` = scrie + sună, fără
   așteptare; `wait(zona, poziție, n)` = așteaptă gardul zonei și citește rezultatele). Bucla e în bibliotecă, testată în SIMULARE;
   OS-ul scrie doar cele două funcții. Ordinea: trimite k, trimite k+1, așteaptă k → `accept_batch`, trimite k+2 în zona lui k, ...
   După o oprire (EOS, eroare, „niciun simbol”) **nu mai trimite nimic**, dar **așteaptă** lotul deja trimis (memoria lui nu se
   atinge cât e la placă) și îi aruncă rezultatele; o eroare la așteptarea lui se raportează separat, nu ascunsă.
   O zonă cerută cât e încă ocupată = refuz (`TwoZones`), nu scriere.
5. **În SIMULARE:** placa (`gpu-sim`) execută loturile trimise abia când driverul așteaptă — cazul cel mai rău pentru o zonă
   rescrisă prea devreme.

## Predicții (scrise acum)
1. **Aceleași rezultate:** textul `[5, 200, 17]` + 12 simboluri (modelul de probă, 3 straturi): cu două zone, loturi de 1 / 4 / 5,
   cu și fără modelul de citire veche (128) + `EachLaunch` → **aceleași simboluri, aceleași 4 cuvinte argmax, cache-urile K/V bit
   cu bit** ca „o poziție pe sonerie”; soneriile = ⌈14 / lot⌉, ca la loturile fără suprapunere.
2. **Suprapunerea:** fiecare lot în afară de primul e scris cât cel de dinainte e la placă (loturi − 1 suprapuneri); niciodată
   mai mult de 2 la placă; zona k = k mod 2.
3. **EOS la mijlocul unui lot:** aceeași secvență ca o poziție pe sonerie cu același EOS; lotul de după a fost trimis, așteptat și
   aruncat (≤ un lot de poziții risipite), nicio eroare.
4. **Greșeli plantate prinse:** (a) ambele loturi la placă în **aceeași zonă** → alte simboluri / alte cache-uri sau oprire (lotul
   k rulează lansările lotului k+1); (b) argmax cu indice în afara vocabularului la poziția 6 → aceeași oprire `Eroare` la 6 ca
   T180e, lotul de după așteptat; (c) `TwoZones` refuză o zonă încă ocupată și un lot prea mare pentru zonă.
5. **Viteza (DEDUS, de verificat pe placă):** placa ~1 075 + ~16 µs pe simbol rămân; scrierea lotului se ascunde după calculul
   lotului de dinainte → **~1 100–1 150 µs pe simbol = ~870–910 simb/s** (față de ~710–800 cu lotul scris tot înainte și 635,90
   azi). Prețul: la oprire, până la un lot de poziții calculate degeaba (cu loturi de 4: ~4,4 ms, o dată pe generare).

## Ce NU face
Nu schimbă niciun program, nicio listă, nici `next_batch` / `accept_batch` / `batch_launches`. Nimic nou în registre sau prin
GSP. Nimic trimis la placă: nota pentru OS vine după, cu „ce e nou pentru placă: două trimiteri la placă în același timp”.
