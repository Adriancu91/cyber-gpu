# C-147 — Planul de performanță, pachetul 1: etapa 0 (măsurători) + viteza maximă pregătită + limitele #651

Sesiunea Cyber-GPU 119 (19 nov 2026, ~23:05–23:40, în cloud, **fără placă**). Comanda: hub `COMENZI.md` C-147 (Adrian, 19 nov 22:33,
completare 22:50). Planul: hub `idei/2026-11-19_plan-performanta-placa.md`. Totul e **adăugat** (nimic redenumit sau șters) și testat
**doar în SIMULARE**; cifrele din teste sunt ale ceasului simulat (250 µs pe lansare) și nu spun nimic despre RTX 3070 Ti.

## 1. Ce măsoară etapa 0 (nimic schimbat în calcule)

| Punct | Ce | Unde în cod | Rânduri în jurnal | Stare |
|---|---|---|---|---|
| (a) | timpul pe **fel de lansare**: cele 16 de pe strat + embedding, normalizarea finală, logit-urile, copierea în RAM. Pentru fiecare fel: suma pe un simbol (median peste simboluri) și o lansare (median) | `nvidia_ga104::kronos_perf::{token_kinds, KindProfile}` | `kp …` (4 + 2 pe fel = 44 la #513), o singură dată la final | PASS (SIMULARE), T135a–b |
| (b) | costul unei **lansări goale** (`i10-cost-lansare`) scris și în jurnal | `kronos_perf::{launch_cost_rows, journal_launch_cost}` | `kl …` (5) | PASS (SIMULARE), T135b |
| (c) | **starea de performanță** a plăcii (doar citire), la sfârșitul lanțului | `nvidia_ga104::gsp::perf::{encode_get_pstate, read_pstate, journal_pstate}` | `ks …` | PASS (SIMULARE), T135d |
| (d) | **copierea mare VRAM → VRAM** (64 MiB, de 3 ori într-o trimitere) cu `copy_b128` → MB/s copiați, traficul memoriei (citire + scriere) și cât e din 608 GB/s | `kronos_perf::VramCopy` | `kc …` (11) | PASS (SIMULARE), T135c |

Cititorul pentru uneltele de pe gazdă: `kronos_perf::parse_journal` (kp, kl, kc); `ks` se citește direct (un rând pe citire).

**Ce include un rând `kp`:** timpul dintre două înregistrări de pe placă. În felul de trimitere de azi (`EachLaunch`, cel care a trecut I14)
acesta e „golirea memoriei rapide + lansarea”, aceeași mărime pe care o adună `kv simbol pe placa lansarile`. Suma tuturor felurilor
= `kv` (verificat în T135a).

**Copierea VRAM → VRAM:** sursa = greutățile modelului, deja pe placă (OS-ul are aceiași octeți în RAM); destinația = 64 MiB liberi.
Trei cuvinte ale destinației (primul, mijlocul, ultimul) se citesc prin fereastră și se compară cu copia din RAM (`kc cuvinte verificate
gresite`). Prima copiere poate plăti pentru memoria rece; contează cea mai rapidă. Atenție la citire: programul `copy_b128` e simplu
(16 octeți pe fir), deci cifra e „cât scoate programul nostru”, o margine de jos a vitezei memoriei, nu plafonul ei.

## 2. Starea de performanță: ce se trimite și ce nu (regula „trei surse”)

| Mesaj | Forma | Surse citite | Verdict |
|---|---|---|---|
| `PERF_GET_CURRENT_PSTATE` (0x20802068), pe subdevice, 4 octeți (`currPstate`, mască de biți P0…P15) | doar citire | (1) NVIDIA 570.144 `ctrl2080perf.h`: comanda, structura, biții; (2) NVIDIA 570.144 `g_subdevice_nvoc.c`: steagurile `0x40048` = `ROUTE_TO_PHYSICAL` → **driverul NVIDIA însuși trimite exact acest mesaj, neschimbat, la GSP-RM**; (3) NVIDIA 570.144 `rpc.c` (`RmRpcPerfGetCurrentPstate`): același unic cuvânt pe calea vGPU. Plicul `GSP_RM_CONTROL` e cel deja dovedit pe placă (I8e–I10) | **se trimite** (la sfârșitul lanțului). Notă cinstită: toate trei sunt fișiere NVIDIA; tinygrad, nouveau și nova nu folosesc comanda |
| `PERF_GET_LEVEL_INFO_V2` (0x2080200b) | cere ca apelantul să completeze domeniile de ceas (`NV2080_CTRL_PERF_GET_CLK_INFO.domain`) | valorile `NV2080_CTRL_CLK_DOMAIN_*` **nu sunt** în antetele publice 570.144 (`ctrl2080clk.h` e gol acolo) | **nu se trimite** — un domeniu ghicit nu se trimite |
| temperatura | — | `ctrl2080thermal.h` e gol în 570.144; nu e publicat `dev_therm.h` pentru GA10x; nouveau citește un registru (familia GP100), necunoscut pentru noi | **nu se citește** — nu există cale documentată (condiția (d) a lui Adrian: „dacă există”) |

Un refuz al plăcii = un rând `ks … = 8000xxxx` (xxxx = codul `NV_STATUS`), iar lanțul merge mai departe ca azi. Modelul nostru de GSP
(`gpu_sim::fake_rm`) nu cunoaște aceste comenzi și le refuză — T135d verifică exact acest drum.

## 3. Viteza maximă cerută oficial (punctul 4) — PREGĂTITĂ, OPRITĂ implicit

**Ce am aflat citind sursele (important):** în driverul NVIDIA pentru plăcile cu GSP, comanda publică `PERF_BOOST` **nu** e trimisă la GSP:
steagurile ei (`0x318`) nu au `ROUTE_TO_PHYSICAL`; partea de pe procesor (`kern_perf_boost.c`, `kperfBoostSet`) o transformă în
**`NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X`** (0x20800a9a, `{NvBool flags; NvU32 duration}`, `ctrl2080internal.h`) și pe aceea o
trimite la GSP-RM. **tinygrad** (driverul lor propriu, `ops_nv.py` `NVDevice.__init__`) trimite la GSP direct comanda publică `PERF_BOOST`
(`{NvU32 flags; NvU32 duration}`, durata `0xffffffff` = „până la ștergere”) — **și nu verifică răspunsul**. Deci nu avem nicio dovadă că
GSP acceptă forma tinygrad.

| | Forma A (ca driverul NVIDIA) | Forma B (ca tinygrad) |
|---|---|---|
| comanda | `INTERNAL_PERF_BOOST_SET_2X` 0x20800a9a | `PERF_BOOST` 0x2080200a |
| parametrii (8 octeți) | octetul 0 = steaguri, octeții 4–7 = durata | octeții 0–3 = steaguri, 4–7 = durata |
| surse | antetul `ctrl2080internal.h` + codul `kern_perf_boost.c` + tabelul de rutare `g_subdevice_nvoc.c` (toate NVIDIA 570.144) | antetul `ctrl2080perf.h` + tinygrad (fără verificarea răspunsului) |
| alegerea noastră | **implicită** (dacă se pornește) | doar dacă A e refuzată și centrul de comandă cere |

Ce am pus în cod (`gsp::perf::BoostPlan`): steagurile ca tinygrad (`BOOST_TO_MAX | CUDA | prioritate mare` = 0x52), **durată finită
600 s** (niciodată „până la ștergere”; peste 3600 s e refuzat), ștergerea la final în aceeași formă (`CLEAR | CUDA` = 0x10, durata 0).
`BoostPlan::default()` are `enabled = false`: `encode_boost` întoarce `Blocked` până când OS-ul îl pornește explicit, doar în modul
ciclu, după ce centrul de comandă confirmă în hub că etapa 0 a arătat o stare joasă (condițiile lui Adrian, C-147 completare 22:50).

**Ce lipsește față de condițiile lui Adrian:** temperatura (d) — nu există cale documentată (§2). Protecția rămâne cea a plăcii
(firmware-ul ei își scade singur ceasurile la căldură) + durata limitată + ștergerea la final. Nu am adăugat un prag de temperatură
pentru că n-avem de unde citi temperatura. Dacă Adrian vrea totuși prag, trebuie o decizie separată (ex. citirea unui registru de
temperatură după nouveau, cu „trei surse”).

**Ordinea propusă în testul de după `os-309`** (dacă `ks` arată o stare joasă): `ks inainte` → cererea (`ks cerere`) → `ks dupa cerere`
→ `kronos --gpu` (aceleași 16 simboluri ca procesorul, `kp`/`kv` înainte/după) → ștergerea (`ks stergerea`) → `ks la final`.

## 4. Kronos #651 (C-150): limitele, în SIMULARE

`kronos_speed::ModelLimits::k651()` (forma din cyberspace `config/conv_w2_mare_24zc.toml`: 24 straturi, 12 capete, E 768, context 512,
vocabular 16 384; de verificat cu `integrare/kronos/MODEL_651.md` când exportul îl scrie):
- **388 de lansări pe simbol** (387 + copierea logit-urilor), zona de comenzi **64 KiB**;
- **pasul valorilor pe sonerie: 389 — peste `0x100` pe care îl folosește OS-ul la #513**: pentru #651 trebuie `0x200`;
- VRAM: greutățile ~190–215 MiB, total ~250–290 MiB (cu memoria convorbirii); urcarea în bucăți de 8 MiB.
`ModelLimits::check` le numește pe toate înainte de urcare (T135e).

## 5. Ce n-am făcut încă din C-147 (în ordinea comenzii)
- punctul 2 (etapa 3a: lipirea calculelor mărunte, 196 → ~87), punctul 3 (etapa 4: `q8_matvec` cu 32 de fire), punctul 5 (unealta NAK
  nouă, circuitele speciale), punctul 6 (studiul câmpurilor de lansare), punctul 7 (estimarea după cifrele etapei 0 — așteaptă `os-309`).
