# Matricea de teste

## Rezultate — sesiunea 3 (23 oct 2026, cloud, fără placă)
| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1–T4 | format, clippy, teste, build no_std | `sh tools/check.sh` | toate trec | 102 trecute, 0 eșuate, 1 ignorat; build OK | PASS |
| T7 | CI pe GitHub | push pe `g0-g2-fundatie` | verde | rulările 7–8 verzi (repo public) | PASS |
| T19 | FWSEC_PROD găsit, ucode necitit | `cargo test -p nvidia-ga104 fwsec` | 11 teste; nicio citire în ucode | conform | PASS (SIMULARE) |
| T20 | regiunile GSP în 8 GiB | `cargo test -p nvidia-ga104 fb_layout` | 6 teste | conform | PASS (aritmetică) |
| T21 | radix3 | `cargo test -p nvidia-ga104 radix3` | 3 teste | conform | PASS (aritmetică) |
| T22 | tag-ul `stabil-2026-10-23` pe GitHub | `git push origin stabil-2026-10-23` | urcat | HTTP 403 (politica sesiunii cloud) | BLOCKED |

## Rezultate — sesiunea 2 (23 oct 2026, cloud, fără placă)
Hardware: container cloud Linux, Intel Xeon 2,1 GHz. **Fără placă NVIDIA.** Toolchain: rustc 1.97.0 stable, `x86_64-unknown-linux-gnu`.

| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1–T4 | format, clippy, teste, build no_std | `sh tools/check.sh` | toate trec | 82 trecute, 0 eșuate, 1 ignorat; build OK | PASS |
| T5 | logică hardware SIMULATĂ (nivel 0, BAR, I2, I3, configurația workstation) | inclus în T3 | zero scrieri; opriri cinstite | conform | PASS (SIMULARE) |
| T7 | CI pe GitHub | push pe `g0-g2-fundatie` | joburile `crates (stable/nightly)` verzi | verzi (rulările 2–4); rularea 1 roșie doar la `kernel-integration` (scos); rularea 5: joburile n-au primit calculator (0 pași), reluată o dată, la fel | PASS până la rularea 4; BLOCKED la 5 |
| T16 | matmul pe blocuri = referința | `cargo test -p gpu-compute` | bit cu bit | bit cu bit, 7 forme | PASS |
| T17 | bf16 | idem | 65 536/65 536 dus-întors; rotunjire = oracol pe 200 000 de valori | conform | PASS |
| T18 | viteză matmul 64×768×768 (orientativ) | `cargo test --release -p gpu-compute -- --ignored --nocapture` | — | referință 1,2 GFLOP/s, blocuri 14,2, bf16 14,5 (gazdă, un fir) | INFO |
| T8, T10–T15 | QEMU, placa fizică | — | — | — | NOT TESTED / BLOCKED |

## Rezultate — sesiunea 1 (23 oct 2026)
Hardware: laptop Lenovo, i5-1135G7, Iris Xe, Windows 11 Pro 26300. **Fără placă NVIDIA.**
Cod: ramura `g0-g2-fundatie`. Toolchain: rustc 1.99.1 `stable-x86_64-pc-windows-gnu`.

| Id | Test | Comandă | Așteptat | Obținut | Clasificare |
|---|---|---|---|---|---|
| T1 | format | `cargo fmt --all -- --check` | fără diferențe | fără diferențe, cod 0 | PASS |
| T2 | analiză statică | `cargo clippy --workspace --all-targets -- -D warnings` | 0 avertismente | 0, cod 0 | PASS |
| T3 | teste pe gazdă | `cargo test --workspace --release` | toate trec | 58 trecute, 0 eșuate | PASS |
| T4 | build `no_std` | `cargo build --workspace --target x86_64-unknown-none --release` | compilează | compilează, cod 0 | PASS |
| T5 | logică hardware SIMULATĂ | inclus în T3 (`gpu-sim`) | nivel 0 fără scrieri; dimensionare restaurată | conform | PASS (SIMULARE) |
| T6 | patch-ul de integrare se aplică | `git apply --check` pe `cyber-universe@2d038de` | se aplică | se aplică, cod 0 | PASS |
| T7 | kernel compilat cu patch-ul | job CI `kernel-integration` | compilează | nerulat local (modificarea clonei OS a fost refuzată de politica de permisiuni) | NOT TESTED |
| T8 | boot QEMU + `gpu` | QEMU | `gpu: nicio placa GA104` | QEMU indisponibil | NOT TESTED |
| T9 | inventar K0 (script) | `tools/k0-inventar-workstation.ps1` pe laptop | rulează, doar citire | rulează; BAR-urile Iris Xe 16 MiB / 256 MiB | PASS (nu e workstation) |
| T10 | identificare RTX 3070 Ti fizic | `gpu identifica` | BOOT_0 = 0x174xxxxx | — | BLOCKED (B1) |
| T11–T15 | memorie, transfer, prima comandă, verificare, repetare | — | — | — | BLOCKED (I4–I10) |

## Plan pentru seara de testare (în ordine; nu se sare peste o etapă eșuată)

### A. Fără workstation (oricând)
1. `sh tools/check.sh` → T1–T4 PASS.
2. Mută `ci/github-actions-ci.yml` în `.github/workflows/ci.yml` (cu un token care are scope-ul `workflow`), push, apoi CI: joburile `crates (stable)`, `crates (nightly)`, `kernel-integration` → T7.
3. Într-o clonă locală `cyber-universe` (de Adrian sau cu acordul lui): aplică patch-ul, `tools/build.sh --test`, scenariul
   `gpu` (mai jos) în QEMU → T8. Așteptat pe serial: `gpu: nicio placa GA104` (QEMU are VGA `1234:1111`).
   Scenariu propus pentru `tests/scenarios.txt` (necesită `test-shell`):
   ```
   gpu       | test-shell       | success | TASTATURA: astept taste;;cyber> gpu;;gpu: nicio placa GA104;;cyber> gpu plan;;I10;;cyber> test-gata;;TEST READY | | keys=g p u ret g p u spc p l a n ret t e s t minus g a t a ret
   ```

### B. Pe workstation — numai în fereastra aprobată de Adrian (Kronos oprit)
Condiții: Kronos oprit și salvat; Adrian prezent; stick cu Release-ul care conține patch-ul; Secure Boot oprit temporar (decizia lui Adrian).
1. **Windows, înainte:** `powershell -ExecutionPolicy Bypass -File tools\k0-inventar-workstation.ps1` → fișierul `rezultate\k0-*.txt`.
   Notează manual din BIOS: IOMMU (AMD-Vi/VT-d), Above 4G Decoding, Resizable BAR, CSM.
2. Pornește de pe stick → `cyber>`.
3. `lspci` → rândul `10de:2482` (și `10de:228b` audio). Poză.
4. `gpu identifica` → poză + `log gpu 40`. PASS dacă: model RTX 3070 Ti, BAR0 mem32, D0, `NV_PMC_BOOT_0` cu `& 0x1ff00000 = 0x17400000`,
   `consecvent true`, **`0 scrieri in placa`**. Notează bitul 29 al BOOT_42.
5. `gpu plan` → poză (după sesiunea 2: oprirea trebuie să fie la **I4**, nu la I2).
5a. **I2 — GFW_BOOT (doar citire)** — necesită în OS o comandă care mapează UC primii **4 MiB** din BAR0 (vezi hub `integrare/cyber-gpu/`):
    `gpu gfw` → poză. PASS: `0x118128` cu bitul 0 = 1, `0x118234 & 0xff = 0xff`, în < 4 s (pe o placă pornită de mult timp: la prima citire),
    `0 scrieri`. FAIL: timeout (notează ultimele valori), `0xbadf…` (eroare de acces) sau `0xffffffff`. **Dacă FAIL: oprire, nu se trece la 5b.**
5b. **I3 — VBIOS (doar citire)**: `gpu vbios` → poză. PASS: primul cuvânt la BAR0+0x300000 = `…aa55` sau `0x4947564e` („NVGI”); lista de imagini
    conține PCI-AT, EFI și cel puțin o FWSEC (`0xe0`); tabela BIT are jetonul `0x70`; numărul de citiri < 300 000; `0 scrieri`; ecranul rămâne viu.
    Comparație: Windows raportează VBIOS 94.04.65.00.3e — dacă OS-ul poate afișa șirul de versiune din imagine, trebuie să coincidă.
    Dacă primul cuvânt nu e o semnătură: **BLOCKED** (umbrirea ROM ar cere o scriere) — se notează valoarea și se discută cu Adrian.
5c. **I3b — FWSEC (doar citire, aceeași fereastră PROM)**, numai dacă 5b a trecut și OS-ul afișează rezultatul lui `fwsec::locate`:
    PASS: tabela PMU conține aplicația `0x85`; descriptorul are versiunea 3; numărul de semnături = biții din `signature_versions`;
    ucode-ul încape în imaginile FWSEC; `0 scrieri`. Notează: numărul de semnături, `ucode_id`, `engine_id_mask`, mărimile IMEM/DMEM.
    FAIL la versiune/pointer = citirea noastră a formatului e greșită → se notează valorile, nu se ghicește.
6. Opțional, numai dacă Adrian acceptă riscul: `gpu dimensiuni da` → mărimi BAR; ecranul trebuie să rămână viu.
7. `reboot` în Windows; verifică că Kronos pornește normal (nvidia-smi OK).

### Ce se colectează pentru fiecare test
identificator, hardware/configurație, commit, comanda, așteptat, obținut, cod de ieșire/jurnal (`log gpu`), PASS/FAIL/BLOCKED/NOT TESTED
→ în acest tabel și în `Cyber-hub/rezultate/`.
