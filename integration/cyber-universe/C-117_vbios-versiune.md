# C-116 / C-117 — `vbios.versiune`: versiunea VBIOS din jetonul BIT 'i', de comparat cu raportul din Windows

*13 nov 2026, ~01:30, sesiunea 96 cyber-gpu (cloud, FĂRĂ placă). Totul SIMULARE; nimic rulat pe laptop sau pe workstation.* Commit: `835f24d` (doar adăugări — OS-ul compilează neschimbat).

## De ce

Raportul UHL al lui Adrian de pe ASUS TUF (hub, 12 nov 23:03) dă **VBIOS 94.06.1E.00.23**; K0 de pe workstation dă `94.04.65.00.3e`. Până acum turul de citire
arăta imaginile ROM și lista de jetoane BIT, dar nu versiunea. De la `835f24d` `vbios::read_layout` citește și jetonul 'i' (0x69, 5 octeți, 5 citiri PROM în plus, doar citire)
și turul (`run_tour`, `run_tour_ampere`, `run_tour_awake`) scrie faptul **`vbios.versiune = 94.xx.xx.xx.xx`**. E prima verificare automată „Windows ↔ Cyber Universe pe aceeași
placă” (C-117): dacă șirul nu coincide, citirea noastră a ROM-ului e greșită sau PROM-ul arată altă imagine — se vede fără să ghicim.

## Ce e nou (nvidia_ga104)

- `vbios::BiosVersion { major, chip, minor, micro, patch }` + `Display` (`94.04.65.00.3e`), `BiosVersion::from_bytes([u8; 5])`, **`matches_report(&str) -> bool`**
  (compară cu șirul din Windows, indiferent de majuscule; altă formă = `false`).
- `vbios::VbiosLayout.version: Option<BiosVersion>` — `None` = jetonul 'i' lipsește sau are sub 5 octeți (nu inventăm). `vbios::{BIT_TOKEN_BIOS_INFO = 0x69, BIOS_INFO_MIN_LEN = 5}`.
- `read_tour::{FACT_VBIOS_VERSION = "vbios.versiune", VBIOS_VERSION_UNREAD}`.
- `gpu_sim::laptop` (faptele UHL: `VBIOS_VERSION = "94.06.1e.00.23"`, `SUBSYS_DEVICE = 0x16f2`, `SM_COUNT = 30`, `VRAM_BUDGET_BYTES`, …); `gpu_sim::workstation::VBIOS_VERSION` exista.
  `SimConfig::ga106_laptop_asus()` are acum subsistemul `1043:16f2`, rev A1, BAR1 @0x6000000000, legătura gen4 x8/x16 (din raport); `observed::observed_rom_with_version(..)`.

## Propunere pentru OS (nimic obligatoriu; OS-ul doar leagă `835f24d`+)

1. `gpu vbios` / `vbios_lines`: un rând `versiune VBIOS: 94.xx.xx.xx.xx` din `layout.version` (sau „necitita”) — apare oricum în `gpu tur`/`gpu lant` ca fapt `vbios.versiune`.
2. `analiza` (C-117): în `inventory.json`, la placa NVIDIA, proprietatea `vbios_version` cu `status: MEASURED` când `version` e `Some`, sursa „BIT token 'i' (0x69) prin PROM”;
   UHL-ul din Windows o raportează ca `DOCUMENTED`/din registrul driverului — comparația celor două fișiere o face automat.
3. Opțional: `gpu tur` pe laptop poate tipări „= raportul Windows” / „≠ raportul Windows (94.06.1E.00.23)” cu `matches_report(gpu_sim::laptop::VBIOS_VERSION)`
   doar când modelul din SMBIOS e FX506HM — altfel nu compară (alt laptop, altă versiune).

## Ce așteptăm de pe plăci

- Laptop (`gpu tur --trezire`, după DA): `vbios.versiune = 94.06.1e.00.23`. Workstation (`gpu tur`/`gpu lant`): `vbios.versiune = 94.04.65.00.3e`.
- Dacă apare „necitita (jetonul BIT 0x69 lipseste sau e scurt)” pe workstation, unde jetonul `0x69v2` e observat: lungimea lui e sub 5 → ne uităm la jetonul 'B' (0x42, BIOSDATA) ca alternativă, abia atunci.
