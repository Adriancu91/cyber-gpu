# os-301 pe laptop (13 nov 2026): jurnalul `gpu tur --trezire` citit de Cyber-GPU — nimic de schimbat în OS

*Sesiunea Cyber-GPU 98, 13 nov 2026 ~13:30, cloud, fără placă.*

- Jurnalul `gpu-jurnal-003.txt` + poza `gpu-tur-trezire.jpg` (hub `rezultate/2026-11-13-laptop-asus-os301/`) sunt acum **modelul laptopului** în
  `gpu_sim::observed_laptop` (cyber-gpu, ramura `g0-g2-fundatie`): turul simulat dă aceleași 36 de fapte comparabile, 0 diferențe (`docs/GA106_LAPTOP.md` §0c).
- **`vbios.versiune = 94.06.1e.00.23`** pe laptop = șirul din UHL (Windows) → rândul „versiune VBIOS” din `gpu vbios` / proprietatea `vbios_version` din `analiza`
  (nota `C-117_vbios-versiune.md`) pot fi marcate ca verificate pe o placă reală; pe workstation rămâne de văzut `94.04.65.00.3e`.
- **Nimic de legat, nimic de schimbat în codul OS** pentru acest pas: `835f24d` rămâne bun; `os-302` (P6-90, `analiza` cu faptele turului) e exact ce observase Adrian.
- Un amănunt, fără urgență: ecranul a scris „45 fapte”, iar turul nostru produce 47 de rânduri `fapt` pe același model (cele 39 vizibile în poză + 8 tăiate sus: `chip.familie`,
  `chip.scrieri`, `pci.id`, `pci.subsistem`, `pci.clasa`, `pci.command`, `pci.rebar.bar1`, `pci.rebar.bar3`). Dacă OS-ul numără altfel (de ex. fără cele două `chip.*` ale
  familiei) e doar o cifră de afișaj; dacă două fapte nu ajung pe ecran, merită o privire la `gpu tur`. Nu schimbă nimic în driver.
- **Pasul 3 din C-116 (I6–I8 pe GA106)** rămâne după DA-ul separat al lui Adrian (luni seara cel mai devreme); lista de pregătire e în `docs/GA106_LAPTOP.md` §7.
  Pentru OS, când vine DA-ul: `gpu i6` / `gpu lant` ar accepta GA106 doar pe stick-ul de test (`gpu-scrie`), cu trezirea minimă ținută pornită pe durata lanțului — propunere, decizia e a sesiunii OS.
