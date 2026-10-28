# Pentru sesiunea OS: `gpu firmware` (I4) și `gpu fwsec` (I3b), ambele doar citire — hub C-36

*28 oct 2026, sesiunea Cyber-GPU 15. Doar o propunere tehnică; comanda-propunere pentru OS e în hub,
`idei/2026-10-28_cyber-universe_gpu-firmware-fwsec.md` (o aprobă centrul de comandă). Nimic de aici nu e aplicat în `cyber-universe`.
Tot ce e mai jos a fost testat **doar în SIMULARE** (pe gazdă), niciodată pe placă sau pe fișierele reale.*

Fixare cyber-gpu: un commit ≥ `2c9c90c` de pe `g0-g2-fundatie` (din copia din hub `integrare/cyber-gpu/snapshot/cyber-gpu.bundle`).
Față de `8313973` (fixarea de acum a OS-ului) sunt **doar adăugări**: modulul `nvidia_ga104::fw_check`, metoda
`nvidia_ga104::fwsec::FwsecLocation::lines()`, modulul `gpu_sim::fw_stick`. Nimic redenumit sau șters.

## 1. `gpu firmware` — fișierele GSP 570.144 de pe stick (I4)
Ce face: citește, pe rând, cele 4 fișiere din `/usb/firmware/nvidia/ga102/gsp/` (+ `/usb/firmware/LICENCE.nvidia`) și afișează pentru
**fiecare** (nu se oprește la primul greșit): lipsă / nu se poate citi / altă mărime / altă sumă SHA-256 / „ok = oficial”; pentru cele
oficiale, și dacă parserul nostru le înțelege antetul. Ultimul rând: `criteriul I4: TRECE` sau `NU TRECE`. **0 accese la placa video**
(nici configurație, nici BAR0). Memorie: un singur fișier odată (cel mai mare ~64 MB, eliberat imediat). Timp: citirea a ~64 MB de pe
stick (cu `usb citire mare` mai repede) + SHA-256 (ESTIMARE < 2 s).

```rust
// kernel/src/drivers/gpu.rs — adapter over the kernel VFS (same as the I4 proposal of 26 Oct).
struct StickFirmware;
impl nvidia_ga104::firmware::FirmwareSource for StickFirmware {
    fn read_file(&mut self, path: &str) -> gpu_core::error::GpuResult<Vec<u8>> {
        crate::fs::read_all(path).map_err(|e| match e {
            crate::fs::FsError::NotFound => gpu_core::error::GpuError::NotPresent,
            _ => gpu_core::error::GpuError::InvalidArgument("citirea de pe stick a esuat"),
        })
    }
}

/// `gpu firmware`: read-only check of the GSP 570.144 files on the stick. No GPU access at all.
pub fn firmware_check() -> Vec<String> {
    use nvidia_ga104::{firmware, fw_check};
    let r = fw_check::check_stick(&mut StickFirmware, &firmware::GSP_570_144);
    let lines = r.lines();
    for l in &lines { crate::info!("gpu: {}", l); }
    lines
}

/// `gpu firmware --sim` (test builds only): the same check on the gpu-sim stand-ins (NOT NVIDIA files).
#[cfg(feature = "test-gpu-sim")]
pub fn firmware_check_sim() -> Vec<String> {
    use gpu_sim::fw_stick;
    use nvidia_ga104::{firmware::STICK_DIR, fw_check};
    let files = fw_stick::files();
    let man = core::array::from_fn(|i| fw_check::pin_of(fw_check::ROLE_ORDER[i], fw_stick::SIM_PATHS[i], &files[i]));
    let mut src = fw_check::MemSource::new();
    for (p, b) in fw_stick::SIM_PATHS.iter().zip(files) { src.insert(format!("{}{}", STICK_DIR, p), b); }
    src.insert(format!("{}{}", STICK_DIR, fw_check::LICENCE), fw_stick::licence());
    fw_check::check_stick(&mut src, &man).lines().into_iter().map(|l| format!("SIMULARE: {}", l)).collect()
}
```

Test QEMU propus: `gpu firmware --sim` → rândul `SIMULARE: criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4)`;
`gpu firmware` (fără fișiere reale în QEMU) → de 4 ori `LIPSESTE de pe stick` și `criteriul I4: NU TRECE (fisiere oficiale 0/4`.
Fișierele reale nu intră niciodată în repo, în CI sau în arhiva stick-ului (le pune Adrian).

## 2. `gpu fwsec` — unde e FWSEC în VBIOS (I3b)
Ce face: după `vbios::read_layout` (ca la `gpu vbios`), cheamă `fwsec::locate` pe aceeași fereastră doar-citire și afișează
`FwsecLocation::lines()` (5 rânduri: imaginile FWSEC, tabela PMU cu aplicațiile, descriptorul v3, semnăturile, `criteriul I3b`).
Citește doar tabelele din fereastra PROM; ucode-ul nu e citit. Ca variantă a lui `InitStep`:

```rust
InitStep::Fwsec => {
    let r = vbios::read_layout(&mut win, &mut journal, &mut clock, vbios::VBIOS_TIMEOUT_US).and_then(|l| {
        nvidia_ga104::fwsec::locate(&mut win, &mut journal, &mut clock, vbios::VBIOS_TIMEOUT_US, &l)
    });
    match r {
        Ok(f) => f.lines().into_iter().map(|l| format!("{}: {}", p, l)).collect(),
        Err(e) => alloc::vec![format!("{}: FWSEC (I3b): {}", p, e)],
    }
}
```
`--sim`: `bar0.load_bytes(regs::PROM_BASE, &rom::asus_like_chain_with_fwsec(&rom::SimFwsec::ga10x_like()))` → rândul
`criteriul I3b: TRECE` (lanțul de tip ASUS, imaginile „NV” la `0x24600`, ca pe placa din workstation).

## 3. `gpu plan`
Rânduri noi propuse (după 5b): ` 5c. gpu fwsec -> poza (I3b: unde e FWSEC; 0 scrieri)` și
` 5d. gpu firmware -> poza (I4: cele 4 fisiere 570.144 de pe stick; 0 accese la placa)`.
