# Propunere pentru sesiunea OS: comanda `gpu firmware` (I4, doar citire)

*26 oct 2026, sesiunea Cyber-GPU 5. Doar o propunere: OS-ul decide (C-19 nu o cere; o poate prelua doar cu „da”-ul lui Adrian,
eventual în același Release pentru fereastra de pe workstation). Nimic de aici nu e aplicat în `cyber-universe`.*

## Ce face
Citește de pe stick cele 4 fișiere firmware GSP 570.144 și verifică pentru fiecare **numele, mărimea și suma SHA-256** față de valorile
fixate în `nvidia-ga104::firmware::GSP_570_144`. **Nu atinge placa video** (0 accese la configurație sau la BAR0) și nu încarcă nimic nicăieri.
Rostul: în fereastra de pe workstation, Adrian vede pe ecran că fișierele de pe stick sunt exact cele oficiale (pasul 8 din
`docs/TEST_MATRIX.md` partea B0).

## Cod (schiță pentru `kernel/src/drivers/gpu.rs`; fixarea cyber-gpu pe un commit ≥ cel din sesiunea 5)
```rust
// Adapter: the cyber-gpu firmware loader reads through the kernel VFS.
struct StickFirmware;
impl nvidia_ga104::firmware::FirmwareSource for StickFirmware {
    fn read_file(&mut self, path: &str) -> gpu_core::GpuResult<alloc::vec::Vec<u8>> {
        crate::fs::read_all(path).map_err(|e| match e {
            crate::fs::FsError::NotFound => gpu_core::error::GpuError::NotPresent,
            _ => gpu_core::error::GpuError::InvalidArgument("firmware: citirea de pe stick a esuat"),
        })
    }
}

/// `gpu firmware`: checks the GSP 570.144 set on the stick. No GPU access.
pub fn cmd_firmware() {
    let mut journal = gpu_core::diag::Journal::new(16);
    let r = nvidia_ga104::firmware::load_gsp_set(
        &mut StickFirmware,
        &nvidia_ga104::firmware::GSP_570_144,
        &mut journal,
    );
    for e in journal.events() { println!("  {e}"); }   // "pas I4: gsp-570.144: ok" ...
    match r {
        Ok(fw) => println!("gpu firmware: 570.144 ok ({} octeti GSP-RM), 0 accese la placa", fw.gsp_rm.len()),
        Err(e) => println!("gpu firmware: {e}"),
    }
}
```
Calea pe stick: `/usb/firmware/nvidia/ga102/gsp/<nume>` (`firmware::STICK_DIR` + `Pinned::path`). Memorie: ~64 MB pentru `gsp-570.144.bin`
(se eliberează la ieșirea din funcție). Timp: SHA-256 pe 64 MB, cu codul nostru (fără SSE), ESTIMARE 1–3 s pe workstation.

## Test în CI (QEMU)
Fără fișierele reale (nu le avem și nu le punem în repo): în QEMU comanda trebuie să spună `gpu firmware: dispozitivul nu exista`
(fișier lipsă = `NotPresent`) — asta verifică doar legătura. Verificarea adevărată e pe workstation, cu stick-ul lui Adrian.

## Compatibilitate
Doar adăugări în cyber-gpu față de `41e44ba`: `firmware::{GSP_570_144, STICK_DIR, FirmwareSource, load_gsp_set, verify, GspFirmware, GspRole, Pinned}`,
`sha256`, `gsp::*`, `Ga104Backend::{set_firmware, has_firmware}`. Schimbări de valoare (nu de formă): `firmware::GSP_VERSION` = `"570.144"`
(era `"535.113.01"`) și căile GSP din `GA104_FIRMWARE`; `init::GA104_PLAN`: I4 e acum „implementat” (planul se oprește la I5), iar mesajul de
blocaj al lui `initialize()` fără firmware este „I4: firmware-ul GSP 570.144 nu a fost dat …”. OS-ul nu folosea niciuna dintre ele (verificat 26 oct).
