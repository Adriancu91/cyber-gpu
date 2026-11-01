//! I6/I7 loader: puts a signed "heavy-secure" (HS) ucode into a falcon over
//! DMA and programs the boot ROM for the signature check — the step that
//! `falcon::load_not_written` stood in for. **Tested only in SIMULARE**
//! (against `gpu_sim::fake_gsp` with synthetic images); never run on the
//! RTX 3070 Ti, and the real firmware files have never been loaded.
//!
//! Sources (read as specification, code written by us, docs/PROVENANCE.md):
//! NVIDIA open-gpu-kernel-modules 570.144 (MIT):
//! - the DMA sequence and the boot-ROM registers: `kernel_gsp_falcon_ga102.c`
//!   (`kgspExecuteHsFalcon_GA102`, `s_dmaTransfer_GA102`) and
//!   `kflcnDisableCtxReq_TU102` (`kernel_falcon_tu102.c`);
//! - the descriptor fields: `kernel_gsp_fwsec.c` (FWSEC, descriptor v3) and
//!   `kernel_gsp_booter.c` (Booter, "BOOT_FROM_HS"), with their range checks;
//! - the signature choice for Booter (`s_patchBooterUcodeSignature`).
//!
//! Register offsets: generated from NVIDIA's MIT headers (`nvidia-rm570`).
//! DEDUS: that the 12-byte `meta_data` of the linux-firmware Booter file is
//! RM's `patchMeta` (fuseVer, engineId, ucodeId) — nouveau `nvfw/hs.h` reads
//! it the same way; the 256-byte alignment we demand of the destinations;
//! the DMA timeout.
//!
//! What the driver does here is only what any loader does: copy opaque bytes
//! (code, data, one signature chosen by index) to where the header says.
//! Nothing looks inside the firmware.
//!
//! Every register write goes through [`GuardedBar0`] (GSP/SEC2 windows only,
//! journaled; on a real BAR only with the `hw-write` permit). Everything is
//! checked **before the first write**. Not here: putting the image in system
//! memory the GPU can reach and enabling PCI Bus Master — the OS does that;
//! [`DmaImage`] only carries the physical address it chose.

use alloc::vec::Vec;

use gpu_core::diag::Event;
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio, poll_until};
use nvidia_rm570::regs::ga102_dev_falcon_second_pri as f2;
use nvidia_rm570::regs::ga102_dev_falcon_v4 as fv4;
use nvidia_rm570::regs::ga102_dev_fbif_v4 as fbif;

use super::falcon::{Engine, GuardedBar0};
use crate::fw_hdr::HsV2;
use crate::fwsec::{FalconUcodeDescV3, RSA3K_SIG_LEN};

/// DMA block: every transfer moves 256 bytes (`DMATRFCMD_SIZE_256B`).
pub const BLOCK: u32 = 256;
/// `FLCN_DMEM_VA_INVALID` (falcon_common.h): DMEM has no virtual address.
pub const DMEM_VA_INVALID: u32 = 0xffff_ffff;
/// DMATRFBASE (32 bits) + DMATRFBASE1 (9 bits) hold the address >> 8.
pub const MAX_DMA_ADDR: u64 = 1 << 49;
/// DEDUS: RM waits with its default GPU timeout; ours, in microseconds.
pub const DMA_US: u64 = 100_000;
/// The register FBIF windows (`dev_g{sp,sec}_addendum.h`: base + 0x600).
const FBIF_REL: u32 = 0x600;
/// ctxDma used by RM on GA102 for the ucode fetch.
const CTX_DMA: u32 = 0;

fn align_up(v: u32) -> Option<u32> {
    v.checked_add(BLOCK - 1).map(|x| x & !(BLOCK - 1))
}

/// Where the pieces of one HS ucode are (RM's `KernelGspFlcnUcodeBootFromHs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HsUcode {
    /// Bytes of the image in the DMA buffer.
    pub size: u32,
    pub code_offset: u32,
    pub imem_pa: u32,
    pub imem_va: u32,
    pub imem_size: u32,
    pub data_offset: u32,
    pub dmem_pa: u32,
    pub dmem_va: u32,
    pub dmem_size: u32,
    /// DMEM address of the (patched) signature for the boot ROM.
    pub sig_dmem_addr: u32,
    pub ucode_id: u8,
    pub engine_id_mask: u32,
}

impl HsUcode {
    /// FWSEC from the VBIOS, descriptor v3 (`kernel_gsp_fwsec.c`).
    pub fn from_fwsec_v3(d: &FalconUcodeDescV3) -> GpuResult<Self> {
        let u = HsUcode {
            size: align_up(d.stored_size)
                .ok_or(GpuError::InvalidArgument("FWSEC: marime imposibila"))?,
            code_offset: 0,
            imem_pa: d.imem_phys_base,
            imem_va: d.imem_virt_base,
            imem_size: d.imem_load_size,
            data_offset: d.imem_load_size,
            dmem_pa: d.dmem_phys_base,
            dmem_va: DMEM_VA_INVALID,
            dmem_size: d.dmem_load_size,
            sig_dmem_addr: d.pkc_data_offset,
            ucode_id: d.ucode_id,
            engine_id_mask: u32::from(d.engine_id_mask),
        };
        u.check()?;
        Ok(u)
    }

    /// Booter (`booter_load-570.144.bin`), one application
    /// (`kernel_gsp_booter.c`, BOOT_FROM_HS branch).
    pub fn from_booter(h: &HsV2, file: &[u8]) -> GpuResult<Self> {
        let [app] = h.apps.as_slice() else {
            return Err(GpuError::InvalidArgument(
                "booter: trebuie exact o aplicatie",
            ));
        };
        let meta = BooterMeta::parse(file, h)?;
        let u = HsUcode {
            size: h.hdr.data_size,
            code_offset: app.offset,
            imem_pa: 0,
            imem_va: app.offset,
            imem_size: app.size,
            data_offset: h.os_data.offset,
            dmem_pa: 0,
            dmem_va: DMEM_VA_INVALID,
            dmem_size: h.os_data.size,
            sig_dmem_addr: h.patch_loc.checked_sub(h.os_data.offset).ok_or(
                GpuError::InvalidArgument("booter: semnatura inaintea datelor"),
            )?,
            ucode_id: u8::try_from(meta.ucode_id)
                .map_err(|_| GpuError::InvalidArgument("booter: ucodeId > 255"))?,
            engine_id_mask: meta.engine_id,
        };
        u.check()?;
        Ok(u)
    }

    /// RM's range checks, plus the ones a 256-byte DMA needs: every block
    /// read stays inside the image, the signature lies in the loaded DMEM.
    pub fn check(&self) -> GpuResult<()> {
        let bad = GpuError::InvalidArgument;
        if self.imem_size == 0 || self.dmem_size == 0 {
            return Err(bad("ucode HS: cod sau date de marime 0"));
        }
        let within = |off: u32, len: u32| -> GpuResult<()> {
            let end = align_up(len)
                .and_then(|l| off.checked_add(l))
                .ok_or(bad("ucode HS: depasire la adunare"))?;
            if end > self.size {
                return Err(GpuError::OutOfBounds {
                    offset: u64::from(off),
                    len: u64::from(len),
                    limit: u64::from(self.size),
                });
            }
            Ok(())
        };
        within(self.code_offset, self.imem_size)?;
        within(self.data_offset, self.dmem_size)?;
        for a in [self.imem_pa, self.imem_va, self.dmem_pa] {
            if !a.is_multiple_of(BLOCK) {
                return Err(bad("ucode HS: destinatie nealiniata la 256 B"));
            }
        }
        if self.dmem_va != DMEM_VA_INVALID && !self.dmem_va.is_multiple_of(BLOCK) {
            return Err(bad("ucode HS: adresa virtuala DMEM nealiniata"));
        }
        let sig_end = self
            .sig_dmem_addr
            .checked_add(RSA3K_SIG_LEN)
            .ok_or(bad("ucode HS: depasire la semnatura"))?;
        if sig_end > self.dmem_size {
            return Err(bad("ucode HS: semnatura in afara DMEM-ului incarcat"));
        }
        Ok(())
    }

    /// Physical address the DMA base register must hold for IMEM:
    /// `image + code_offset - imem_va` (RM), which must be 256-aligned
    /// because the register keeps only `addr >> 8`.
    fn imem_src(&self, image: &DmaImage) -> GpuResult<u64> {
        let a = image
            .phys
            .checked_add(u64::from(self.code_offset))
            .and_then(|a| a.checked_sub(u64::from(self.imem_va)))
            .ok_or(GpuError::InvalidArgument("DMA: adresa codului imposibila"))?;
        dma_base_ok(a)
    }

    fn dmem_src(&self, image: &DmaImage) -> GpuResult<(u64, u32)> {
        let mut a = image
            .phys
            .checked_add(u64::from(self.data_offset))
            .ok_or(GpuError::InvalidArgument("DMA: adresa datelor imposibila"))?;
        let mut mem_off = 0;
        if self.dmem_va != DMEM_VA_INVALID {
            a = a
                .checked_sub(u64::from(self.dmem_va))
                .ok_or(GpuError::InvalidArgument("DMA: adresa datelor imposibila"))?;
            mem_off = self.dmem_va;
        }
        Ok((dma_base_ok(a)?, mem_off))
    }
}

fn dma_base_ok(a: u64) -> GpuResult<u64> {
    if !a.is_multiple_of(u64::from(BLOCK)) || a >= MAX_DMA_ADDR {
        return Err(GpuError::InvalidArgument(
            "DMA: sursa nealiniata la 256 B sau peste 2^49",
        ));
    }
    Ok(a)
}

/// RM's `patchMeta` of a Booter image, read from the file's `meta_data`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BooterMeta {
    pub fuse_ver: u32,
    pub engine_id: u32,
    pub ucode_id: u32,
}

impl BooterMeta {
    pub fn parse(file: &[u8], h: &HsV2) -> GpuResult<Self> {
        let m = file
            .get(h.meta_data.clone())
            .ok_or(GpuError::InvalidArgument("booter: meta_data in afara"))?;
        if m.len() != 12 {
            return Err(GpuError::InvalidArgument(
                "booter: meta_data trebuie sa aiba 12 octeti",
            ));
        }
        let w = |i: usize| u32::from_le_bytes([m[i], m[i + 1], m[i + 2], m[i + 3]]);
        Ok(BooterMeta {
            fuse_ver: w(0),
            engine_id: w(4),
            ucode_id: w(8),
        })
    }
}

/// Index of the Booter signature for a board whose SEC2 fuse holds
/// `fuse_version` (RM `s_patchBooterUcodeSignature`: counted from the end).
/// The fuse is read by [`crate::fuse::booter_fuse_version`] (SEC2, `meta.ucode_id`).
pub fn booter_signature_index(num_sig: u32, fuse_version: u32) -> GpuResult<u32> {
    if num_sig == 0 {
        return Err(GpuError::InvalidArgument("booter: nicio semnatura"));
    }
    if num_sig == 1 {
        return Ok(0);
    }
    if fuse_version > num_sig - 1 {
        return Err(GpuError::Blocked(
            "booter: nicio semnatura pentru versiunea din siguranta placii",
        ));
    }
    Ok(num_sig - 1 - fuse_version)
}

/// The bytes to put in DMA-able system memory for Booter: the image with the
/// chosen signature copied to `patch_loc`, padded to whole 256-byte blocks.
pub fn booter_image(h: &HsV2, file: &[u8], fuse_version: u32) -> GpuResult<Vec<u8>> {
    if h.patch_sig != 0 {
        // RM refuses any other value as well.
        return Err(GpuError::InvalidArgument("booter: patch_sig diferit de 0"));
    }
    let idx = booter_signature_index(h.num_sig, fuse_version)? as usize;
    let len = h.sig_size();
    let sigs = &file[h.sig_prod.clone()];
    let sig = &sigs[idx * len..(idx + 1) * len];
    let mut img = Vec::from(&file[h.hdr.data()]);
    let at = h.patch_loc as usize;
    img[at..at + len].copy_from_slice(sig);
    pad(&mut img)?;
    Ok(img)
}

/// The bytes for FWSEC: the ucode as read from the VBIOS (`stored_size`
/// bytes) with `signature` (chosen with `fwsec::signature_index`) copied to
/// DMEM `pkc_data_offset`, padded to whole blocks.
pub fn fwsec_image(d: &FalconUcodeDescV3, ucode: &[u8], signature: &[u8]) -> GpuResult<Vec<u8>> {
    if ucode.len() != d.stored_size as usize || signature.len() != RSA3K_SIG_LEN as usize {
        return Err(GpuError::InvalidArgument(
            "FWSEC: marimea codului sau a semnaturii nu se potriveste",
        ));
    }
    let u = HsUcode::from_fwsec_v3(d)?;
    let at = (u.data_offset + u.sig_dmem_addr) as usize;
    let mut img = Vec::from(ucode);
    let dst = img
        .get_mut(at..at + signature.len())
        .ok_or(GpuError::OutOfBounds {
            offset: at as u64,
            len: signature.len() as u64,
            limit: ucode.len() as u64,
        })?;
    dst.copy_from_slice(signature);
    pad(&mut img)?;
    Ok(img)
}

/// FWSEC command "FRTS" (`FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS`,
/// RM `kernel_gsp_frts_tu102.c`).
pub const FWSEC_CMD_FRTS: u32 = 0x15;
/// Interface entry id of the DMEM mapper.
pub const FWSEC_IFACE_DMEMMAPPER: u32 = 0x4;
/// `sizeof(FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3)`: 4+2+2+14*4.
pub const DMEM_MAPPER_LEN: u32 = 64;
/// Offsets inside the DMEM mapper (C layout of the same struct).
pub const MAPPER_CMD_IN_OFFSET: usize = 8;
pub const MAPPER_CMD_IN_SIZE: usize = 12;
pub const MAPPER_INIT_CMD: usize = 44;
/// `sizeof(FWSECLIC_FRTS_CMD)`: READ_VBIOS_DESC (24, 8-aligned) + FRTS_REGION_DESC (20), padded to 48.
pub const FRTS_CMD_LEN: usize = 48;
/// FRTS region size RM asks for: 1 MiB in 4 KiB units.
pub const FRTS_SIZE_4K: u32 = 0x100;

fn rd32(b: &[u8], o: usize) -> GpuResult<u32> {
    let x = b
        .get(
            o..o.checked_add(4)
                .ok_or(GpuError::InvalidArgument("FWSEC: offset"))?,
        )
        .ok_or(GpuError::OutOfBounds {
            offset: o as u64,
            len: 4,
            limit: b.len() as u64,
        })?;
    Ok(u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
}

/// The bytes of `FWSECLIC_FRTS_CMD` for an FRTS region at VRAM `frts_offset`
/// (RM `s_prepareForFwsec_TU102`: read-VBIOS v1 with flags 2, region v1 of
/// 1 MiB in VRAM, media type 2 = FB).
pub fn frts_command(frts_offset: u64) -> GpuResult<[u8; FRTS_CMD_LEN]> {
    if frts_offset == 0
        || !frts_offset.is_multiple_of(0x1000)
        || frts_offset >> 12 > u64::from(u32::MAX)
    {
        return Err(GpuError::InvalidArgument(
            "FRTS: offset 0, nealiniat la 4 KiB sau prea mare",
        ));
    }
    let mut c = [0u8; FRTS_CMD_LEN];
    let mut put = |o: usize, v: u32| c[o..o + 4].copy_from_slice(&v.to_le_bytes());
    put(0, 1); // readVbiosDesc.version
    put(4, 24); // readVbiosDesc.size
    // gfwImageOffset (u64 at 8) = 0, gfwImageSize (16) = 0
    put(20, 2); // flags = FWSECLIC_READ_VBIOS_STRUCT_FLAGS
    put(24, 1); // frtsRegionDesc.version
    put(28, 20); // frtsRegionDesc.size
    put(32, (frts_offset >> 12) as u32);
    put(36, FRTS_SIZE_4K);
    put(40, 2); // FWSECLIC_FRTS_REGION_MEDIA_FB
    Ok(c)
}

/// RM `s_vbiosPatchInterfaceData`: in FWSEC's data (`data` = the DMEM
/// image), follow the interface table at `interface_offset` to the DMEM
/// mapper, set `init_cmd` and copy `cmd` into its input buffer. Stricter
/// than RM: a too-small input buffer, or one overlapping `keep_out` (the
/// signature), is refused instead of only logged.
pub fn patch_fwsec_command(
    data: &mut [u8],
    interface_offset: u32,
    init_cmd: u32,
    cmd: &[u8],
    keep_out: core::ops::Range<usize>,
) -> GpuResult<()> {
    let bad = GpuError::InvalidArgument;
    let io = interface_offset as usize;
    let hdr = data
        .get(io..io + 4)
        .ok_or(bad("FWSEC: interfata in afara datelor"))?;
    let count = usize::from(hdr[3]);
    if count < 2 {
        return Err(bad("FWSEC: prea putine intrari in interfata"));
    }
    let mut mapper = None;
    for i in 0..count {
        let e = io + 4 + 8 * i;
        if rd32(data, e)? == FWSEC_IFACE_DMEMMAPPER {
            mapper = Some(rd32(data, e + 4)? as usize);
        }
    }
    let m = mapper.ok_or(bad("FWSEC: lipseste DMEM mapper-ul"))?;
    if m.checked_add(DMEM_MAPPER_LEN as usize)
        .is_none_or(|e| e > data.len())
    {
        return Err(bad("FWSEC: DMEM mapper-ul in afara datelor"));
    }
    let in_off = rd32(data, m + MAPPER_CMD_IN_OFFSET)? as usize;
    let in_size = rd32(data, m + MAPPER_CMD_IN_SIZE)? as usize;
    if in_size < cmd.len() {
        return Err(bad("FWSEC: bufferul de comanda prea mic"));
    }
    let end = in_off
        .checked_add(cmd.len())
        .filter(|&e| e <= data.len())
        .ok_or(bad("FWSEC: bufferul de comanda in afara datelor"))?;
    let overlaps = |a: usize, b: usize| a < keep_out.end && keep_out.start < b;
    if overlaps(in_off, end) || overlaps(m + MAPPER_INIT_CMD, m + MAPPER_INIT_CMD + 4) {
        return Err(bad("FWSEC: comanda s-ar suprapune peste semnatura"));
    }
    data[m + MAPPER_INIT_CMD..m + MAPPER_INIT_CMD + 4].copy_from_slice(&init_cmd.to_le_bytes());
    data[in_off..end].copy_from_slice(cmd);
    Ok(())
}

/// The FWSEC image for I6: [`fwsec_image`] plus the FRTS command for the
/// region at VRAM `frts_offset` (from `fb_layout`).
pub fn fwsec_frts_image(
    d: &FalconUcodeDescV3,
    ucode: &[u8],
    signature: &[u8],
    frts_offset: u64,
) -> GpuResult<Vec<u8>> {
    let mut img = fwsec_image(d, ucode, signature)?;
    let u = HsUcode::from_fwsec_v3(d)?;
    let (ds, de) = (
        u.data_offset as usize,
        (u.data_offset + u.dmem_size) as usize,
    );
    let sig = u.sig_dmem_addr as usize..(u.sig_dmem_addr + RSA3K_SIG_LEN) as usize;
    patch_fwsec_command(
        &mut img[ds..de],
        d.interface_offset,
        FWSEC_CMD_FRTS,
        &frts_command(frts_offset)?,
        sig,
    )?;
    Ok(img)
}

fn pad(img: &mut Vec<u8>) -> GpuResult<()> {
    let n = u32::try_from(img.len())
        .ok()
        .and_then(align_up)
        .ok_or(GpuError::InvalidArgument("imagine prea mare"))?;
    img.resize(n as usize, 0);
    Ok(())
}

/// An image the OS copied to physically contiguous system memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaImage {
    pub phys: u64,
    pub len: u32,
}

fn poll_dma<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
    done: impl Fn(u32) -> bool,
) -> GpuResult<()> {
    let off = e.base() + fv4::NV_PFALCON_FALCON_DMATRFCMD;
    let mut err = None;
    let r = poll_until(clock, DMA_US, 10, || match bar.read(off) {
        Ok(v) if done(v) => Some(()),
        Ok(_) => None,
        Err(x) => {
            err = Some(x);
            Some(())
        }
    });
    match err {
        Some(x) => Err(x),
        None => r,
    }
}

fn not_full(v: u32) -> bool {
    fv4::NV_PFALCON_FALCON_DMATRFCMD_FULL.get(v) == fv4::NV_PFALCON_FALCON_DMATRFCMD_FULL_FALSE
}

fn idle(v: u32) -> bool {
    fv4::NV_PFALCON_FALCON_DMATRFCMD_IDLE.get(v) == fv4::NV_PFALCON_FALCON_DMATRFCMD_IDLE_TRUE
}

/// `s_dmaTransfer_GA102`: base once, then one command per 256-byte block,
/// never writing while the request queue is full; wait for idle at the end.
#[allow(clippy::too_many_arguments)]
fn transfer<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
    mut dest: u32,
    mut mem_off: u32,
    src: u64,
    size: u32,
    cmd: u32,
) -> GpuResult<()> {
    let b = e.base();
    poll_dma(bar, clock, e, not_full)?;
    bar.write(b + fv4::NV_PFALCON_FALCON_DMATRFBASE, (src >> 8) as u32)?;
    bar.write(
        b + fv4::NV_PFALCON_FALCON_DMATRFBASE1,
        ((src >> 40) as u32) & 0x1ff,
    )?;
    let mut done = 0;
    while done < size {
        poll_dma(bar, clock, e, not_full)?;
        bar.write(
            b + fv4::NV_PFALCON_FALCON_DMATRFMOFFS,
            fv4::NV_PFALCON_FALCON_DMATRFMOFFS_OFFS.set(0, dest),
        )?;
        bar.write(b + fv4::NV_PFALCON_FALCON_DMATRFFBOFFS, mem_off)?;
        bar.write(b + fv4::NV_PFALCON_FALCON_DMATRFCMD, cmd)?;
        done += BLOCK;
        dest += BLOCK;
        mem_off += BLOCK;
    }
    poll_dma(bar, clock, e, idle)
}

/// Loads `u` from `image` into falcon `e` and programs the boot ROM; the
/// caller then starts the CPU (`falcon::run_to_halt`). The falcon must be
/// reset and on its falcon core (as `falcon::fwsec_frts` / `booter_load`
/// do before calling their `load` step).
pub fn dma_load<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
    u: &HsUcode,
    image: &DmaImage,
) -> GpuResult<()> {
    // All checks first: nothing is written if any of them fails.
    u.check()?;
    if image.len < u.size {
        return Err(GpuError::OutOfBounds {
            offset: 0,
            len: u64::from(u.size),
            limit: u64::from(image.len),
        });
    }
    let imem_src = u.imem_src(image)?;
    let (dmem_src, dmem_off) = u.dmem_src(image)?;
    let end = image.phys.checked_add(u64::from(image.len));
    if end.is_none_or(|x| x > MAX_DMA_ADDR) {
        return Err(GpuError::InvalidArgument("DMA: imaginea trece de 2^49"));
    }
    let b = e.base();
    let imem_blocks =
        fv4::NV_PFALCON_FALCON_HWCFG_IMEM_SIZE.get(bar.read(b + fv4::NV_PFALCON_FALCON_HWCFG)?);
    let imem_end = u64::from(u.imem_pa) + u64::from(align_up(u.imem_size).unwrap_or(u32::MAX));
    if imem_end > u64::from(imem_blocks) * u64::from(BLOCK) {
        return Err(GpuError::OutOfBounds {
            offset: u64::from(u.imem_pa),
            len: u64::from(u.imem_size),
            limit: u64::from(imem_blocks) * u64::from(BLOCK),
        });
    }

    // kflcnDisableCtxReq: physical addresses without a context, no block bind.
    let fb = b + FBIF_REL;
    let ctl = bar.read(fb + fbif::NV_PFALCON_FBIF_CTL)?;
    bar.write(
        fb + fbif::NV_PFALCON_FBIF_CTL,
        fbif::NV_PFALCON_FBIF_CTL_ALLOW_PHYS_NO_CTX
            .set(ctl, fbif::NV_PFALCON_FBIF_CTL_ALLOW_PHYS_NO_CTX_ALLOW),
    )?;
    bar.write(b + fv4::NV_PFALCON_FALCON_DMACTL, 0)?;
    // TRANSCFG(ctxDma): coherent system memory, physical addresses.
    let tc_off = fb + fbif::NV_PFALCON_FBIF_TRANSCFG(CTX_DMA);
    let mut tc = bar.read(tc_off)?;
    tc = fbif::NV_PFALCON_FBIF_TRANSCFG_TARGET
        .set(tc, fbif::NV_PFALCON_FBIF_TRANSCFG_TARGET_COHERENT_SYSMEM);
    tc = fbif::NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE
        .set(tc, fbif::NV_PFALCON_FBIF_TRANSCFG_MEM_TYPE_PHYSICAL);
    bar.write(tc_off, tc)?;

    let mut cmd =
        fv4::NV_PFALCON_FALCON_DMATRFCMD_WRITE.set(0, fv4::NV_PFALCON_FALCON_DMATRFCMD_WRITE_FALSE);
    cmd =
        fv4::NV_PFALCON_FALCON_DMATRFCMD_SIZE.set(cmd, fv4::NV_PFALCON_FALCON_DMATRFCMD_SIZE_256B);
    cmd = fv4::NV_PFALCON_FALCON_DMATRFCMD_CTXDMA.set(cmd, CTX_DMA);
    let imem_cmd = fv4::NV_PFALCON_FALCON_DMATRFCMD_SEC.set(
        fv4::NV_PFALCON_FALCON_DMATRFCMD_IMEM.set(cmd, fv4::NV_PFALCON_FALCON_DMATRFCMD_IMEM_TRUE),
        1,
    );
    transfer(
        bar,
        clock,
        e,
        u.imem_pa,
        u.imem_va,
        imem_src,
        u.imem_size,
        imem_cmd,
    )?;
    let mut dmem_cmd = cmd;
    if u.dmem_va != DMEM_VA_INVALID {
        dmem_cmd = fv4::NV_PFALCON_FALCON_DMATRFCMD_SET_DMTAG
            .set(dmem_cmd, fv4::NV_PFALCON_FALCON_DMATRFCMD_SET_DMTAG_TRUE);
    }
    transfer(
        bar,
        clock,
        e,
        u.dmem_pa,
        dmem_off,
        dmem_src,
        u.dmem_size,
        dmem_cmd,
    )?;

    // Boot ROM: where the signature is, which engine and ucode, RSA-3K.
    let r = e.riscv_base();
    bar.write(r + f2::NV_PFALCON2_FALCON_BROM_PARAADDR(0), u.sig_dmem_addr)?;
    bar.write(r + f2::NV_PFALCON2_FALCON_BROM_ENGIDMASK, u.engine_id_mask)?;
    bar.write(
        r + f2::NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID,
        f2::NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID_VAL.set(0, u32::from(u.ucode_id)),
    )?;
    bar.write(
        r + f2::NV_PFALCON2_FALCON_MOD_SEL,
        f2::NV_PFALCON2_FALCON_MOD_SEL_ALGO.set(0, f2::NV_PFALCON2_FALCON_MOD_SEL_ALGO_RSA3K),
    )?;
    bar.write(b + fv4::NV_PFALCON_FALCON_BOOTVEC, u.imem_va)?;
    bar.journal().push(Event::Step {
        name: "I6/I7 ucode HS incarcat prin DMA",
        ok: true,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::falcon::{booter_load_dma, fwsec_frts_dma};
    use super::*;
    use gpu_core::diag::Journal;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::{FakeGsp, GSP_BASE, SEC2_BASE, SimFault, SimHsImage, SimUcode};

    const PHYS: u64 = 0x1_2340_0000;
    const BOOTER_PHYS: u64 = 0x1_2380_0000;

    /// SIMULARE: synthetic bytes, never NVIDIA firmware.
    fn pattern(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31) ^ seed).collect()
    }

    fn fwsec_desc() -> FalconUcodeDescV3 {
        FalconUcodeDescV3 {
            stored_size: 0x1400,
            pkc_data_offset: 0x100,
            imem_phys_base: 0,
            imem_load_size: 0x1000,
            imem_virt_base: 0,
            dmem_phys_base: 0,
            dmem_load_size: 0x400,
            engine_id_mask: 0x400,
            ucode_id: 5,
            signature_count: 1,
            signature_versions: 1,
            ..Default::default()
        }
    }

    /// The FWSEC image in SIM system memory + what the "boot ROM" accepts.
    fn fwsec_board() -> (FakeGsp, HsUcode, DmaImage, Vec<u8>) {
        let d = fwsec_desc();
        let img = fwsec_image(&d, &pattern(0x1400, 0x11), &pattern(384, 0x5a)).unwrap();
        let u = HsUcode::from_fwsec_v3(&d).unwrap();
        let mut g = FakeGsp::new();
        g.sim_expect(SimHsImage {
            falcon: GSP_BASE,
            ucode: SimUcode::FwsecFrts,
            imem_pa: 0,
            code: img[..0x1000].to_vec(),
            dmem_pa: 0,
            data: img[0x1000..0x1400].to_vec(),
            sig_dmem_addr: 0x100,
            ucode_id: 5,
            engine_id_mask: 0x400,
            bootvec: 0,
        });
        let image = DmaImage {
            phys: PHYS,
            len: img.len() as u32,
        };
        g.sysmem = Some((PHYS, img.clone()));
        (g, u, image, img)
    }

    fn put(f: &mut [u8], o: usize, v: u32) {
        f[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// SIMULARE: a synthetic Booter file (layout as `fw_hdr::HsV2::parse`
    /// reads it): blob 0x2000 B at 0x1000; os code 0..0x400, os data
    /// 0x400..0xc00, one app (code) 0x1000..0x1800; 3 signatures of 384 B;
    /// patch_loc 0x500 (DMEM 0x100); meta = fuse 0, engine 0x4, ucode 2.
    fn booter_file() -> Vec<u8> {
        let mut f = pattern(0x3000, 0x77);
        for (i, v) in [0x10de, 1, 0x3000, 0x40, 0x1000, 0x2000].iter().enumerate() {
            put(&mut f, 4 * i, *v);
        }
        let h = 0x40;
        // sig_off, sig_size, loc_ptr, sig_ptr, meta_off, meta_size, num_sig_ptr, lh_off
        for (i, v) in [0x200, 3 * 384, 0x100, 0x104, 0x110, 12, 0x108, 0x120]
            .iter()
            .enumerate()
        {
            put(&mut f, h + 4 * i, *v);
        }
        put(&mut f, 0x100, 0x500); // patch_loc
        put(&mut f, 0x104, 0); // patch_sig
        put(&mut f, 0x108, 3); // num_sig (word 6 of the HS header points here)
        put(&mut f, 0x110, 0); // fuse_ver
        put(&mut f, 0x114, 0x4); // engine_id
        put(&mut f, 0x118, 2); // ucode_id
        for (i, v) in [0, 0x400, 0x400, 0x800, 1, 0x1000, 0x800]
            .iter()
            .enumerate()
        {
            put(&mut f, 0x120 + 4 * i, *v);
        }
        for k in 0..3 {
            f[0x200 + k * 384..0x200 + (k + 1) * 384].fill(0xc0 + k as u8);
        }
        f
    }

    fn add_booter(g: &mut FakeGsp, fuse: u32) -> (HsUcode, DmaImage) {
        let f = booter_file();
        let h = HsV2::parse(&f).unwrap();
        let u = HsUcode::from_booter(&h, &f).unwrap();
        let img = booter_image(&h, &f, fuse).unwrap();
        // What a correct loader puts there: the signature for fuse 0 (the
        // last one, counted from the end) at DMEM 0x100.
        let mut good = Vec::from(&f[0x1000..0x3000]);
        good[0x500..0x500 + 384].copy_from_slice(&f[0x200 + 2 * 384..0x200 + 3 * 384]);
        g.sim_expect(SimHsImage {
            falcon: SEC2_BASE,
            ucode: SimUcode::BooterLoad,
            imem_pa: 0,
            code: good[0x1000..0x1800].to_vec(),
            dmem_pa: 0,
            data: good[0x400..0xc00].to_vec(),
            sig_dmem_addr: 0x100,
            ucode_id: 2,
            engine_id_mask: 0x4,
            bootvec: 0x1000,
        });
        // Both images in one SIM system memory: FWSEC first, Booter after.
        let (base, mut bytes) = g.sysmem.take().unwrap();
        bytes.resize((BOOTER_PHYS - base) as usize, 0);
        bytes.extend_from_slice(&img);
        g.sysmem = Some((base, bytes));
        (
            u,
            DmaImage {
                phys: BOOTER_PHYS,
                len: img.len() as u32,
            },
        )
    }

    fn frts(g: &FakeGsp) -> (u64, u64) {
        g.frts
    }

    #[test]
    fn descriptors_follow_rm() {
        let u = HsUcode::from_fwsec_v3(&fwsec_desc()).unwrap();
        assert_eq!((u.size, u.code_offset, u.data_offset), (0x1400, 0, 0x1000));
        assert_eq!(
            (u.sig_dmem_addr, u.ucode_id, u.engine_id_mask),
            (0x100, 5, 0x400)
        );
        assert_eq!(u.dmem_va, DMEM_VA_INVALID);
        let f = booter_file();
        let h = HsV2::parse(&f).unwrap();
        let b = HsUcode::from_booter(&h, &f).unwrap();
        assert_eq!(
            (b.code_offset, b.imem_va, b.imem_size),
            (0x1000, 0x1000, 0x800)
        );
        assert_eq!(
            (b.data_offset, b.dmem_size, b.sig_dmem_addr),
            (0x400, 0x800, 0x100)
        );
        assert_eq!((b.ucode_id, b.engine_id_mask), (2, 4));
        // Signature choice counted from the end (RM).
        assert_eq!(booter_signature_index(3, 0).unwrap(), 2);
        assert_eq!(booter_signature_index(3, 2).unwrap(), 0);
        assert!(matches!(
            booter_signature_index(3, 3),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(booter_signature_index(1, 7).unwrap(), 0);
    }

    #[test]
    fn fwsec_then_booter_over_dma_simulated() {
        let (mut g, u, image, img) = fwsec_board();
        let (bu, bimage) = add_booter(&mut g, 0);
        let mut j = Journal::new(4096);
        let mut c = SimClock::default();
        let fr = frts(&g);
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            let w = fwsec_frts_dma(&mut b, &mut c, &u, &image, fr).unwrap();
            assert_eq!((w.start, w.end), fr);
            booter_load_dma(&mut b, &mut c, &bu, &bimage, 0x1_0000_0000).unwrap();
        }
        assert!(g.riscv_active());
        assert_eq!(g.booter_arg, Some(0x1_0000_0000));
        assert_eq!(g.dma_faults, 0);
        // 16 + 4 blocks for FWSEC, 8 + 8 for Booter.
        assert_eq!(g.dma_blocks, 36);
        // Every write inside GSP/SEC2 and journaled with before/after.
        assert!(g.writes.iter().all(|&(o, _)| {
            (0x110000..=0x113fff).contains(&o) || (0x840000..=0x843fff).contains(&o)
        }));
        assert_eq!(j.writes(), g.writes.len());
        // FWSEC's DMEM holds the patched signature at 0x100. (SEC2 was the
        // last falcon loaded; the GSP falcon still holds FWSEC: no reset.)
        assert_eq!(
            &g.dmem(GSP_BASE)[0x100..0x100 + 384],
            &pattern(384, 0x5a)[..]
        );
        assert_eq!(&g.imem(GSP_BASE)[..0x1000], &img[..0x1000]);
    }

    #[test]
    fn wrong_signature_or_brom_values_are_refused_by_the_simulated_rom() {
        // Booter signature for another fuse version: the ROM refuses.
        let (mut g, u, image, _) = fwsec_board();
        let (bu, bimage) = add_booter(&mut g, 1);
        let mut j = Journal::new(4096);
        let mut c = SimClock::default();
        let fr = frts(&g);
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        fwsec_frts_dma(&mut b, &mut c, &u, &image, fr).unwrap();
        assert!(matches!(
            booter_load_dma(&mut b, &mut c, &bu, &bimage, 0x1_0000_0000),
            Err(GpuError::HardwareFault { code: 0xdead })
        ));
        // Planted mistakes in the boot-ROM values: each one caught.
        for bad in [
            HsUcode {
                sig_dmem_addr: 0x104,
                ..u
            },
            HsUcode { ucode_id: 6, ..u },
            HsUcode {
                engine_id_mask: 0x200,
                ..u
            },
            HsUcode {
                dmem_pa: 0x100,
                ..u
            },
        ] {
            let (mut g, _, image, _) = fwsec_board();
            let fr = frts(&g);
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            assert!(matches!(
                fwsec_frts_dma(&mut b, &mut c, &bad, &image, fr),
                Err(GpuError::HardwareFault { code: 0 })
            ));
            assert!(!g.riscv_active());
        }
    }

    #[test]
    fn nothing_is_written_when_a_check_fails() {
        let cases: [(HsUcode, DmaImage); 5] = {
            let (_, u, image, _) = fwsec_board();
            [
                // Source not 256-aligned (the base register keeps addr >> 8).
                (
                    u,
                    DmaImage {
                        phys: PHYS + 0x40,
                        ..image
                    },
                ),
                // Image shorter than the ucode.
                (
                    u,
                    DmaImage {
                        len: 0x1000,
                        ..image
                    },
                ),
                // Above 2^49.
                (
                    u,
                    DmaImage {
                        phys: MAX_DMA_ADDR,
                        ..image
                    },
                ),
                // Signature outside the loaded DMEM.
                (
                    HsUcode {
                        sig_dmem_addr: 0x300,
                        ..u
                    },
                    image,
                ),
                // IMEM destination past the IMEM size reported by HWCFG.
                (
                    HsUcode {
                        imem_pa: 0xf800,
                        ..u
                    },
                    image,
                ),
            ]
        };
        for (u, image) in cases {
            let (mut g, _, _, _) = fwsec_board();
            let mut j = Journal::new(64);
            let mut c = SimClock::default();
            {
                let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
                assert!(dma_load(&mut b, &mut c, Engine::Gsp, &u, &image).is_err());
            }
            assert!(g.writes.is_empty(), "{u:?} {image:?}");
        }
    }

    #[test]
    fn dma_faults_are_reported_not_skipped() {
        for fault in [SimFault::DmaQueueFull, SimFault::DmaNeverIdle] {
            let (mut g, u, image, _) = fwsec_board();
            g.fault = fault;
            let fr = frts(&g);
            let mut j = Journal::new(4096);
            let mut c = SimClock::default();
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            assert!(matches!(
                fwsec_frts_dma(&mut b, &mut c, &u, &image, fr),
                Err(GpuError::Timeout { .. })
            ));
        }
        // The OS did not put the image in system memory: DMA reads nothing,
        // the ROM refuses, FWSEC creates no WPR2 — an error, not a success.
        let (mut g, u, image, _) = fwsec_board();
        g.sysmem = None;
        let fr = frts(&g);
        let mut j = Journal::new(4096);
        let mut c = SimClock::default();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            assert!(matches!(
                fwsec_frts_dma(&mut b, &mut c, &u, &image, fr),
                Err(GpuError::HardwareFault { code: 0 })
            ));
        }
        assert_eq!(g.dma_faults, 20);
    }

    /// SIMULARE: FWSEC with an interface table in its data (hand-written
    /// positions): header at DMEM 0x300 (2 entries), DMEM mapper at 0x340,
    /// command input buffer at 0x380 (0x40 bytes); signature at 0x100.
    fn fwsec_iface_desc() -> FalconUcodeDescV3 {
        FalconUcodeDescV3 {
            stored_size: 0x1600,
            dmem_load_size: 0x600,
            interface_offset: 0x300,
            ..fwsec_desc()
        }
    }

    fn fwsec_iface_ucode() -> Vec<u8> {
        let mut u = pattern(0x1600, 0x22);
        let d = 0x1000;
        u[d + 0x300..d + 0x304].copy_from_slice(&[1, 4, 8, 2]);
        put(&mut u, d + 0x304, 1);
        put(&mut u, d + 0x308, 0);
        put(&mut u, d + 0x30c, 4);
        put(&mut u, d + 0x310, 0x340);
        put(&mut u, d + 0x340 + 8, 0x380);
        put(&mut u, d + 0x340 + 12, 0x40);
        u
    }

    /// What the DMEM must hold after a correct patch, built by hand.
    fn fwsec_iface_expected(frts_at: u64) -> Vec<u8> {
        let mut data = fwsec_iface_ucode()[0x1000..0x1600].to_vec();
        put(&mut data, 0x340 + 44, 0x15);
        for (i, v) in [
            1,
            24,
            0,
            0,
            0,
            2,
            1,
            20,
            (frts_at >> 12) as u32,
            0x100,
            2,
            0,
        ]
        .iter()
        .enumerate()
        {
            put(&mut data, 0x380 + 4 * i, *v);
        }
        data[0x100..0x100 + 384].copy_from_slice(&pattern(384, 0x5a));
        data
    }

    fn fwsec_iface_board(img: Vec<u8>, expected_data: Vec<u8>) -> (FakeGsp, HsUcode, DmaImage) {
        let u = HsUcode::from_fwsec_v3(&fwsec_iface_desc()).unwrap();
        let mut g = FakeGsp::new();
        g.frts = (0, 0); // FWSEC must take the region from the command
        g.fwsec_mapper_at = Some(0x340);
        g.sim_expect(SimHsImage {
            falcon: GSP_BASE,
            ucode: SimUcode::FwsecFrts,
            imem_pa: 0,
            code: fwsec_iface_ucode()[..0x1000].to_vec(),
            dmem_pa: 0,
            data: expected_data,
            sig_dmem_addr: 0x100,
            ucode_id: 5,
            engine_id_mask: 0x400,
            bootvec: 0,
        });
        let image = DmaImage {
            phys: PHYS,
            len: img.len() as u32,
        };
        g.sysmem = Some((PHYS, img));
        (g, u, image)
    }

    const FRTS_AT: (u64, u64) = ((8 << 30) - (2 << 20), (8 << 30) - (1 << 20));

    #[test]
    fn fwsec_frts_command_creates_wpr2_where_asked_simulated() {
        let d = fwsec_iface_desc();
        let sig = pattern(384, 0x5a);
        let img = fwsec_frts_image(&d, &fwsec_iface_ucode(), &sig, FRTS_AT.0).unwrap();
        assert_eq!(&img[0x1000..0x1600], &fwsec_iface_expected(FRTS_AT.0)[..]);
        let (mut g, u, image) = fwsec_iface_board(img, fwsec_iface_expected(FRTS_AT.0));
        let mut j = Journal::new(4096);
        let mut c = SimClock::default();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let w = fwsec_frts_dma(&mut b, &mut c, &u, &image, FRTS_AT).unwrap();
        assert_eq!((w.start, w.end), FRTS_AT);
    }

    #[test]
    fn fwsec_without_or_with_a_wrong_command_is_an_error() {
        let d = fwsec_iface_desc();
        let sig = pattern(384, 0x5a);
        let mut j = Journal::new(4096);
        let mut c = SimClock::default();
        // Command for another region than the layout asks: the driver sees
        // WPR2 in the wrong place.
        let other = FRTS_AT.0 - (1 << 20);
        let img = fwsec_frts_image(&d, &fwsec_iface_ucode(), &sig, other).unwrap();
        let (mut g, u, image) = fwsec_iface_board(img, fwsec_iface_expected(other));
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        assert!(matches!(
            fwsec_frts_dma(&mut b, &mut c, &u, &image, FRTS_AT),
            Err(GpuError::VerifyFailed { .. })
        ));
        // No command patched (image accepted by the ROM, init_cmd not FRTS):
        // FWSEC reports an error code, never a success.
        let img = fwsec_image(&d, &fwsec_iface_ucode(), &sig).unwrap();
        let data = img[0x1000..0x1600].to_vec();
        let (mut g, u, image) = fwsec_iface_board(img, data);
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        assert!(matches!(
            fwsec_frts_dma(&mut b, &mut c, &u, &image, FRTS_AT),
            Err(GpuError::HardwareFault { code: 0xe1 })
        ));
    }

    #[test]
    fn broken_fwsec_interfaces_are_refused() {
        let cmd = frts_command(FRTS_AT.0).unwrap();
        let data = || fwsec_iface_ucode()[0x1000..0x1600].to_vec();
        let sig = 0x100..0x280;
        let mut ok = data();
        assert!(patch_fwsec_command(&mut ok, 0x300, FWSEC_CMD_FRTS, &cmd, sig.clone()).is_ok());
        let mut x = data();
        x[0x303] = 1; // one entry
        assert!(patch_fwsec_command(&mut x, 0x300, FWSEC_CMD_FRTS, &cmd, sig.clone()).is_err());
        let mut x = data();
        put(&mut x, 0x30c, 3); // no DMEM mapper
        assert!(patch_fwsec_command(&mut x, 0x300, FWSEC_CMD_FRTS, &cmd, sig.clone()).is_err());
        let mut x = data();
        put(&mut x, 0x340 + 12, 0x20); // input buffer too small
        assert!(patch_fwsec_command(&mut x, 0x300, FWSEC_CMD_FRTS, &cmd, sig.clone()).is_err());
        let mut x = data();
        put(&mut x, 0x340 + 8, 0x200); // input buffer over the signature
        assert!(patch_fwsec_command(&mut x, 0x300, FWSEC_CMD_FRTS, &cmd, sig.clone()).is_err());
        let mut x = data();
        put(&mut x, 0x340 + 8, 0x5f0); // past the data
        assert!(patch_fwsec_command(&mut x, 0x300, FWSEC_CMD_FRTS, &cmd, sig.clone()).is_err());
        let mut x = data();
        assert!(patch_fwsec_command(&mut x, 0x5fe, FWSEC_CMD_FRTS, &cmd, sig).is_err());
        assert!(frts_command(0).is_err());
        assert!(frts_command(0x1234).is_err());
    }

    #[test]
    fn broken_inputs_are_refused() {
        let d = fwsec_desc();
        assert!(fwsec_image(&d, &pattern(0x1000, 0), &pattern(384, 0)).is_err());
        assert!(fwsec_image(&d, &pattern(0x1400, 0), &pattern(100, 0)).is_err());
        assert!(
            HsUcode::from_fwsec_v3(&FalconUcodeDescV3 {
                pkc_data_offset: 0x300,
                ..d
            })
            .is_err()
        );
        assert!(
            HsUcode::from_fwsec_v3(&FalconUcodeDescV3 {
                imem_load_size: 0x1400,
                ..d
            })
            .is_err()
        );
        let mut f = booter_file();
        put(&mut f, 0x40 + 20, 8); // meta_size 8
        let h = HsV2::parse(&f).unwrap();
        assert!(HsUcode::from_booter(&h, &f).is_err());
        let mut f = booter_file();
        put(&mut f, 0x120 + 16, 2); // two apps
        put(&mut f, 0x120 + 28, 0x1800);
        put(&mut f, 0x120 + 32, 0x100);
        let h = HsV2::parse(&f).unwrap();
        assert!(HsUcode::from_booter(&h, &f).is_err());
        let f = booter_file();
        let h = HsV2 {
            patch_sig: 1,
            ..HsV2::parse(&f).unwrap()
        };
        assert!(booter_image(&h, &f, 0).is_err());
    }

    #[test]
    fn booter_with_the_shape_of_the_official_570_144_files_loads() {
        // C-39: header words as in the official files; meta and every byte
        // outside the headers are SIMULARE filler (gpu-sim::fw_stick).
        use gpu_sim::fw_stick::booter_570_144;
        for (load, imem, dmem) in [
            (true, 0x8900, (0x8a00, 0x6200)),
            (false, 0x4f00, (0x5000, 0x4d00)),
        ] {
            let f = booter_570_144(load);
            let h = HsV2::parse(&f).unwrap();
            let u = HsUcode::from_booter(&h, &f).unwrap();
            assert_eq!((u.code_offset, u.imem_size), (0x100, imem));
            assert_eq!((u.data_offset, u.dmem_size), dmem);
            assert_eq!(u.sig_dmem_addr, 0x10);
            // fuse 0 -> the last of the 2 signatures, copied at patch_loc.
            let img = booter_image(&h, &f, 0).unwrap();
            let at = h.patch_loc as usize;
            assert_eq!(&img[at..at + 384], &f[0x3c + 384..0x3c + 768]);
            assert!(img.len().is_multiple_of(256) && img.len() >= h.hdr.data_size as usize);
        }
    }
}
