<#
K0 - inventarul workstation-ului, DOAR CITIRE, din Windows.

Ruleaza numai intr-o fereastra aprobata de Adrian (workstation-ul e al lui Kronos).
Nu opreste antrenarea, nu schimba nimic: citeste WMI/CIM, registrul PnP si `nvidia-smi -q`.
Rezultatul: un fisier text in rezultate\k0-<calculator>-<data>.txt (de copiat in docs/HARDWARE_WORKSTATION.md).

  powershell -ExecutionPolicy Bypass -File tools\k0-inventar-workstation.ps1
#>
param([string]$OutDir = (Join-Path $PSScriptRoot "..\rezultate"))

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$out = Join-Path $OutDir ("k0-{0}-{1}.txt" -f $env:COMPUTERNAME, $stamp)
$lines = New-Object System.Collections.Generic.List[string]
function W([string]$s) { $lines.Add($s) | Out-Null; Write-Host $s }

W "# K0 inventar ($stamp) - doar citire"
$os = Get-CimInstance Win32_OperatingSystem
W ("OS: {0} {1} build {2}" -f $os.Caption, $os.Version, $os.BuildNumber)
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
W ("CPU: {0} - {1} nuclee, {2} fire" -f $cpu.Name.Trim(), $cpu.NumberOfCores, $cpu.NumberOfLogicalProcessors)
$bb = Get-CimInstance Win32_BaseBoard
W ("Placa de baza: {0} {1}" -f $bb.Manufacturer, $bb.Product)
$bios = Get-CimInstance Win32_BIOS
W ("BIOS: {0} {1} ({2})" -f $bios.Manufacturer, $bios.SMBIOSBIOSVersion, $bios.ReleaseDate)
$ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
W ("RAM: {0:N1} GiB" -f ($ram / 1GB))

try { W ("Secure Boot: {0}" -f (Confirm-SecureBootUEFI)) } catch { W "Secure Boot: necunoscut (ruleaza ca administrator sau sistem non-UEFI)" }

try {
    $dg = Get-CimInstance -Namespace root\Microsoft\Windows\DeviceGuard -ClassName Win32_DeviceGuard -ErrorAction Stop
    W ("VBS stare: {0} (0=oprit, 1=configurat, 2=ruleaza)" -f $dg.VirtualizationBasedSecurityStatus)
    W ("Proprietati disponibile: {0} (3 = protectie DMA / IOMMU disponibila)" -f (($dg.AvailableSecurityProperties) -join ","))
} catch { W "DeviceGuard: indisponibil" }
W "IOMMU in BIOS (AMD-Vi / VT-d): DE NOTAT MANUAL din setup-ul BIOS"
W "Above 4G Decoding / Resizable BAR in BIOS: DE NOTAT MANUAL (si vezi BAR1 mai jos)"

W ""
W "## Placi video"
$gpus = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue
foreach ($g in $gpus) {
    W ("- {0} [{1}] stare {2}" -f $g.FriendlyName, $g.InstanceId, $g.Status)
    $loc = (Get-PnpDeviceProperty -InstanceId $g.InstanceId -KeyName DEVPKEY_Device_LocationInfo -ErrorAction SilentlyContinue).Data
    if ($loc) { W ("  locatie: {0}" -f $loc) }
    $pnp = Get-CimInstance Win32_PnPEntity -Filter ("PNPDeviceID='{0}'" -f ($g.InstanceId -replace '\\', '\\'))
    if ($pnp) {
        $res = Get-CimAssociatedInstance -InputObject $pnp -ResultClassName Win32_DeviceMemoryAddress -ErrorAction SilentlyContinue | Sort-Object StartingAddress -Unique
        foreach ($r in $res) {
            $start = [UInt64]$r.StartingAddress
            $end = [UInt64]$r.EndingAddress
            $size = $end - $start + 1
            W ("  memorie: {0:X} - {1:X} ({2:N0} MiB)" -f $start, $end, ($size / 1MB))
        }
        $io = Get-CimAssociatedInstance -InputObject $pnp -ResultClassName Win32_PortResource -ErrorAction SilentlyContinue | Sort-Object StartingAddress -Unique
        foreach ($p in $io) { W ("  I/O: {0:X} - {1:X}" -f [UInt64]$p.StartingAddress, [UInt64]$p.EndingAddress) }
    }
}

W ""
W "## nvidia-smi"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    & nvidia-smi --query-gpu=name,pci.bus_id,pci.device_id,pci.sub_device_id,vbios_version,driver_version,memory.total --format=csv | ForEach-Object { W $_ }
    # BAR1 total: 256 MiB = fara Resizable BAR, ~8192 MiB = ReBAR activ
    & nvidia-smi -q -d MEMORY,PCI | Select-String -Pattern "BAR1|Total|Link|Gen|Width|Bus Id|Device Id" | ForEach-Object { W ("  " + $_.Line.Trim()) }
} else {
    W "nvidia-smi nu exista pe acest calculator"
}

$lines | Set-Content -Encoding utf8 $out
Write-Host ""
Write-Host "Scris: $out"
