[CmdletBinding()]
param([ValidateSet('check', 'disable', 'restore')][string]$Action = 'check')

# Explicitly approved one-device isolation. Never targets GPU display or motherboard audio.
$ErrorActionPreference = 'Stop'
$instance = 'PCI\VEN_10DE&DEV_22BA&SUBSYS_51021462&REV_A1\4&2106B648&0&0109'
$directory = 'B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11\matched-audio-isolation-v1'
$beforePath = Join-Path $directory 'audio-before.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Get-TargetState {
    $device = Get-PnpDevice -InstanceId $instance -ErrorAction Stop
    $hardware = (Get-PnpDeviceProperty -InstanceId $instance -KeyName 'DEVPKEY_Device_HardwareIds').Data
    if ($device.Class -ne 'System' -or -not ($hardware -match '^PCI\\VEN_10DE&DEV_22BA')) {
        throw 'Exact approved NVIDIA PCI audio controller identity/class mismatch.'
    }
    [pscustomobject]@{
        InstanceId = $device.InstanceId
        Class = $device.Class
        FriendlyName = $device.FriendlyName
        Status = $device.Status
        ProblemCode = [int](Get-PnpDeviceProperty -InstanceId $instance -KeyName 'DEVPKEY_Device_ProblemCode').Data
        UTC = [DateTime]::UtcNow.ToString('o')
        WindowsBootUTC = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
    }
}

if ($Action -eq 'check') {
    Get-TargetState | ConvertTo-Json -Depth 4
    exit 0
}
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator access is required for this approved device-state change.'
}
if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
    throw 'Approved diagnostic directory missing; no device changed.'
}
$current = Get-TargetState
if ($Action -eq 'disable') {
    if ($current.WindowsBootUTC -ne '2026-10-11T08:25:48.5000000Z' -or $current.ProblemCode -ne 0) {
        throw 'Reviewed boot or original enabled state changed; no device changed.'
    }
    if (Test-Path -LiteralPath $beforePath) {
        throw 'One-shot audio isolation already claimed; no automatic repeat.'
    }
    [IO.File]::WriteAllText($beforePath, ($current | ConvertTo-Json -Depth 4), $utf8)
    Disable-PnpDevice -InstanceId $instance -Confirm:$false -ErrorAction Stop
    $expected = 22
} else {
    $before = Get-Content -LiteralPath $beforePath -Raw | ConvertFrom-Json
    if ($before.InstanceId -ne $instance -or $before.ProblemCode -ne 0) {
        throw 'Original audio state does not authorize enabling this exact device.'
    }
    if ($current.ProblemCode -eq 22) {
        Enable-PnpDevice -InstanceId $instance -Confirm:$false -ErrorAction Stop
    } elseif ($current.ProblemCode -ne 0) {
        throw 'Unexpected device problem; refusing a blind device-state change.'
    }
    $expected = 0
}
for ($i = 0; $i -lt 20; $i++) {
    $after = Get-TargetState
    if ($after.ProblemCode -eq $expected) { break }
    Start-Sleep -Milliseconds 250
}
if ($after.ProblemCode -ne $expected) {
    throw "Device-state verification failed: expected problem-code$expected, got $($after.ProblemCode)."
}
$receiptPath = Join-Path $directory ("audio-" + $Action + ".json")
[IO.File]::WriteAllText($receiptPath, ($after | ConvertTo-Json -Depth 4), $utf8)
$after | ConvertTo-Json -Depth 4
