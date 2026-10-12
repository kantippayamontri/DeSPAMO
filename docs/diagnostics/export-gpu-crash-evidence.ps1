[CmdletBinding()]
param(
    [string]$Destination = 'B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11',
    [switch]$CheckOnly
)

# Local CPU-only evidence collection. Does not launch CUDA, restart training,
# upload dumps, install software, modify drivers, or change registry settings.
$ErrorActionPreference = 'Stop'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$sources = @(
    'C:\WINDOWS\Minidump\101126-13859-01.dmp',
    'C:\WINDOWS\Minidump\101126-9468-01.dmp'
)

if ($CheckOnly) {
    [pscustomobject]@{
        Elevated = $elevated
        Destination = $Destination
        ArchiveDriveAvailable = (Test-Path -LiteralPath 'B:\DeSpaMo')
        ExpectedDumps = $sources
    } | ConvertTo-Json -Depth 4
    exit 0
}

if (-not $elevated) {
    throw 'Open Windows PowerShell as Administrator, then run this script again.'
}
if (-not (Test-Path -LiteralPath 'B:\DeSpaMo')) {
    throw 'Expected B:\DeSpaMo archive location is unavailable. No files copied.'
}
$fullDestination = [IO.Path]::GetFullPath($Destination)
if (-not $fullDestination.StartsWith('B:\DeSpaMo\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Evidence destination must be inside B:\DeSpaMo\.'
}
foreach ($source in $sources) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required dump missing or unreadable: $source"
    }
}
New-Item -ItemType Directory -Path $fullDestination -Force | Out-Null

$inventory = @()
foreach ($source in $sources) {
    $target = Join-Path $fullDestination ([IO.Path]::GetFileName($source))
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $sourceHash) {
            throw "Existing destination differs; preserve and inspect: $target"
        }
    } else {
        Copy-Item -LiteralPath $source -Destination $target
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $sourceHash) {
        throw "Copied dump failed SHA256 verification: $target"
    }
    $inventory += [pscustomobject]@{
        Source = $source
        Copy = $target
        Bytes = (Get-Item -LiteralPath $target).Length
        SHA256 = $sourceHash.ToLowerInvariant()
    }
}

$start = [datetime]::Parse('2026-10-11T00:00:00Z').ToLocalTime()
$ids = @(14, 153, 41, 6008, 1001, 1019, 4101, 161, 162)
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$start; Id=$ids} |
    Where-Object {$_.ProviderName -match 'nvlddmkm|Kernel-Power|WER-System|EventLog|Display|volmgr'} |
    ForEach-Object {
        [xml]$xml = $_.ToXml()
        [pscustomobject]@{
            UTC = $_.TimeCreated.ToUniversalTime().ToString('o')
            ID = $_.Id
            Provider = $_.ProviderName
            Message = $_.Message
            EventDataXML = $xml.Event.EventData.InnerXml
        }
    })

$os = Get-CimInstance Win32_OperatingSystem
$gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate)
$receipt = [pscustomobject]@{
    Status = 'local_evidence_exported_hash_verified'
    CollectedUTC = [datetime]::UtcNow.ToString('o')
    Dumps = $inventory
    WindowsVersion = $os.Version
    WindowsBuild = $os.BuildNumber
    LastBootUTC = $os.LastBootUpTime.ToUniversalTime().ToString('o')
    GPUs = $gpu
    GPUWorkLaunched = $false
    DriverOrRegistryChanges = $false
    TrainingHoldReleased = $false
    DumpUploadPerformed = $false
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $fullDestination 'manifest.json'),
    ($receipt | ConvertTo-Json -Depth 8), $utf8)
[IO.File]::WriteAllText((Join-Path $fullDestination 'system-events.json'),
    ($events | ConvertTo-Json -Depth 8), $utf8)

Write-Host "Local evidence ready: $fullDestination"
Write-Host 'Open each copied .dmp in WinDbg. Run !analyze -v and lmvm nvlddmkm.'
Write-Host 'Share MODULE_NAME, IMAGE_NAME, FAILURE_BUCKET_ID and STACK_TEXT, or save local analysis text.'
Write-Host 'Training and automatic recovery remain stopped.'
