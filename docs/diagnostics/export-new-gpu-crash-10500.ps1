[CmdletBinding()]
param([switch]$CheckOnly)

# One local evidence copy only. No GPU work, uploads, settings changes or retries.
$ErrorActionPreference = 'Stop'
$source = 'C:\Windows\Minidump\101126-10500-01.dmp'
$directory = 'B:\DeSpaMo\diagnostics\gpu-crashes-2026-10-11'
$target = Join-Path $directory '101126-10500-01.dmp'
$manifest = Join-Path $directory 'manifest-10500.json'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$elevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($CheckOnly) {
    [pscustomobject]@{
        Elevated = $elevated
        Source = $source
        Target = $target
        DestinationExists = (Test-Path -LiteralPath $directory -PathType Container)
    } | ConvertTo-Json
    exit 0
}
if (-not $elevated) {
    throw 'Administrator access is required to read this protected dump.'
}
if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
    throw 'Expected existing B: diagnostics directory is unavailable.'
}

$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
if (Test-Path -LiteralPath $target) {
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $sourceHash) {
        throw 'Existing exported dump differs; refusing to overwrite it.'
    }
} else {
    Copy-Item -LiteralPath $source -Destination $target
}
if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $sourceHash) {
    throw 'Exported dump SHA256 does not match protected original.'
}
$receipt = [pscustomobject]@{
    Status = 'single_dump_exported_hash_verified'
    CollectedUTC = [DateTime]::UtcNow.ToString('o')
    Source = $source
    Copy = $target
    Bytes = (Get-Item -LiteralPath $target).Length
    SHA256 = $sourceHash.ToLowerInvariant()
    GPUWorkLaunched = $false
    DriverOrRegistryChanges = $false
    TrainingHoldReleased = $false
    DumpUploadPerformed = $false
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($manifest, ($receipt | ConvertTo-Json -Depth 4), $utf8)
$receipt | ConvertTo-Json -Depth 4
