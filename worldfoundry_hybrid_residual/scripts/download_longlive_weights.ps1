param(
    [string]$DownloadRoot = "E:\Codex_work\ssh_experiment\external_sources\LongLive-1.3B"
)

$ErrorActionPreference = "Stop"

$resolvedRoot = [IO.Path]::GetFullPath($DownloadRoot)
$stage = Join-Path $resolvedRoot "transfer_parts"
$models = Join-Path $resolvedRoot "models"
New-Item -ItemType Directory -Force $stage, $models | Out-Null

$specs = @(
    @{
        Name = "longlive_base.pt"
        Url = "https://huggingface.co/Efficient-Large-Model/LongLive-1.3B/resolve/main/models/longlive_base.pt?download=true"
        Sha256 = "10a2aa8fcf89c77d9033f4c117405412a690e289625766619d293f0c5a208ee7"
    },
    @{
        Name = "lora.pt"
        Url = "https://huggingface.co/Efficient-Large-Model/LongLive-1.3B/resolve/main/models/lora.pt?download=true"
        Sha256 = "c4e43b87d62d4b0614b496773639f1ab170a7ee486dc23407901e9d3a5ebc07a"
    }
)

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$running = @()
foreach ($spec in $specs) {
    $destination = Join-Path $models $spec.Name
    if (Test-Path -LiteralPath $destination) {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant()
        if ($actual -ne $spec.Sha256) {
            throw "Existing $destination has an unexpected SHA-256."
        }
        Write-Host "[verified-existing] $($spec.Name)"
        continue
    }

    $part = Join-Path $stage "$($spec.Name).part"
    if (-not (Test-Path -LiteralPath $part)) {
        New-Item -ItemType File $part | Out-Null
    }
    $stdout = Join-Path $stage "$($spec.Name).$timestamp.stdout.log"
    $stderr = Join-Path $stage "$($spec.Name).$timestamp.stderr.log"
    $arguments = @(
        "-L", "--fail", "--retry", "20", "--retry-delay", "5",
        "--connect-timeout", "30", "--speed-time", "120",
        "--speed-limit", "1024", "-C", "-", "-o", $part, $spec.Url
    )
    $process = Start-Process -FilePath "$env:SystemRoot\System32\curl.exe" `
        -ArgumentList $arguments -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{ Spec = $spec; Part = $part; Process = $process }
    Write-Host "[started] $($spec.Name) pid=$($process.Id) bytes=$((Get-Item $part).Length)"
}

foreach ($item in $running) {
    $item.Process.WaitForExit()
    # The file hash is authoritative. Windows PowerShell can expose a null
    # ExitCode for an asynchronously started process even after WaitForExit.
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $item.Part).Hash.ToLowerInvariant()
    if ($actual -ne $item.Spec.Sha256) {
        throw "SHA-256 mismatch for $($item.Spec.Name): $actual"
    }

    $destination = [IO.Path]::GetFullPath((Join-Path $models $item.Spec.Name))
    $expectedPrefix = $resolvedRoot.TrimEnd('\') + '\'
    if (-not $destination.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to move outside download root: $destination"
    }
    Move-Item -LiteralPath $item.Part -Destination $destination
    Write-Host "[verified-complete] $($item.Spec.Name) $actual"
}

Get-ChildItem -LiteralPath $models -File | Select-Object Name, Length, LastWriteTime
