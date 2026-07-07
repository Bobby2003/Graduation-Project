# Install NVIDIA CUDA Toolkit 12.4.0 (Windows x64) using the official network installer.
# MUST run in an elevated PowerShell: right-click -> Run as administrator
#
# Silent:  toolkit installs without GUI (may still show brief console).
# If you prefer GUI, run the downloaded .exe without "-s".

$ErrorActionPreference = "Stop"
$url = "https://developer.download.nvidia.com/compute/cuda/12.4.0/network_installers/cuda_12.4.0_windows_network.exe"
$out = Join-Path $env:TEMP "cuda_12.4.0_windows_network.exe"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Run this script as Administrator (right-click PowerShell -> Run as administrator)." -ForegroundColor Red
    exit 1
}

Write-Host "Downloading: $url"
curl.exe -L --retry 3 -o $out $url
if (-not (Test-Path $out) -or (Get-Item $out).Length -lt 20MB) {
    throw "Download failed or file too small: $out"
}

Write-Host "Starting silent install (this can take several minutes)..."
$proc = Start-Process -FilePath $out -ArgumentList @("-s") -Wait -PassThru
Write-Host "Installer exit code: $($proc.ExitCode)"

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
if (Test-Path $nvcc) {
    Write-Host "OK: CUDA 12.4 found." -ForegroundColor Green
    & $nvcc --version
} else {
    Write-Host "nvcc not found at default path. Check installer log under `$env:TEMP or run the .exe without -s for GUI." -ForegroundColor Yellow
}
