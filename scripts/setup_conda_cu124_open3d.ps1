# One-shot setup: conda env (Python 3.10), PyTorch cu124, local Open3D wheel, requirements-conda-env.txt
#
# Usage (PowerShell, repo root optional):
#   .\scripts\setup_conda_cu124_open3d.ps1
#   .\scripts\setup_conda_cu124_open3d.ps1 -Open3dWheel "D:\pkgs\open3d-....whl"
#
# Wheel resolution order:
#   1) -Open3dWheel argument
#   2) $env:OPEN3D_WHEEL
#   3) First file matching repo\wheels\open3d*.whl
#   4) Repo parent directory: ..\open3d*.whl

param(
    [string] $EnvName = "graduation-cu124-o3d",
    [string] $Open3dWheel = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

function Resolve-Open3dWheel {
    param([string] $Explicit)
    if ($Explicit -and (Test-Path -LiteralPath $Explicit)) { return (Resolve-Path -LiteralPath $Explicit).Path }
    if ($env:OPEN3D_WHEEL -and (Test-Path -LiteralPath $env:OPEN3D_WHEEL)) {
        return (Resolve-Path -LiteralPath $env:OPEN3D_WHEEL).Path
    }
    $inWheels = Get-ChildItem -Path (Join-Path $RepoRoot "wheels") -Filter "open3d*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($inWheels) { return $inWheels.FullName }
    $parent = Split-Path -Parent $RepoRoot
    $sibling = Get-ChildItem -Path $parent -Filter "open3d*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($sibling) { return $sibling.FullName }
    return $null
}

$wheelPath = Resolve-Open3dWheel -Explicit $Open3dWheel
if (-not $wheelPath) {
    Write-Error @"
Could not find an Open3D .whl file. Do one of the following:
  - Copy your wheel to: $(Join-Path $RepoRoot 'wheels\')
  - Place it next to the repo folder (sibling of Graduation-Project): ..\open3d-*.whl
  - Or set environment variable OPEN3D_WHEEL to the full path
  - Or pass -Open3dWheel 'C:\path\to\open3d-....whl'
"@
}

Write-Host "Repo: $RepoRoot"
Write-Host "Open3D wheel: $wheelPath"

$pyExe = $null
$candidates = @(
    "$env:USERPROFILE\anaconda3\envs\$EnvName\python.exe",
    "$env:USERPROFILE\miniconda3\envs\$EnvName\python.exe"
)
foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c) { $pyExe = $c; break }
}
if (-not $pyExe) {
    Write-Host "Creating conda env: $EnvName (python=3.10)"
    conda create -n $EnvName python=3.10 pip -y
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { $pyExe = $c; break }
    }
}
if (-not $pyExe) {
    try {
        $condaBase = (conda info --base 2>$null).Trim()
        if ($condaBase) {
            $p = Join-Path $condaBase "envs\$EnvName\python.exe"
            if (Test-Path -LiteralPath $p) { $pyExe = $p }
        }
    } catch { }
}
if (-not $pyExe) {
    Write-Error "Could not locate python.exe for env $EnvName. Run: conda create -n $EnvName python=3.10 pip -y"
}

$env:PYTHONNOUSERSITE = "1"
Write-Host "Using: $pyExe"

Write-Host "Installing PyTorch CUDA 12.4 (large download)..."
& $pyExe -m pip install --upgrade pip
& $pyExe -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu124"

Write-Host "Installing Open3D from wheel..."
& $pyExe -m pip install --force-reinstall "$wheelPath"

$req = Join-Path $RepoRoot "requirements-conda-env.txt"
Write-Host "Installing $req ..."
& $pyExe -m pip install -r $req

Write-Host "Smoke test:"
& $pyExe (Join-Path $RepoRoot "test_o3d_cuda.py")
Write-Host "Done. Activate: conda activate $EnvName"
