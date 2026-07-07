# Check GPU driver, toolkit, and common CUDA-related variables.
# Windows: LD_LIBRARY_PATH is rarely used; DLL search uses PATH and CUDA_PATH.

Write-Host "=== nvidia-smi ===" -ForegroundColor Cyan
try {
    nvidia-smi
} catch {
    Write-Host "nvidia-smi failed or not in PATH: $_" -ForegroundColor Red
}

Write-Host "`n=== nvcc --version ===" -ForegroundColor Cyan
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    nvcc --version
} else {
    Write-Host "nvcc not found in PATH (CUDA Toolkit may be missing or not added to PATH)." -ForegroundColor Yellow
}

Write-Host "`n=== Environment variables ===" -ForegroundColor Cyan
foreach ($k in @("CUDA_HOME", "CUDA_PATH", "CUDA_PATH_V12_4", "LD_LIBRARY_PATH", "PATH")) {
    $v = [Environment]::GetEnvironmentVariable($k, "Process")
    if ($k -eq "PATH") {
        $short = if ($v.Length -gt 200) { $v.Substring(0, 200) + "..." } else { $v }
        Write-Host "${k}: $short"
    } else {
        Write-Host "${k}: $(if ($v) { $v } else { '<empty>' })"
    }
}

Write-Host "`n=== Typical Windows CUDA install path (v12.4) ===" -ForegroundColor Cyan
$cuda124 = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if (Test-Path $cuda124) {
    Write-Host "Found: $cuda124"
} else {
    Write-Host "Not found: $cuda124 (install CUDA Toolkit 12.4 if you need nvcc / CUDA_HOME)." -ForegroundColor Yellow
}
