# Conda activate hook for PowerShell (same intent as .bat).
# Install:
#   New-Item -ItemType Directory -Force -Path "$env:CONDA_PREFIX\etc\conda\activate.d" | Out-Null
#   Copy-Item -Force "$PSScriptRoot\conda_activate_open3d_cuda_paths.ps1" "$env:CONDA_PREFIX\etc\conda\activate.d\"

$cuda124 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
if (Test-Path "$cuda124\bin") {
    $env:CUDA_PATH = $cuda124
    $env:CUDA_HOME = $cuda124
    $env:PATH = "$cuda124\bin;$cuda124\lib\x64;$env:PATH"
} else {
    Remove-Item Env:CUDA_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:CUDA_HOME -ErrorAction SilentlyContinue
}

$torchLib = Join-Path $env:CONDA_PREFIX 'Lib\site-packages\torch\lib'
if (Test-Path $torchLib) {
    $env:PATH = "$torchLib;$env:PATH"
}

$o3d = Join-Path $env:CONDA_PREFIX 'Lib\site-packages\open3d'
if (Test-Path $o3d) {
    $env:PATH = "$o3d;$env:PATH"
}

$condaLibBin = Join-Path $env:CONDA_PREFIX 'Library\bin'
if (Test-Path $condaLibBin) {
    $env:PATH = "$condaLibBin;$env:PATH"
}
