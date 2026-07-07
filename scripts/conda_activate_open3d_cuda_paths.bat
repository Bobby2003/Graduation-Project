@echo off
rem =============================================================================
rem Conda activate hook: CUDA / Open3D DLL search path (Windows)
rem
rem Install (after: conda activate graduation-cu124-o3d):
rem   mkdir "%CONDA_PREFIX%\etc\conda\activate.d" 2>nul
rem   copy /Y "%~dp0conda_activate_open3d_cuda_paths.bat" "%CONDA_PREFIX%\etc\conda\activate.d\"
rem
rem Notes for this project:
rem   - CUDA Toolkit 12.4 may be absent; then nvcc stays unavailable but GPU
rem     wheels (PyTorch cu124, Open3D CUDA) can still use driver + bundled DLLs.
rem   - PyTorch ships cudart/cublas etc. under Lib\site-packages\torch\lib .
rem   - Open3D wheel often ships tbb*.dll next to Lib\site-packages\open3d .
rem   - LD_LIBRARY_PATH is ignored on Windows; do not rely on it here.
rem =============================================================================

set "CUDA_V12_4=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

if exist "%CUDA_V12_4%\bin" (
  set "CUDA_PATH=%CUDA_V12_4%"
  set "CUDA_HOME=%CUDA_V12_4%"
  set "PATH=%CUDA_V12_4%\bin;%CUDA_V12_4%\lib\x64;%PATH%"
) else (
  rem Toolkit not installed at default path (your machine was in this state).
  rem Clear stale values from parent shell if any:
  set "CUDA_PATH="
  set "CUDA_HOME="
)

if exist "%CONDA_PREFIX%\Lib\site-packages\torch\lib" (
  set "PATH=%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"
)

if exist "%CONDA_PREFIX%\Lib\site-packages\open3d" (
  set "PATH=%CONDA_PREFIX%\Lib\site-packages\open3d;%PATH%"
)

if exist "%CONDA_PREFIX%\Library\bin" (
  set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
)
