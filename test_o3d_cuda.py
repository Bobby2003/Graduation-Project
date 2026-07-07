"""
Smoke test: Open3D import and Open3D Core CUDA (CUDA:0).
Run inside the conda env: python test_o3d_cuda.py
"""
from __future__ import annotations

import platform
import sys


def main() -> int:
    print("Python:", sys.version.split()[0], "| executable:", sys.executable)
    print("Platform:", platform.platform())

    try:
        import open3d as o3d
    except ImportError as e:
        print("FAIL: cannot import open3d:", e)
        return 1

    print("open3d version:", o3d.__version__)

    try:
        import open3d.core as o3c
    except ImportError as e:
        print("FAIL: cannot import open3d.core:", e)
        return 1

    avail = o3c.cuda.is_available()
    print("open3d.core.cuda.is_available():", avail)

    try:
        n = o3c.cuda.device_count()
        print("open3d.core.cuda.device_count():", n)
    except Exception as e:
        print("device_count() error:", e)
        n = 0

    if not avail or n < 1:
        print(
            "NOTE: Open3D reports no CUDA device. Common causes:\n"
            "  - Wheel is CPU-only (official PyPI Windows wheels often have no CUDA).\n"
            "  - Wheel CUDA major != driver-supported runtime (use a matching custom build).\n"
            "  - Missing CUDA runtime DLLs on PATH (install NVIDIA driver; for toolkit apps set CUDA_PATH).\n"
        )
        return 2

    try:
        import numpy as np

        dev = o3c.Device("CUDA:0")
        t = o3c.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), device=dev)
        t = t + 1.0
        host = t.cpu().numpy()
        print("CUDA:0 tensor ok, cpu view:", host)
    except Exception as e:
        print("FAIL: CUDA:0 tensor op:", e)
        return 3

    print("OK: Open3D installed and CUDA:0 usable for open3d.core operations.")

    try:
        import torch

        print(
            "PyTorch (optional):",
            torch.__version__,
            "cuda=",
            torch.version.cuda,
            "is_available=",
            torch.cuda.is_available(),
        )
    except ImportError:
        print("PyTorch: not installed (optional; use cu124 index if you need GPU torch).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
