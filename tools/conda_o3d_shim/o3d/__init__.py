"""
Conda 侧「import o3d」与「import open3d」指向同一模块，并挂上 GPU 默认设备辅助函数。

安装：把整个 o3d 文件夹复制到当前环境的 site-packages 下，例如：
  copy /E /I tools\\conda_o3d_shim\\o3d "%CONDA_PREFIX%\\Lib\\site-packages\\o3d"

使用：
  import o3d
  dev = o3d.default_o3c_device()
  import open3d.core as o3c
  t = o3c.Tensor(..., device=dev)

前提：open3d 必须能正常 import（需完整 wheel，含 open3d.cpu）；本包不解决缺 cpu 的裁剪 wheel。
"""
from __future__ import annotations

import sys

import open3d as _open3d
import open3d.core as o3c


def default_o3c_device() -> o3c.Device:
    """优先 CUDA:0，否则 CPU:0。"""
    try:
        if o3c.cuda.is_available() and o3c.cuda.device_count() > 0:
            return o3c.Device("CUDA:0")
    except Exception:
        pass
    return o3c.Device("CPU:0")


# 挂到 open3d 模块上，这样 import o3d / import open3d 都能用 o3d.default_o3c_device
setattr(_open3d, "default_o3c_device", default_o3c_device)

# 让 `import o3d` 与 `import open3d` 为同一模块（geometry / core 等子模块不变）
sys.modules["o3d"] = _open3d
