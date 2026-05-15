"""
OpenNI2 进程级运行时：oniInitialize / oniShutdown 在整进程内只能配对一次。
Django 中 AR 扫描页与 Pointcloud2mesh 的 UnifiedDepthScanner 会各自加载 OrbbecCameraSDK；
若多个 Python 包装对象各自调用 oniShutdown，极易触发原生堆损坏并直接结束解释器进程。

本模块用单一引用计数，使任意数量的 OrbbecCameraSDK 实例共享同一运行时生命周期。
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.RLock()
_refcount = 0


def openni_runtime_attach(lib: Any, api_version: int, status_ok: int) -> bool:
    """
    在持锁情况下按需调用 oniInitialize，并增加引用计数。
    返回 True 表示运行时可用；失败时不会增加计数。
    """
    global _refcount
    with _lock:
        if _refcount == 0:
            print("初始化OpenNI SDK...")
            status = lib.oniInitialize(api_version)
            if int(status) != int(status_ok):
                print(f"❌ 初始化失败，错误码: {status}")
                return False
            print("✅ OpenNI SDK初始化成功")
        _refcount += 1
        return True


def openni_runtime_detach(lib: Any) -> None:
    """减少引用计数；仅当计数归零时调用 oniShutdown。"""
    global _refcount
    with _lock:
        if _refcount <= 0:
            return
        _refcount -= 1
        if _refcount == 0:
            lib.oniShutdown()
            print("✅ OpenNI SDK已关闭（进程级运行时已释放）")
        else:
            print(f"✅ OpenNI 运行时仍被占用 refcount={_refcount}，未调用 oniShutdown")
