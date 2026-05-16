"""
Django 与 ALL/Center_pipeline 之间的进程内桥接（同一 Python 进程可读内存 mesh）。

独立脚本跑 Center_pipeline 时，本桥接无法跨进程读内存，需改为 HTTP/IPC。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ALL_ROOT = Path(__file__).resolve().parents[2]
_all_str = str(_ALL_ROOT)
if _all_str not in sys.path:
    sys.path.insert(0, _all_str)

_center_mod: Any = None
_CENTER_IMPORT_ERROR: str | None = None


def _ensure_center() -> bool:
    global _center_mod, _CENTER_IMPORT_ERROR
    if _CENTER_IMPORT_ERROR is not None:
        return False
    if _center_mod is not None:
        return True
    try:
        import Center_pipeline as cp  # noqa: WPS433

        _center_mod = cp
        _CENTER_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _CENTER_IMPORT_ERROR = repr(exc)
        _center_mod = None
        return False


def _enrich_status_with_pipeline_debug(st: dict[str, Any]) -> dict[str, Any]:
    """把嵌套的 pipeline.last_error 提到顶层，便于浏览器 Network 里直接看到。"""
    if not isinstance(st, dict):
        return st
    pl = st.get("pipeline")
    if isinstance(pl, dict) and pl.get("last_error") is not None:
        st["pipeline_last_error"] = pl["last_error"]
    if isinstance(pl, dict) and not st.get("running", False):
        if pl.get("last_error") is not None:
            st["center_hint"] = (
                "采集线程已停：见 pipeline_last_error。常见：GPU ICP / CUDA 失败（可将 Pointcloud2mesh/config.py "
                "中 tracking_backend 改为 cpu_rgbd）、或深度相机被其他程序占用。"
                "开发请加 runserver --noreload。"
            )
        else:
            st["center_hint"] = (
                "采集已停止但未写入 pipeline.last_error：多为后台输入线程已退出或相机无帧。"
                "请用 runserver --noreload 单进程调试，并确认深度相机未被其他程序占用。"
            )
    return st


class CenterBridge:
    def start_mapping(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.start_pipeline(input_mode="scanner")
            if isinstance(st, dict):
                merged = {"ok": True, **st}
                return _enrich_status_with_pipeline_debug(merged)
            return st
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def stop_mapping(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.stop_pipeline()
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug({"ok": True, **st})
            return st
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def status(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.get_pipeline_status()
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug({"ok": True, **st})
            return {"ok": True, "detail": st}
        except Exception as exc:
            return {"ok": False, "running": False, "error": repr(exc)}

    def get_latest_mesh_snapshot(self, max_vertices: int = 60_000) -> dict[str, Any]:
        if not _ensure_center():
            return {
                "ok": False,
                "running": False,
                "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED",
                "mesh": None,
            }
        try:
            return _center_mod.get_latest_mesh_api_payload(max_vertices=max_vertices)
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "mesh": None}

    def get_latest_pose(self) -> dict[str, Any]:
        if not _ensure_center():
            return {
                "ok": False,
                "tracking_success": False,
                "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED",
            }
        try:
            pose = _center_mod.get_pose_latest()
            if isinstance(pose, dict):
                return {"ok": True, **pose}
            return {"ok": True, "detail": pose}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "tracking_success": False}

    def reset_center_reconstruction(self, **kwargs: Any) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.reset_center_reconstruction(**kwargs)
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug(st)
            return st
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "running": False}

    def resume_center_recovery(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.resume_center_recovery()
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug(st)
            return st
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "running": False}


center_bridge = CenterBridge()
