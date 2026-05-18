"""
Django 与 ALL/Center_pipeline 之间的进程内桥接（同一 Python 进程可读内存 mesh）。

独立脚本跑 Center_pipeline 时，本桥接无法跨进程读内存，需改为 HTTP/IPC。
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote
from typing import Any

_ALL_ROOT = Path(__file__).resolve().parents[2]
_all_str = str(_ALL_ROOT)
if _all_str not in sys.path:
    sys.path.insert(0, _all_str)

_center_mod: Any = None
_CENTER_IMPORT_ERROR: str | None = None

# 与 urlpatterns 中 path 前缀一致（桥接不写 Django reverse，前端用同源相对路径）
_MATERIAL_SEGMENT_URL_PREFIX = "/api/center/material/segment"

_MATERIAL_NOT_READY = "MATERIAL_NOT_READY"


def _material_segment_model_url(segment_key: str, version: int | None) -> str:
    enc = quote(str(segment_key), safe="")
    base = f"{_MATERIAL_SEGMENT_URL_PREFIX}/{enc}/"
    if version is None:
        return base
    return f"{base}?version={int(version)}"


def _enrich_material_targets(payload: dict[str, Any]) -> dict[str, Any]:
    """为每个 segment 填入可经由 Django 下载的 model_url（GET segment）。"""
    if not isinstance(payload, dict):
        return payload
    version = payload.get("version")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        return payload
    for t in targets:
        if not isinstance(t, dict):
            continue
        sk = t.get("segment_key")
        if sk is None:
            continue
        t["model_url"] = _material_segment_model_url(str(sk), int(version) if version is not None else None)
    return payload


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

    def start_imu_tracking(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.start_imu_pipeline()
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug(st)
            return st
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def stop_imu_tracking(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "running": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.stop_imu_pipeline()
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug(st)
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

    def clear_published_material(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            st = _center_mod.clear_published_material()
            if isinstance(st, dict):
                return _enrich_status_with_pipeline_debug(st)
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

    def get_pose_latest(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            data = _center_mod.get_pose_latest()
            if isinstance(data, dict):
                return {"ok": True, **data}
            return {"ok": True, "detail": data}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def get_material_targets(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            payload = _center_mod.get_material_targets()
            if not isinstance(payload, dict):
                return {"ok": False, "error": "INVALID_MATERIAL_TARGETS"}
            version = payload.get("version")
            targets = payload.get("targets")
            if version is None or not isinstance(targets, list) or len(targets) == 0:
                merged = dict(payload)
                merged["ok"] = False
                merged["error"] = _MATERIAL_NOT_READY
                return merged
            enriched = _enrich_material_targets(payload)
            return {"ok": True, **enriched}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "targets": [], "version": None}

    def get_material_library(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED", "materials": []}
        try:
            data = _center_mod.get_material_library()
            if isinstance(data, dict):
                return {"ok": True, **data}
            return {"ok": True, "materials": data}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "materials": []}

    def get_material_status(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            data = _center_mod.get_material_status()
            if isinstance(data, dict):
                version = data.get("version")
                if version is None:
                    merged = dict(data)
                    merged["ok"] = False
                    merged["error"] = _MATERIAL_NOT_READY
                    return merged
                return {"ok": True, **data}
            return {"ok": True, "detail": data}
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "version": None, "status": {}}

    def get_material_scene_latest(self) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            data = _center_mod.get_material_scene_latest()
            if not isinstance(data, dict):
                return {"ok": False, "error": "INVALID_MATERIAL_SCENE_META"}
            status = data.get("status")
            if status == "not_ready":
                merged = dict(data)
                merged["ok"] = False
                merged["error"] = _MATERIAL_NOT_READY
                return merged
            return {"ok": True, **data}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def change_material(
        self,
        segment_key: str,
        material_id: str,
        tiling: float = 1.0,
    ) -> dict[str, Any]:
        if not _ensure_center():
            return {"ok": False, "error": _CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED"}
        try:
            result = _center_mod.change_material(segment_key, material_id, tiling=tiling)
            if isinstance(result, dict):
                return {"ok": True, **result}
            return {"ok": True, "detail": result}
        except RuntimeError as exc:
            err = str(exc)
            code = _MATERIAL_NOT_READY if "Material scene is not ready" in err else "CENTER_MATERIAL_RUNTIME_ERROR"
            return {"ok": False, "error": code, "message": err}
        except ValueError as exc:
            return {"ok": False, "error": "MATERIAL_CHANGE_FAILED", "message": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

    def get_material_scene_file_path(self, version: int | None = None) -> Path:
        if not _ensure_center():
            raise RuntimeError(_CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED")
        return Path(_center_mod.get_material_scene_file(version))

    def get_material_segment_file_path(self, segment_key: str, version: int | None = None) -> Path:
        if not _ensure_center():
            raise RuntimeError(_CENTER_IMPORT_ERROR or "CENTER_IMPORT_FAILED")
        return Path(_center_mod.get_material_segment_file(segment_key, version))


center_bridge = CenterBridge()
