"""统一 Reality Scan（last_reality_scan）服务端清洗与校验。"""
from __future__ import annotations

from typing import Any

# 与 API / 文档保持一致（views 引用这些常量）
MAX_WEB_SURFACES = 12
MAX_DEPTH_SURFACES = 64
MAX_RAW_REF_LEN = 300
MAX_META_STR_LEN = 120

RAW_REF_MAX_LEN = MAX_RAW_REF_LEN


def clamp_meta_str(value: Any, default: str, maxlen: int = 48) -> str:
    if not isinstance(value, str):
        return default
    s = value.strip()
    if not s:
        return default
    return s[:maxlen]


# 仅保留白名单字段，避免超大嵌套污染 JSONField
SURFACE_ALLOWED_KEYS = frozenset(
    {
        "id",
        "type",
        "confidence",
        "bounds",
        "center",
        "size",
        "normal",
        "material_pack",
    }
)

MESH_SUMMARY_ALLOWED_KEYS = frozenset({"vertices", "faces", "unit", "scale_unit"})

DEVICE_STRING_KEYS = frozenset({"type", "name", "serial"})
DEVICE_MAX_LEN = MAX_META_STR_LEN


def sanitize_raw_ref(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    return s[:RAW_REF_MAX_LEN]


def clean_surface(surface: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in SURFACE_ALLOWED_KEYS:
        if k not in surface:
            continue
        v = surface[k]
        if k == "id" and isinstance(v, str):
            out[k] = v.strip()[:128]
        elif k == "type" and isinstance(v, str):
            out[k] = v.strip()[:64]
        elif k == "material_pack" and isinstance(v, str):
            out[k] = v.strip()[:64]
        elif k == "confidence" and isinstance(v, (int, float)):
            out[k] = float(v)
        elif k == "bounds" and isinstance(v, dict):
            out[k] = _clean_bounds(v)
        elif k in ("center", "size", "normal") and isinstance(v, list):
            out[k] = _clean_num_list(v, 4 if k == "size" else 3)
    return out


def _clean_bounds(b: dict[str, Any]) -> dict[str, float]:
    r: dict[str, float] = {}
    for key in ("x", "y", "w", "h"):
        if key in b and isinstance(b[key], (int, float)):
            r[key] = float(b[key])
    return r


def _clean_num_list(arr: list[Any], max_len: int) -> list[float]:
    out: list[float] = []
    for i, x in enumerate(arr[:max_len]):
        if isinstance(x, (int, float)):
            out.append(float(x))
        else:
            out.append(0.0)
    while len(out) < max_len:
        out.append(0.0)
    return out


def clean_surfaces_list(surfaces: list[Any], *, max_items: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in surfaces[:max_items]:
        if isinstance(item, dict):
            cleaned = clean_surface(item)
            if not cleaned:
                continue
            if not (
                cleaned.get("type")
                or cleaned.get("bounds")
                or cleaned.get("center")
            ):
                continue
            out.append(cleaned)
    return out


def clean_mesh_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    r: dict[str, Any] = {}
    for k in MESH_SUMMARY_ALLOWED_KEYS:
        if k not in value:
            continue
        v = value[k]
        if k in ("vertices", "faces") and isinstance(v, int) and v >= 0:
            r[k] = v
        elif k in ("unit", "scale_unit") and isinstance(v, str):
            r[k] = v.strip()[:32]
    return r or None


def clean_device_metadata(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict) or not value:
        return None
    out: dict[str, str] = {}
    for k in DEVICE_STRING_KEYS:
        if k not in value:
            continue
        v = value[k]
        if isinstance(v, str):
            out[k] = v.strip()[:DEVICE_MAX_LEN]
        elif v is not None:
            out[k] = str(v)[:DEVICE_MAX_LEN]
    return out or None
