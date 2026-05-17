"""用户界面偏好（存 UserProfile.private_realm_json['ui_settings']）。"""

from __future__ import annotations

from typing import Any

AVATAR_PRESETS = [
    "🎮",
    "🤖",
    "👾",
    "🛸",
    "⚡",
    "🔮",
    "🌌",
    "🦾",
    "🧿",
    "💠",
    "🎯",
    "🛡️",
]

DEFAULT_UI_SETTINGS: dict[str, Any] = {
    "avatar_emoji": "🎮",
    "depth_driver": "openni2",
    "gesture_sensitivity": 70,
    "depth_range": "100-3000",
    "fps_limit": 30,
    "theme": "cyberpunk",
    "hud_opacity": 85,
    "scanlines": True,
    "particles": False,
    "two_factor_enabled": False,
}

DEPTH_DRIVER_CHOICES = [
    ("openni2", "OpenNI 2.0 (推荐)"),
    ("native_sdk", "Native SDK"),
    ("directshow", "DirectShow"),
]

DEPTH_RANGE_CHOICES = [
    ("100-3000", "100 - 3000mm (默认)"),
    ("200-5000", "200 - 5000mm (远距离)"),
    ("50-1500", "50 - 1500mm (近距离)"),
]

THEME_CHOICES = [
    ("cyberpunk", "Cyberpunk (默认)"),
    ("dark_matrix", "Dark Matrix"),
    ("neon_night", "Neon Night"),
    ("minimal_gray", "Minimal Gray"),
]


def format_user_public_id(user) -> str:
    """稳定、可复现的展示用 ID（非安全令牌）。"""
    uid = int(getattr(user, "pk", None) or 0)
    a = uid & 0xFFFF
    b = (uid * 7919) & 0xFFFF
    c = (uid * 104729) & 0xFFFF
    return f"#{a:04X}-{b:04X}-{c:04X}"


def _raw_realm_dict(profile) -> dict:
    raw = getattr(profile, "private_realm_json", None) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def get_ui_settings(profile) -> dict[str, Any]:
    raw = _raw_realm_dict(profile)
    stored = raw.get("ui_settings")
    out = dict(DEFAULT_UI_SETTINGS)
    if isinstance(stored, dict):
        for key in DEFAULT_UI_SETTINGS:
            if key in stored and stored[key] is not None:
                out[key] = stored[key]
    emoji = out.get("avatar_emoji")
    if not emoji or emoji not in AVATAR_PRESETS:
        out["avatar_emoji"] = DEFAULT_UI_SETTINGS["avatar_emoji"]
    return out


def save_ui_settings(profile, patch: dict[str, Any]) -> dict[str, Any]:
    allowed = set(DEFAULT_UI_SETTINGS.keys())
    clean: dict[str, Any] = {}
    for key, val in patch.items():
        if key not in allowed:
            continue
        clean[key] = val

    if "avatar_emoji" in clean and clean["avatar_emoji"] not in AVATAR_PRESETS:
        clean["avatar_emoji"] = DEFAULT_UI_SETTINGS["avatar_emoji"]

    if "gesture_sensitivity" in clean:
        try:
            clean["gesture_sensitivity"] = max(0, min(100, int(clean["gesture_sensitivity"])))
        except (TypeError, ValueError):
            clean["gesture_sensitivity"] = DEFAULT_UI_SETTINGS["gesture_sensitivity"]

    if "hud_opacity" in clean:
        try:
            clean["hud_opacity"] = max(30, min(100, int(clean["hud_opacity"])))
        except (TypeError, ValueError):
            clean["hud_opacity"] = DEFAULT_UI_SETTINGS["hud_opacity"]

    if "fps_limit" in clean:
        try:
            clean["fps_limit"] = max(15, min(60, int(clean["fps_limit"])))
        except (TypeError, ValueError):
            clean["fps_limit"] = DEFAULT_UI_SETTINGS["fps_limit"]

    if "depth_driver" in clean:
        valid = {c[0] for c in DEPTH_DRIVER_CHOICES}
        if clean["depth_driver"] not in valid:
            clean["depth_driver"] = DEFAULT_UI_SETTINGS["depth_driver"]

    if "depth_range" in clean:
        valid = {c[0] for c in DEPTH_RANGE_CHOICES}
        if clean["depth_range"] not in valid:
            clean["depth_range"] = DEFAULT_UI_SETTINGS["depth_range"]

    if "theme" in clean:
        valid = {c[0] for c in THEME_CHOICES}
        if clean["theme"] not in valid:
            clean["theme"] = DEFAULT_UI_SETTINGS["theme"]

    if "scanlines" in clean:
        clean["scanlines"] = bool(clean["scanlines"])
    if "particles" in clean:
        clean["particles"] = bool(clean["particles"])
    if "two_factor_enabled" in clean:
        clean["two_factor_enabled"] = bool(clean["two_factor_enabled"])

    merged = get_ui_settings(profile)
    merged.update(clean)

    raw = _raw_realm_dict(profile)
    raw["ui_settings"] = merged
    profile.private_realm_json = raw
    profile.save(update_fields=["private_realm_json"])
    return merged
