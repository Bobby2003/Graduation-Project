"""私人位面编辑器：房间模板 / 材质协议 / 灯光方案（配置存 UserProfile.private_realm_json）"""

from .realm_scan import is_web_simulator_reality_scan

PRIVATE_REALM_DEFAULTS = {
    "room_template": "basic_room",
    "material_pack": "cyber_neon",
    "lighting": "cyan_cold",
    "decorations": [],
}

ROOM_TEMPLATE_ALIASES = {
    "bedroom_small": "basic_room",
}

ROOM_TEMPLATE_CHOICES = [
    ("basic_room", "基础校准房"),
    ("bedroom_small", "旧版小房间（渲染同基础校准房）"),
    ("living_room", "客厅"),
    ("lab", "实验室"),
    ("underground_bunker", "地下基地"),
    ("starship_cabin", "星舰舱室"),
    ("wasteland_safehouse", "废土安全屋"),
    ("cyber_apartment", "赛博公寓"),
    ("magic_library", "魔法图书馆"),
]

MATERIAL_PACK_CHOICES = [
    ("cyber_neon", "赛博霓虹协议"),
    ("wasteland_rust", "废土铁锈滤镜"),
    ("starship_alloy", "星舰合金材质包"),
    ("magic_stone", "魔法石墙"),
    ("forest_temple", "森林神殿"),
    ("deep_sea", "深海基地"),
    ("pixel_retro", "像素复古"),
    ("hacker_matrix", "黑客矩阵"),
]

LIGHTING_CHOICES = [
    ("cyan_cold", "冷蓝工业光"),
    ("purple_neon", "霓虹紫"),
    ("warm_amber", "暖黄氛围"),
    ("alert_red", "警报红"),
    ("moonlit", "月光冷白"),
]


def normalize_room_template(value: str) -> str:
    """将旧 key / 别名规范为当前主模板 key（不写库，仅合并展示与前端注入）。"""
    if not value or not isinstance(value, str):
        return PRIVATE_REALM_DEFAULTS["room_template"]
    v = value.strip()
    return ROOM_TEMPLATE_ALIASES.get(v, v)


def merged_private_realm(profile) -> dict:
    base = dict(PRIVATE_REALM_DEFAULTS)
    raw = getattr(profile, "private_realm_json", None) or {}
    if isinstance(raw, dict):
        base.update(raw)
    dec = base.get("decorations")
    if not isinstance(dec, list):
        base["decorations"] = []
    for k in PRIVATE_REALM_DEFAULTS:
        if k not in base or base[k] is None:
            base[k] = PRIVATE_REALM_DEFAULTS[k]
    base["room_template"] = normalize_room_template(base.get("room_template", ""))
    if is_web_simulator_reality_scan(base.get("last_reality_scan")):
        base = dict(base)
        base.pop("last_reality_scan", None)
    return base
