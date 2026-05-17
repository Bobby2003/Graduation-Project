import json
import secrets
import time
from collections import defaultdict

from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages

from .forms import RealmRegisterForm
from django.db.models import Q, Sum
from django.http import FileResponse, HttpResponseForbidden, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_GET, require_POST

from .game_services import ensure_default_loadout, ensure_user_missions
from .progress_service import handle_progress_event
from .realm_scan import (
    MAX_DEPTH_SURFACES,
    MAX_WEB_SURFACES,
    clamp_meta_str,
    clean_device_metadata,
    clean_mesh_summary,
    clean_surfaces_list,
    clear_web_simulator_scan_from_profile,
    sanitize_raw_ref,
)
from .realm_catalog import (
    LIGHTING_CHOICES,
    MATERIAL_PACK_CHOICES,
    PRIVATE_REALM_DEFAULTS,
    ROOM_TEMPLATE_CHOICES,
    merged_private_realm,
)
from .center_bridge import center_bridge
from .user_settings import (
    AVATAR_PRESETS,
    DEPTH_DRIVER_CHOICES,
    DEPTH_RANGE_CHOICES,
    THEME_CHOICES,
    format_user_public_id,
    get_ui_settings,
    save_ui_settings,
)
from .models import (
    Achievement,
    EquipmentItem,
    MarketProduct,
    Mission,
    Realm,
    UserAchievement,
    UserEquippedItem,
    UserInventory,
    UserMission,
    UserProfile,
)

# 装备 icon 键 → 展示用符号（仅响应输出，不写入 MySQL）
EQUIP_ICON_MAP = {
    "sword": "\u2694",
    "dagger": "\U0001F5E1",
    "armor": "\U0001F6E1",
    "core": "\U0001F4A0",
    "mobility": "\U0001F300",
    "mask": "\U0001F3AD",
    "bow": "\U0001F3F9",
}

ACHIEVEMENT_CATEGORY_ICONS = {
    "realm_explorer": "\U0001F310",
    "combat_legend": "\u2694",
    "economy_lord": "\U0001F4B0",
    "collector": "\U0001F4E6",
    "sprint2_desktop": "\U0001F4BB",
    "sprint3_training": "\u2699",
    "reality_override": "\U0001F4F7",
}


def _new_reality_scan_id() -> str:
    return f"scan_{timezone.now().strftime('%Y%m%d')}_{secrets.token_hex(4)}"


def _merge_private_realm_core(
    profile: UserProfile,
    *,
    room_template: str,
    material_pack: str,
    lighting: str,
    decorations: list,
) -> None:
    """Merge core realm fields into private_realm_json without dropping extension keys."""
    raw = getattr(profile, "private_realm_json", None) or {}
    if not isinstance(raw, dict):
        raw = {}
    out = dict(raw)
    out["room_template"] = room_template
    out["material_pack"] = material_pack
    out["lighting"] = lighting
    out["decorations"] = decorations
    profile.private_realm_json = out


def _equip_icon(key: str) -> str:
    return EQUIP_ICON_MAP.get(key or "", "\u00B7")


def _mission_kind_class(kind: str) -> str:
    return {
        Mission.KIND_MAIN: "main",
        Mission.KIND_DAILY: "daily",
        Mission.KIND_WEEKLY: "weekly",
        Mission.KIND_SIDE: "side",
    }.get(kind, "side")


def _mission_btn(um: UserMission):
    if um.status == UserMission.STATUS_COMPLETED:
        return "COMPLETED", True
    if um.status == UserMission.STATUS_IN_PROGRESS:
        return "IN PROGRESS", False
    if um.mission.kind == Mission.KIND_MAIN:
        return "START", False
    return "AVAILABLE", False


# ── 注册 / 登录 / 仪表盘 ──────────────────────────────────────


def user_register(request):
    if request.method == "POST":
        form = RealmRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.get_or_create(
                user=user, defaults={"combat_power": 1000, "credits": 500}
            )
            ensure_user_missions(user)
            ensure_default_loadout(user)
            messages.success(request, "注册成功，请使用新账号登录。")
            return redirect("login")
        for err in form.non_field_errors():
            messages.error(request, err)
        for field in form:
            for err in field.errors:
                label = field.label or field.name
                messages.error(request, f"{label}: {err}")
    else:
        form = RealmRegisterForm()
    return render(request, "register.html", {"form": form})


def user_login(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            UserProfile.objects.get_or_create(user=user)
            ensure_user_missions(user)
            ensure_default_loadout(user)
            return redirect("dashboard")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


def user_logout(request):
    logout(request)
    return redirect("login")


@login_required
def dashboard(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    ensure_user_missions(request.user)
    ensure_default_loadout(request.user)
    realms = Realm.objects.all()
    inventory = UserInventory.objects.filter(user=request.user).select_related("loot")
    recent_achievements = (
        UserAchievement.objects.filter(user=request.user)
        .select_related("achievement")
        .order_by("-unlocked_at")[:5]
    )
    context = {
        "username": request.user.username,
        "profile": profile,
        "realms": realms,
        "inventory": inventory,
        "recent_achievements": recent_achievements,
    }
    return render(request, "user_center.html", context)


PROGRESS_HUB_TABS = frozenset(
    {"missions", "achievements", "equipment", "market", "ranking"}
)
WORLDS_HUB_TABS = frozenset({"plaza", "realm", "editor"})
GUIDE_HUB_TABS = frozenset({"tutorial", "training"})


def _hub_tab(request, allowed, default):
    tab = (request.GET.get("tab") or default).strip()
    if tab not in allowed:
        return default
    return tab


def _market_context():
    return {"products": MarketProduct.objects.filter(is_active=True)}


def _ranking_context():
    return {
        "top_profiles": UserProfile.objects.select_related("user").order_by(
            "-combat_power", "user__id"
        )[:50]
    }


def market(request):
    return redirect(reverse("progress_hub") + "?tab=market")


def ranking(request):
    return redirect(reverse("progress_hub") + "?tab=ranking")


def _missions_context(request):
    ensure_user_missions(request.user)
    user = request.user
    missions_qs = Mission.objects.filter(is_active=True).order_by("sort_order", "id")
    um_map = {
        um.mission_id: um
        for um in UserMission.objects.filter(user=user).select_related("mission")
    }

    rows_by_kind = defaultdict(list)
    for m in missions_qs:
        um = um_map.get(m.id)
        if um is None:
            um, _ = UserMission.objects.get_or_create(
                user=user,
                mission=m,
                defaults={
                    "progress": 0,
                    "status": UserMission.STATUS_AVAILABLE,
                },
            )
            um_map[m.id] = um
        btn_text, btn_done = _mission_btn(um)
        pct = um.progress_percent
        if m.target > 0:
            prog_text = f"{um.progress}/{m.target}"
        else:
            prog_text = "—"
        rows_by_kind[m.kind].append(
            {
                "mission": m,
                "um": um,
                "kind_class": _mission_kind_class(m.kind),
                "pct": pct,
                "prog_text": prog_text,
                "btn_text": btn_text,
                "btn_done": btn_done,
            }
        )

    section_meta = [
        (Mission.KIND_MAIN, "MAIN QUESTS // 主线任务"),
        (Mission.KIND_DAILY, "DAILY OPS // 日常任务"),
        (Mission.KIND_WEEKLY, "WEEKLY CONTRACTS // 周常契约"),
        (Mission.KIND_SIDE, "SIDE OPS // 支线任务"),
    ]
    mission_sections = [
        {"kind": k, "title": title, "rows": rows_by_kind[k]}
        for k, title in section_meta
        if rows_by_kind[k]
    ]

    daily_total = missions_qs.filter(kind=Mission.KIND_DAILY).count()
    daily_done = sum(
        1
        for m in missions_qs.filter(kind=Mission.KIND_DAILY)
        if um_map.get(m.id)
        and um_map[m.id].status == UserMission.STATUS_COMPLETED
    )
    all_um = UserMission.objects.filter(user=user).select_related("mission")
    completed_n = all_um.filter(status=UserMission.STATUS_COMPLETED).count()
    pending_credits = (
        all_um.filter(~Q(status=UserMission.STATUS_COMPLETED)).aggregate(
            t=Sum("mission__reward_credits")
        )["t"]
        or 0
    )
    day_pct = int(100 * daily_done / daily_total) if daily_total else 0

    stats = {
        "daily_total": daily_total,
        "completed_total": completed_n,
        "pending_credits": pending_credits,
        "daily_pct": day_pct,
    }
    return {"mission_sections": mission_sections, "stats": stats}


@login_required
def missions(request):
    return redirect(reverse("progress_hub") + "?tab=missions")


def _achievements_context(request):
    user = request.user
    achievements_qs = Achievement.objects.filter(is_active=True).order_by(
        "category_key", "sort_order", "id"
    )
    unlocked = set(
        UserAchievement.objects.filter(user=user).values_list(
            "achievement_id", flat=True
        )
    )
    ua_dates = {
        ua.achievement_id: ua.unlocked_at
        for ua in UserAchievement.objects.filter(user=user).select_related(
            "achievement"
        )
    }

    grouped = defaultdict(list)
    for a in achievements_qs:
        grouped[a.category_key].append(a)

    categories = []
    for key, items in grouped.items():
        cat_unlocked = sum(1 for a in items if a.id in unlocked)
        row_items = [
            {
                "a": a,
                "unlocked": a.id in unlocked,
                "unlocked_at": ua_dates.get(a.id),
            }
            for a in items
        ]
        categories.append(
            {
                "key": key,
                "title": items[0].category_title if items else key,
                "icon": ACHIEVEMENT_CATEGORY_ICONS.get(key, "\u2726"),
                "unlocked": cat_unlocked,
                "total": len(items),
                "items": row_items,
            }
        )
    categories.sort(key=lambda c: c["key"])

    total_n = achievements_qs.count()
    unlocked_n = len(unlocked)
    pct = int(100 * unlocked_n / total_n) if total_n else 0

    return {
        "achievement_categories": categories,
        "total_achievements": total_n,
        "unlocked_count": unlocked_n,
        "weekly_new_achievements": 0,
        "completion_pct": pct,
    }


@login_required
def achievements(request):
    return redirect(reverse("progress_hub") + "?tab=achievements")


@login_required
def progress_hub(request):
    tab = _hub_tab(request, PROGRESS_HUB_TABS, "missions")
    ctx = {
        "nav_active": "progress_hub",
        "hub_tab": tab,
    }
    ctx.update(_market_context())
    ctx.update(_ranking_context())
    if request.user.is_authenticated:
        ctx.update(_missions_context(request))
        ctx.update(_achievements_context(request))
        ctx.update(_equipment_context(request))
    else:
        ctx.update(
            {
                "mission_sections": [],
                "stats": {
                    "daily_total": 0,
                    "completed_total": 0,
                    "pending_credits": 0,
                    "daily_pct": 0,
                },
                "achievement_categories": [],
                "total_achievements": 0,
                "unlocked_count": 0,
                "completion_pct": 0,
                "slot_rows": [],
                "catalog_rows": [],
                "loadout_power": 0,
            }
        )
    return render(request, "progress_hub.html", ctx)


def _equipment_context(request):
    ensure_default_loadout(request.user)
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)

    equipped = {
        e.slot: e
        for e in UserEquippedItem.objects.filter(user=user).select_related("item")
    }
    slot_rows = []
    loadout_power = 0
    first_equipped = True
    for slot, label in EquipmentItem.SLOT_CHOICES:
        ue = equipped.get(slot)
        item = ue.item if ue else None
        if item:
            loadout_power += item.power_bonus
        is_active = item is not None and first_equipped
        if item is not None:
            first_equipped = False
        slot_rows.append(
            {
                "slot": slot,
                "label": label,
                "item": item,
                "icon": _equip_icon(item.icon) if item else "",
                "power": item.power_bonus if item else 0,
                "equipped": item is not None,
                "is_active": is_active,
            }
        )

    catalog = EquipmentItem.objects.all().order_by("sort_order", "id")
    catalog_rows = []
    for it in catalog:
        ue = equipped.get(it.slot)
        catalog_rows.append(
            {
                "item": it,
                "icon": _equip_icon(it.icon),
                "is_equipped": ue is not None and ue.item_id == it.id,
            }
        )

    detail = None
    for slot, _lbl in EquipmentItem.SLOT_CHOICES:
        ue = equipped.get(slot)
        if ue:
            detail = ue.item
            break

    detail_icon = _equip_icon(detail.icon) if detail else ""

    return {
        "profile": profile,
        "slot_rows": slot_rows,
        "catalog_rows": catalog_rows,
        "detail_item": detail,
        "detail_icon": detail_icon,
        "loadout_power": loadout_power,
    }


@login_required
def equipment(request):
    return redirect(reverse("progress_hub") + "?tab=equipment")


@login_required
@require_POST
def equipment_equip(request):
    """将图鉴中的装备写入对应槽位（含主武器 / 副武器切换）"""
    raw = request.POST.get("item_id")
    try:
        item_id = int(raw)
    except (TypeError, ValueError):
        return redirect(reverse("progress_hub") + "?tab=equipment")
    item = EquipmentItem.objects.filter(pk=item_id).first()
    if item is None:
        return redirect(reverse("progress_hub") + "?tab=equipment")
    UserEquippedItem.objects.update_or_create(
        user=request.user,
        slot=item.slot,
        defaults={"item": item},
    )
    loadout = (
        UserEquippedItem.objects.filter(user=request.user).aggregate(
            s=Sum("item__power_bonus")
        )["s"]
        or 0
    )
    UserProfile.objects.filter(user=request.user).update(
        combat_power=1000 + int(loadout)
    )
    return redirect(reverse("progress_hub") + "?tab=equipment")


@login_required
def settings_page(request):
    """Settings page — must not be named ``settings`` (shadows ``django.conf.settings``)."""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    ui = get_ui_settings(profile)
    return render(
        request,
        "settings.html",
        {
            "profile": profile,
            "user_public_id": format_user_public_id(request.user),
            "ui_settings": ui,
            "ui_settings_json": mark_safe(json.dumps(ui, ensure_ascii=False)),
            "avatar_presets_json": mark_safe(json.dumps(AVATAR_PRESETS, ensure_ascii=False)),
            "depth_driver_choices": DEPTH_DRIVER_CHOICES,
            "depth_range_choices": DEPTH_RANGE_CHOICES,
            "theme_choices": THEME_CHOICES,
        },
    )


@login_required
@require_POST
def settings_save_api(request):
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "invalid_json"}, status=400)
    if not isinstance(body, dict):
        return JsonResponse({"ok": False, "error": "invalid_body"}, status=400)

    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    ui = save_ui_settings(profile, body)
    return JsonResponse({"ok": True, "ui_settings": ui})


@login_required
@require_POST
def settings_change_password_api(request):
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "invalid_json"}, status=400)

    old_password = body.get("old_password") or ""
    new_password = body.get("new_password") or ""
    confirm = body.get("confirm_password") or ""

    if not request.user.check_password(old_password):
        return JsonResponse({"ok": False, "error": "当前密码不正确"}, status=400)
    if not new_password:
        return JsonResponse({"ok": False, "error": "新密码不能为空"}, status=400)
    if new_password != confirm:
        return JsonResponse({"ok": False, "error": "两次输入的新密码不一致"}, status=400)

    try:
        validate_password(new_password, request.user)
    except ValidationError as exc:
        return JsonResponse(
            {"ok": False, "error": "；".join(exc.messages)},
            status=400,
        )

    request.user.set_password(new_password)
    request.user.save(update_fields=["password"])
    update_session_auth_hash(request, request.user)
    return JsonResponse({"ok": True})


@login_required
@require_GET
def settings_export_api(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    ui = get_ui_settings(profile)
    inventory = list(
        UserInventory.objects.filter(user=request.user)
        .select_related("loot")
        .values("quantity", "loot__name", "loot__rarity")
    )
    achievements = list(
        UserAchievement.objects.filter(user=request.user)
        .select_related("achievement")
        .values("unlocked_at", "achievement__name", "achievement__code")
    )
    payload = {
        "exported_at": timezone.now().isoformat(),
        "user": {
            "username": request.user.username,
            "email": request.user.email,
            "public_id": format_user_public_id(request.user),
        },
        "profile": {
            "level": profile.level,
            "exp": profile.exp,
            "combat_power": profile.combat_power,
            "credits": profile.credits,
            "display_title": profile.display_title,
            "realm_visibility": profile.realm_visibility,
        },
        "ui_settings": ui,
        "private_realm": merged_private_realm(profile),
        "inventory": inventory,
        "achievements": achievements,
    }
    response = JsonResponse(payload, json_dumps_params={"ensure_ascii": False, "indent": 2})
    filename = f"ar_realm_export_{request.user.username}.json"
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@login_required
@require_POST
def settings_delete_account_api(request):
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "invalid_json"}, status=400)

    password = body.get("password") or ""
    confirm_username = (body.get("confirm_username") or "").strip()

    if confirm_username != request.user.username:
        return JsonResponse({"ok": False, "error": "用户名确认不一致"}, status=400)
    if not request.user.check_password(password):
        return JsonResponse({"ok": False, "error": "密码不正确"}, status=400)

    user = request.user
    logout(request)
    user.delete()
    return JsonResponse({"ok": True, "redirect": reverse("login")})


def _desktop_realm_bootstrap_context(request, force_room_template=None):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    clear_web_simulator_scan_from_profile(profile)
    realm_quests = Mission.objects.filter(
        kind=Mission.KIND_MAIN, is_active=True
    ).order_by("sort_order", "id")[:5]
    realm_cfg = merged_private_realm(profile)
    if force_room_template:
        realm_cfg = dict(realm_cfg)
        realm_cfg["room_template"] = force_room_template
    private_realm_json = mark_safe(json.dumps(realm_cfg, ensure_ascii=False))
    return {
        "profile": profile,
        "realm_quests": realm_quests,
        "private_realm_json": private_realm_json,
        "debug_allowed": settings.DEBUG or getattr(request.user, "is_staff", False),
    }


@login_required
def realm(request):
    return render(request, "realm.html", _desktop_realm_bootstrap_context(request))


@login_required
def training(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    raw = getattr(profile, "private_realm_json", None) or {}
    if isinstance(raw, dict) and raw.get("room_template"):
        persistent_room = raw.get("room_template")
    else:
        persistent_room = merged_private_realm(profile).get("room_template") or "bedroom_small"
    ctx = _desktop_realm_bootstrap_context(request, force_room_template="basic_room")
    ctx["persistent_room_template"] = persistent_room
    return render(request, "training.html", ctx)


@login_required
@require_POST
def realm_save_api(request):
    """JSON API：更新 private_realm_json（供 Three.js 桌面位面保存）"""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "INVALID_JSON"}, status=400)

    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "INVALID_PAYLOAD"}, status=400)

    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    valid_templates = {c[0] for c in ROOM_TEMPLATE_CHOICES}
    valid_materials = {c[0] for c in MATERIAL_PACK_CHOICES}
    valid_light = {c[0] for c in LIGHTING_CHOICES}

    current = dict(merged_private_realm(profile))

    if "room_template" in payload:
        v = payload["room_template"]
        if v not in valid_templates:
            return JsonResponse({"ok": False, "error": "INVALID_ROOM_TEMPLATE"}, status=400)
        current["room_template"] = v

    mat = payload.get("material_pack")
    if mat is None:
        mat = payload.get("material_protocol")
    if mat is not None:
        if mat not in valid_materials:
            return JsonResponse({"ok": False, "error": "INVALID_MATERIAL_PACK"}, status=400)
        current["material_pack"] = mat

    if "lighting" in payload:
        v = payload["lighting"]
        if v not in valid_light:
            return JsonResponse({"ok": False, "error": "INVALID_LIGHTING"}, status=400)
        current["lighting"] = v

    if "decorations" in payload:
        dec = payload["decorations"]
        if not isinstance(dec, list):
            return JsonResponse({"ok": False, "error": "INVALID_DECORATIONS"}, status=400)
        current["decorations"] = dec

    _merge_private_realm_core(
        profile,
        room_template=current["room_template"],
        material_pack=current["material_pack"],
        lighting=current["lighting"],
        decorations=current["decorations"],
    )
    profile.save(update_fields=["private_realm_json"])
    return JsonResponse({"ok": True, "realm": merged_private_realm(profile)})


@login_required
@require_POST
def center_start_api(request):
    data = center_bridge.start_mapping()
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def center_start_imu_api(request):
    """AR/VR 沉浸漫游：仅启动 IMU 位姿，不打开深度相机。"""
    data = center_bridge.start_imu_tracking()
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def center_stop_imu_api(request):
    data = center_bridge.stop_imu_tracking()
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def center_stop_api(request):
    data = center_bridge.stop_mapping()
    return JsonResponse(data, safe=False)


@login_required
@require_GET
def center_status_api(request):
    data = center_bridge.status()
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def center_resume_api(request):
    data = center_bridge.resume_center_recovery()
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def center_reset_reconstruction_api(request):
    data = center_bridge.reset_center_reconstruction()
    return JsonResponse(data, safe=False)


@login_required
@require_GET
def center_latest_mesh_api(request):
    try:
        max_v = int(request.GET.get("max_vertices", 60000))
    except (TypeError, ValueError):
        max_v = 60000
    max_v = max(4096, min(max_v, 250_000))
    data = center_bridge.get_latest_mesh_snapshot(max_vertices=max_v)
    return JsonResponse(data, safe=False)


@login_required
@require_GET
def center_pose_latest_api(request):
    data = center_bridge.get_pose_latest()
    return JsonResponse(data, safe=False)


def _center_material_version_param(request) -> tuple[int | None, str | None]:
    """Returns (version_or_none_for_latest, None) or (_, error_code)."""
    raw = request.GET.get("version")
    if raw is None or raw == "":
        return None, None
    try:
        return int(raw), None
    except (TypeError, ValueError):
        return None, "INVALID_VERSION"


@login_required
@require_GET
def center_material_targets_api(request):
    data = center_bridge.get_material_targets()
    return JsonResponse(data, safe=False)


@login_required
@require_GET
def center_material_library_api(request):
    data = center_bridge.get_material_library()
    return JsonResponse(data, safe=False)


@login_required
@require_GET
def center_material_status_api(request):
    data = center_bridge.get_material_status()
    return JsonResponse(data, safe=False)


@login_required
@require_GET
def center_material_scene_latest_api(request):
    data = center_bridge.get_material_scene_latest()
    return JsonResponse(data, safe=False)


@login_required
@require_POST
def center_material_change_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "INVALID_JSON"}, status=400)

    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "INVALID_PAYLOAD"}, status=400)

    segment_key = payload.get("segment_key")
    material_id = payload.get("material_id")
    if not isinstance(segment_key, str) or not segment_key.strip():
        return JsonResponse({"ok": False, "error": "INVALID_SEGMENT_KEY"}, status=400)
    if not isinstance(material_id, str) or not material_id.strip():
        return JsonResponse({"ok": False, "error": "INVALID_MATERIAL_ID"}, status=400)

    tiling_raw = payload.get("tiling", 1.0)
    try:
        tiling = float(tiling_raw)
    except (TypeError, ValueError):
        return JsonResponse({"ok": False, "error": "INVALID_TILING"}, status=400)

    data = center_bridge.change_material(segment_key.strip(), material_id.strip(), tiling=tiling)
    status_code = 200
    if data.get("ok") is False:
        if data.get("error") == "MATERIAL_CHANGE_FAILED":
            status_code = 400
    return JsonResponse(data, safe=False, status=status_code)


@login_required
@require_GET
def center_material_scene_glb(request):
    ver, ver_err = _center_material_version_param(request)
    if ver_err:
        return JsonResponse({"ok": False, "error": ver_err}, status=400)
    try:
        path = center_bridge.get_material_scene_file_path(ver)
    except (KeyError, FileNotFoundError) as exc:
        return JsonResponse(
            {"ok": False, "error": "MATERIAL_SCENE_FILE_NOT_FOUND", "message": str(exc)},
            status=404,
        )
    except RuntimeError as exc:
        return JsonResponse(
            {"ok": False, "error": "CENTER_IMPORT_FAILED", "message": str(exc)},
            status=503,
        )

    if not path.is_file():
        return JsonResponse({"ok": False, "error": "MATERIAL_SCENE_FILE_MISSING"}, status=404)

    name = f"center_material_scene_{ver or 'latest'}.glb"
    return FileResponse(path.open("rb"), as_attachment=False, filename=name, content_type="model/gltf-binary")


@login_required
@require_GET
def center_material_segment_download(request, segment_key: str):
    ver, ver_err = _center_material_version_param(request)
    if ver_err:
        return JsonResponse({"ok": False, "error": ver_err}, status=400)
    try:
        path = center_bridge.get_material_segment_file_path(segment_key, ver)
    except RuntimeError as exc:
        return JsonResponse(
            {"ok": False, "error": "CENTER_UNAVAILABLE", "message": str(exc)},
            status=503,
        )
    except ValueError as exc:
        return JsonResponse(
            {"ok": False, "error": "INVALID_SEGMENT_KEY", "message": str(exc)},
            status=400,
        )
    except KeyError as exc:
        return JsonResponse(
            {"ok": False, "error": "MATERIAL_SEGMENT_NOT_FOUND", "message": str(exc)},
            status=404,
        )

    if not path.is_file():
        return JsonResponse({"ok": False, "error": "MATERIAL_SEGMENT_FILE_MISSING"}, status=404)

    safe_name = segment_key.replace("/", "__") + ".glb"
    return FileResponse(path.open("rb"), as_attachment=False, filename=safe_name, content_type="model/gltf-binary")


@login_required
@require_POST
def progress_event_api(request):
    """Sprint 2：客户端上报位面行为事件 → 任务 / 成就 / 统计"""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "INVALID_JSON"}, status=400)

    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "INVALID_PAYLOAD"}, status=400)

    event = payload.get("event")
    event_payload = payload.get("payload") or {}
    if not isinstance(event_payload, dict):
        return JsonResponse({"ok": False, "error": "INVALID_PAYLOAD"}, status=400)

    if not isinstance(event, str) or not event:
        return JsonResponse({"ok": False, "error": "INVALID_EVENT"}, status=400)

    result = handle_progress_event(request.user, event, event_payload)
    status = 200 if result.get("ok") else 400
    return JsonResponse(result, status=status)


@login_required
def reality_override(request):
    """现实覆写页已下线：重定向至绿洲（浏览器模拟扫描入口已隐藏）。"""
    return redirect("realm")


@login_required
@require_POST
def reality_override_save_api(request):
    """浏览器模拟扫描已下线，不再写入 last_reality_scan。"""
    return JsonResponse(
        {"ok": False, "error": "REALITY_OVERRIDE_DISABLED"},
        status=410,
    )


@login_required
@require_POST
def depth_scan_import_api(request):
    """深度相机 / 扫描器管线：写入统一 last_reality_scan，不覆盖 room_template / lighting / decorations。"""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "INVALID_JSON"}, status=400)

    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "error": "INVALID_PAYLOAD"}, status=400)

    surfaces = payload.get("surfaces", [])
    if not isinstance(surfaces, list):
        return JsonResponse({"ok": False, "error": "INVALID_SURFACES"}, status=400)
    surfaces_clean = clean_surfaces_list(surfaces, max_items=MAX_DEPTH_SURFACES)

    mesh_summary = clean_mesh_summary(payload.get("mesh_summary"))

    valid_materials = dict(MATERIAL_PACK_CHOICES)
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    raw = getattr(profile, "private_realm_json", None) or {}
    if not isinstance(raw, dict):
        raw = {}
    out = dict(raw)

    default_mat = out.get("material_pack") or PRIVATE_REALM_DEFAULTS["material_pack"]
    material_pack = payload.get("material_pack") or default_mat
    if material_pack not in valid_materials:
        return JsonResponse({"ok": False, "error": "INVALID_MATERIAL_PACK"}, status=400)

    scan_id = payload.get("id") if isinstance(payload.get("id"), str) else None
    if scan_id:
        scan_id = scan_id.strip()[:80]
    else:
        scan_id = _new_reality_scan_id()

    scan: dict = {
        "id": scan_id,
        "mode": clamp_meta_str(payload.get("mode"), "sensor", 32),
        "source": clamp_meta_str(payload.get("source"), "depth_camera", 48),
        "pipeline": clamp_meta_str(
            payload.get("pipeline"), "depth_reconstruction_v1", 64
        ),
        "captured_at": timezone.now().isoformat(),
        "material_pack": material_pack,
        "coordinate_space": clamp_meta_str(
            payload.get("coordinate_space"), "metric_room", 32
        ),
        "surfaces": surfaces_clean,
        "mesh_summary": mesh_summary,
        "raw_ref": sanitize_raw_ref(payload.get("raw_ref")),
    }
    dev = clean_device_metadata(payload.get("device"))
    if dev:
        scan["device"] = dev

    out["last_reality_scan"] = scan
    out["material_pack"] = material_pack
    profile.private_realm_json = out
    profile.save(update_fields=["private_realm_json"])

    progress_result = handle_progress_event(
        request.user,
        "reality_scan_completed",
        {
            "mode": scan["mode"],
            "source": scan["source"],
            "surface_count": len(surfaces_clean),
            "pipeline": scan["pipeline"],
        },
    )

    return JsonResponse(
        {
            "ok": True,
            "last_reality_scan": scan,
            "realm": merged_private_realm(profile),
            "progress": progress_result,
        }
    )


@login_required
def depth_scan_tester(request):
    """Sprint 6C：无真实设备时调试 metric 导入（仅 DEBUG 或 staff）。"""
    if not (settings.DEBUG or request.user.is_staff):
        return HttpResponseForbidden(
            "Depth Scan Import Tester is only available when DEBUG=True or for staff users."
        )
    return render(
        request,
        "depth_scan_tester.html",
        {
            "nav_active": "depth_scan_tester",
            "debug_allowed": True,
        },
    )


@login_required
@require_POST
def reality_scan_clear_api(request):
    """移除 last_reality_scan，不修改 room_template / material_pack / lighting / decorations。"""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    raw = getattr(profile, "private_realm_json", None) or {}
    if not isinstance(raw, dict):
        raw = {}
    realm = dict(raw)
    realm.pop("last_reality_scan", None)
    profile.private_realm_json = realm
    profile.save(update_fields=["private_realm_json"])
    return JsonResponse({"ok": True, "realm": merged_private_realm(profile)})


def _my_realm_context(request):
    """私人位面总览（配置存 MySQL UserProfile）"""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    clear_web_simulator_scan_from_profile(profile)
    realm_config = merged_private_realm(profile)
    last_scan = (
        realm_config.get("last_reality_scan")
        if isinstance(realm_config.get("last_reality_scan"), dict)
        else None
    )
    scan_source_key = (last_scan or {}).get("source") or ""
    scan_source_labels = {
        "web_simulator": "WEB SIMULATOR（浏览器演示管线）",
        "depth_camera": "DEPTH CAMERA（深度相机 / 扫描器）",
    }
    scan_source_label = scan_source_labels.get(scan_source_key)
    if scan_source_label is None:
        if last_scan and not scan_source_key:
            scan_source_label = "未标注（兼容旧 scan JSON）"
        elif scan_source_key:
            scan_source_label = scan_source_key.replace("_", " ").upper()
        else:
            scan_source_label = "—"
    decorations_json = json.dumps(
        realm_config.get("decorations", []), ensure_ascii=False, indent=2
    )
    return {
        "profile": profile,
        "realm_config": realm_config,
        "scan_source_label": scan_source_label,
        "room_templates": ROOM_TEMPLATE_CHOICES,
        "material_packs": MATERIAL_PACK_CHOICES,
        "lightings": LIGHTING_CHOICES,
        "decorations_json": decorations_json,
        "room_label": dict(ROOM_TEMPLATE_CHOICES).get(
            realm_config.get("room_template"), realm_config.get("room_template")
        ),
        "material_label": dict(MATERIAL_PACK_CHOICES).get(
            realm_config.get("material_pack"), realm_config.get("material_pack")
        ),
        "lighting_label": dict(LIGHTING_CHOICES).get(
            realm_config.get("lighting"), realm_config.get("lighting")
        ),
    }


@login_required
def my_realm(request):
    return redirect(reverse("worlds") + "?tab=realm")


@login_required
def realm_editor(request):
    """位面编辑器：保存 room / material / lighting / decorations 到 private_realm_json"""
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    valid_templates = {c[0] for c in ROOM_TEMPLATE_CHOICES}
    valid_materials = {c[0] for c in MATERIAL_PACK_CHOICES}
    valid_light = {c[0] for c in LIGHTING_CHOICES}
    valid_vis = {c[0] for c in UserProfile.REALM_VISIBILITY_CHOICES}

    if request.method == "POST":
        room_template = request.POST.get("room_template", "")
        material_pack = request.POST.get("material_pack", "")
        lighting = request.POST.get("lighting", "")
        visibility = request.POST.get("realm_visibility", UserProfile.VISIBILITY_PRIVATE)
        dec_raw = (request.POST.get("decorations_json") or "").strip()
        decorations = []
        if dec_raw:
            try:
                parsed = json.loads(dec_raw)
                if isinstance(parsed, list):
                    decorations = parsed
            except (ValueError, TypeError):
                decorations = []

        if room_template not in valid_templates:
            room_template = ROOM_TEMPLATE_CHOICES[0][0]
        if material_pack not in valid_materials:
            material_pack = MATERIAL_PACK_CHOICES[0][0]
        if lighting not in valid_light:
            lighting = LIGHTING_CHOICES[0][0]
        if visibility not in valid_vis:
            visibility = UserProfile.VISIBILITY_PRIVATE

        _merge_private_realm_core(
            profile,
            room_template=room_template,
            material_pack=material_pack,
            lighting=lighting,
            decorations=decorations,
        )
        profile.realm_visibility = visibility
        profile.save(update_fields=["private_realm_json", "realm_visibility"])
        return redirect(reverse("worlds") + "?tab=editor&saved=1")

    return redirect(reverse("worlds") + "?tab=editor")


def privacy_policy(request):
    return render(request, "privacy.html", {"nav_active": "privacy"})


def terms_of_service(request):
    return render(request, "terms.html", {"nav_active": "terms"})


def security_center(request):
    return render(request, "security.html", {"nav_active": "security"})


def disclaimer(request):
    return render(request, "disclaimer.html", {"nav_active": "disclaimer"})


def devices(request):
    return redirect(reverse("settings") + "#devices")


@login_required
def material_library(request):
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    private_realm = merged_private_realm(profile)
    current_material_pack = private_realm.get("material_pack") or "cyber_neon"
    return render(
        request,
        "material_library.html",
        {
            "nav_active": "material_library",
            "material_packs": MATERIAL_PACK_CHOICES,
            "room_templates": ROOM_TEMPLATE_CHOICES,
            "private_realm": private_realm,
            "current_material_pack": current_material_pack,
        },
    )


@login_required
def worlds_plaza(request):
    tab = _hub_tab(request, WORLDS_HUB_TABS, "plaza")
    ctx = _my_realm_context(request)
    ctx["nav_active"] = "worlds"
    ctx["hub_tab"] = tab
    ctx["editor_saved"] = request.GET.get("saved") == "1"
    return render(request, "worlds.html", ctx)


def guide_hub(request):
    tab = _hub_tab(request, GUIDE_HUB_TABS, "tutorial")
    return render(
        request,
        "guide_hub.html",
        {"nav_active": "guide", "hub_tab": tab},
    )


def tutorial(request):
    return redirect(reverse("guide") + "?tab=tutorial")


def about(request):
    return render(request, "about.html")


def error_404(request, exception):
    return render(request, "404.html", status=404)
