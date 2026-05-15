"""Sprint 2：进度事件 → 任务推进 + 成就解锁 + 统计"""
from __future__ import annotations

from typing import Any

from django.contrib.auth.models import User
from django.db import transaction

from .models import (
    Achievement,
    Mission,
    UserAchievement,
    UserEventStat,
    UserMission,
    UserProfile,
)

ALLOWED_EVENTS = frozenset(
    {
        "realm_entered",
        "desktop_moved",
        "pointer_locked",
        "look_moved",
        "material_changed",
        "realm_saved",
        "training_completed",
        "reality_scan_completed",
        "reality_override_applied",
    }
)


def _unlock_achievement(
    user: User, profile: UserProfile, code: str, newly: list[dict[str, Any]]
) -> None:
    ach = Achievement.objects.filter(code=code, is_active=True).first()
    if not ach:
        return
    _, created = UserAchievement.objects.get_or_create(user=user, achievement=ach)
    if not created:
        return
    profile.exp += ach.reward_exp
    profile.credits += ach.reward_credits
    newly.append(
        {
            "code": ach.code,
            "name": ach.name,
            "description": ach.description,
            "icon_key": ach.icon_key or "*",
            "reward_exp": ach.reward_exp,
            "reward_credits": ach.reward_credits,
        }
    )


def _advance_event_missions(
    user: User, profile: UserProfile, event: str, newly: list[dict[str, Any]]
) -> None:
    qs = Mission.objects.filter(is_active=True, target_event=event).exclude(
        target_event=""
    )
    for mission in qs:
        um, _ = UserMission.objects.get_or_create(
            user=user,
            mission=mission,
            defaults={
                "progress": 0,
                "status": UserMission.STATUS_AVAILABLE,
            },
        )
        if um.status == UserMission.STATUS_COMPLETED:
            continue
        if um.status == UserMission.STATUS_AVAILABLE:
            um.status = UserMission.STATUS_IN_PROGRESS
        um.progress += 1
        if um.progress >= mission.target:
            um.progress = mission.target
            um.status = UserMission.STATUS_COMPLETED
            profile.exp += mission.reward_exp
            profile.credits += mission.reward_credits
            newly.append(
                {
                    "code": mission.code,
                    "title": mission.title,
                    "reward_exp": mission.reward_exp,
                    "reward_credits": mission.reward_credits,
                }
            )
        um.save(update_fields=["progress", "status"])


@transaction.atomic
def handle_progress_event(user: User, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    if event not in ALLOWED_EVENTS:
        return {"ok": False, "error": "INVALID_EVENT"}

    profile, _ = UserProfile.objects.select_for_update().get_or_create(user=user)

    newly_missions: list[dict[str, Any]] = []
    newly_achievements: list[dict[str, Any]] = []

    stat, _ = UserEventStat.objects.select_for_update().get_or_create(
        user=user,
        event=event,
        defaults={"count": 0, "data": {}},
    )
    stat.count += 1
    data = dict(stat.data or {})
    if event == "material_changed":
        mat = payload.get("material_pack") or payload.get("material_protocol")
        mats = set(data.get("materials", []))
        if mat:
            mats.add(str(mat))
        data["materials"] = list(mats)
    stat.data = data
    stat.save(update_fields=["count", "data"])

    _advance_event_missions(user, profile, event, newly_missions)

    if event == "realm_entered":
        _unlock_achievement(user, profile, "first_realm_entry", newly_achievements)
    if event == "desktop_moved":
        _unlock_achievement(user, profile, "desktop_walker", newly_achievements)
    if event == "material_changed":
        _unlock_achievement(user, profile, "reality_rewriter", newly_achievements)
        mats = data.get("materials", [])
        if isinstance(mats, list) and len(mats) >= 3:
            _unlock_achievement(user, profile, "style_collector", newly_achievements)
    if event == "realm_saved":
        _unlock_achievement(user, profile, "realm_designer", newly_achievements)

    if event == "training_completed":
        _unlock_achievement(user, profile, "training_protocol_complete", newly_achievements)

    if event == "reality_scan_completed":
        _unlock_achievement(user, profile, "reality_observer", newly_achievements)
    if event == "reality_override_applied":
        _unlock_achievement(user, profile, "override_executor", newly_achievements)

    profile.save(update_fields=["exp", "credits"])

    return {
        "ok": True,
        "event": event,
        "newly_completed_missions": newly_missions,
        "newly_unlocked_achievements": newly_achievements,
        "profile": {"exp": profile.exp, "credits": profile.credits},
    }
