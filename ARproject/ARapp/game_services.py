"""为新用户初始化任务等游戏状态（数据在 MySQL）"""
from django.contrib.auth.models import User

from .models import Mission, UserMission, UserEquippedItem, EquipmentItem


def ensure_user_missions(user: User) -> None:
    for m in Mission.objects.filter(is_active=True).order_by('sort_order', 'id'):
        UserMission.objects.get_or_create(
            user=user,
            mission=m,
            defaults={'progress': 0, 'status': UserMission.STATUS_AVAILABLE},
        )


def ensure_default_loadout(user: User) -> None:
    """每位面装备槽：若为空则装备标记为默认的装备（仅用于演示）"""
    defaults = EquipmentItem.objects.filter(is_default_loadout=True)
    for item in defaults:
        UserEquippedItem.objects.get_or_create(
            user=user,
            slot=item.slot,
            defaults={'item': item},
        )
