"""Sprint 2：桌面位面新手任务 + 成就（事件驱动）"""
from django.core.management.base import BaseCommand

from ARapp.models import Achievement, Mission


class Command(BaseCommand):
    help = "Seed Sprint2 event missions + desktop realm achievements"

    def handle(self, *args, **options):
        achievements = [
            {
                "code": "first_realm_entry",
                "category_key": "sprint2_desktop",
                "category_title": "DESKTOP REALM // 新手",
                "name": "初入绿洲",
                "description": "第一次进入 AR REALM 私人位面。",
                "rarity": "COMMON",
                "reward_exp": 100,
                "reward_credits": 100,
                "icon_key": "OASIS",
                "sort_order": 0,
            },
            {
                "code": "desktop_walker",
                "category_key": "sprint2_desktop",
                "category_title": "DESKTOP REALM // 新手",
                "name": "桌面行者",
                "description": "使用 DESKTOP REALM 完成一次 WASD 移动。",
                "rarity": "COMMON",
                "reward_exp": 100,
                "reward_credits": 100,
                "icon_key": "WASD",
                "sort_order": 1,
            },
            {
                "code": "reality_rewriter",
                "category_key": "sprint2_desktop",
                "category_title": "DESKTOP REALM // 新手",
                "name": "现实重写者",
                "description": "第一次应用材质协议。",
                "rarity": "RARE",
                "reward_exp": 150,
                "reward_credits": 200,
                "icon_key": "MAT",
                "sort_order": 2,
            },
            {
                "code": "realm_designer",
                "category_key": "sprint2_desktop",
                "category_title": "DESKTOP REALM // 新手",
                "name": "位面设计师",
                "description": "第一次保存私人位面配置。",
                "rarity": "RARE",
                "reward_exp": 200,
                "reward_credits": 300,
                "icon_key": "SAVE",
                "sort_order": 3,
            },
            {
                "code": "style_collector",
                "category_key": "sprint2_desktop",
                "category_title": "DESKTOP REALM // 新手",
                "name": "风格收藏家",
                "description": "尝试 3 种不同材质协议。",
                "rarity": "EPIC",
                "reward_exp": 300,
                "reward_credits": 500,
                "icon_key": "GEM",
                "sort_order": 4,
            },
            {
                "code": "training_protocol_complete",
                "category_key": "sprint3_training",
                "category_title": "TRAINING GROUND // 接入训练",
                "name": "协议校准者",
                "description": "完成 Desktop Realm 接入训练。",
                "rarity": "RARE",
                "reward_exp": 300,
                "reward_credits": 500,
                "icon_key": "CAL",
                "sort_order": 0,
            },
            {
                "code": "reality_observer",
                "category_key": "reality_override",
                "category_title": "REALITY OVERRIDE // 现实覆写",
                "name": "现实观测者",
                "description": "你完成了第一次现实空间扫描，系统已记录可覆写表面。",
                "rarity": "RARE",
                "reward_exp": 250,
                "reward_credits": 400,
                "icon_key": "OBS",
                "sort_order": 0,
            },
            {
                "code": "override_executor",
                "category_key": "reality_override",
                "category_title": "REALITY OVERRIDE // 现实覆写",
                "name": "覆写执行者",
                "description": "你第一次将材质协议覆盖到现实扫描结果上。",
                "rarity": "EPIC",
                "reward_exp": 300,
                "reward_credits": 500,
                "icon_key": "OVR",
                "sort_order": 1,
            },
        ]
        for row in achievements:
            code = row.pop("code")
            Achievement.objects.update_or_create(code=code, defaults=row)

        missions = [
            {
                "code": "mission_enter_realm",
                "kind": Mission.KIND_MAIN,
                "title": "首次接入",
                "description": "进入一次 DESKTOP REALM 私人位面。",
                "target": 1,
                "target_event": "realm_entered",
                "reward_exp": 50,
                "reward_credits": 50,
                "sort_order": -50,
            },
            {
                "code": "mission_move_calibration",
                "kind": Mission.KIND_MAIN,
                "title": "移动校准",
                "description": "使用 WASD 完成一次移动。",
                "target": 1,
                "target_event": "desktop_moved",
                "reward_exp": 50,
                "reward_credits": 50,
                "sort_order": -49,
            },
            {
                "code": "mission_pointer_sync",
                "kind": Mission.KIND_MAIN,
                "title": "视角同步",
                "description": "完成一次鼠标锁定（点击画布）。",
                "target": 1,
                "target_event": "pointer_locked",
                "reward_exp": 50,
                "reward_credits": 50,
                "sort_order": -48,
            },
            {
                "code": "mission_material_override",
                "kind": Mission.KIND_MAIN,
                "title": "材质覆写",
                "description": "切换一次材质协议。",
                "target": 1,
                "target_event": "material_changed",
                "reward_exp": 80,
                "reward_credits": 100,
                "sort_order": -47,
            },
            {
                "code": "mission_save_realm",
                "kind": Mission.KIND_MAIN,
                "title": "位面保存",
                "description": "保存一次私人位面配置。",
                "target": 1,
                "target_event": "realm_saved",
                "reward_exp": 120,
                "reward_credits": 150,
                "sort_order": -46,
            },
            {
                "code": "mission_training_complete",
                "kind": Mission.KIND_SIDE,
                "title": "训练协议",
                "description": "完成一次 Desktop Realm 接入训练。",
                "target": 1,
                "target_event": "training_completed",
                "reward_exp": 150,
                "reward_credits": 200,
                "sort_order": -45,
            },
            {
                "code": "mission_reality_scan_once",
                "kind": Mission.KIND_SIDE,
                "title": "现实扫描",
                "description": "启动 Reality Override，并完成一次空间表面扫描。",
                "target": 1,
                "target_event": "reality_scan_completed",
                "reward_exp": 120,
                "reward_credits": 180,
                "sort_order": -44,
            },
            {
                "code": "mission_reality_override_apply",
                "kind": Mission.KIND_SIDE,
                "title": "协议覆写",
                "description": "将一种材质协议应用到扫描到的现实表面。",
                "target": 1,
                "target_event": "reality_override_applied",
                "reward_exp": 180,
                "reward_credits": 260,
                "sort_order": -43,
            },
        ]
        Mission.objects.filter(
            code__in=("mission_reality_scan", "mission_reality_apply")
        ).update(is_active=False)
        for row in missions:
            code = row.pop("code")
            Mission.objects.update_or_create(
                code=code,
                defaults={**row, "is_active": True, "reward_item_label": ""},
            )

        self.stdout.write(self.style.SUCCESS("seed_sprint2_progress 完成"))
