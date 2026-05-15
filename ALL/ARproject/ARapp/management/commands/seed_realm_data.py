"""
向 MySQL 写入绿洲静态配置：位面、黑市、装备图鉴、任务、成就。
可重复执行：已存在相同 code/slug 的跳过或更新。
"""
from django.core.management.base import BaseCommand

from ARapp.models import (
    Achievement,
    EquipmentItem,
    Loot,
    MarketProduct,
    Mission,
    Realm,
)


class Command(BaseCommand):
    help = "Seed Realm / Market / Equipment / Mission / Achievement (MySQL)"

    def handle(self, *args, **options):
        self._seed_realms()
        self._seed_loot()
        self._seed_market()
        self._seed_equipment()
        self._seed_missions()
        self._seed_achievements()
        self.stdout.write(self.style.SUCCESS("seed_realm_data 完成"))

    def _seed_realms(self):
        if Realm.objects.exists():
            return
        Realm.objects.create(
            name="霓虹边境",
            description="赛博废土与高塔都市交界的开放位面。",
            rarity="Epic",
            bonus_effect="全属性 +5%",
            preview_color="#00f2ff",
            is_unlocked_default=True,
        )
        Realm.objects.create(
            name="虚空回廊",
            description="不稳定的裂隙空间，适合高阶探索者。",
            rarity="Legendary",
            bonus_effect="技能伤害 +8%",
            preview_color="#bc00ff",
            is_unlocked_default=False,
        )
        self.stdout.write("  + Realm ×2")

    def _seed_loot(self):
        if Loot.objects.exists():
            return
        Loot.objects.create(
            name="位面密钥碎片",
            loot_type="Material",
            rarity="Rare",
            base_price=120,
            icon_code="KEY",
        )
        self.stdout.write("  + Loot ×1")

    def _seed_market(self):
        defaults = [
            ("量子肾上腺素 v2.0", "战斗向增益药剂，短时提升反应与技能回转 15%。", 1200, ""),
            ("相位折射镜片", "护甲插件：降低受到的指向性技能命中判定。", 850, "cyan"),
            ("废弃的机甲核心", "从高危位面回收的动力核心，可合成高阶协议。", 3500, ""),
        ]
        for i, (name, desc, price, accent) in enumerate(defaults):
            MarketProduct.objects.get_or_create(
                name=name,
                defaults={
                    "description": desc,
                    "price_credits": price,
                    "accent": accent,
                    "sort_order": i,
                },
            )
        self.stdout.write("  + MarketProduct")

    def _seed_equipment(self):
        items = [
            (
                "星陨刃",
                EquipmentItem.SLOT_WEAPON_MAIN,
                "Legendary",
                "由坠落星核锻造的能量长刃，可撕裂虚拟位面的防御层。",
                "sword",
                3500,
                "战力",
                "+3,500",
                "暴击",
                "+18%",
                True,
                0,
            ),
            (
                "副武装·脉冲匕",
                EquipmentItem.SLOT_WEAPON_OFF,
                "Rare",
                "近身副武器，适合断后与快速位移。",
                "dagger",
                1800,
                "战力",
                "+1,800",
                "攻速",
                "+10%",
                True,
                1,
            ),
            (
                "零点护甲",
                EquipmentItem.SLOT_ARMOR,
                "Epic",
                "吸收首发伤害的相位护盾。",
                "armor",
                2200,
                "战力",
                "+2,200",
                "减伤",
                "+12%",
                True,
                2,
            ),
            (
                "幻影核心",
                EquipmentItem.SLOT_CORE,
                "Epic",
                "短时间内制造残影分身，迷惑锁定系统。",
                "core",
                2200,
                "战力",
                "+2,200",
                "闪避",
                "+25%",
                True,
                3,
            ),
            (
                "裂隙推进器",
                EquipmentItem.SLOT_MOBILITY,
                "Rare",
                "短距离空间跃迁组件。",
                "mobility",
                3500,
                "移速",
                "+20%",
                "冲刺冷却",
                "-15%",
                True,
                4,
            ),
            (
                "黑曜面具",
                EquipmentItem.SLOT_COSMETIC,
                "Rare",
                "隐藏身份标识，降低被追踪概率。",
                "mask",
                800,
                "隐匿",
                "+15%",
                "魅力",
                "+800",
                True,
                5,
            ),
            (
                "星轨长弓",
                EquipmentItem.SLOT_WEAPON_MAIN,
                "Epic",
                "远程主武器，适合开阔位面。",
                "bow",
                2800,
                "战力",
                "+2,800",
                "射程",
                "+20%",
                False,
                10,
            ),
        ]
        for row in items:
            (
                name,
                slot,
                rarity,
                desc,
                icon_key,
                power,
                s1n,
                s1v,
                s2n,
                s2v,
                default_eq,
                sort_o,
            ) = row
            EquipmentItem.objects.get_or_create(
                name=name,
                slot=slot,
                defaults={
                    "rarity": rarity,
                    "description": desc,
                    "icon": icon_key,
                    "power_bonus": power,
                    "stat_label_1": s1n,
                    "stat_value_1": s1v,
                    "stat_label_2": s2n,
                    "stat_value_2": s2v,
                    "is_default_loadout": default_eq,
                    "sort_order": sort_o,
                },
            )
        self.stdout.write("  + EquipmentItem")

    def _seed_missions(self):
        data = [
            (
                "awaken_link",
                Mission.KIND_MAIN,
                "觉醒：神经链接",
                "首次接入 AR REALM，激活玩家身份与绿洲链路。",
                1,
                500,
                200,
                "初号核心",
                0,
            ),
            (
                "pick_realm",
                Mission.KIND_MAIN,
                "选择初始位面",
                "在终端中锁定你的第一个常驻位面协议。",
                10,
                1200,
                500,
                "位面密钥碎片",
                1,
            ),
            (
                "reality_tutorial",
                Mission.KIND_MAIN,
                "现实重构入门",
                "在现实重构引擎中完成一次世界皮肤 / 材质覆写预览。",
                4,
                2000,
                1000,
                "幻影核心碎片",
                2,
            ),
            (
                "daily_realm_login",
                Mission.KIND_DAILY,
                "每日登录绿洲",
                "当日首次通过任意控制模式进入 REALM 视图。",
                1,
                100,
                50,
                "",
                0,
            ),
            (
                "daily_market",
                Mission.KIND_DAILY,
                "黑市访客",
                "访问黑市交易所并查看至少 5 件商品。",
                1,
                80,
                30,
                "",
                1,
            ),
            (
                "daily_public_realm",
                Mission.KIND_DAILY,
                "公共位面访客",
                "访问一个公开位面节点并完成一次交互。",
                2,
                150,
                80,
                "",
                2,
            ),
            (
                "daily_equipment",
                Mission.KIND_DAILY,
                "装备强化",
                "在工坊中对任意装备进行一次强化。",
                1,
                200,
                100,
                "",
                3,
            ),
            (
                "weekly_raid",
                Mission.KIND_WEEKLY,
                "周常：裂隙征伐",
                "本周在副本或竞技中累计取得 50 次胜利判定。",
                50,
                5000,
                2000,
                "传说宝箱",
                0,
            ),
            (
                "weekly_power",
                Mission.KIND_WEEKLY,
                "战力飙升",
                "本周将战力提升 5,000 点。",
                5000,
                8000,
                3000,
                "强化石×10",
                1,
            ),
        ]
        for row in data:
            code, kind, title, desc, target, exp, cred, label, sort_o = row
            Mission.objects.get_or_create(
                code=code,
                defaults={
                    "kind": kind,
                    "title": title,
                    "description": desc,
                    "target": target,
                    "reward_exp": exp,
                    "reward_credits": cred,
                    "reward_item_label": label,
                    "sort_order": sort_o,
                },
            )
        # 演示进度：与旧静态页接近
        m_awaken = Mission.objects.filter(code="awaken_link").first()
        if m_awaken:
            m_awaken.target = 1
            m_awaken.save(update_fields=["target"])
        m_pick = Mission.objects.filter(code="pick_realm").first()
        if m_pick:
            m_pick.target = 10
            m_pick.save(update_fields=["target"])
        self.stdout.write("  + Mission")

    def _seed_achievements(self):
        rows = [
            (
                "realm_explorer",
                "REALM EXPLORER // 位面探索",
                "first_oasis",
                "首入绿洲",
                "第一次进入 AR REALM 世界链路。",
                "LEGENDARY",
                2000,
                5000,
                "",
                0,
            ),
            (
                "realm_explorer",
                "REALM EXPLORER // 位面探索",
                "realm_wanderer",
                "位面漫游者",
                "访问 5 个不同位面区域。",
                "EPIC",
                500,
                1000,
                "",
                1,
            ),
            (
                "realm_explorer",
                "REALM EXPLORER // 位面探索",
                "rift_crosser",
                "裂隙穿越者",
                "完成第一次跨位面传送。",
                "RARE",
                200,
                300,
                "",
                2,
            ),
            (
                "realm_explorer",
                "REALM EXPLORER // 位面探索",
                "world_reskin",
                "世界重构者",
                "首次对现实空间应用「世界皮肤 / 材质覆写」。",
                "EPIC",
                800,
                1200,
                "",
                3,
            ),
            (
                "realm_explorer",
                "REALM EXPLORER // 位面探索",
                "star_gate",
                "星门开启者",
                "解锁一处隐藏位面入口。",
                "RARE",
                800,
                2000,
                "",
                4,
            ),
            (
                "combat_legend",
                "COMBAT LEGEND // 战斗传说",
                "first_kill",
                "首杀宣告",
                "在竞技场或副本中取得首次击败。",
                "EPIC",
                1000,
                2000,
                "",
                0,
            ),
            (
                "combat_legend",
                "COMBAT LEGEND // 战斗传说",
                "combo_master",
                "连击大师",
                "单次战斗打出 30 段以上连击。",
                "RARE",
                300,
                500,
                "",
                1,
            ),
            (
                "combat_legend",
                "COMBAT LEGEND // 战斗传说",
                "perfect_block",
                "完美格挡",
                "在无伤状态下完成一场首领战。",
                "COMMON",
                50,
                50,
                "",
                2,
            ),
            (
                "combat_legend",
                "COMBAT LEGEND // 战斗传说",
                "realm_war_god",
                "位面战神",
                "在跨位面赛事中进入前 1% 排名。",
                "LEGENDARY",
                5000,
                10000,
                "",
                3,
            ),
            (
                "economy_lord",
                "ECONOMY LORD // 经济领主",
                "millionaire",
                "百万富翁",
                "累计获得超过 1,000,000 信用点。",
                "EPIC",
                3000,
                0,
                "专属称号",
                0,
            ),
            (
                "economy_lord",
                "ECONOMY LORD // 经济领主",
                "black_market_regular",
                "黑市常客",
                "在黑市完成 50 次交易。",
                "RARE",
                500,
                1000,
                "",
                1,
            ),
            (
                "economy_lord",
                "ECONOMY LORD // 经济领主",
                "first_gold",
                "第一桶金",
                "赚取第一个 1,000 信用点。",
                "COMMON",
                100,
                200,
                "",
                2,
            ),
            (
                "collector",
                "COLLECTOR // 收藏家",
                "skin_hunter",
                "外观猎人",
                "收集 20 款角色外观或涂装。",
                "EPIC",
                2000,
                0,
                "限定边框",
                0,
            ),
            (
                "collector",
                "COLLECTOR // 收藏家",
                "plugin_set",
                "插件全制霸",
                "集齐一套同名插件套装（4/4）。",
                "RARE",
                400,
                800,
                "",
                1,
            ),
            (
                "collector",
                "COLLECTOR // 收藏家",
                "pet_contract",
                "伙伴契约",
                "获得第一只 AI 伙伴或宠物节点。",
                "COMMON",
                50,
                100,
                "",
                2,
            ),
        ]
        for (
            ckey,
            ctitle,
            code,
            name,
            desc,
            rarity,
            exp,
            cred,
            note,
            sort_o,
        ) in rows:
            Achievement.objects.get_or_create(
                code=code,
                defaults={
                    "category_key": ckey,
                    "category_title": ctitle,
                    "name": name,
                    "description": desc,
                    "rarity": rarity,
                    "reward_exp": exp,
                    "reward_credits": cred,
                    "reward_note": note,
                    "sort_order": sort_o,
                },
            )
        self.stdout.write("  + Achievement")
