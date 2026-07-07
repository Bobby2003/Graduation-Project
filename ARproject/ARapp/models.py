from django.db import models
from django.contrib.auth.models import User


class Realm(models.Model):
    """位面 / 材质贴图包"""
    RARITY_CHOICES = [
        ('Common', '普通'),
        ('Rare', '稀有'),
        ('Epic', '史诗'),
        ('Legendary', '传说'),
    ]

    name = models.CharField(max_length=100, verbose_name="位面名称")
    description = models.TextField(verbose_name="位面描述")
    rarity = models.CharField(max_length=20, choices=RARITY_CHOICES, default='Common')
    bonus_effect = models.CharField(max_length=100, verbose_name="战斗加成", help_text="例如: 物理防御 +15%")
    preview_color = models.CharField(max_length=7, default="#00f2ff", verbose_name="预览色(HEX)")
    is_unlocked_default = models.BooleanField(default=False, verbose_name="是否默认解锁")

    def __str__(self):
        return self.name


class Loot(models.Model):
    """战利品 / 交易素材"""
    TYPE_CHOICES = [
        ('Material', '基础材料'),
        ('Texture', '贴图碎片'),
        ('Core', '能量核心'),
        ('Blueprint', '武器图纸'),
    ]

    name = models.CharField(max_length=100)
    loot_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    rarity = models.CharField(max_length=20, choices=Realm.RARITY_CHOICES)
    base_price = models.IntegerField(default=100, verbose_name="商店回收价")
    icon_code = models.CharField(max_length=50, default="BOX", verbose_name="图标代号(避免 MySQL utf8 存 emoji)")

    def __str__(self):
        return f"{self.name} ({self.get_rarity_display()})"


class UserInventory(models.Model):
    """用户背包 / 资产库"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="inventory")
    loot = models.ForeignKey(Loot, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1, verbose_name="持有数量")
    acquired_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "用户背包"
        unique_together = ('user', 'loot')  # 确保同一个用户同种物品只占一行

    def __str__(self):
        return f"{self.user.username} 的 {self.loot.name}"


class UserProfile(models.Model):
    """扩展用户信息 (战力、等级、货币)"""
    VISIBILITY_PRIVATE = "private"
    VISIBILITY_FRIENDS = "friends"
    VISIBILITY_PUBLIC = "public"
    REALM_VISIBILITY_CHOICES = [
        (VISIBILITY_PRIVATE, "私有"),
        (VISIBILITY_FRIENDS, "好友"),
        (VISIBILITY_PUBLIC, "公开"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    level = models.IntegerField(default=1)
    exp = models.PositiveIntegerField(default=0, verbose_name="经验值")
    combat_power = models.IntegerField(default=1000)
    credits = models.IntegerField(default=500, verbose_name="信用点(货币)")
    display_title = models.CharField(
        max_length=100, default="初始接入者", verbose_name="称号"
    )
    current_realm = models.ForeignKey(Realm, on_delete=models.SET_NULL, null=True, blank=True)
    realm_space_level = models.PositiveSmallIntegerField(
        default=1, verbose_name="私人位面空间等级"
    )
    realm_visibility = models.CharField(
        max_length=20,
        choices=REALM_VISIBILITY_CHOICES,
        default=VISIBILITY_PRIVATE,
        verbose_name="私人位面访问权限",
    )
    private_realm_json = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="私人位面配置",
        help_text="room_template / material_pack / lighting / decorations 等",
    )

    def __str__(self):
        return f"{self.user.username} 的角色数据"


# ── 黑市商品（静态配置数据，存 MySQL）────────────────────────────


class MarketProduct(models.Model):
    """黑市上架商品（非玩家背包实例）"""

    name = models.CharField(max_length=120)
    description = models.TextField()
    price_credits = models.PositiveIntegerField(default=0)
    image_placeholder = models.CharField(
        max_length=80, default="IMG_DATA_LOST", verbose_name="占位图文字"
    )
    accent = models.CharField(
        max_length=20,
        blank=True,
        help_text="cyan / purple / 空，用于前端样式类",
    )
    sort_order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["sort_order", "id"]

    def __str__(self):
        return self.name


# ── 装备图鉴与玩家当前装配 ───────────────────────────────────────


class EquipmentItem(models.Model):
    """装备定义（图鉴）；玩家装配见 UserEquippedItem"""

    SLOT_WEAPON_MAIN = "weapon_main"
    SLOT_WEAPON_OFF = "weapon_off"
    SLOT_ARMOR = "armor"
    SLOT_CORE = "core"
    SLOT_MOBILITY = "mobility"
    SLOT_COSMETIC = "cosmetic"
    SLOT_CHOICES = [
        (SLOT_WEAPON_MAIN, "主武器"),
        (SLOT_WEAPON_OFF, "副武器"),
        (SLOT_ARMOR, "护甲"),
        (SLOT_CORE, "技能核心"),
        (SLOT_MOBILITY, "移动装置"),
        (SLOT_COSMETIC, "外观"),
    ]

    name = models.CharField(max_length=120)
    slot = models.CharField(max_length=20, choices=SLOT_CHOICES)
    rarity = models.CharField(max_length=20, choices=Realm.RARITY_CHOICES, default="Rare")
    description = models.TextField()
    icon = models.CharField(
        max_length=32,
        default="sword",
        help_text="模板用键，如 sword / core，避免 4 字节 emoji 写入非 utf8mb4 表",
    )
    power_bonus = models.PositiveIntegerField(default=0, verbose_name="战力加成展示值")
    stat_label_1 = models.CharField(max_length=50, blank=True)
    stat_value_1 = models.CharField(max_length=50, blank=True)
    stat_label_2 = models.CharField(max_length=50, blank=True)
    stat_value_2 = models.CharField(max_length=50, blank=True)
    is_default_loadout = models.BooleanField(
        default=False, verbose_name="新用户默认装入此槽位"
    )
    sort_order = models.IntegerField(default=0)

    class Meta:
        ordering = ["sort_order", "id"]

    def __str__(self):
        return f"{self.name} ({self.get_slot_display()})"


class UserEquippedItem(models.Model):
    """每个槽位至多一件"""

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="equipped_items"
    )
    slot = models.CharField(max_length=20, choices=EquipmentItem.SLOT_CHOICES)
    item = models.ForeignKey(EquipmentItem, on_delete=models.CASCADE)

    class Meta:
        unique_together = [("user", "slot")]

    def __str__(self):
        return f"{self.user.username} · {self.get_slot_display()} · {self.item.name}"


# ── 任务定义与玩家进度 ───────────────────────────────────────────


class Mission(models.Model):
    KIND_MAIN = "main"
    KIND_DAILY = "daily"
    KIND_WEEKLY = "weekly"
    KIND_SIDE = "side"
    KIND_CHOICES = [
        (KIND_MAIN, "主线"),
        (KIND_DAILY, "日常"),
        (KIND_WEEKLY, "周常"),
        (KIND_SIDE, "支线"),
    ]

    code = models.SlugField(unique=True, help_text="稳定键，如 awaken_link")
    kind = models.CharField(max_length=10, choices=KIND_CHOICES)
    title = models.CharField(max_length=200)
    description = models.TextField()
    target = models.PositiveIntegerField(default=1, verbose_name="目标进度值")
    reward_exp = models.PositiveIntegerField(default=0)
    reward_credits = models.PositiveIntegerField(default=0)
    reward_item_label = models.CharField(max_length=120, blank=True)
    sort_order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True, verbose_name="是否启用")
    target_event = models.CharField(
        max_length=80,
        blank=True,
        db_index=True,
        verbose_name="进度事件键",
        help_text="与 /api/progress/event/ 的 event 对齐；空则不由事件推进",
    )

    class Meta:
        ordering = ["sort_order", "id"]

    def __str__(self):
        return self.title


class UserMission(models.Model):
    STATUS_AVAILABLE = "available"
    STATUS_IN_PROGRESS = "in_progress"
    STATUS_COMPLETED = "completed"
    STATUS_CHOICES = [
        (STATUS_AVAILABLE, "可接"),
        (STATUS_IN_PROGRESS, "进行中"),
        (STATUS_COMPLETED, "已完成"),
    ]

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="user_missions"
    )
    mission = models.ForeignKey(Mission, on_delete=models.CASCADE)
    progress = models.PositiveIntegerField(default=0)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_AVAILABLE
    )

    class Meta:
        unique_together = [("user", "mission")]

    def __str__(self):
        return f"{self.user.username} · {self.mission.title}"

    @property
    def progress_percent(self) -> int:
        if self.mission.target <= 0:
            return 0
        return min(100, int(100 * self.progress / self.mission.target))


class UserEventStat(models.Model):
    """玩家行为统计（材质尝试集合、事件计数等）"""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="event_stats")
    event = models.CharField(max_length=80, db_index=True)
    count = models.PositiveIntegerField(default=0)
    data = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = [("user", "event")]

    def __str__(self):
        return f"{self.user.username} · {self.event} · {self.count}"


# ── 成就定义与玩家解锁 ───────────────────────────────────────────


class Achievement(models.Model):
    category_key = models.CharField(max_length=40, db_index=True)
    category_title = models.CharField(max_length=120)
    code = models.SlugField(unique=True)
    name = models.CharField(max_length=120)
    description = models.TextField()
    rarity = models.CharField(max_length=20, default="COMMON")
    reward_exp = models.PositiveIntegerField(default=0)
    reward_credits = models.PositiveIntegerField(default=0)
    reward_note = models.CharField(max_length=200, blank=True)
    sort_order = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True, verbose_name="是否启用")
    icon_key = models.CharField(
        max_length=20,
        blank=True,
        default="",
        verbose_name="HUD 短标",
        help_text="ASCII 短标，如 OASIS / GEM，避免 emoji 入库",
    )

    class Meta:
        ordering = ["category_key", "sort_order", "id"]

    def __str__(self):
        return self.name


class UserAchievement(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="user_achievements"
    )
    achievement = models.ForeignKey(Achievement, on_delete=models.CASCADE)
    unlocked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("user", "achievement")]

    def __str__(self):
        return f"{self.user.username} · {self.achievement.name}"