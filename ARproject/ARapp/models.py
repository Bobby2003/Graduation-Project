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
    icon_code = models.CharField(max_length=50, default="📦", verbose_name="图标符号")

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
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    level = models.IntegerField(default=1)
    combat_power = models.IntegerField(default=1000)
    credits = models.IntegerField(default=500, verbose_name="信用点(货币)")
    current_realm = models.ForeignKey(Realm, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} 的角色数据"