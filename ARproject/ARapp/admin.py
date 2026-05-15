from django.contrib import admin

from .models import (
    Achievement,
    EquipmentItem,
    Loot,
    MarketProduct,
    Mission,
    Realm,
    UserAchievement,
    UserEquippedItem,
    UserEventStat,
    UserInventory,
    UserMission,
    UserProfile,
)


@admin.register(Realm)
class RealmAdmin(admin.ModelAdmin):
    list_display = ("name", "rarity", "is_unlocked_default")


@admin.register(Loot)
class LootAdmin(admin.ModelAdmin):
    list_display = ("name", "loot_type", "rarity", "base_price", "icon_code")


@admin.register(UserInventory)
class UserInventoryAdmin(admin.ModelAdmin):
    list_display = ("user", "loot", "quantity")


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "display_title",
        "level",
        "exp",
        "combat_power",
        "credits",
        "realm_space_level",
        "realm_visibility",
        "current_realm",
    )


@admin.register(MarketProduct)
class MarketProductAdmin(admin.ModelAdmin):
    list_display = ("name", "price_credits", "sort_order", "is_active", "accent")


@admin.register(EquipmentItem)
class EquipmentItemAdmin(admin.ModelAdmin):
    list_display = ("name", "slot", "rarity", "power_bonus", "is_default_loadout")


@admin.register(UserEquippedItem)
class UserEquippedItemAdmin(admin.ModelAdmin):
    list_display = ("user", "slot", "item")


@admin.register(Mission)
class MissionAdmin(admin.ModelAdmin):
    list_display = (
        "code",
        "kind",
        "title",
        "target",
        "target_event",
        "is_active",
        "sort_order",
    )


@admin.register(UserMission)
class UserMissionAdmin(admin.ModelAdmin):
    list_display = ("user", "mission", "progress", "status")


@admin.register(Achievement)
class AchievementAdmin(admin.ModelAdmin):
    list_display = (
        "code",
        "name",
        "category_key",
        "icon_key",
        "rarity",
        "is_active",
        "sort_order",
    )


@admin.register(UserEventStat)
class UserEventStatAdmin(admin.ModelAdmin):
    list_display = ("user", "event", "count")
    search_fields = ("user__username", "event")


@admin.register(UserAchievement)
class UserAchievementAdmin(admin.ModelAdmin):
    list_display = ("user", "achievement", "unlocked_at")
