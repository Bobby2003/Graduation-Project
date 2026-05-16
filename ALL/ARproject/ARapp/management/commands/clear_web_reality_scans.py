"""移除所有用户 private_realm_json 中的浏览器模拟 last_reality_scan（保留深度相机 scan）。"""
from django.contrib.auth.models import User

from django.core.management.base import BaseCommand

from ARapp.realm_scan import clear_web_simulator_scan_from_profile
from ARapp.models import UserProfile


class Command(BaseCommand):
    help = "Clear web_simulator / reality_override_web last_reality_scan from all user profiles."

    def handle(self, *args, **options):
        cleared = 0
        for user in User.objects.iterator():
            profile, _ = UserProfile.objects.get_or_create(user=user)
            if clear_web_simulator_scan_from_profile(profile):
                cleared += 1
        self.stdout.write(self.style.SUCCESS(f"Cleared web simulator scan for {cleared} user(s)."))
