from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import Realm, Loot, UserInventory, UserProfile
from django.shortcuts import render

# 1. 注册
def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.get_or_create(user=user, combat_power=1000, credits=500)
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})


# 2. 登录
def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')  # 登录后跳转到仪表盘
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})


# 3. 登出
def user_logout(request):
    logout(request)
    return redirect('login')


# 4. 仪表盘 (Dashboard)
@login_required
def dashboard(request):
    # 获取当前用户的 Profile 和位面信息
    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    realms = Realm.objects.all()
    inventory = UserInventory.objects.filter(user=request.user).select_related('loot')

    context = {
        'username': request.user.username,
        'profile': profile,
        'realms': realms,
        'inventory': inventory,
    }
    # 渲染之前写好的个人中心模板作为 Dashboard
    return render(request, 'user_center.html', context)

# 黑市视图
def market(request):
    # 这里以后可以从你的数据库查物品，现在先只返回网页
    return render(request, 'market.html')

# 战力榜单视图
def ranking(request):
    # 这里以后可以从 UserProfile 查 Combat Power 排序
    return render(request, 'ranking.html')

def error_404(request, exception):
    return render(request, '404.html', status=404)


import threading
import atexit
import time
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# 导入你的引擎类（确保 scanner_engine.py 在同一目录下）
from .scanner_engine import DepthScannerEngine

# ==========================================================
# AR 硬件单例管理器
# ==========================================================
class ARScannerManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.engine = DepthScannerEngine()
        self.is_active = False

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = ARScannerManager()
                atexit.register(cls._instance.shutdown)
        return cls._instance

    def start_camera(self):
        with self._lock:
            if not self.is_active:
                print("[SYSTEM] 正在冷启动奥比中光深度相机驱动...")
                if self.engine.init_device():
                    self.is_active = True
                    return True
                return False
            return True

    def shutdown(self):
        if self.is_active:
            print("[SYSTEM] 释放硬件资源...")
            self.engine.cleanup()
            self.is_active = False

scanner_manager = ARScannerManager.get_instance()

# ==========================================================
# 修正后的视图函数（对应你的 urls.py 名称）
# ==========================================================

def view_scanner(request):
    """
    匹配 urls.py 中的 path('scanner/', views.view_scanner, name='scanner')
    """
    # 预热硬件
    scanner_manager.start_camera()
    return render(request, 'scanner.html')


def video_feed_gen():
    """MJPEG 流生成器"""
    manager = ARScannerManager.get_instance()
    while True:
        if manager.is_active:
            # 调用 scanner_engine.py 中的 process_frame
            res = manager.engine.get_processed_frame()
            if res:
                frame_bytes, fps = res
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.5)

def video_feed(request):
    """
    匹配 urls.py 中的 path('video_feed/', views.video_feed, name='video_feed')
    """
    return StreamingHttpResponse(
        video_feed_gen(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )