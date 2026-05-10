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

# 任务系统视图
def missions(request):
    return render(request, 'missions.html')

# 成就系统视图
def achievements(request):
    return render(request, 'achievements.html')

# 装备系统视图
def equipment(request):
    return render(request, 'equipment.html')

# 系统设置视图
def settings(request):
    return render(request, 'settings.html')

# 新手引导视图
def tutorial(request):
    return render(request, 'tutorial.html')

# 关于页面视图
def about(request):
    return render(request, 'about.html')

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


def video_feed_gen(request):
    """MJPEG 流生成器"""
    manager = ARScannerManager.get_instance()
    frame_count = 0
    last_log_time = time.time()

    while True:
        # 检查请求是否已断开
        try:
            # Django 会自动处理断开连接
            pass
        except GeneratorExit:
            break

        frame_count += 1

        # 始终生成帧（模拟模式或真实相机）
        try:
            res = manager.engine.get_processed_frame()
            if res and res[0]:
                frame_bytes, fps = res
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # 生成备用黑色帧
                import numpy as np
                import cv2
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ok, buf = cv2.imencode('.jpg', frame)
                if ok:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        except Exception as e:
            print(f"[VideoFeed] 帧生成错误: {e}")

        # 每5秒打印一次状态
        current_time = time.time()
        if current_time - last_log_time > 5:
            print(f"[VideoFeed] 运行中, 已生成 {frame_count} 帧, is_active={manager.is_active}")
            last_log_time = current_time

        # 控制帧率 ~30fps
        time.sleep(0.033)

def video_feed(request):
    """
    匹配 urls.py 中的 path('video_feed/', views.video_feed, name='video_feed')
    """
    return StreamingHttpResponse(
        video_feed_gen(request),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )