import threading
import atexit
import time
from .scanner_engine import DepthScannerEngine


class ScannerManager:
    _instance = None
    _lock = threading.Lock()  # 线程锁，防止初始化冲突

    def __new__(cls):
        """确保整个 Django 进程只有一个管理实例"""
        with cls._lock:
            if cls._instance is None:
                print("[SYSTEM] 正在初始化 AR 硬件单例驱动...")
                cls._instance = super(ScannerManager, cls).__new__(cls)
                cls._instance.engine = DepthScannerEngine()
                cls._instance.is_initialized = False
            return cls._instance

    def start(self):
        """受保护的启动逻辑"""
        with self._lock:
            if not self.is_initialized:
                success = self.engine.init_device()
                if success:
                    print("[SYSTEM] ✅ AR 硬件连接成功")
                    self.is_initialized = True
                    # 注册退出钩子：当 Python 进程结束（Ctrl+C）时自动断开相机
                    atexit.register(self.stop)
                else:
                    print("[SYSTEM] ❌ AR 硬件初始化失败")
                return success
            return True

    def get_frame(self):
        """获取帧的回调"""
        if not self.is_initialized:
            if not self.start(): return None
        return self.engine.get_processed_frame()

    def get_fps(self):
        return self.engine.last_be_fps

    def stop(self):
        """安全释放硬件"""
        with self._lock:
            if self.is_initialized:
                print("[SYSTEM] 正在安全释放 AR 硬件...")
                self.engine.cleanup()
                self.is_initialized = False


# 实例化全局唯一的管理器
ar_scanner = ScannerManager()