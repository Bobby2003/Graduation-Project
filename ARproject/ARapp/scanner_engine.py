import cv2
import numpy as np
import time
import threading
from .orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
from .utils import guided_filter, _fill_all_holes_for_filter

class DepthScannerEngine:
    """核心深度处理引擎：深度流走 Orbbec SDK，彩色流走 OpenCV（供手势识别）"""

    def __init__(self):
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())

        self.MIN_DEPTH = 100
        self.MAX_DEPTH = 3000
        self.EMA_ALPHA = 0.5
        self.ema_depth = None
        self.last_be_fps = 0
        self.is_running = False

        # OpenCV 彩色相机
        self.cv_camera = None
        self.cv_camera_index = None
        self.cv_camera_initialized = False

        # 彩色帧共享区（供手势引擎读取）
        self._latest_color_frame = None
        self._color_lock = threading.Lock()
        self._color_thread = None

    def init_device(self):
        """初始化设备：深度走 SDK，彩色走 OpenCV"""
        print("[Engine] 正在初始化深度设备...")

        # 尝试初始化深度相机（可失败）
        depth_available = False
        if self.sdk.initialize():
            if self.sdk.open_device():
                if self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH):
                    if self.sdk.start_stream(self.sdk.depth_stream_handle):
                        depth_available = True
                        print("[Engine] ✅ 深度流已启动")
                    else:
                        print("[Engine] ❌ 深度流启动失败")
                else:
                    print("[Engine] ❌ 深度流创建失败")
            else:
                print("[Engine] ❌ 深度设备打开失败")
        else:
            print("[Engine] ❌ SDK 初始化失败")

        # 即使深度相机不可用，也继续初始化 OpenCV 彩色摄像头
        print("[Engine] 正在尝试使用 OpenCV 打开彩色摄像头...")
        if not self._init_opencv_camera():
            print("[Engine] ⚠️ OpenCV 彩色摄像头不可用")
        else:
            print(f"[Engine] ✅ OpenCV 彩色摄像头已就绪，索引={self.cv_camera_index}")

        # 启动彩色采集线程（如果有摄像头）
        if self.cv_camera_initialized:
            self._color_thread = threading.Thread(
                target=self._color_capture_loop,
                daemon=True
            )
            self._color_thread.start()

        self.is_running = True

        # 只要有彩色摄像头就返回成功
        if self.cv_camera_initialized:
            print("[Engine] ✅ 引擎初始化完成（仅彩色模式）")
            return True

        # 没有深度也没有彩色，返回失败
        if not depth_available and not self.cv_camera_initialized:
            print("[Engine] ❌ 没有任何可用相机")
            return False

        return True

    def _init_opencv_camera(self):
        """初始化 OpenCV 摄像头，自动尝试多个索引"""
        candidate_indexes = [0, 1, 2, 3]

        for idx in candidate_indexes:
            try:
                print(f"[Engine] 尝试打开摄像头索引 {idx} ...")
                cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

                # 如果 CAP_DSHOW 不行，可回退
                if not cam.isOpened():
                    cam.release()
                    cam = cv2.VideoCapture(idx)

                if not cam.isOpened():
                    cam.release()
                    continue

                # 设置分辨率和帧率
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cam.set(cv2.CAP_PROP_FPS, 30)

                # 预热几帧
                ok = False
                test_frame = None
                for _ in range(5):
                    ret, frame = cam.read()
                    if ret and frame is not None and frame.size > 0:
                        ok = True
                        test_frame = frame
                        break
                    time.sleep(0.05)

                if ok:
                    self.cv_camera = cam
                    self.cv_camera_index = idx
                    self.cv_camera_initialized = True
                    print(f"[Engine] 彩色摄像头测试成功: {test_frame.shape[1]}x{test_frame.shape[0]}")
                    return True
                else:
                    cam.release()

            except Exception as e:
                print(f"[Engine] 打开摄像头索引 {idx} 失败: {e}")

        self.cv_camera = None
        self.cv_camera_index = None
        self.cv_camera_initialized = False
        return False

    def _color_capture_loop(self):
        """
        独立守护线程：持续从 OpenCV 读取彩色帧，写入共享区。
        手势引擎直接从共享区取帧，无需自己打开摄像头。
        """
        print("[Engine] OpenCV 彩色帧采集线程已启动")

        while self.is_running:
            try:
                if not self.cv_camera_initialized or self.cv_camera is None:
                    time.sleep(0.1)
                    continue

                ret, frame = self.cv_camera.read()
                if ret and frame is not None:
                    # OpenCV 输出本来就是 BGR
                    with self._color_lock:
                        self._latest_color_frame = frame.copy()
                else:
                    # 读取失败时稍等，避免空转
                    time.sleep(0.02)

            except Exception as e:
                print(f"[Engine] ⚠️ OpenCV 彩色帧读取异常: {e}")
                time.sleep(0.1)

        print("[Engine] OpenCV 彩色帧采集线程已退出")

    def get_latest_color_frame(self):
        """
        供 consumers.py / 手势引擎调用。
        返回最新的 BGR 彩色帧（numpy array），无帧时返回 None。
        """
        with self._color_lock:
            if self._latest_color_frame is not None:
                return self._latest_color_frame.copy()
        return None

    def remove_flying_pixels(self, d, depth_thresh=100.0, erode_px=2):
        diff_x = np.abs(np.diff(d, axis=1, append=0))
        diff_y = np.abs(np.diff(d, axis=0, append=0))
        edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)
        kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        edge_dilated = cv2.dilate(edge, kernel)
        d_out = d.copy()
        d_out[edge_dilated > 0] = 0.0
        return d_out

    def get_processed_frame(self):
        """获取深度帧并处理，返回 (JPG字节, FPS)"""
        start_time = time.time()

        try:
            # 如果SDK没有初始化，先尝试初始化
            if not self.sdk.is_initialized:
                if not self.sdk.initialize():
                    return self._generate_demo_frame()

            res = self.sdk.capture_depth_frame()
            if res is None:
                return self._generate_demo_frame()

            raw, _ = res
            if raw is None or len(raw) == 0:
                return self._generate_demo_frame()

            d = raw.astype(np.float32)
            d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0
            d = self.remove_flying_pixels(d)

            valid_mask = (d > 0).astype(np.uint8)
            d_for_gf = _fill_all_holes_for_filter(d)
            guide = cv2.normalize(d_for_gf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_gf = guided_filter(guide, d_for_gf, r=4, eps=50.0)
            d_gf[valid_mask == 0] = 0.0

            disp = np.clip(d_gf / self.MAX_DEPTH * 255, 0, 255).astype(np.uint8)
            color_map = cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET)

            self.last_be_fps = 1.0 / max((time.time() - start_time), 1e-6)

            ok, buffer = cv2.imencode('.jpg', color_map)
            if not ok:
                return self._generate_demo_frame()

            return buffer.tobytes(), self.last_be_fps

        except Exception as e:
            # 任何异常都返回模拟帧
            print(f"[Engine] 帧获取异常: {e}")
            return self._generate_demo_frame()

    def _generate_demo_frame(self):
        """生成模拟点云图 - 当相机不可用时显示"""
        width, height = 640, 480

        # 创建深色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [5, 8, 15]  # 深蓝黑色背景

        # 生成随机点云效果
        num_points = 800
        points = []

        # 生成多层点云（模拟不同深度）
        for layer in range(3):
            layer_depth = 0.3 + layer * 0.2
            layer_num = num_points // 3

            # 中心区域更密集
            cx, cy = width // 2 + np.random.randint(-50, 50), height // 2 + np.random.randint(-50, 50)

            for _ in range(layer_num):
                # 在椭圆区域内随机分布
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.exponential(100)
                x = int(cx + radius * np.cos(angle) + np.random.randn() * 30)
                y = int(cy + radius * np.sin(angle) + np.random.randn() * 30)

                if 0 <= x < width and 0 <= y < height:
                    # 颜色根据深度变化：近处青色，远处紫色
                    t = layer_depth
                    r = int(0 + t * 100)
                    g = int(200 - t * 100)
                    b = int(255 - t * 50)
                    points.append((x, y, r, g, b))

        # 绘制点
        for x, y, r, g, b in points:
            size = np.random.randint(1, 3)
            cv2.circle(frame, (x, y), size, (b, g, r), -1)

        # 添加扫描线效果
        scan_y = int((time.time() * 50) % height)
        cv2.line(frame, (0, scan_y), (width, scan_y), (0, 100, 100), 1)
        cv2.line(frame, (0, scan_y + 1), (width, scan_y + 1), (0, 50, 50), 1)

        # 添加网格线（透视效果）
        grid_color = (20, 25, 35)
        for i in range(0, width, 80):
            cv2.line(frame, (i, 0), (i, height), grid_color, 1)
        for i in range(0, height, 80):
            cv2.line(frame, (0, i), (width, i), grid_color, 1)

        # 添加中心准星
        cx, cy = width // 2, height // 2
        cv2.circle(frame, (cx, cy), 30, (0, 200, 200), 1)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.line(frame, (cx - 50, cy), (cx - 20, cy), (0, 200, 200), 1)
        cv2.line(frame, (cx + 20, cy), (cx + 50, cy), (0, 200, 200), 1)
        cv2.line(frame, (cx, cy - 50), (cx, cy - 20), (0, 200, 200), 1)
        cv2.line(frame, (cx, cy + 20), (cx, cy + 50), (0, 200, 200), 1)

        # 添加文字提示
        cv2.putText(frame, "SIMULATION MODE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 150), 1)
        cv2.putText(frame, "Waiting for depth camera...", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        ok, buffer = cv2.imencode('.jpg', frame)
        if ok:
            return buffer.tobytes(), 0
        return None, 0

    def cleanup(self):
        """释放资源"""
        self.is_running = False

        # 给线程一点退出时间
        if self._color_thread is not None and self._color_thread.is_alive():
            self._color_thread.join(timeout=1.0)

        # 释放 OpenCV 摄像头
        if self.cv_camera is not None:
            try:
                self.cv_camera.release()
                print("[Engine] OpenCV 彩色摄像头已释放")
            except Exception as e:
                print(f"[Engine] 释放 OpenCV 摄像头失败: {e}")
            finally:
                self.cv_camera = None
                self.cv_camera_initialized = False

        # 清空共享帧
        with self._color_lock:
            self._latest_color_frame = None

        # 清理 SDK
        self.sdk.cleanup()
        print("[Engine] 引擎资源已释放")