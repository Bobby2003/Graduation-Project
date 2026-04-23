import cv2
import numpy as np
import time
import threading
from .orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
from .utils import guided_filter, _fill_all_holes_for_filter


class DepthScannerEngine:
    """核心深度处理引擎，同时提供深度流（建模）和彩色流（手势识别）"""

    def __init__(self):
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())
        self.MIN_DEPTH = 100
        self.MAX_DEPTH = 3000
        self.EMA_ALPHA = 0.5
        self.ema_depth = None
        self.last_be_fps = 0
        self.is_running = False

        # 彩色帧共享区（供手势引擎读取）
        self._latest_color_frame = None
        self._color_lock = threading.Lock()
        self._color_thread = None

    def init_device(self):
        """初始化设备，同时开启深度流和彩色流"""
        if not self.sdk.initialize():
            return False
        if not self.sdk.open_device():
            return False

        # 开启深度流
        if not self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH):
            return False
        if not self.sdk.start_stream(self.sdk.depth_stream_handle):
            return False

        # 开启彩色流
        print("[Engine] 正在开启彩色传感器流（供手势识别使用）...")
        color_ok = False
        if self.sdk.create_stream(self.sdk.ONI_SENSOR_COLOR):
            if self.sdk.start_stream(self.sdk.color_stream_handle):
                print("[Engine] ✅ 彩色流已就绪，手势识别可用")
                color_ok = True
            else:
                print("[Engine] ⚠️ 彩色流启动失败，手势识别将不可用")
        else:
            print("[Engine] ⚠️ 彩色流创建失败，手势识别将不可用")

        if color_ok:
            self._color_thread = threading.Thread(
                target=self._color_capture_loop,
                daemon=True
            )
            self._color_thread.start()

        self.is_running = True
        return True

    def _color_capture_loop(self):
        """
        独立守护线程：持续从奥比中光读取彩色帧，写入共享区。
        手势引擎直接从共享区取帧，无需自己打开摄像头。
        """
        print("[Engine] 彩色帧采集线程已启动")
        while self.is_running:
            try:
                frame_info = self.sdk.read_frame(self.sdk.color_stream_handle)
                if frame_info and frame_info.get('data_type') == 'color':
                    # orbbec_sdk 返回 RGB，转为 BGR 供 OpenCV/MediaPipe 使用
                    rgb_frame = frame_info['data']
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    with self._color_lock:
                        self._latest_color_frame = bgr_frame
                    self.sdk.release_frame(frame_info)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[Engine] ⚠️ 彩色帧读取异常: {e}")
                time.sleep(0.1)
        print("[Engine] 彩色帧采集线程已退出")

    def get_latest_color_frame(self):
        """
        供 consumers.py 手势引擎调用。
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
        res = self.sdk.capture_depth_frame()
        if res is None:
            return None

        raw, _ = res
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

        self.last_be_fps = 1.0 / (time.time() - start_time)

        _, buffer = cv2.imencode('.jpg', color_map)
        return buffer.tobytes(), self.last_be_fps

    def cleanup(self):
        self.is_running = False
        self.sdk.cleanup()
        print("[Engine] 引擎资源已释放")