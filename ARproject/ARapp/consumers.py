import json
import asyncio
import cv2
import math
import time
import threading
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer

mp_hands = mp.solutions.hands

# ── 手势算法辅助函数 ──

def is_pinch(lm):
    ps = math.hypot(
        lm.landmark[0].x - lm.landmark[9].x,
        lm.landmark[0].y - lm.landmark[9].y
    )
    return ps > 1e-5 and math.hypot(
        lm.landmark[4].x - lm.landmark[8].x,
        lm.landmark[4].y - lm.landmark[8].y
    ) < ps * 0.25

def is_pointing(lm):
    index_up = lm.landmark[8].y < lm.landmark[6].y
    others_bent = all(
        lm.landmark[t].y > lm.landmark[m].y
        for t, m in [(12, 10), (16, 14), (20, 18)]
    )
    return index_up and others_bent

def get_pointer_pos(lm):
    return {
        'x': lm.landmark[8].x,
        'y': lm.landmark[8].y
    }

def is_open_palm(lm):
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(
        lm.landmark[t].y < lm.landmark[p].y
        for t, p in zip(tips, pips)
    ) >= 4

def is_fist(lm):
    tips, mcps = [8, 12, 16, 20], [5, 9, 13, 17]
    return sum(
        lm.landmark[t].y > lm.landmark[m].y
        for t, m in zip(tips, mcps)
    ) >= 4

# ── 全局后台手势追踪引擎 ──

class BackgroundGestureEngine:
    def __init__(self):
        self.clients = set()
        self.is_running = False

        self.prev_palm_y = None

        self._lock = threading.Lock()
        self._loop = None
        self._thread = None

        # 调试用：避免 None 帧疯狂刷屏
        self._last_no_frame_log = 0
        self._last_frame_log = 0

    def start(self, loop):
        if self.is_running:
            return

        self._loop = loop
        self.is_running = True

        print("[AR System] 👐 启动手势识别引擎（对接 OpenCV 彩色流）...")

        self._thread = threading.Thread(
            target=self._run_in_thread,
            daemon=True
        )
        self._thread.start()

    def add_client(self, client, loop):
        with self._lock:
            self.clients.add(client)

        if not self.is_running:
            self.start(loop)

    def remove_client(self, client):
        with self._lock:
            self.clients.discard(client)

    def _get_color_frame(self):
        """
        从 scanner_singleton 获取 OpenCV 彩色帧。

        关键点：
        必须调用 ar_scanner.get_color_frame()
        不要调用 ar_scanner.engine.get_latest_color_frame()

        因为 ar_scanner.get_color_frame() 会自动触发：
        ar_scanner.start() -> engine.init_device() -> 打开 OpenCV 摄像头 -> 启动采集线程
        """
        try:
            from ARapp.scanner_singleton import ar_scanner

            frame = ar_scanner.get_color_frame()

            if frame is None:
                now = time.time()
                if now - self._last_no_frame_log > 2.0:
                    print("[AR System] ⚠️ 暂未获取到 OpenCV 彩色帧")
                    self._last_no_frame_log = now
                return None

            now = time.time()
            if now - self._last_frame_log > 5.0:
                print(f"[AR System] ✅ 已获取 OpenCV 彩色帧: shape={frame.shape}")
                self._last_frame_log = now

            return frame

        except Exception as e:
            now = time.time()
            if now - self._last_no_frame_log > 2.0:
                print(f"[AR System] ❌ 获取 OpenCV 彩色帧异常: {e}")
                self._last_no_frame_log = now
            return None

    def _run_in_thread(self):
        """
        在独立守护线程中运行 MediaPipe 手势识别。
        帧源来自 scanner_engine.py 里的 OpenCV 摄像头采集线程。
        """
        print("[AR System] ✅ 手势识别线程已启动，准备初始化 AR 硬件...")

        # 关键：主动启动硬件，确保 scanner_engine.init_device() 被调用
        try:
            from ARapp.scanner_singleton import ar_scanner

            if not ar_scanner.start():
                print("[AR System] ❌ AR 硬件启动失败，手势识别无法运行")
                self.is_running = False
                return

            print("[AR System] ✅ AR 硬件已启动，等待 OpenCV 彩色帧...")

        except Exception as e:
            print(f"[AR System] ❌ AR 硬件启动异常: {e}")
            self.is_running = False
            return

        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        print("[AR System] ✅ MediaPipe Hands 已就绪")

        while self.is_running:
            # 无客户端时降低功耗
            with self._lock:
                has_clients = bool(self.clients)

            if not has_clients:
                time.sleep(0.1)
                continue

            # 从 OpenCV 彩色共享区取帧
            frame = self._get_color_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                # OpenCV 是 BGR，MediaPipe 需要 RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

            except Exception as e:
                print(f"[AR System] ❌ MediaPipe 处理帧异常: {e}")
                time.sleep(0.05)
                continue

            # ── 手势解析 ──
            right_gesture = 'none'
            left_gesture = 'none'
            scroll = None
            pointer = None
            all_landmarks = {}

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):
                    label = handedness.classification[0].label

                    # 注意：
                    # 如果你的画面没有镜像，MediaPipe 的 handedness 通常可直接用。
                    # 如果你前端或后端对画面做了水平镜像，这里可能需要左右互换。
                    is_user_right = (label == 'Right')
                    is_user_left = (label == 'Left')

                    lm_list = [{'x': p.x, 'y': p.y} for p in lm.landmark]

                    side = 'right' if is_user_right else 'left'
                    all_landmarks[side] = lm_list

                    if is_user_right:
                        if is_pinch(lm):
                            right_gesture = 'pinch'
                        elif is_pointing(lm):
                            right_gesture = 'point'
                            pointer = get_pointer_pos(lm)

                    elif is_user_left:
                        if is_open_palm(lm):
                            left_gesture = 'open'

                            curr_y = lm.landmark[9].y
                            if self.prev_palm_y is not None:
                                delta = curr_y - self.prev_palm_y

                                # 图像坐标：y 增大 = 往下，y 减小 = 往上
                                if delta > 0.02:
                                    scroll = 'down'
                                elif delta < -0.02:
                                    scroll = 'up'

                            self.prev_palm_y = curr_y

                        else:
                            if is_fist(lm):
                                left_gesture = 'fist'
                            self.prev_palm_y = None

            if 'left' not in all_landmarks:
                self.prev_palm_y = None

            data = json.dumps({
                'right': right_gesture,
                'left': left_gesture,
                'scroll': scroll,
                'pointer': pointer,
                'landmarks': all_landmarks
            })

            # ── 广播至所有已连接客户端 ──
            with self._lock:
                active_clients = list(self.clients)

            for client in active_clients:
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        client.send(data),
                        self._loop
                    )

            time.sleep(0.033)  # 约 30FPS

        hands.close()
        print("[AR System] 手势引擎已安全停止")

# 全局唯一引擎实例
engine = BackgroundGestureEngine()

# ── WebSocket 消费者接口 ──

class GestureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        loop = asyncio.get_event_loop()
        engine.add_client(self, loop)

        print("[WS] ✅ 客户端已接入战术手势链路")

    async def disconnect(self, code):
        engine.remove_client(self)
        print(f"[WS] ❌ 链路断开 (Code: {code})")

    async def receive(self, text_data=None, bytes_data=None):
        # 预留：可接收前端控制指令
        pass