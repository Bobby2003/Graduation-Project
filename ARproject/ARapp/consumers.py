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
    ps = math.hypot(lm.landmark[0].x - lm.landmark[9].x,
                    lm.landmark[0].y - lm.landmark[9].y)
    return ps > 1e-5 and math.hypot(lm.landmark[4].x - lm.landmark[8].x,
                                     lm.landmark[4].y - lm.landmark[8].y) < ps * 0.25


def is_pointing(lm):
    index_up = lm.landmark[8].y < lm.landmark[6].y
    others_bent = all(lm.landmark[t].y > lm.landmark[m].y
                      for t, m in [(12, 10), (16, 14), (20, 18)])
    return index_up and others_bent


def get_pointer_pos(lm):
    return {'x': lm.landmark[8].x, 'y': lm.landmark[8].y}


def is_open_palm(lm):
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(lm.landmark[t].y < lm.landmark[p].y
               for t, p in zip(tips, pips)) >= 4


def is_fist(lm):
    tips, mcps = [8, 12, 16, 20], [5, 9, 13, 17]
    return sum(lm.landmark[t].y > lm.landmark[m].y
               for t, m in zip(tips, mcps)) >= 4


# ── 全局后台手势追踪引擎 ──

class BackgroundGestureEngine:
    def __init__(self):
        self.clients = set()
        self.is_running = False
        self.prev_palm_y = None
        self._lock = threading.Lock()
        self._loop = None

    def start(self, loop):
        if self.is_running:
            return
        self._loop = loop
        self.is_running = True
        print("[AR System] 👐 启动手势识别引擎（对接奥比中光彩色流）...")
        thread = threading.Thread(target=self._run_in_thread, daemon=True)
        thread.start()

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
        从奥比中光引擎共享区获取彩色帧。
        不 fallback 到 cv2.VideoCapture，避免摄像头独占冲突。
        """
        try:
            from ARapp.scanner_singleton import ar_scanner
            return ar_scanner.engine.get_latest_color_frame()
        except Exception:
            return None

    def _run_in_thread(self):
        """
        在独立守护线程中运行 MediaPipe 手势识别。
        帧源来自奥比中光彩色流共享区，不独立打开摄像头。
        """
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        print("[AR System] ✅ 手势识别线程已就绪，等待奥比中光彩色帧...")

        while self.is_running:
            # 无客户端时降低功耗
            with self._lock:
                has_clients = bool(self.clients)

            if not has_clients:
                time.sleep(0.1)
                continue

            # 从奥比中光共享区取彩色帧
            frame = self._get_color_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # ── 手势解析 ──
            right_gesture = 'none'
            left_gesture = 'none'
            scroll = None
            pointer = None
            all_landmarks = {}

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, handedness in zip(result.multi_hand_landmarks,
                                          result.multi_handedness):
                    label = handedness.classification[0].label
                    # 头戴第一人称视角，无镜像，Right=用户右手，Left=用户左手
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