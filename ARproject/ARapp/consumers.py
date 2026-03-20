import json
import asyncio
import cv2
import math
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer

mp_hands = mp.solutions.hands


# ── 右手手势逻辑 ──

def is_pinch(lm):
    """拇指尖与食指尖距离小于手掌尺寸的 25%"""
    ps = math.hypot(lm.landmark[0].x - lm.landmark[9].x,
                    lm.landmark[0].y - lm.landmark[9].y)
    return ps > 1e-5 and math.hypot(lm.landmark[4].x - lm.landmark[8].x,
                                    lm.landmark[4].y - lm.landmark[8].y) < ps * 0.25


def is_pointing(lm):
    """食指伸直，其余手指弯曲"""
    index_up = lm.landmark[8].y < lm.landmark[6].y
    others_bent = all(
        lm.landmark[t].y > lm.landmark[m].y
        for t, m in [(12, 10), (16, 14), (20, 18)]
    )
    return index_up and others_bent


def get_pointer_pos(lm):
    """返回食指尖归一化坐标"""
    return {'x': lm.landmark[8].x, 'y': lm.landmark[8].y}


# ── 左手手势逻辑 ──

def is_open_palm(lm):
    """判断是否为张开的手掌"""
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(lm.landmark[t].y < lm.landmark[p].y for t, p in zip(tips, pips)) >= 4


def is_fist(lm):
    """判断是否为握拳"""
    tips, mcps = [8, 12, 16, 20], [5, 9, 13, 17]
    return sum(lm.landmark[t].y > lm.landmark[m].y for t, m in zip(tips, mcps)) >= 4


class GestureConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.running = True
        self.prev_palm_y = None  # 用于记录左手手掌上一帧的 Y 坐标
        asyncio.ensure_future(self.stream_gestures())

    async def disconnect(self, code):
        self.running = False

    async def stream_gestures(self):
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)  # 水平翻转摄像头
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                right_gesture = 'none'
                left_gesture = 'none'
                scroll = None
                pointer = None
                all_landmarks = {}

                if result.multi_hand_landmarks and result.multi_handedness:
                    for lm, handedness in zip(result.multi_hand_landmarks,
                                              result.multi_handedness):

                        # MediaPipe 镜像翻转后：Label "Left" 是用户的右手，"Right" 是左手
                        label = handedness.classification[0].label
                        is_user_right = (label == 'Left')
                        is_user_left = (label == 'Right')

                        lm_list = [{'x': p.x, 'y': p.y} for p in lm.landmark]
                        side = 'right' if is_user_right else 'left'
                        all_landmarks[side] = lm_list

                        # 处理右手：负责指向和点击（point/pinch）
                        if is_user_right:
                            if is_pinch(lm):
                                right_gesture = 'pinch'
                            elif is_pointing(lm):
                                right_gesture = 'point'
                                pointer = get_pointer_pos(lm)

                        # 处理左手：负责滚动控制（scroll）
                        elif is_user_left:
                            if is_open_palm(lm):
                                left_gesture = 'open'
                                # 使用中指根部（landmark 9）作为手掌中心参考点
                                curr_y = lm.landmark[9].y

                                if self.prev_palm_y is not None:
                                    delta = curr_y - self.prev_palm_y
                                    # 阈值设置（0.015-0.03），根据实际灵敏度需求调整
                                    if delta > 0.02:  # 手掌向下移动 -> 触发向下滚动
                                        scroll = 'down'
                                    elif delta < -0.02:  # 手掌向上移动 -> 触发向上滚动
                                        scroll = 'up'

                                self.prev_palm_y = curr_y  # 更新位置
                            else:
                                if is_fist(lm):
                                    left_gesture = 'fist'
                                # 如果不是张开的手掌（比如握拳或收回手），重置参考点
                                self.prev_palm_y = None

                # 如果画面中没有左手，也重置参考点
                if 'left' not in all_landmarks:
                    self.prev_palm_y = None

                await self.send(json.dumps({
                    'right': right_gesture,
                    'left': left_gesture,
                    'scroll': scroll,
                    'pointer': pointer,
                    'landmarks': all_landmarks
                }))

                await asyncio.sleep(0.033)  # 约 30 FPS
        finally:
            cap.release()
            hands.close()