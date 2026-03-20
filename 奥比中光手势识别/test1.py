import cv2
import math
import sys
import time

try:
    import mediapipe as mp
except ImportError:
    print("❌ 缺少 mediapipe 库！请运行：pip install mediapipe")
    sys.exit(1)


class HoloSystem:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.menu_is_open = False
        self.trigger_hand_label = None
        self.fist_ready = {"Left": None, "Right": None}

        self.current_page = None
        # back按钮放在画面水平中央
        self.back_btn = [540, 20, 140, 50]

        self.pinch_state = {"Left": False, "Right": False}
        self.tap_times = {"Left": [], "Right": []}
        self.DOUBLE_TAP_WINDOW = 0.6
        self.select_flash = {"btn": None, "time": 0.0}
        self.debug_state = {"Left": "", "Right": ""}

        self.buttons = [
            {"text": "Display Mode", "color": (0, 180, 220)},
            {"text": "Network",      "color": (0, 160, 200)},
            {"text": "Security",     "color": (0, 140, 180)},
            {"text": "Close System", "color": (180, 60,  60)},
        ]

        self.page_content = {
            "Display Mode": ["Brightness : 80%", "Resolution : 1920x1080", "Refresh    : 60Hz", "HDR        : ON"],
            "Network":      ["Status     : Connected", "IP         : 192.168.x.x", "Signal     : Strong", "Firewall   : Active"],
            "Security":     ["Auth       : Biometric", "Encrypt    : AES-256", "Last Scan  : Today", "Threats    : 0"],
            "Close System": ["Shutting down...", "Save state  : YES", "Confirm     : ---"],
        }

    def palm_size(self, lm):
        return math.hypot(lm.landmark[0].x - lm.landmark[9].x,
                          lm.landmark[0].y - lm.landmark[9].y)

    def dist(self, lm, a, b):
        return math.hypot(lm.landmark[a].x - lm.landmark[b].x,
                          lm.landmark[a].y - lm.landmark[b].y)

    def is_fist(self, lm):
        tips = [8, 12, 16, 20]
        mcps = [5,  9, 13, 17]
        return sum(lm.landmark[t].y > lm.landmark[m].y for t, m in zip(tips, mcps)) >= 4

    def is_open_palm(self, lm):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        if sum(lm.landmark[t].y < lm.landmark[p].y for t, p in zip(tips, pips)) < 4:
            return False
        ps = self.palm_size(lm)
        return ps > 1e-5 and sum(
            self.dist(lm, a, b) > ps * 0.35 for a, b in [(8, 12), (12, 16), (16, 20)]
        ) >= 2

    def is_pinch(self, lm):
        ps = self.palm_size(lm)
        return ps > 1e-5 and self.dist(lm, 4, 8) < ps * 0.25

    def check_double_tap(self, label, pinching, now):
        was = self.pinch_state[label]
        self.pinch_state[label] = pinching
        if pinching and not was:
            self.tap_times[label].append(now)
            self.tap_times[label] = self.tap_times[label][-2:]
            taps = self.tap_times[label]
            if len(taps) == 2 and (taps[1] - taps[0]) < self.DOUBLE_TAP_WINDOW:
                self.tap_times[label] = []
                return True
        return False

    def get_btn_rect(self, i):
        return [50, 160 + i * 90, 300, 70]

    def hit_test(self, cursor, rect):
        if not cursor:
            return False
        bx, by, bw, bh = rect
        return bx < cursor[0] < bx + bw and by < cursor[1] < by + bh

    def draw_cursor(self, frame, cursor):
        if cursor:
            cv2.circle(frame, cursor, 14, (0, 255, 255), 2)
            cv2.circle(frame, cursor, 4,  (0, 255, 255), -1)
            cv2.putText(frame, "double pinch = select",
                        (cursor[0] + 18, cursor[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    def draw_menu(self, frame, cursor, now):
        ov = frame.copy()
        cv2.rectangle(ov, (20, 100), (420, 570), (20, 20, 20), -1)
        cv2.rectangle(ov, (20, 100), (420, 570), (0, 255, 255), 2)
        cv2.putText(ov, "SYSTEM SETTINGS", (50, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        for i, btn in enumerate(self.buttons):
            rect = self.get_btn_rect(i)
            bx, by, bw, bh = rect
            hover = self.hit_test(cursor, rect)
            flash = (self.select_flash["btn"] == btn["text"] and
                     now - self.select_flash["time"] < 0.35)
            bg = (255, 255, 255) if flash else (btn["color"] if hover else (60, 60, 60))
            tc = (0, 0, 0) if flash else ((255, 255, 255) if hover else (200, 200, 200))
            cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), bg, -1)
            if hover:
                cv2.rectangle(ov, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            cv2.putText(ov, btn["text"], (bx + 18, by + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, tc, 2)

        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
        self.draw_cursor(frame, cursor)

    def draw_page(self, frame, cursor, now):
        h, w = frame.shape[:2]
        name = self.current_page
        ov = frame.copy()

        cv2.rectangle(ov, (0, 0), (w, h), (12, 12, 12), -1)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

        # 标题栏
        cv2.rectangle(frame, (0, 0), (w, 90), (25, 25, 25), -1)
        cv2.line(frame, (0, 90), (w, 90), (0, 255, 255), 2)

        # 标题文字放右侧，避开中间的back按钮
        cv2.putText(frame, name.upper(), (750, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # back按钮（水平居中）
        bx, by, bw, bh = self.back_btn
        back_hover = self.hit_test(cursor, self.back_btn)
        back_flash = (self.select_flash["btn"] == "__back__" and
                      now - self.select_flash["time"] < 0.35)
        bbg = (255, 255, 255) if back_flash else ((0, 200, 100) if back_hover else (50, 50, 50))
        btc = (0, 0, 0) if (back_flash or back_hover) else (200, 200, 200)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bbg, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 100), 2)
        cv2.putText(frame, "< BACK", (bx + 12, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, btc, 2)

        # 内容列表
        lines = self.page_content.get(name, [])
        for j, line in enumerate(lines):
            cv2.putText(frame, line, (160, 160 + j * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (220, 220, 220), 2)
        cv2.line(frame, (140, 120), (140, 120 + len(lines) * 60), (0, 255, 255), 2)

        self.draw_cursor(frame, cursor)

    def run(self):
        print("🚀 启动")
        print("打开菜单：握拳 → 张开手掌")
        print("关闭菜单：同一只手再次握拳")
        print("选择按钮：另一只手食指+拇指快速双击捏合")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cursor = None
            now = time.time()

            if result.multi_hand_landmarks:
                for lm, hd in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label  = hd.classification[0].label
                    fist   = self.is_fist(lm)
                    opened = self.is_open_palm(lm)
                    pinch  = self.is_pinch(lm)

                    self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

                    if not self.menu_is_open:
                        if fist:
                            self.fist_ready[label] = now
                            self.debug_state[label] = "FIST✊"
                            px = int(lm.landmark[0].x * w)
                            py = max(int(lm.landmark[0].y * h) - 20, 30)
                            cv2.putText(frame, "FIST - OPEN NOW!",
                                        (px - 110, py),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
                        elif opened:
                            t = self.fist_ready[label]
                            if t is not None and (now - t) < 3.0:
                                self.menu_is_open = True
                                self.trigger_hand_label = label
                                self.fist_ready[label] = None
                                self.debug_state[label] = "OPEN✋"
                                print(f"✅ 菜单已打开，触发手: {label}")
                            else:
                                self.debug_state[label] = "open(no fist)"
                        else:
                            if self.fist_ready[label] and (now - self.fist_ready[label]) > 3.0:
                                self.fist_ready[label] = None
                            self.debug_state[label] = "..."

                    else:
                        if label == self.trigger_hand_label:
                            if fist:
                                self.menu_is_open = False
                                self.trigger_hand_label = None
                                self.current_page = None
                                self.fist_ready = {"Left": None, "Right": None}
                                self.debug_state[label] = "FIST→CLOSE"
                                print("👋 菜单已关闭（握拳）")
                            else:
                                self.debug_state[label] = "HOLDING✋"
                                px = int(lm.landmark[9].x * w)
                                py = max(int(lm.landmark[9].y * h) - 60, 30)
                                cv2.putText(frame, "FIST TO CLOSE",
                                            (px - 90, py),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cursor = (
                                int(lm.landmark[8].x * w),
                                int(lm.landmark[8].y * h)
                            )
                            double_tapped = self.check_double_tap(label, pinch, now)
                            self.debug_state[label] = "PINCH👌" if pinch else "cursor👆"

                            if double_tapped:
                                if self.current_page is not None:
                                    if self.hit_test(cursor, self.back_btn):
                                        self.select_flash = {"btn": "__back__", "time": now}
                                        self.current_page = None
                                        print("🔙 返回主菜单")
                                else:
                                    for i, btn in enumerate(self.buttons):
                                        if self.hit_test(cursor, self.get_btn_rect(i)):
                                            self.select_flash = {"btn": btn["text"], "time": now}
                                            self.current_page = btn["text"]
                                            print(f"✅ 进入页面: {btn['text']}")
                                            break

            if self.menu_is_open:
                if self.current_page is None:
                    self.draw_menu(frame, cursor, now)
                else:
                    self.draw_page(frame, cursor, now)

            cv2.putText(frame,
                        f"L: {self.debug_state['Left']}  R: {self.debug_state['Right']}",
                        (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            cv2.putText(frame,
                        f"Menu: {'OPEN' if self.menu_is_open else 'CLOSED'}"
                        f"  Page: {self.current_page or 'main'}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

            cv2.imshow("Holo UI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HoloSystem().run()