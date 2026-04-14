import time
import threading

from common.types import MapPacket
from common.timing import FPSCounter, Timer

class TrackingWorker(threading.Thread):
    """
    高优先级 tracking 线程：
    - 永远优先处理最新帧
    - 不追历史
    - 成功结果送 mapping
    """

    def __init__(
        self,
        latest_frame_slot,
        tracker,
        shared_state,
        mapping_queue,
        stop_event,
        sleep_ms=1,
        enable_log=True,
        mapping_stride=1,
    ):
        super().__init__(daemon=True)
        self.latest_frame_slot = latest_frame_slot
        self.tracker = tracker
        self.shared_state = shared_state
        self.mapping_queue = mapping_queue
        self.stop_event = stop_event
        self.sleep_ms = sleep_ms
        self.enable_log = enable_log
        self.mapping_stride = max(1, int(mapping_stride))

        self.fps_counter = FPSCounter()
        self.processed_frames = 0
        self.pushed_to_mapping = 0
        self.failed_frames = 0
        self.last_frame_id = None

    def _log(self, msg: str):
        if self.enable_log:
            print(msg)

    def run(self):
        self._log("[TrackingWorker] started")

        while not self.stop_event.is_set():
            frame = self.latest_frame_slot.get_latest()

            if frame is None:
                time.sleep(self.sleep_ms / 1000.0)
                continue

            timer = Timer()
            timer.start()

            try:
                tracking = self.tracker.track(frame)
            except Exception as e:
                self.failed_frames += 1
                self._log(f"[TrackingWorker] exception on frame={getattr(frame, 'frame_id', 'unknown')}: {e}")
                continue

            elapsed = timer.stop()
            fps = self.fps_counter.tick()

            self.shared_state.set_latest_tracking(tracking)

            self.processed_frames += 1
            self.last_frame_id = frame.frame_id

            if not tracking.success:
                self.failed_frames += 1

            should_push_mapping = (
                tracking.success and
                (frame.frame_id % self.mapping_stride == 0)
            )

            if should_push_mapping:
                pkt = MapPacket(frame=frame, tracking=tracking)
                self.mapping_queue.put_drop_oldest(pkt)
                self.pushed_to_mapping += 1

            # 控制日志量：不是每帧都狂打
            if self.enable_log and (self.processed_frames % 10 == 0 or not tracking.success):
                self._log(
                    "[TrackingWorker] "
                    f"frame={frame.frame_id} "
                    f"mode={tracking.mode} "
                    f"success={tracking.success} "
                    f"score={tracking.score:.3f} "
                    f"valid={frame.valid_pixel_count} "
                    f"time={elapsed*1000:.2f}ms "
                    f"fps={fps:.2f}"
                )

        self._log("[TrackingWorker] stopped")

    def get_stats(self) -> dict:
        return {
            "processed_frames": self.processed_frames,
            "pushed_to_mapping": self.pushed_to_mapping,
            "failed_frames": self.failed_frames,
            "last_frame_id": self.last_frame_id,
            "fps": self.fps_counter.fps,
        }