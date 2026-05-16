import time
import threading
from typing import Callable, Optional

from ..common.types import MapPacket
from ..common.timing import FPSCounter, Timer

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
        logger=None,
        tracking_enabled_fn: Optional[Callable[[], bool]] = None,
        tracking_result_observer: Optional[Callable[..., None]] = None,
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
        self.logger = logger

        self._tracking_enabled_fn = tracking_enabled_fn or (lambda: True)
        self._tracking_result_observer = tracking_result_observer

        self.fps_counter = FPSCounter()
        self.processed_frames = 0
        self.pushed_to_mapping = 0
        self.failed_frames = 0
        self.last_frame_id = None

        self.last_error = None
        self.last_error_frame_id = None

    def _log(self, msg: str, level: str = "status", force: bool = False):
        if not self.enable_log:
            return

        if self.logger is None:
            return

        if level == "warning":
            self.logger.warning(msg, force=force)
        elif level == "debug":
            self.logger.debug(msg, force=force)
        elif level == "profile":
            if hasattr(self.logger, "profile"):
                self.logger.profile(msg, force=force)
            else:
                self.logger.status(msg, force=force)
        else:
            self.logger.status(msg, force=force)

    def run(self):
        self._log("started", force=True)

        while not self.stop_event.is_set():
            frame = self.latest_frame_slot.get_latest()

            if frame is None:
                time.sleep(self.sleep_ms / 1000.0)
                continue

            if self.last_frame_id == frame.frame_id:
                time.sleep(self.sleep_ms / 1000.0)
                continue

            if not self._tracking_enabled_fn():
                time.sleep(self.sleep_ms / 1000.0)
                continue

            timer = Timer()
            timer.start()

            try:
                tracking = self.tracker.track(frame)
            except Exception as e:
                self.failed_frames += 1
                self.last_error = repr(e)
                self.last_error_frame_id = getattr(frame, "frame_id", None)
                self._log(
                    f"exception on frame={getattr(frame, 'frame_id', 'unknown')}: {e}",
                    level="warning",
                    force=True,
                )
                continue

            elapsed = timer.stop()
            fps = self.fps_counter.tick()

            self.shared_state.set_latest_tracking(tracking)

            if self._tracking_result_observer is not None:
                try:
                    self._tracking_result_observer(tracking)
                except Exception:
                    pass

            self.processed_frames += 1
            self.last_frame_id = frame.frame_id

            if not tracking.success:
                self.failed_frames += 1

            extras = tracking.extras if tracking.extras is not None else {}

            fitness = float(extras.get("fitness", 1.0))
            rmse = float(extras.get("inlier_rmse", 0.0))
            icp_estimation = extras.get("icp_estimation", "")

            mapping_quality_ok = True

            if extras.get("tracker_backend") == "gpu_icp":
                mapping_quality_ok = (
                    fitness >= 0.20 and
                    rmse <= 0.05 and
                    icp_estimation != "icp_failed"
                )

            should_push_mapping = (
                tracking.success and
                mapping_quality_ok and
                (
                    tracking.mode == "init"
                    or (frame.frame_id % self.mapping_stride == 0)
                )
            )

            if should_push_mapping:
                pkt = MapPacket(frame=frame, tracking=tracking)
                self.mapping_queue.put_drop_oldest(pkt)
                self.pushed_to_mapping += 1

            # 控制日志量：不是每帧都狂打
            if self.enable_log and (self.processed_frames % 10 == 0 or not tracking.success):
                if self.logger is not None:
                    self.logger.tracking_state(
                        f"frame={frame.frame_id} "
                        f"mode={tracking.mode} "
                        f"success={tracking.success} "
                        f"score={tracking.score:.3f} "
                        f"valid={frame.valid_pixel_count} "
                        f"time={elapsed * 1000:.2f}ms "
                        f"fps={fps:.2f}",
                        frame_id=frame.frame_id,
                        force=not tracking.success,
                    )

        self._log("stopped", force=True)

    def get_stats(self) -> dict:
        return {
            "processed_frames": self.processed_frames,
            "pushed_to_mapping": self.pushed_to_mapping,
            "failed_frames": self.failed_frames,
            "last_frame_id": self.last_frame_id,
            "fps": self.fps_counter.fps,
            "last_error": self.last_error,
            "last_error_frame_id": self.last_error_frame_id,
        }