import threading
from queue import Empty
from typing import Callable, Optional

from ..common.timing import FPSCounter, Timer

class MappingWorker(threading.Thread):
    def __init__(
        self,
        mapping_queue,
        mapper,
        shared_state,
        stop_event,
        enable_log=True,
        logger=None,
        mapping_enabled_fn: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(daemon=True)
        self.mapping_queue = mapping_queue
        self.mapper = mapper
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.enable_log = enable_log
        self.logger = logger
        self._mapping_enabled_fn = mapping_enabled_fn or (lambda: True)

        self.fps_counter = FPSCounter()
        self.processed_packets = 0
        self.failed_packets = 0
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
            try:
                pkt = self.mapping_queue.get(timeout=0.05)
            except Empty:
                continue

            if not self._mapping_enabled_fn():
                continue

            timer = Timer()
            timer.start()

            try:
                map_snapshot = self.mapper.update(pkt)
            except Exception as e:
                self.failed_packets += 1
                self.last_error = repr(e)
                self.last_error_frame_id = getattr(pkt.frame, "frame_id", None)
                self._log(
                    f"exception on frame={getattr(pkt.frame, 'frame_id', 'unknown')}: {e}",
                    level="warning",
                    force=True,
                )
                continue

            elapsed = timer.stop()
            fps = self.fps_counter.tick()

            self.shared_state.set_latest_map(map_snapshot)

            self.processed_packets += 1
            self.last_frame_id = pkt.frame.frame_id

            if self.enable_log and (self.processed_packets % 10 == 0):
                map_data = map_snapshot.map_data if map_snapshot is not None else {}
                mesh_v = map_data.get("mesh_vertex_count", 0) if isinstance(map_data, dict) else 0
                integrated = map_data.get("integrated", False) if isinstance(map_data, dict) else False

                if self.logger is not None:
                    self.logger.mapping_state(
                        f"frame={pkt.frame.frame_id} "
                        f"integrated={integrated} "
                        f"mesh_vertices={mesh_v} "
                        f"time={elapsed * 1000:.2f}ms "
                        f"fps={fps:.2f}",
                        frame_id=pkt.frame.frame_id,
                    )

        self._log("stopped", force=True)

    def get_stats(self) -> dict:
        return {
            "processed_packets": self.processed_packets,
            "failed_packets": self.failed_packets,
            "last_frame_id": self.last_frame_id,
            "fps": self.fps_counter.fps,
            "integrated_frames": self.mapper.integrated_frames,
            "last_error": self.last_error,
            "last_error_frame_id": self.last_error_frame_id,
        }