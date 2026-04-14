import threading
from queue import Empty

from common.timing import FPSCounter, Timer

class MappingWorker(threading.Thread):
    def __init__(
        self,
        mapping_queue,
        mapper,
        shared_state,
        stop_event,
        enable_log=True,
    ):
        super().__init__(daemon=True)
        self.mapping_queue = mapping_queue
        self.mapper = mapper
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.enable_log = enable_log

        self.fps_counter = FPSCounter()
        self.processed_packets = 0
        self.failed_packets = 0
        self.last_frame_id = None

    def _log(self, msg: str):
        if self.enable_log:
            print(msg)

    def run(self):
        self._log("[MappingWorker] started")

        while not self.stop_event.is_set():
            try:
                pkt = self.mapping_queue.get(timeout=0.05)
            except Empty:
                continue

            timer = Timer()
            timer.start()

            try:
                map_snapshot = self.mapper.update(pkt)
            except Exception as e:
                self.failed_packets += 1
                self._log(f"[MappingWorker] exception on frame={pkt.frame.frame_id}: {e}")
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

                self._log(
                    "[MappingWorker] "
                    f"frame={pkt.frame.frame_id} "
                    f"integrated={integrated} "
                    f"mesh_vertices={mesh_v} "
                    f"time={elapsed * 1000:.2f}ms "
                    f"fps={fps:.2f}"
                )

        self._log("[MappingWorker] stopped")

    def get_stats(self) -> dict:
        return {
            "processed_packets": self.processed_packets,
            "failed_packets": self.failed_packets,
            "last_frame_id": self.last_frame_id,
            "fps": self.fps_counter.fps,
            "integrated_frames": self.mapper.integrated_frames,
        }