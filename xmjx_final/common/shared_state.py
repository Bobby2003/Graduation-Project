import threading
from typing import Optional
from common.types import TrackingResult, MapSnapshot

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_tracking: Optional[TrackingResult] = None
        self._latest_map: Optional[MapSnapshot] = None

    def set_latest_tracking(self, tracking: TrackingResult):
        with self._lock:
            self._latest_tracking = tracking

    def get_latest_tracking(self) -> Optional[TrackingResult]:
        with self._lock:
            return self._latest_tracking

    def set_latest_map(self, map_snapshot: MapSnapshot):
        with self._lock:
            self._latest_map = map_snapshot

    def get_latest_map(self) -> Optional[MapSnapshot]:
        with self._lock:
            return self._latest_map