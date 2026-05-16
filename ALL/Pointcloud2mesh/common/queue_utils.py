import threading
from queue import Queue, Full, Empty

class LatestFrameSlot:
    """
    只保留最新一帧。
    tracking 总是拿最新的，避免被历史帧拖死。
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._item = None

    def put(self, item):
        with self._lock:
            self._item = item

    def get_latest(self):
        with self._lock:
            item = self._item
            self._item = None
            return item

class BoundedDropQueue:
    """
    有界队列，满时丢最旧。
    适合 mapping 这种异步消费者。
    """
    def __init__(self, maxsize=4):
        self._q = Queue(maxsize=maxsize)

    def put_drop_oldest(self, item):
        try:
            self._q.put_nowait(item)
        except Full:
            try:
                self._q.get_nowait()
            except Empty:
                pass
            self._q.put_nowait(item)

    def get(self, timeout=None):
        return self._q.get(timeout=timeout)

    def empty(self):
        return self._q.empty()

    def qsize(self):
        return self._q.qsize()

    def clear(self):
        """Drop all pending packets (used after reconstruction reset)."""
        while True:
            try:
                self._q.get_nowait()
            except Empty:
                break