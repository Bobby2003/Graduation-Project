import time

class FPSCounter:
    def __init__(self, update_interval=1.0):
        self.update_interval = update_interval
        self.last_t = time.perf_counter()
        self.count = 0
        self.fps = 0.0

    def tick(self):
        now = time.perf_counter()
        self.count += 1
        dt = now - self.last_t

        if dt >= self.update_interval:
            self.fps = self.count / dt
            self.count = 0
            self.last_t = now

        return self.fps

class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.perf_counter()

    def stop(self):
        if self.t0 is None:
            return 0.0
        return time.perf_counter() - self.t0