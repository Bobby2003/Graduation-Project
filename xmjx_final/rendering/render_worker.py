import threading
import time

class RenderWorker(threading.Thread):
    """
    独立渲染线程：
    - 定频读取 shared_state 最新 tracking / map
    - 更新 Open3D 窗口
    - 不消费历史任务
    """

    def __init__(
        self,
        shared_state,
        renderer,
        stop_event,
        render_fps=20.0,
        enable_status_log=False,
        status_log_interval=30,
    ):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.renderer = renderer
        self.stop_event = stop_event
        self.render_interval = 1.0 / max(render_fps, 1e-6)
        self.enable_status_log = enable_status_log
        self.status_log_interval = max(1, int(status_log_interval))

        self.render_count = 0

    def run(self):
        try:
            while not self.stop_event.is_set():
                tracking = self.shared_state.get_latest_tracking()
                map_snapshot = self.shared_state.get_latest_map()

                alive = self.renderer.render(tracking, map_snapshot)
                self.render_count += 1

                if self.enable_status_log and (self.render_count % self.status_log_interval == 0):
                    self.renderer.print_status(tracking, map_snapshot)

                if alive is False:
                    print("[RenderWorker] window closed, stopping pipeline")
                    self.stop_event.set()
                    break

                time.sleep(self.render_interval)

        finally:
            self.renderer.close()