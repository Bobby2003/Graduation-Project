import threading
import time
import traceback

class RenderWorker(threading.Thread):
    """
    渲染执行器：
    - 可作为线程 run()
    - 也可在主线程反复调用 run_once()
    """

    def __init__(
        self,
        shared_state,
        renderer,
        stop_event,
        render_fps=20.0,
        enable_status_log=False,
        status_log_interval=30,
        logger=None,
    ):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.renderer = renderer
        self.stop_event = stop_event
        self.render_interval = 1.0 / max(render_fps, 1e-6)
        self.enable_status_log = enable_status_log
        self.status_log_interval = max(1, int(status_log_interval))
        self.logger = logger

        self.render_count = 0
        self._last_render_ts = 0.0

    def _log(self, msg: str, level: str = "status", force: bool = False):
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

    def run_once(self, force=False):
        """
        执行一次渲染。

        Returns:
            True  -> 继续
            False -> 窗口关闭或异常，建议停止 pipeline
        """
        try:
            now = time.monotonic()
            if (not force) and (now - self._last_render_ts < self.render_interval):
                return True

            self._last_render_ts = now

            tracking = self.shared_state.get_latest_tracking()
            map_snapshot = self.shared_state.get_latest_map()

            alive = self.renderer.render(tracking, map_snapshot)
            self.render_count += 1

            if self.enable_status_log and (self.render_count % self.status_log_interval == 0):
                self.renderer.print_status(tracking, map_snapshot)

            if alive is False:
                self._log(
                    "window closed, stopping pipeline",
                    level="status",
                    force=True,
                )
                self.stop_event.set()
                return False

            return True

        except Exception:
            self._log(
                "exception in render:\n" + traceback.format_exc(),
                level="warning",
                force=True,
            )
            self.stop_event.set()
            return False

    def run(self):
        try:
            while not self.stop_event.is_set():
                ok = self.run_once()
                if not ok:
                    break

                time.sleep(0.001)
        finally:
            self.renderer.close()