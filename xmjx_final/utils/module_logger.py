import sys
import threading

class ModuleLogger:
    """
    模块级日志工具。

    它只负责：
    - 按类别开关日志
    - 按 frame_id interval 节流
    - 统一日志前缀格式

    文件落盘不在这里做。
    文件落盘由 utils/log_redirect.py 的 LogRedirectManager 负责。
    """

    _write_lock = threading.Lock()

    def __init__(self, name: str, logging_config):
        self.name = str(name)
        self.cfg = logging_config

    def _category_enabled(self, category_name: str) -> bool:
        if self.cfg is None:
            return True

        if not getattr(self.cfg, "enable_console", True):
            return False

        category_cfg = getattr(self.cfg, category_name, None)
        if category_cfg is None:
            return True

        return bool(getattr(category_cfg, "enabled", True))

    def _category_interval(self, category_name: str) -> int:
        if self.cfg is None:
            return 1

        category_cfg = getattr(self.cfg, category_name, None)
        if category_cfg is None:
            return 1

        return max(1, int(getattr(category_cfg, "interval", 1)))

    def _should_print(self, category_name: str, frame_id=None, force: bool = False) -> bool:
        if force:
            return self._category_enabled(category_name)

        if not self._category_enabled(category_name):
            return False

        if frame_id is None:
            return True

        interval = self._category_interval(category_name)
        return int(frame_id) % interval == 0

    def _print(self, level: str, message: str):
        text = f"[{self.name}][{level}] {message}\n"

        with ModuleLogger._write_lock:
            sys.stdout.write(text)
            sys.stdout.flush()

    def profile(self, message: str, frame_id=None, force: bool = False):
        if self._should_print("profile", frame_id=frame_id, force=force):
            self._print("PROFILE", message)

    def debug(self, message: str, frame_id=None, force: bool = False):
        if self._should_print("debug", frame_id=frame_id, force=force):
            self._print("DEBUG", message)

    def status(self, message: str, frame_id=None, force: bool = False):
        if self._should_print("status", frame_id=frame_id, force=force):
            self._print("STATUS", message)

    def warning(self, message: str, frame_id=None, force: bool = True):
        if self._should_print("warning", frame_id=frame_id, force=force):
            self._print("WARNING", message)

    def save(self, message: str, frame_id=None, force: bool = True):
        if self._should_print("save", frame_id=frame_id, force=force):
            self._print("SAVE", message)