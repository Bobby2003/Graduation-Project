import sys
from pathlib import Path
from datetime import datetime
import threading

class TeeStream:
    """
    将输出同时写到终端和日志文件
    """

    def __init__(self, console_stream, file_stream):
        self.console_stream = console_stream
        self.file_stream = file_stream
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            if self.console_stream:
                self.console_stream.write(data)
                self.console_stream.flush()
            if self.file_stream:
                self.file_stream.write(data)
                self.file_stream.flush()

    def flush(self):
        with self._lock:
            if self.console_stream:
                self.console_stream.flush()
            if self.file_stream:
                self.file_stream.flush()

    def isatty(self):
        if self.console_stream:
            return self.console_stream.isatty()
        return False

class LogRedirectManager:
    def __init__(self):
        self.log_file = None
        self.stdout_backup = None
        self.stderr_backup = None
        self.log_path = None
        self.enabled = False

    def start(self, log_dir, log_prefix="log", enabled=True):
        if not enabled:
            return None

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"{log_prefix}_{timestamp}.txt"

        self.log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)

        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr

        sys.stdout = TeeStream(self.stdout_backup, self.log_file)
        sys.stderr = TeeStream(self.stderr_backup, self.log_file)

        self.enabled = True

        sys.stdout.write(f"✅ 日志已保存到: {self.log_path}\n")
        sys.stdout.flush()

        return self.log_path

    def stop(self):
        if not self.enabled:
            return

        try:
            if self.stdout_backup is not None:
                sys.stdout = self.stdout_backup
            if self.stderr_backup is not None:
                sys.stderr = self.stderr_backup
        finally:
            if self.log_file is not None:
                self.log_file.flush()
                self.log_file.close()
                self.log_file = None

        self.enabled = False