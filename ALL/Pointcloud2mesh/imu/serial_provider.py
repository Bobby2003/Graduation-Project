import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IMUSample:
    timestamp: float
    quaternion_wxyz: np.ndarray
    raw_frame: bytes


class SerialIMUProvider:
    """
    Serial reader for the IMU protocol in communication spec.

    Frame layout:
    - header: 0x7E 0x23
    - length: total frame length in bytes
    - command: 0x16 for quaternion
    - payload: four little-endian float32 values, w/x/y/z
    - checksum: low byte of sum(header through byte before checksum)
    """

    HEADER = b"\x7e\x23"
    CMD_QUATERNION = 0x16

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        read_window_sec: float = 0.5,
        logger=None,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.read_window_sec = float(read_window_sec)
        self.logger = logger
        self._serial = None
        self._buffer = bytearray()

    def _log_warning(self, msg: str):
        if self.logger is not None:
            self.logger.warning(msg, force=True)

    def open(self):
        if self._serial is not None:
            return

        try:
            import serial
        except Exception as exc:
            raise RuntimeError(
                "pyserial is required for serial IMU support. Install package 'pyserial'."
            ) from exc

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

    def close(self):
        if self._serial is None:
            return
        try:
            self._serial.close()
        finally:
            self._serial = None
            self._buffer.clear()

    def read_quaternion_sample(self, timeout_sec: Optional[float] = None) -> Optional[IMUSample]:
        if timeout_sec is None:
            timeout_sec = self.read_window_sec

        self.open()
        deadline = time.perf_counter() + float(timeout_sec)

        while time.perf_counter() < deadline:
            waiting = getattr(self._serial, "in_waiting", 0) or 1
            chunk = self._serial.read(waiting)
            if chunk:
                self._buffer.extend(chunk)

            sample = self._pop_quaternion_sample()
            if sample is not None:
                return sample

            time.sleep(0.001)

        return None

    def _pop_quaternion_sample(self) -> Optional[IMUSample]:
        while True:
            start = self._buffer.find(self.HEADER)
            if start < 0:
                if len(self._buffer) > 1:
                    del self._buffer[:-1]
                return None

            if start > 0:
                del self._buffer[:start]

            if len(self._buffer) < 4:
                return None

            frame_len = int(self._buffer[2])
            if frame_len < 5:
                del self._buffer[0]
                continue

            if len(self._buffer) < frame_len:
                return None

            frame = bytes(self._buffer[:frame_len])
            del self._buffer[:frame_len]

            if not self._valid_checksum(frame):
                self._log_warning("serial IMU checksum mismatch; dropping frame")
                continue

            if frame[3] != self.CMD_QUATERNION:
                continue

            if len(frame) != 0x15:
                self._log_warning(f"unexpected quaternion frame length: {len(frame)}")
                continue

            quat = np.asarray(struct.unpack("<ffff", frame[4:20]), dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if norm <= 1e-12 or not np.isfinite(norm):
                self._log_warning("invalid IMU quaternion; dropping frame")
                continue

            quat /= norm
            return IMUSample(
                timestamp=time.perf_counter(),
                quaternion_wxyz=quat,
                raw_frame=frame,
            )

    @staticmethod
    def _valid_checksum(frame: bytes) -> bool:
        if len(frame) < 2:
            return False
        expected = sum(frame[:-1]) & 0xFF
        return expected == frame[-1]
