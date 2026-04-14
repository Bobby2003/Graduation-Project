import time
from typing import Optional
import numpy as np

from common.types import RGBDFrame

class InputAdapter:
    """
    将上游 scanner 输出统一适配为 RGBDFrame。

    当前重点兼容：
    - UnifiedDepthScanner.get_rgb_and_depth()

    统一时间语义：
    - device_timestamp: 主时间轴
    - host_timestamp: 仅用于调试/兜底
    """

    def __init__(
        self,
        scanner,
        timestamp_unit: str = "auto",
        validate: bool = False,
        print_timestamp_debug_once: bool = False,
    ):
        self.scanner = scanner
        self.timestamp_unit = timestamp_unit
        self.validate_frame = validate
        self.print_timestamp_debug_once = print_timestamp_debug_once

        self._last_raw_ts: Optional[float] = None
        self._last_norm_ts: Optional[float] = None
        self._frame_counter_fallback = 0
        self._printed_ts_debug = False

    def _normalize_device_timestamp(self, ts) -> Optional[float]:
        if ts is None:
            return None

        ts = float(ts)

        if self.timestamp_unit == "sec":
            return ts
        elif self.timestamp_unit == "ms":
            return ts / 1e3
        elif self.timestamp_unit == "us":
            return ts / 1e6
        elif self.timestamp_unit == "auto":
            # 启发式判断：
            # sec: 一般 < 1e6
            # ms/us: 常见设备时间戳往往更大
            if ts >= 1e12:
                return ts / 1e6
            if ts >= 1e9:
                return ts / 1e6
            if ts >= 1e6:
                return ts / 1e6
            if ts >= 1e3:
                return ts / 1e3
            return ts
        else:
            raise ValueError(f"Unknown timestamp_unit: {self.timestamp_unit}")

    def _extract_frame_id(self, data: dict) -> int:
        if "frame_id" in data:
            return int(data["frame_id"])
        if "frame_index" in data:
            return int(data["frame_index"])

        self._frame_counter_fallback += 1
        return self._frame_counter_fallback

    def _extract_timestamp(self, data: dict) -> Optional[float]:
        if "device_timestamp" in data:
            return self._normalize_device_timestamp(data["device_timestamp"])
        if "timestamp" in data:
            return self._normalize_device_timestamp(data["timestamp"])
        return None

    def _extract_color(self, data: dict) -> np.ndarray:
        if "color_rgb" in data:
            color = data["color_rgb"]
        elif "color" in data:
            color = data["color"]
        else:
            raise KeyError("No color field found in input data")

        color = np.asarray(color)
        if color.dtype != np.uint8:
            color = color.astype(np.uint8, copy=False)
        return color

    def _extract_depth(self, data: dict) -> np.ndarray:
        if "depth" in data:
            depth = data["depth"]
        elif "raw_depth" in data:
            depth = data["raw_depth"]
        else:
            raise KeyError("No depth field found in input data")

        depth = np.asarray(depth)
        if depth.dtype not in (np.float32, np.float64, np.uint16):
            depth = depth.astype(np.float32, copy=False)
        return depth

    def _extract_mask(self, data: dict, depth: np.ndarray) -> np.ndarray:
        if "final_mask" in data:
            mask = data["final_mask"]
        elif "mask" in data:
            mask = data["mask"]
        else:
            mask = (depth > 0).astype(np.uint8)

        mask = np.asarray(mask)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8, copy=False)
        return mask

    def _print_timestamp_debug(self, data, norm_ts):
        if self._printed_ts_debug or (not self.print_timestamp_debug_once):
            return

        raw_ts = None
        if isinstance(data, dict):
            raw_ts = data.get("device_timestamp", data.get("timestamp", None))

        print(
            "[InputAdapter] timestamp debug: "
            f"raw={raw_ts}, normalized={norm_ts}, unit_mode={self.timestamp_unit}"
        )
        self._printed_ts_debug = True

    def get_frame(self) -> Optional[RGBDFrame]:
        host_ts = time.perf_counter()
        data = self.scanner.get_rgb_and_depth()

        if data is None:
            return None

        # 兼容极简旧接口：(color, depth)
        if isinstance(data, tuple) and len(data) == 2:
            color, depth = data
            color = np.asarray(color)
            depth = np.asarray(depth)

            self._frame_counter_fallback += 1
            device_ts = host_ts

            frame = RGBDFrame(
                frame_id=self._frame_counter_fallback,
                device_timestamp=device_ts,
                host_timestamp=host_ts,
                color=color.astype(np.uint8, copy=False),
                depth=depth,
                mask=(depth > 0).astype(np.uint8),
                extras={
                    "raw_input": data,
                    "timestamp_source": "host_fallback",
                },
            )
            if self.validate_frame:
                frame.validate()
            return frame

        if not isinstance(data, dict):
            raise TypeError(
                f"scanner.get_rgb_and_depth() must return dict or (color, depth), got {type(data)}"
            )

        frame_id = self._extract_frame_id(data)
        device_ts = self._extract_timestamp(data)
        color = self._extract_color(data)
        depth = self._extract_depth(data)
        mask = self._extract_mask(data, depth)

        if device_ts is None:
            # 如果上游没给设备时间戳，则暂时回退到 host time
            device_ts = host_ts
            ts_source = "host_fallback"
        else:
            ts_source = "device"

        frame = RGBDFrame(
            frame_id=frame_id,
            device_timestamp=float(device_ts),
            host_timestamp=float(host_ts),
            color=color,
            depth=depth,
            mask=mask,
            extras={
                "raw_input": data,
                "timestamp_source": ts_source,
                "frame_info": data.get("frame_info", None),
                "raw_depth": data.get("raw_depth", None),
            }
        )

        if self.validate_frame:
            frame.validate()

        self._last_raw_ts = data.get("device_timestamp", data.get("timestamp", None))
        self._last_norm_ts = frame.device_timestamp
        self._print_timestamp_debug(data, frame.device_timestamp)

        return frame

    def debug_timestamp_info(self) -> dict:
        return {
            "timestamp_unit": self.timestamp_unit,
            "last_raw_ts": self._last_raw_ts,
            "last_norm_ts": self._last_norm_ts,
        }