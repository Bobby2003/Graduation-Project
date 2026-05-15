from dataclasses import dataclass, field
from typing import Optional, Any, Dict
import numpy as np

def make_identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float64)

@dataclass
class RGBDFrame:
    frame_id: int
    device_timestamp: float
    host_timestamp: Optional[float]
    color: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self):
        return self.depth.shape

    @property
    def valid_pixel_count(self) -> int:
        return int((self.mask > 0).sum())

    @property
    def has_valid_depth(self) -> bool:
        return self.valid_pixel_count > 0

    def validate(self):
        if self.color is None or self.depth is None or self.mask is None:
            raise ValueError("RGBDFrame contains None fields")

        if self.depth.ndim != 2:
            raise ValueError(f"depth must be HxW, got shape={self.depth.shape}")

        if self.mask.ndim != 2:
            raise ValueError(f"mask must be HxW, got shape={self.mask.shape}")

        h, w = self.depth.shape
        if self.mask.shape != (h, w):
            raise ValueError(
                f"mask shape {self.mask.shape} does not match depth shape {(h, w)}"
            )

        if self.color.ndim != 3 or self.color.shape[2] != 3:
            raise ValueError(f"color must be HxWx3, got shape={self.color.shape}")

        if self.color.shape[0] != h or self.color.shape[1] != w:
            raise ValueError(
                f"color shape {self.color.shape[:2]} does not match depth shape {(h, w)}"
            )

@dataclass
class TrackingResult:
    frame_id: int
    timestamp: float
    success: bool
    T_wc: np.ndarray = field(default_factory=make_identity_pose)
    score: float = 0.0
    mode: str = "tracking"   # init / tracking / lost / relocalized
    extras: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if self.T_wc.shape != (4, 4):
            raise ValueError(f"T_wc must be 4x4, got shape={self.T_wc.shape}")

@dataclass
class MapPacket:
    frame: RGBDFrame
    tracking: TrackingResult

@dataclass
class MapSnapshot:
    frame_id: int
    timestamp: float
    map_data: Any
    extras: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)