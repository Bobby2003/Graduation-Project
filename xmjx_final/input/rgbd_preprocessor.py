import open3d as o3d
import numpy as np

class RGBDPreprocessor:
    """
    将 numpy 的 color/depth 预处理为 Open3D RGBDImage。
    这里迁移自你的原型 preprocess_frame()，但拆成独立模块。
    """

    def __init__(
        self,
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        input_color_is_bgr=False,
        depth_scale=None,
        depth_trunc=2.0,
        min_valid_ratio=0.01,
    ):
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        self.input_color_is_bgr = input_color_is_bgr
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.min_valid_ratio = min_valid_ratio

    def get_intrinsic(self):
        return self.intrinsic

    def _ensure_intrinsic_matches(self, h, w):
        if (w != self.intrinsic.width) or (h != self.intrinsic.height):
            fx, fy = self.intrinsic.get_focal_length()
            cx, cy = self.intrinsic.get_principal_point()
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            print(f"[RGBDPreprocessor] intrinsic resized to {w}x{h}")

    def _auto_depth_scale(self, depth_np):
        valid = depth_np[depth_np > 0]
        vmax = float(np.percentile(valid, 95)) if valid.size > 0 else 0.0
        # 简单启发式：如果 95 分位非常大，通常原始深度单位是 mm
        self.depth_scale = 1000.0 if vmax > 20 else 1.0
        print(
            f"[RGBDPreprocessor] auto depth_scale: "
            f"dtype={depth_np.dtype}, p95={vmax:.3f}, depth_scale={self.depth_scale}"
        )

    def preprocess(self, color_np, depth_np):
        """
        Returns:
            rgbd, info
            - rgbd: Open3D RGBDImage or None
            - info: dict
        """
        if color_np is None or depth_np is None:
            return None, {"reason": "color_or_depth_is_none"}

        if self.input_color_is_bgr and color_np.ndim == 3 and color_np.shape[2] == 3:
            color_np = color_np[:, :, ::-1].copy()

        color_np = np.ascontiguousarray(color_np)
        depth_np = np.ascontiguousarray(depth_np)

        if self.depth_scale is None:
            self._auto_depth_scale(depth_np)

        if depth_np.dtype not in (np.uint16, np.float32, np.float64):
            depth_np = depth_np.astype(np.float32)

        if np.issubdtype(depth_np.dtype, np.floating):
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        h, w = depth_np.shape[:2]
        self._ensure_intrinsic_matches(h, w)

        valid_ratio = float(np.count_nonzero(depth_np)) / float(depth_np.size)
        if valid_ratio < self.min_valid_ratio:
            return None, {
                "reason": "too_few_valid_depth",
                "valid_ratio": valid_ratio,
            }

        color_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(depth_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        return rgbd, {
            "valid_ratio": valid_ratio,
            "depth_scale": self.depth_scale,
            "depth_trunc": self.depth_trunc,
            "shape": (h, w),
        }