import open3d as o3d
import numpy as np
import cv2

class RGBDPreprocessor:
    """
    将 numpy 的 color/depth 预处理为 Open3D RGBDImage。

    这个版本与旧版的关键区别：
    1. 真正按目标 width / height 对输入图像做 resize
    2. intrinsic 固定为目标分辨率对应的内参
    3. 不再根据输入图像尺寸反向修改 intrinsic
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
        logger=None,
    ):
        # 目标输出分辨率
        self.target_width = int(width)
        self.target_height = int(height)

        # 目标分辨率下对应的相机内参
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.target_width,
            self.target_height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )

        self.input_color_is_bgr = input_color_is_bgr
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.min_valid_ratio = min_valid_ratio
        self.logger = logger

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

    def get_intrinsic(self):
        return self.intrinsic

    def _auto_depth_scale(self, depth_np):
        valid = depth_np[depth_np > 0]
        vmax = float(np.percentile(valid, 95)) if valid.size > 0 else 0.0

        # 启发式判断：
        # 如果 95 分位大于 20，通常原始深度是 mm
        self.depth_scale = 1000.0 if vmax > 20 else 1.0

        self._log(
            f"auto depth_scale: dtype={depth_np.dtype}, "
            f"p95={vmax:.3f}, depth_scale={self.depth_scale}",
            force=True,
        )

    def _resize_if_needed(self, color_np, depth_np):
        """
        若输入分辨率与目标分辨率不同，则执行 resize：
        - color: INTER_LINEAR
        - depth: INTER_NEAREST
        """
        h, w = depth_np.shape[:2]

        if color_np.shape[:2] != (h, w):
            return None, None, {
                "reason": "color_depth_shape_mismatch",
                "color_shape": tuple(color_np.shape),
                "depth_shape": tuple(depth_np.shape),
            }

        if (w, h) == (self.target_width, self.target_height):
            return color_np, depth_np, None

        resized_color = cv2.resize(
            color_np,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        resized_depth = cv2.resize(
            depth_np,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return resized_color, resized_depth, {
            "resized": True,
            "input_shape": (h, w),
            "output_shape": (self.target_height, self.target_width),
        }

    def preprocess(self, color_np, depth_np):
        """
        Returns:
            rgbd, info
            - rgbd: Open3D RGBDImage or None
            - info: dict
        """
        if color_np is None or depth_np is None:
            return None, {"reason": "color_or_depth_is_none"}

        if not isinstance(color_np, np.ndarray) or not isinstance(depth_np, np.ndarray):
            return None, {
                "reason": "input_not_numpy_array",
                "color_type": str(type(color_np)),
                "depth_type": str(type(depth_np)),
            }

        # 颜色通道处理：如果输入是 BGR，转成 RGB
        if self.input_color_is_bgr and color_np.ndim == 3 and color_np.shape[2] == 3:
            color_np = color_np[:, :, ::-1].copy()

        # 保证内存连续
        color_np = np.ascontiguousarray(color_np)
        depth_np = np.ascontiguousarray(depth_np)

        # 自动推断 depth_scale，只做一次
        if self.depth_scale is None:
            self._auto_depth_scale(depth_np)

        # 深度 dtype 规范化
        if depth_np.dtype not in (np.uint16, np.float32, np.float64):
            depth_np = depth_np.astype(np.float32)

        if np.issubdtype(depth_np.dtype, np.floating):
            depth_np = np.nan_to_num(
                depth_np,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)

        # resize 到目标分辨率
        color_np, depth_np, resize_info = self._resize_if_needed(color_np, depth_np)
        if color_np is None or depth_np is None:
            return None, resize_info

        h, w = depth_np.shape[:2]

        # 再次确保输出尺寸正确
        if (w, h) != (self.target_width, self.target_height):
            return None, {
                "reason": "resize_failed",
                "expected_shape": (self.target_height, self.target_width),
                "actual_shape": (h, w),
            }

        # 计算有效深度比例
        valid_ratio = float(np.count_nonzero(depth_np)) / float(depth_np.size)
        if valid_ratio < self.min_valid_ratio:
            return None, {
                "reason": "too_few_valid_depth",
                "valid_ratio": valid_ratio,
                "shape": (h, w),
            }

        # 若彩色不是 uint8，可尝试转成 uint8
        if color_np.dtype != np.uint8:
            if np.issubdtype(color_np.dtype, np.floating):
                color_np = np.clip(color_np, 0.0, 255.0).astype(np.uint8)
            else:
                color_np = color_np.astype(np.uint8)

        color_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(depth_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False,
        )

        info = {
            "valid_ratio": valid_ratio,
            "depth_scale": self.depth_scale,
            "depth_trunc": self.depth_trunc,
            "shape": (h, w),
            "target_shape": (self.target_height, self.target_width),
            "resized": resize_info is not None and resize_info.get("resized", False),
        }

        if resize_info is not None:
            info.update(resize_info)

        return rgbd, info