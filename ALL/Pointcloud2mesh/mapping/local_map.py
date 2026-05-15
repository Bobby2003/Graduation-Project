import threading
import numpy as np
import open3d as o3d

class LocalMap:
    def __init__(
        self,
        voxel_size=0.03,
        max_points=120000,
        local_radius=2.5,
        min_points_for_tracking=1500,
        logger=None,
    ):
        self.voxel_size = float(voxel_size)
        self.max_points = int(max_points)
        self.local_radius = float(local_radius)
        self.min_points_for_tracking = int(min_points_for_tracking)
        self.logger = logger

        self._lock = threading.Lock()
        self._pcd = o3d.t.geometry.PointCloud()

    def has_enough_points(self):
        with self._lock:
            if self._pcd.is_empty():
                return False
            return len(self._pcd.point.positions) >= self.min_points_for_tracking

    def get_tracking_target(self, camera_pos_world=None):
        """
        返回给 tracker 用的 target 点云，坐标系是 world。

        重要：
        如果提供了 camera_pos_world，但相机附近 local_radius 内点数不足，
        直接返回 None，而不是退回整张地图。

        原因：
        退回整张地图会让 ICP 在全局范围内找错误对应，
        容易把当前帧错误吸附到旧地图另一块区域，导致模型交叉/重叠。
        """
        with self._lock:
            if self._pcd.is_empty():
                return None

            pcd = self._pcd.clone()

        if pcd.is_empty():
            return None

        if camera_pos_world is not None:
            pts = pcd.point.positions.cpu().numpy()
            if pts.shape[0] <= 0:
                return None

            c = np.asarray(camera_pos_world, dtype=np.float32).reshape(1, 3)
            dist = np.linalg.norm(pts - c, axis=1)
            mask = dist < self.local_radius
            cnt = int(np.count_nonzero(mask))

            if cnt < self.min_points_for_tracking:
                if self.logger is not None:
                    self.logger.warning(
                        f"target rejected: "
                        f"nearby_points={cnt} < min={self.min_points_for_tracking}, "
                        f"radius={self.local_radius:.2f}. "
                        f"Do not fallback to full map."
                    )
                return None

            mask_t = o3d.core.Tensor(mask, dtype=o3d.core.Dtype.Bool)
            pcd = pcd.select_by_mask(mask_t)

        return pcd

    def integrate_frame_pcd(self, pcd_cam, T_wc):
        """
        把当前帧点云从 camera 坐标变到 world 坐标，然后加入 local map。
        pcd_cam: o3d.t.geometry.PointCloud, camera coordinate
        T_wc: 4x4, camera -> world
        """
        if pcd_cam is None or pcd_cam.is_empty():
            return None

        pcd_world = pcd_cam.clone()
        pcd_world.transform(T_wc)

        # 本帧增量，后续可以传给下一层
        delta_pcd_world = pcd_world.clone()

        with self._lock:
            if self._pcd.is_empty():
                self._pcd = pcd_world
            else:
                self._pcd = self._pcd.append(pcd_world)

            self._pcd = self._pcd.voxel_down_sample(self.voxel_size)

            n = len(self._pcd.point.positions)
            if n > self.max_points:
                # 简单限流：随机保留 max_points
                idx = np.random.choice(n, self.max_points, replace=False)
                idx_t = o3d.core.Tensor(idx, dtype=o3d.core.Dtype.Int64)
                self._pcd = self._pcd.select_by_index(idx_t)

        return delta_pcd_world

    def clear(self):
        with self._lock:
            self._pcd = o3d.t.geometry.PointCloud()