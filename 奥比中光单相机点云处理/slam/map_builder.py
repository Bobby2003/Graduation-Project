import threading
import numpy as np
import open3d as o3d
import cv2
from common import MIN_DEPTH, MAX_DEPTH, refine_icp


class MapBuilder:
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size
        self.global_pcd = o3d.geometry.PointCloud()
        self.lock       = threading.Lock()
        self._dirty     = False
        self._last_kf   = None

    def _colorize(self, kf, stride):
        pcd = kf.get_pcd(stride=stride)
        if len(pcd.points) == 0:
            return None
        rgb    = cv2.cvtColor(kf.color_bgr, cv2.COLOR_BGR2RGB)
        colors = rgb[::stride, ::stride].reshape(-1, 3).astype(np.float64) / 255.0
        n = len(pcd.points)
        if len(colors) > n:   colors = colors[:n]
        elif len(colors) < n: colors = np.vstack([colors, np.zeros((n-len(colors), 3))])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def add_keyframe(self, kf):
        pcd = self._colorize(kf, stride=2)
        if pcd is None:
            return

        if self._last_kf is not None:
            T_rel = np.linalg.inv(self._last_kf.pose) @ kf.pose
            _, ok, fitness = refine_icp(
                self._last_kf.get_pcd(stride=3), kf.get_pcd(stride=3), T_rel)
            if not ok:
                print(f"[Map] KF{kf.kf_id} 对齐质量差(fitness:{fitness:.2f})，跳过")
                return

        pcd.transform(kf.pose)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        with self.lock:
            self.global_pcd += pcd
            self._dirty = True
        self._last_kf = kf

    def rebuild(self, keyframes):
        new_pcd = o3d.geometry.PointCloud()
        prev_kf = None
        skipped = 0
        for kf in keyframes:
            pcd = self._colorize(kf, stride=3)
            if pcd is None:
                continue
            if prev_kf is not None:
                T_rel = np.linalg.inv(prev_kf.pose) @ kf.pose
                _, ok, fitness = refine_icp(
                    prev_kf.get_pcd(stride=3), kf.get_pcd(stride=3), T_rel)
                if not ok:
                    skipped += 1
                    continue
            pcd.transform(kf.pose)
            new_pcd += pcd
            prev_kf = kf
        new_pcd = new_pcd.voxel_down_sample(self.voxel_size)
        with self.lock:
            self.global_pcd = new_pcd
            self._dirty     = True
        print(f"[Map] 重建完成: {len(new_pcd.points)} 点  跳过:{skipped}帧")

    def get_snapshot(self):
        with self.lock:
            self._dirty = False
            snap = o3d.geometry.PointCloud()
            snap.points = o3d.utility.Vector3dVector(
                np.asarray(self.global_pcd.points).copy())
            snap.colors = o3d.utility.Vector3dVector(
                np.asarray(self.global_pcd.colors).copy())
            return snap

    def is_dirty(self):
        return self._dirty

    def save(self, path):
        with self.lock:
            o3d.io.write_point_cloud(path, self.global_pcd)
        print(f"[Map] 已保存: {path}  ({len(self.global_pcd.points)} 点)")