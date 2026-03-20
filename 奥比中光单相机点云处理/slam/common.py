import numpy as np
import open3d as o3d
import cv2
import os

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
try:
    o3d.utility.set_num_threads(os.cpu_count())
except Exception:
    pass

# ── 相机参数 ──────────────────────────────────
FX, FY        = 525.0, 525.0
CX, CY        = 319.5, 239.5
WIDTH, HEIGHT = 640, 480
K             = np.array([[FX,0,CX],[0,FY,CY],[0,0,1]], dtype=np.float64)
DEPTH_SCALE   = 0.001
MIN_DEPTH, MAX_DEPTH = 0.3, 8.0

# ── 运动合理性上限 ────────────────────────────
MAX_TRANS = 0.40   # 40cm
MAX_ROT   = 0.80   # ~46°


class KeyFrame:
    def __init__(self, kf_id, pose, depth_m, color_bgr, kps, descs):
        self.kf_id      = kf_id
        self.pose       = pose.copy()
        self.depth_m    = depth_m
        self.color_bgr  = color_bgr
        self.kps        = kps
        self.descs      = descs
        self.bow_vec    = None
        self._pcd_cache = {}

    def get_pcd(self, stride=3):
        if stride in self._pcd_cache:
            return self._pcd_cache[stride]
        rows, cols = self.depth_m.shape
        u, v  = np.meshgrid(np.arange(0,cols,stride), np.arange(0,rows,stride))
        d     = self.depth_m[::stride, ::stride]
        valid = (d > MIN_DEPTH) & (d < MAX_DEPTH)
        z = d[valid]; x = (u[valid]-CX)*z/FX; y = (v[valid]-CY)*z/FY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack([x,y,z], axis=-1))
        self._pcd_cache[stride] = pcd
        return pcd


def refine_icp(pcd_ref, pcd_cur, T_init, max_dist=0.03, max_iter=50):
    for pcd in [pcd_ref, pcd_cur]:
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20))
    result = o3d.pipelines.registration.registration_icp(
        pcd_cur, pcd_ref, max_dist, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return result.transformation, result.fitness > 0.5, result.fitness


def is_motion_valid(T_rel):
    t = np.linalg.norm(T_rel[:3, 3])
    r = np.arccos(np.clip((np.trace(T_rel[:3,:3])-1)/2, -1, 1))
    if t > MAX_TRANS or r > MAX_ROT:
        print(f"[Motion] 异常拒绝: t={t*100:.1f}cm r={np.degrees(r):.1f}°")
        return False
    return True
