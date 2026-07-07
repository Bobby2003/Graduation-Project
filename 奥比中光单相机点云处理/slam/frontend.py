import cv2
import numpy as np
from common import K, MIN_DEPTH, MAX_DEPTH, WIDTH, HEIGHT, DEPTH_SCALE


class FrontEnd:
    def __init__(self):
        self.orb     = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, gray):
        return self.orb.detectAndCompute(gray, None)

    def match(self, d1, d2):
        if d1 is None or d2 is None or len(d1)<2 or len(d2)<2:
            return []
        pairs = self.matcher.knnMatch(d1, d2, k=2)
        return [m for p in pairs if len(p)==2
                for m,n in [p] if m.distance < 0.75*n.distance]

    def pnp(self, kps_ref, kps_cur, matches, depth_ref):
        if len(matches) < 8:
            return False, None
        pts3d, pts2d = [], []
        valid_depth_count = 0
        bad_points = []  # debug

        for m in matches:
            u, v = kps_ref[m.queryIdx].pt
            ui, vi = int(round(u)), int(round(v))
            # 检查边界
            if not (0 <= vi < HEIGHT and 0 <= ui < WIDTH):
                bad_points.append(f"out_of_bound({u},{v})")
                continue
            z = depth_ref[vi, ui]
            if z < MIN_DEPTH or z > MAX_DEPTH or np.isnan(z) or np.isinf(z):
                bad_points.append(f"invalid_z={z:.3f}")
                continue
            valid_depth_count += 1
            z = z * DEPTH_SCALE
            pts3d.append([(u - K[0, 2]) * z / K[0, 0],
                          (v - K[1, 2]) * z / K[1, 1], z])
            pts2d.append(kps_cur[m.trainIdx].pt)

        print(f"[PnP] matches={len(matches)} valid_depth={valid_depth_count} "
              f"bad={len(bad_points)} ({bad_points[:5]})")

        if len(pts3d) < 6:
            return False, None

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(pts3d, dtype=np.float64),
            np.array(pts2d, dtype=np.float64),
            K, None,
            iterationsCount=100, reprojectionError=2.0, confidence=0.99)

        if not ok or inliers is None or len(inliers) < 6:
            return False, None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        T = np.linalg.inv(T)  # world→camera → camera motion
        return True, T
