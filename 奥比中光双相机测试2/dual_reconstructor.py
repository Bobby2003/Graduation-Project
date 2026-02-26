import cv2
import numpy as np
import open3d as o3d
import time
from orbbec_sdk import OrbbecCameraSDK


class PointCloudStitcher:
    def __init__(self):
        # 标定内参数 - 请确保这些值接近你相机的真实值
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5
        self.sift = cv2.SIFT_create(nfeatures=2000)

    def get_3d_point(self, u, v, depth_map):
        """
        修正：从深度图中提取特定点的 3D 坐标
        """
        # 1. 确保坐标在图像范围内并获取标量数值
        u_i, v_i = int(round(u)), int(round(v))
        h, w = depth_map.shape
        if not (0 <= u_i < w and 0 <= v_i < h):
            return None

        d_val = depth_map[v_i, u_i]  # 获取该点的深度值 (uint16)
        r = d_val / 1000.0  # 转为米 (float)

        # 2. 现在 r 是一个数字，可以安全地进行 if 判断
        if r <= 0.1 or r > 4.5:
            return None

        # 3. 球面投影数学模型
        theta = (u - self.cx) / self.fx
        phi = (v - self.cy) / self.fy
        denom = np.sqrt(theta ** 2 + phi ** 2 + 1)

        x = r * (theta / denom)
        y = r * (phi / denom)
        z = r * (1 / denom)
        return np.array([x, y, z])

    def depth_to_pcd(self, depth, transform=None):
        """矢量化处理整张深度图转点云"""
        h, w = depth.shape
        u_coords, v_coords = np.meshgrid(np.arange(0, w, 2), np.arange(0, h, 2))
        r_vals = depth[v_coords, u_coords].astype(float) / 1000.0

        # 这里使用 & 是位运算，支持 numpy 数组判断
        mask = (r_vals > 0.1) & (r_vals < 4.5)
        u_v, v_v, r_v = u_coords[mask], v_coords[mask], r_vals[mask]

        theta = (u_v - self.cx) / self.fx
        phi = (v_v - self.cy) / self.fy
        denom = np.sqrt(theta ** 2 + phi ** 2 + 1)

        pts = np.stack([
            r_v * (theta / denom),
            r_v * (phi / denom),
            r_v * (1 / denom)
        ], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if transform is not None:
            pcd.transform(transform)
        return pcd

    def visualize_2d_matches(self, c0, c1, d0, d1, kp0, kp1, matches):
        """展示 2D 特征匹配图"""
        match_img = cv2.drawMatches(c0, kp0, c1, kp1, matches[:40], None, flags=2)

        def to_color(d):
            norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        d_vis = np.hstack([to_color(d0), to_color(d1)])
        # 确保宽度一致以便堆叠
        if match_img.shape[1] != d_vis.shape[1]:
            d_vis = cv2.resize(d_vis, (match_img.shape[1], d_vis.shape[0]))

        cv2.imshow("Top: RGB Matches | Bottom: Depth Keypoints", np.vstack([match_img, d_vis]))
        cv2.waitKey(1)

    def filter_occlusion(self, pcd1, depth0, threshold=0.03):
        """射线法剔除分层重影点"""
        pts1 = np.asarray(pcd1.points)
        num_pts = pts1.shape[0]
        h, w = depth0.shape

        r_dist = np.linalg.norm(pts1, axis=1)
        z = pts1[:, 2]
        z[z == 0] = 0.001

        u = (pts1[:, 0] * self.fx / z + self.cx).astype(int)
        v = (pts1[:, 1] * self.fy / z + self.cy).astype(int)

        mask_dup = np.zeros(num_pts, dtype=bool)
        for i in range(num_pts):
            if 0 <= u[i] < w and 0 <= v[i] < h:
                d0_val = depth0[v[i], u[i]] / 1000.0
                if d0_val > 0.1 and abs(r_dist[i] - d0_val) < threshold:
                    mask_dup[i] = True

        return pcd1.select_by_index(np.where(~mask_dup)[0]), \
               pcd1.select_by_index(np.where(mask_dup)[0])

    def calculate_transformation(self, c0, c1, d0, d1):
        kp0, des0 = self.sift.detectAndCompute(c0, None)
        kp1, des1 = self.sift.detectAndCompute(c1, None)

        if des0 is None or des1 is None: return None, "未能提取特征点"

        bf = cv2.BFMatcher()
        raw_matches = bf.knnMatch(des0, des1, k=2)

        good_matches, src_pts, dst_pts = [], [], []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                # 传入坐标和对应的深度图
                p0_3d = self.get_3d_point(kp0[m.queryIdx].pt[0], kp0[m.queryIdx].pt[1], d0)
                p1_3d = self.get_3d_point(kp1[m.trainIdx].pt[0], kp1[m.trainIdx].pt[1], d1)

                if p0_3d is not None and p1_3d is not None:
                    dst_pts.append(p0_3d);
                    src_pts.append(p1_3d)
                    good_matches.append(m)

        self.visualize_2d_matches(c0, c1, d0, d1, kp0, kp1, good_matches)

        if len(src_pts) < 4: return None, f"有效 3D 匹配点不足({len(src_pts)})"

        retval, trans, _ = cv2.estimateAffine3D(np.array(src_pts), np.array(dst_pts), ransacThreshold=0.03)
        if retval:
            T = np.eye(4);
            T[:3, :] = trans
            return T, "求解成功"
        return None, "RANSAC 失败"


if __name__ == "__main__":
    stitcher = PointCloudStitcher()
    sdk0 = OrbbecCameraSDK();
    sdk1 = OrbbecCameraSDK()
    sdk0.initialize();
    sdk1.initialize()
    devs = sdk0.get_devices()

    if len(devs) >= 2:
        sdk0.open(devs[0]['uri']);
        sdk1.open(devs[1]['uri'])
        cap0 = cv2.VideoCapture(0);
        cap1 = cv2.VideoCapture(1)
        try:
            print("🚀 正在捕捉数据并展示 2D 特征匹配图...")
            time.sleep(2)
            # 获取深度图并做镜像处理（如果你的相机需要）
            d0, d1 = cv2.flip(sdk0.get_depth(), 1), cv2.flip(sdk1.get_depth(), 1)
            _, c0 = cap0.read();
            _, c1 = cap1.read()

            T, msg = stitcher.calculate_transformation(c0, c1, d0, d1)
            print(f"提示: {msg}")

            if T is not None:
                pcd0 = stitcher.depth_to_pcd(d0)
                pcd1_full = stitcher.depth_to_pcd(d1, transform=T)

                # 剔除分层
                pcd1_kept, pcd1_red = stitcher.filter_occlusion(pcd1_full, d0, threshold=0.03)

                # 设置颜色
                pcd0.paint_uniform_color([0.2, 0.7, 0.2])  # 绿色 (Cam0)
                pcd1_red.paint_uniform_color([1.0, 0.1, 0.1])  # 红色 (被剔除的分层)
                pcd1_kept.paint_uniform_color([0.1, 0.1, 1.0])  # 蓝色 (Cam1 保留)

                print("\n1. 诊断视图已开启：红色代表被射线检查剔除的重复点层。")
                o3d.visualization.draw_geometries([pcd0, pcd1_kept, pcd1_red], window_name="Step 1: Overlap Diagnosis")

                print("2. 最终清理视图已开启：仅显示非重叠区域。")
                o3d.visualization.draw_geometries([pcd0, pcd1_kept], window_name="Step 2: Cleaned Result")
            else:
                print("未获得有效的坐标转换。请确保两个相机能看到相同的物体。")
                cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()
            sdk0.close();
            sdk1.close();
            cap0.release();
            cap1.release()
    else:
        print("未检测到足够的相机设备。")