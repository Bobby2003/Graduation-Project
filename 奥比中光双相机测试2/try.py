import cv2
import numpy as np
import open3d as o3d
import time
import os
from orbbec_sdk import OrbbecCameraSDK


class PrecisionDualReconstructor:
    def __init__(self):
        self.fx, self.fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5
        self.sift = cv2.SIFT_create(nfeatures=3000)  # 增加特征点数量
        self.output_dir = "precision_model"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        self.last_T = np.eye(4)

    def depth_to_pcd(self, depth, color=None):
        """恢复全分辨率点云，后续再进行体素降采样以保证空间分布均匀"""
        h, w = depth.shape
        # 直接生成全量坐标（不使用 step，保证精度）
        u, v = np.meshgrid(np.arange(0, w), np.arange(0, h))
        z = depth.astype(float) / 1000.0
        mask = (z > 0.2) & (z < 3.5)

        z_f, u_f, v_f = z[mask], u[mask], v[mask]
        x_f = (u_f - self.cx) * z_f / self.fx
        y_f = (v_f - self.cy) * z_f / self.fy

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.stack([x_f, y_f, z_f], axis=-1))

        if color is not None:
            c_res = cv2.resize(color, (w, h))
            colors = c_res[mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1])

        # 使用体素下采样（Voxel Downsample）代替步长抽样，这能让点云分布更科学
        return pcd.voxel_down_sample(voxel_size=0.005)  # 5mm 体素间距

    def get_3d_point(self, u, v, depth_map):
        u_i, v_i = int(round(u)), int(round(v))
        h, w = depth_map.shape
        if not (0 <= u_i < w and 0 <= v_i < h): return None
        d = depth_map[v_i, u_i]
        if d < 200 or d > 4000: return None
        z = d / 1000.0
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])

    def rough_alignment(self, c0, c1, d0, d1):
        """增强版特征匹配"""
        kp0, des0 = self.sift.detectAndCompute(c0, None)
        kp1, des1 = self.sift.detectAndCompute(c1, None)
        if des0 is None or des1 is None: return None

        matches = cv2.BFMatcher().knnMatch(des0, des1, k=2)
        dst_pts, src_pts = [], []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                p0 = self.get_3d_point(kp0[m.queryIdx].pt[0], kp0[m.queryIdx].pt[1], d0)
                p1 = self.get_3d_point(kp1[m.trainIdx].pt[0], kp1[m.trainIdx].pt[1], d1)
                if p0 is not None and p1 is not None:
                    dst_pts.append(p0);
                    src_pts.append(p1)

        if len(src_pts) < 10: return None
        retval, trans, _ = cv2.estimateAffine3D(np.array(src_pts), np.array(dst_pts), ransacThreshold=0.02)
        if retval:
            T = np.eye(4);
            T[:3, :] = trans
            return T
        return None

    def run(self):
        sdk0 = OrbbecCameraSDK();
        sdk1 = OrbbecCameraSDK()
        sdk0.initialize();
        sdk1.initialize()
        devs = sdk0.get_devices()
        if len(devs) < 2: return
        sdk0.open(devs[0]['uri']);
        sdk1.open(devs[1]['uri'])
        cap0 = cv2.VideoCapture(0);
        cap1 = cv2.VideoCapture(1)

        try:
            time.sleep(2)
            d0, d1 = cv2.flip(sdk0.get_depth(), 1), cv2.flip(sdk1.get_depth(), 1)
            _, c0 = cap0.read();
            _, c1 = cap1.read()

            pcd0 = self.depth_to_pcd(d0, c0)
            pcd1 = self.depth_to_pcd(d1, c1)

            # 1. 初始矩阵
            T_init = self.rough_alignment(c0, c1, d0, d1)
            if T_init is None:
                T_init = self.last_T
                print("⚠️ 粗配准失败，使用历史值")
            else:
                self.last_T = T_init
                print("✅ 粗配准精度：良好")

            # 2. 优化法线估计（这是消除重影的关键）
            # 半径必须根据体素大小调整，这里体素 5mm，搜索半径用 2cm 比较合适
            pcd0.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

            # 3. 双阶段高精度 ICP
            print("🚀 正在注入高精度匹配...")
            # 第一阶段：大容错 (3cm)
            reg_rough = o3d.pipelines.registration.registration_icp(
                pcd1, pcd0, 0.03, T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            # 第二阶段：极小容错 (1cm) 进一步精修
            reg_fine = o3d.pipelines.registration.registration_icp(
                pcd1, pcd0, 0.01, reg_rough.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            pcd1.transform(reg_fine.transformation)

            # --- 保存结果 ---
            combined = pcd0 + pcd1
            # 解决 PLY 无法打开问题：使用强制二进制编码并过滤无效点
            combined.remove_non_finite_points()
            o3d.io.write_point_cloud(os.path.join(self.output_dir, "fused.ply"), combined, write_ascii=False)


            # 生成 STL
            self.save_as_stl(combined)

            o3d.visualization.draw_geometries([pcd0, pcd1], window_name="LazyMan Precision View")

        finally:
            sdk0.close();
            sdk1.close();
            cap0.release();
            cap1.release()

    def save_as_stl(self, pcd):
        print("正在构建 STL 表面...")
        # 更加精细的重建参数
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        # 裁剪掉低密度生成的野点（Poisson 重建特有的边缘包裹）
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        o3d.io.write_triangle_mesh(os.path.join(self.output_dir, "fused.stl"), mesh)
        print("✅ STL 导出完成")


if __name__ == "__main__":
    app = PrecisionDualReconstructor()
    app.run()