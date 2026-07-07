"""
奥比中光Astra双相机彩色3D重建系统 - 精确平移计算版（修复版）
修复相机对象属性错误
"""

import cv2
import numpy as np
import open3d as o3d
import json
import os
import gc
import traceback
import time
import sys
import warnings
import copy
from datetime import datetime
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')

# 导入独立的SDK接口层
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SDK接口层导入失败: {e}")
    SDK_AVAILABLE = False


class PreciseCalibrator:
    """精确标定器 - 通过图像匹配计算实际平移距离"""

    def __init__(self):
        # 特征检测器
        self.detector = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.01,
            edgeThreshold=10
        )

        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # 相机内参（Astra标准参数）
        self.K = np.array([
            [525.0, 0, 319.5],
            [0, 525.0, 239.5],
            [0, 0, 1]
        ])

        # 标定结果
        self.transform = np.eye(4)
        self.calculated_distance = 0.16  # 默认16cm
        self.calibration_confidence = 0.0

        print("初始化精确标定器 - 通过图像匹配计算实际平移距离")

    def extract_and_match_features(self, image1, image2):
        """提取和匹配特征点"""
        if image1 is None or image2 is None:
            return None, None, []

        # 转换为灰度
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # 提取特征
        kp1, desc1 = self.detector.detectAndCompute(gray1, None)
        kp2, desc2 = self.detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            return kp1, kp2, []

        # 特征匹配
        matches = self.matcher.match(desc1, desc2)

        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)

        return kp1, kp2, matches

    def compute_translation_from_matches(self, kp1, kp2, matches, depth1, depth2):
        """从匹配点计算平移距离"""
        if len(matches) < 10:
            print("  匹配点不足，无法计算平移")
            return 0.16, 0.0  # 返回默认值

        print(f"  使用 {len(matches)} 个匹配点计算平移...")

        # 提取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # 使用RANSAC过滤误匹配
        if len(matches) >= 8:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            inliers = mask.ravel() == 1
            pts1 = pts1[inliers]
            pts2 = pts2[inliers]
            print(f"  RANSAC内点: {np.sum(inliers)}/{len(matches)}")

        if len(pts1) < 10:
            print("  内点不足，使用默认平移")
            return 0.16, 0.0

        # 计算每个匹配点的视差和深度
        disparities = []
        calculated_distances = []
        valid_points = []

        for (u1, v1), (u2, v2) in zip(pts1, pts2):
            u1_int, v1_int = int(round(u1)), int(round(v1))
            u2_int, v2_int = int(round(u2)), int(round(v2))

            # 检查边界
            if not (0 <= u1_int < 640 and 0 <= v1_int < 480 and
                    0 <= u2_int < 640 and 0 <= v2_int < 480):
                continue

            # 获取深度值
            d1 = depth1[v1_int, u1_int]
            d2 = depth2[v2_int, u2_int]

            # 深度有效性检查
            if d1 < 0.3 or d1 > 4.0 or d2 < 0.3 or d2 > 4.0:
                continue

            # 计算视差（像素）
            disparity_x = u2 - u1
            disparity_y = v2 - v1

            # 使用视差和深度计算实际距离
            # 根据相机几何：baseline = depth * disparity / focal_length
            focal_length = self.K[0, 0]  # fx

            if abs(disparity_x) > 1:  # 只使用x方向视差（水平相机）
                # 计算相机间距
                baseline = d1 * abs(disparity_x) / focal_length

                disparities.append(disparity_x)
                calculated_distances.append(baseline)
                valid_points.append(((u1, v1, d1), (u2, v2, d2), baseline))

        if len(calculated_distances) < 5:
            print("  有效匹配点不足，使用默认平移")
            return 0.16, 0.0

        # 计算统计信息
        calculated_distances = np.array(calculated_distances)
        disparities = np.array(disparities)

        # 使用中值作为估计（对异常值更鲁棒）
        estimated_distance = np.median(calculated_distances)
        median_disparity = np.median(disparities)

        # 计算置信度（距离估计的一致性）
        std_distance = np.std(calculated_distances)
        confidence = max(0, 1.0 - std_distance / estimated_distance) if estimated_distance > 0 else 0

        print(f"  计算统计:")
        print(f"    估计距离: {estimated_distance:.4f}m ({estimated_distance*100:.1f}cm)")
        print(f"    中值视差: {median_disparity:.1f}像素")
        print(f"    距离范围: [{calculated_distances.min():.4f}, {calculated_distances.max():.4f}]m")
        print(f"    距离标准差: {std_distance:.4f}m")
        print(f"    置信度: {confidence:.3f}")

        # 如果估计距离不合理，使用默认值
        if estimated_distance < 0.05 or estimated_distance > 0.5:  # 5cm-50cm范围
            print(f"  ⚠️ 估计距离不合理，使用默认值")
            return 0.16, 0.0

        return estimated_distance, confidence

    def compute_transform_from_3d_points(self, pts1_3d, pts2_3d):
        """从3D点对计算变换矩阵"""
        if len(pts1_3d) < 10 or len(pts2_3d) < 10:
            return np.eye(4)

        # 计算质心
        centroid1 = np.mean(pts1_3d, axis=0)
        centroid2 = np.mean(pts2_3d, axis=0)

        # 去中心化
        pts1_centered = pts1_3d - centroid1
        pts2_centered = pts2_3d - centroid2

        # 计算H矩阵
        H = pts1_centered.T @ pts2_centered

        # SVD分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = Vt.T @ U.T

        # 确保右手坐标系
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 计算平移向量
        t = centroid2 - R @ centroid1

        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        return transform

    def calibrate_with_images(self, image1, depth1, image2, depth2):
        """使用图像进行标定"""
        print("通过图像匹配计算相机间距...")

        # 1. 特征提取和匹配
        kp1, kp2, matches = self.extract_and_match_features(image1, image2)

        if len(matches) < 20:
            print("  ⚠️ 匹配点不足")
            return False

        print(f"  找到 {len(matches)} 个匹配点")

        # 2. 计算平移距离
        estimated_distance, confidence = self.compute_translation_from_matches(
            kp1, kp2, matches, depth1, depth2
        )

        self.calculated_distance = estimated_distance
        self.calibration_confidence = confidence

        print(f"  计算得到的相机间距: {estimated_distance*100:.1f}cm")

        # 3. 从匹配点创建3D点对
        pts1_3d = []
        pts2_3d = []

        for match in matches[:100]:  # 最多使用100个匹配点
            u1, v1 = kp1[match.queryIdx].pt
            u2, v2 = kp2[match.trainIdx].pt

            u1_int, v1_int = int(round(u1)), int(round(v1))
            u2_int, v2_int = int(round(u2)), int(round(v2))

            # 检查边界
            if not (0 <= u1_int < 640 and 0 <= v1_int < 480 and
                    0 <= u2_int < 640 and 0 <= v2_int < 480):
                continue

            # 获取深度值
            d1 = depth1[v1_int, u1_int]
            d2 = depth2[v2_int, u2_int]

            # 深度有效性检查
            if d1 < 0.3 or d1 > 4.0 or d2 < 0.3 or d2 > 4.0:
                continue

            # 转换为3D坐标
            point1_3d = self.pixel_to_3d(u1, v1, d1)
            point2_3d = self.pixel_to_3d(u2, v2, d2)

            pts1_3d.append(point1_3d)
            pts2_3d.append(point2_3d)

        # 4. 从3D点对计算变换矩阵
        if len(pts1_3d) >= 10:
            pts1_3d = np.array(pts1_3d)
            pts2_3d = np.array(pts2_3d)

            transform = self.compute_transform_from_3d_points(pts1_3d, pts2_3d)

            # 检查变换矩阵的合理性
            translation_norm = np.linalg.norm(transform[:3, 3])

            # 如果计算出的平移与估计距离相差太大，使用估计距离
            if abs(translation_norm - estimated_distance) > 0.05:  # 5cm差异
                print(f"  ⚠️ 变换矩阵平移({translation_norm:.3f}m)与估计距离({estimated_distance:.3f}m)不一致")
                print(f"    使用估计距离构建变换矩阵")

                # 使用估计距离构建变换矩阵（假设只有X方向平移）
                self.transform = np.eye(4)
                self.transform[0, 3] = estimated_distance
            else:
                self.transform = transform
                print(f"  使用3D点对计算的变换矩阵")
        else:
            # 没有足够的3D点对，使用估计距离构建变换矩阵
            print("  3D点对不足，使用估计距离构建变换矩阵")
            self.transform = np.eye(4)
            self.transform[0, 3] = estimated_distance

        print(f"  最终变换矩阵:\n{self.transform}")
        return True

    def pixel_to_3d(self, u, v, depth):
        """像素坐标转3D坐标"""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.array([x, y, z])

    def get_transform(self):
        """获取变换矩阵"""
        return self.transform.copy()

    def get_calculated_distance(self):
        """获取计算的距离"""
        return self.calculated_distance


class AccurateDualCameraReconstructor:
    """精确双相机重建器 - 基于图像匹配计算实际平移"""

    def __init__(self, sdk_path=None):
        """初始化重建器"""
        if not SDK_AVAILABLE:
            print("❌ SDK接口层不可用，无法使用相机功能")
            return

        self.sdk_path = sdk_path if sdk_path else get_default_sdk_path()

        # 相机实例
        self.camera1_sdk = None
        self.camera2_sdk = None
        self.cameras_initialized = False

        # 彩色摄像头
        self.cv_camera1 = None
        self.cv_camera2 = None

        # 标定器
        self.calibrator = PreciseCalibrator()

        # 相机参数
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525.0, 525.0, 319.5, 239.5
        )

        # 变换矩阵（通过图像匹配计算）
        self.camera_transform = np.eye(4)
        self.calculated_distance = 0.16  # 默认16cm

        # 重建参数
        self.params = {
            # 标定参数
            'calibration_frames': 5,  # 标定使用的帧数
            'min_matches': 20,  # 最小匹配点数

            # 采集参数
            'capture_frames': 20,  # 采集帧数
            'capture_interval': 0.5,  # 采集间隔

            # 深度参数
            'depth_min': 0.3,  # 最小深度
            'depth_max': 3.0,  # 最大深度

            # 点云参数
            'voxel_size': 0.005,  # 体素大小
            'remove_outliers': True,

            # 配准参数
            'registration_method': 'feature_based',  # feature_based, icp, hybrid
            'icp_threshold': 0.02,  # ICP阈值

            # 融合参数
            'fusion_method': 'voxel_average',  # voxel_average, overlap_remove

            # 输出参数
            'save_calibration_images': True,
            'save_pointclouds': True,
            'save_mesh': True,
            'export_formats': ['ply', 'stl', 'obj'],
        }

        self.output_dir = None

        print("🎯 精确双相机重建器初始化完成")
        print("   基于图像匹配计算实际相机间距")

    def setup_cameras(self):
        """设置双相机"""
        if not SDK_AVAILABLE:
            return False

        print("=" * 60)
        print("设置双相机")
        print("=" * 60)

        try:
            # 初始化SDK
            self.camera1_sdk = OrbbecCameraSDK(self.sdk_path)
            self.camera2_sdk = OrbbecCameraSDK(self.sdk_path)

            if not self.camera1_sdk.initialize() or not self.camera2_sdk.initialize():
                return False

            # 打开设备
            success = True
            if not self.camera1_sdk.open_device(device_index=0):
                success = False

            # 尝试不同索引打开相机2
            if not self.camera2_sdk.open_device(device_index=1):
                if not self.camera2_sdk.open_device(device_index=0):
                    success = False

            if not success:
                return False

            # 创建深度流
            if not self.camera1_sdk.create_stream(self.camera1_sdk.ONI_SENSOR_DEPTH):
                return False
            if not self.camera2_sdk.create_stream(self.camera2_sdk.ONI_SENSOR_DEPTH):
                return False

            # 启动深度流
            if not self.camera1_sdk.start_stream() or not self.camera2_sdk.start_stream():
                return False

            # 设置彩色摄像头
            self.cv_camera1 = cv2.VideoCapture(0)
            self.cv_camera2 = cv2.VideoCapture(1)

            for idx, cam in enumerate([self.cv_camera1, self.cv_camera2], 1):
                if not cam.isOpened():
                    for cam_idx in range(4):
                        cam.release()
                        cam = cv2.VideoCapture(cam_idx)
                        if cam.isOpened():
                            print(f"✅ 彩色摄像头{idx}使用索引{cam_idx}")
                            break

                if cam.isOpened():
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cam.set(cv2.CAP_PROP_FPS, 15)

            self.cameras_initialized = True
            return True

        except Exception as e:
            print(f"❌ 相机设置失败: {e}")
            return False

    def capture_frames(self):
        """采集帧 - 修复版"""
        try:
            # 采集深度
            result1 = self.camera1_sdk.capture_depth_frame(timeout=1000)
            result2 = self.camera2_sdk.capture_depth_frame(timeout=1000)

            if not result1 or not result2:
                print("⚠️ 深度帧读取失败")
                return None, None, None, None

            depth1_array, _ = result1
            depth2_array, _ = result2

            depth1 = depth1_array.astype(np.float32) * 0.001
            depth2 = depth2_array.astype(np.float32) * 0.001

            # 采集彩色
            color1, color2 = None, None
            if self.cv_camera1 and self.cv_camera1.isOpened():
                ret, frame = self.cv_camera1.read()
                if ret and frame is not None:
                    if frame.shape[:2] != (480, 640):
                        color1 = cv2.resize(frame, (640, 480))
                    else:
                        color1 = frame.copy()

            if self.cv_camera2 and self.cv_camera2.isOpened():
                ret, frame = self.cv_camera2.read()
                if ret and frame is not None:
                    if frame.shape[:2] != (480, 640):
                        color2 = cv2.resize(frame, (640, 480))
                    else:
                        color2 = frame.copy()

            return depth1, color1, depth2, color2

        except Exception as e:
            print(f"❌ 采集失败: {e}")
            return None, None, None, None

    def calibrate_from_images(self):
        """通过图像匹配进行标定"""
        print("=" * 60)
        print("通过图像匹配计算相机间距")
        print("=" * 60)

        calib_dir = os.path.join(self.output_dir, "calibration_images")
        os.makedirs(calib_dir, exist_ok=True)

        calibration_results = []

        for i in range(self.params['calibration_frames']):
            print(f"\n采集标定帧 {i+1}/{self.params['calibration_frames']}...")

            depth1, color1, depth2, color2 = self.capture_frames()
            if depth1 is None or depth2 is None or color1 is None or color2 is None:
                print("  采集失败，跳过")
                continue

            # 保存图像
            if self.params['save_calibration_images']:
                cv2.imwrite(os.path.join(calib_dir, f"calib_{i:02d}_cam1.png"), color1)
                cv2.imwrite(os.path.join(calib_dir, f"calib_{i:02d}_cam2.png"), color2)

            # 进行标定
            print("  进行图像匹配标定...")
            success = self.calibrator.calibrate_with_images(color1, depth1, color2, depth2)

            if success:
                transform = self.calibrator.get_transform()
                distance = self.calibrator.get_calculated_distance()
                confidence = self.calibrator.calibration_confidence

                calibration_results.append({
                    'transform': transform,
                    'distance': distance,
                    'confidence': confidence,
                    'frame_index': i
                })

                print(f"  标定结果: {distance*100:.1f}cm, 置信度: {confidence:.3f}")

            if i < self.params['calibration_frames'] - 1:
                time.sleep(0.5)

        if len(calibration_results) == 0:
            print("❌ 标定失败，使用默认16cm间距")
            self.camera_transform = np.eye(4)
            self.camera_transform[0, 3] = 0.16
            self.calculated_distance = 0.16
            return False

        # 选择最佳标定结果（最高置信度）
        best_result = max(calibration_results, key=lambda x: x['confidence'])

        self.camera_transform = best_result['transform']
        self.calculated_distance = best_result['distance']

        print(f"\n✅ 图像匹配标定完成")
        print(f"   最佳结果: 第{best_result['frame_index']+1}帧")
        print(f"   计算间距: {self.calculated_distance*100:.1f}cm")
        print(f"   置信度: {best_result['confidence']:.3f}")
        print(f"   变换矩阵:\n{self.camera_transform}")

        # 保存标定结果
        calib_result = {
            'calculated_distance': float(self.calculated_distance),
            'transform_matrix': self.camera_transform.tolist(),
            'best_frame': best_result['frame_index'],
            'confidence': float(best_result['confidence']),
            'all_results': [
                {
                    'frame': r['frame_index'],
                    'distance': float(r['distance']),
                    'confidence': float(r['confidence'])
                }
                for r in calibration_results
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(self.output_dir, "calibration_result.json"), 'w') as f:
            json.dump(calib_result, f, indent=2)

        print(f"   标定结果已保存")
        return True

    def capture_scene(self):
        """采集场景"""
        print("=" * 60)
        print("采集场景数据")
        print("=" * 60)

        capture_dir = os.path.join(self.output_dir, "capture")
        os.makedirs(capture_dir, exist_ok=True)

        all_pcds_cam1 = []
        all_pcds_cam2 = []

        for i in range(self.params['capture_frames']):
            print(f"采集帧 {i+1}/{self.params['capture_frames']}...")

            depth1, color1, depth2, color2 = self.capture_frames()
            if depth1 is None or depth2 is None:
                continue

            # 保存图像
            if color1 is not None:
                cv2.imwrite(os.path.join(capture_dir, f"frame_{i:03d}_cam1.png"), color1)
            if color2 is not None:
                cv2.imwrite(os.path.join(capture_dir, f"frame_{i:03d}_cam2.png"), color2)

            # 创建点云
            pcd1 = self.depth_to_pointcloud(depth1, color1)
            pcd2 = self.depth_to_pointcloud(depth2, color2)

            if pcd1 is not None and len(pcd1.points) > 0:
                pcd1 = self.process_pointcloud(pcd1)
                all_pcds_cam1.append(pcd1)

            if pcd2 is not None and len(pcd2.points) > 0:
                pcd2 = self.process_pointcloud(pcd2)
                all_pcds_cam2.append(pcd2)

            if i < self.params['capture_frames'] - 1:
                time.sleep(self.params['capture_interval'])

        # 合并点云
        merged_pcd1 = None
        if len(all_pcds_cam1) > 0:
            merged_pcd1 = all_pcds_cam1[0]
            for pcd in all_pcds_cam1[1:]:
                merged_pcd1 += pcd
            merged_pcd1 = merged_pcd1.voxel_down_sample(self.params['voxel_size'])

        merged_pcd2 = None
        if len(all_pcds_cam2) > 0:
            merged_pcd2 = all_pcds_cam2[0]
            for pcd in all_pcds_cam2[1:]:
                merged_pcd2 += pcd
            merged_pcd2 = merged_pcd2.voxel_down_sample(self.params['voxel_size'])

        print(f"\n✅ 采集完成")
        print(f"   相机1点数: {len(merged_pcd1.points) if merged_pcd1 else 0}")
        print(f"   相机2点数: {len(merged_pcd2.points) if merged_pcd2 else 0}")

        return merged_pcd1, merged_pcd2

    def depth_to_pointcloud(self, depth, color=None):
        """深度图转点云"""
        if depth is None:
            return None

        depth_mm = (depth * 1000).astype(np.uint16)
        depth_image = o3d.geometry.Image(depth_mm)

        if color is not None:
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            color_image = o3d.geometry.Image(color_rgb)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image, depth_image,
                depth_scale=1000.0,
                depth_trunc=self.params['depth_max'],
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_image, self.intrinsic,
                depth_scale=1000.0,
                depth_trunc=self.params['depth_max']
            )

        # 范围裁剪
        points = np.asarray(pcd.points)
        valid_mask = (points[:, 2] >= self.params['depth_min']) & \
                     (points[:, 2] <= self.params['depth_max'])
        pcd = pcd.select_by_index(np.where(valid_mask)[0])

        return pcd

    def process_pointcloud(self, pcd):
        """处理点云"""
        if pcd is None or len(pcd.points) == 0:
            return None

        # 降采样
        pcd = pcd.voxel_down_sample(self.params['voxel_size'])

        # 去除离群点
        if self.params['remove_outliers'] and len(pcd.points) > 0:
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            pcd = pcd.select_by_index(ind)

        return pcd

    def register_and_fuse_pointclouds(self, pcd1, pcd2):
        """配准和融合点云"""
        print("=" * 60)
        print("点云配准和融合")
        print("=" * 60)

        if pcd1 is None or pcd2 is None:
            print("❌ 点云数据不足")
            return None

        # 应用变换矩阵
        pcd2_transformed = copy.deepcopy(pcd2)
        pcd2_transformed.transform(self.camera_transform)

        print(f"  应用变换矩阵:")
        print(f"    平移: [{self.camera_transform[0,3]:.4f}, "
              f"{self.camera_transform[1,3]:.4f}, "
              f"{self.camera_transform[2,3]:.4f}]m")
        print(f"    计算间距: {self.calculated_distance*100:.1f}cm")

        # 检查配准效果
        self.check_registration_quality(pcd1, pcd2_transformed)

        # 融合点云
        fused_pcd = self.fuse_pointclouds(pcd1, pcd2_transformed)

        return fused_pcd

    def check_registration_quality(self, pcd1, pcd2):
        """检查配准质量"""
        print("  检查配准质量...")

        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)

        if len(points1) == 0 or len(points2) == 0:
            return

        # 构建KD树查找最近邻
        tree = KDTree(points1)
        distances, indices = tree.query(points2, k=1)

        # 计算统计
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        std_distance = np.std(distances)

        # 计算重叠比例（距离小于阈值的点）
        overlap_threshold = 0.01  # 1cm
        overlap_ratio = np.sum(distances < overlap_threshold) / len(distances)

        print(f"    点云2到点云1的平均距离: {mean_distance:.4f}m")
        print(f"    点云2到点云1的中值距离: {median_distance:.4f}m")
        print(f"    距离标准差: {std_distance:.4f}m")
        print(f"    重叠比例（距离<1cm）: {overlap_ratio:.2%}")

        # 判断配准质量
        if mean_distance < 0.02 and overlap_ratio > 0.3:  # 平均距离<2cm且重叠>30%
            print("  ✅ 配准质量良好")
        elif mean_distance < 0.05 and overlap_ratio > 0.1:  # 平均距离<5cm且重叠>10%
            print("  ⚠️ 配准质量一般，可能存在轻微分层")
        else:
            print("  ❌ 配准质量差，可能存在严重分层")

    def fuse_pointclouds(self, pcd1, pcd2):
        """融合点云"""
        print("  融合点云...")

        if self.params['fusion_method'] == 'voxel_average':
            # 简单合并后体素降采样
            fused = pcd1 + pcd2
        elif self.params['fusion_method'] == 'overlap_remove':
            # 重叠去除融合
            fused = self.overlap_remove_fusion(pcd1, pcd2)
        else:
            # 简单合并
            fused = pcd1 + pcd2

        # 最终降采样
        fused = fused.voxel_down_sample(self.params['voxel_size'])

        print(f"  融合后点数: {len(fused.points)}")
        return fused

    def overlap_remove_fusion(self, pcd1, pcd2):
        """重叠去除融合"""
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)

        if len(points1) == 0:
            return pcd2
        if len(points2) == 0:
            return pcd1

        # 构建KD树
        tree1 = KDTree(points1)

        # 找到pcd2中与pcd1不重叠的点
        distances, indices = tree1.query(points2, k=1)
        overlap_threshold = self.params['voxel_size'] * 2

        non_overlap_mask = distances > overlap_threshold
        non_overlap_points = points2[non_overlap_mask]

        # 合并非重叠点
        all_points = np.vstack([points1, non_overlap_points])

        fused = o3d.geometry.PointCloud()
        fused.points = o3d.utility.Vector3dVector(all_points)

        # 如果有颜色，处理颜色
        if pcd1.has_colors() and pcd2.has_colors():
            colors1 = np.asarray(pcd1.colors)
            colors2 = np.asarray(pcd2.colors)
            non_overlap_colors = colors2[non_overlap_mask]
            all_colors = np.vstack([colors1, non_overlap_colors])
            fused.colors = o3d.utility.Vector3dVector(all_colors)

        return fused

    def create_mesh(self, pcd):
        """创建网格"""
        if pcd is None or len(pcd.points) < 1000:
            return None

        print("创建网格...")

        try:
            # 估计法线
            pcd.estimate_normals()

            # Poisson重建
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8
            )

            # 移除低密度区域
            if len(densities) > 0:
                threshold = np.quantile(densities, 0.01)
                vertices_to_remove = densities < threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)

            # 计算法线和平滑
            mesh.compute_vertex_normals()
            mesh = mesh.filter_smooth_simple(number_of_iterations=2)
            mesh.compute_vertex_normals()

            print(f"✅ 网格创建完成: {len(mesh.vertices)}顶点, {len(mesh.triangles)}面片")
            return mesh

        except Exception as e:
            print(f"❌ 网格创建失败: {e}")
            return None

    def save_results(self, fused_pcd, mesh):
        """保存结果"""
        print("=" * 60)
        print("保存结果")
        print("=" * 60)

        vis_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

        # 保存点云
        if fused_pcd is not None and self.params['save_pointclouds']:
            o3d.io.write_point_cloud(os.path.join(vis_dir, "fused_pointcloud.ply"), fused_pcd)
            print("✅ 融合点云已保存")

        # 保存网格
        if mesh is not None and self.params['save_mesh']:
            for fmt in self.params['export_formats']:
                if fmt == 'ply':
                    o3d.io.write_triangle_mesh(os.path.join(vis_dir, "model.ply"), mesh)
                elif fmt == 'stl':
                    o3d.io.write_triangle_mesh(os.path.join(vis_dir, "model.stl"), mesh)
                elif fmt == 'obj':
                    o3d.io.write_triangle_mesh(os.path.join(vis_dir, "model.obj"), mesh)

            print(f"✅ 网格已保存为 {self.params['export_formats']} 格式")

        # 生成最终报告
        self.generate_final_report(fused_pcd, mesh, vis_dir)

        print(f"\n📁 所有结果已保存到: {vis_dir}")

    def generate_final_report(self, fused_pcd, mesh, output_dir):
        """生成最终报告"""
        report_path = os.path.join(output_dir, "final_report.txt")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("精确双相机重建报告 - 基于图像匹配计算平移\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"重建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"计算得到的相机间距: {self.calculated_distance*100:.1f}cm\n\n")

                f.write("变换矩阵:\n")
                for row in self.camera_transform:
                    f.write(f"  [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}, {row[3]:.6f}]\n")

                # 点云统计
                if fused_pcd is not None:
                    points = np.asarray(fused_pcd.points)
                    f.write(f"\n点云统计:\n")
                    f.write(f"  点数: {len(points)}\n")
                    f.write(f"  X范围: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}] m\n")
                    f.write(f"  Y范围: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}] m\n")
                    f.write(f"  Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}] m\n")
                    f.write(f"  重建尺寸: {points[:, 0].max()-points[:, 0].min():.3f} × "
                           f"{points[:, 1].max()-points[:, 1].min():.3f} × "
                           f"{points[:, 2].max()-points[:, 2].min():.3f} m\n")

                # 网格统计
                if mesh is not None:
                    f.write(f"\n网格统计:\n")
                    f.write(f"  顶点数: {len(mesh.vertices)}\n")
                    f.write(f"  面片数: {len(mesh.triangles)}\n")

                f.write(f"\n📁 输出文件:\n")
                for file in sorted(os.listdir(output_dir)):
                    if file.endswith(('.ply', '.stl', '.obj', '.txt', '.json')):
                        file_path = os.path.join(output_dir, file)
                        file_size = os.path.getsize(file_path)
                        f.write(f"  - {file} ({file_size/1024:.1f} KB)\n")

                f.write(f"\n💡 技术原理:\n")
                f.write("  1. 通过图像特征匹配计算两个相机之间的视差\n")
                f.write("  2. 根据视差和深度值计算实际相机间距\n")
                f.write("  3. 使用计算得到的间距进行点云配准\n")

                f.write(f"\n🔧 如果仍有分层问题:\n")
                f.write("  1. 检查calibration_images/目录中的匹配图像\n")
                f.write("  2. 查看calibration_result.json中的计算距离\n")
                f.write("  3. 调整相机位置使视野有更多重叠\n")
                f.write("  4. 增加calibration_frames参数使用更多帧进行标定\n")

            print(f"✅ 最终报告已保存: {report_path}")

        except Exception as e:
            print(f"⚠️ 生成报告失败: {e}")

    def run_reconstruction(self):
        """运行重建"""
        print("=" * 70)
        print("精确双相机彩色3D重建系统")
        print("基于图像匹配计算实际相机间距")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"accurate_reconstruction_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")

        # 保存配置
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.params, f, indent=2)

        # 检查SDK
        if not SDK_AVAILABLE:
            print("❌ SDK不可用")
            return False

        # 设置相机
        if not self.setup_cameras():
            print("❌ 相机设置失败")
            return False

        try:
            # 步骤1: 图像匹配标定
            print("\n" + "=" * 60)
            print("步骤1: 图像匹配标定（计算实际相机间距）")
            print("=" * 60)

            self.calibrate_from_images()

            # 步骤2: 采集场景
            print("\n" + "=" * 60)
            print("步骤2: 采集场景")
            print("=" * 60)

            pcd1, pcd2 = self.capture_scene()

            if pcd1 is None and pcd2 is None:
                print("❌ 采集失败")
                self.cleanup()
                return False

            # 步骤3: 配准和融合
            print("\n" + "=" * 60)
            print("步骤3: 点云配准和融合")
            print("=" * 60)

            fused_pcd = self.register_and_fuse_pointclouds(pcd1, pcd2)

            if fused_pcd is None:
                print("❌ 融合失败")
                self.cleanup()
                return False

            # 步骤4: 创建网格
            print("\n" + "=" * 60)
            print("步骤4: 创建网格")
            print("=" * 60)

            mesh = self.create_mesh(fused_pcd)

            # 步骤5: 保存结果
            print("\n" + "=" * 60)
            print("步骤5: 保存结果")
            print("=" * 60)

            self.save_results(fused_pcd, mesh)

            print("\n" + "=" * 70)
            print("🎉 精确重建完成！")
            print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")
            print("=" * 70)

            # 显示关键信息
            print(f"\n📊 重建统计:")
            print(f"  计算相机间距: {self.calculated_distance*100:.1f}cm")
            print(f"  融合点云点数: {len(fused_pcd.points) if fused_pcd else 0}")
            print(f"  网格顶点数: {len(mesh.vertices) if mesh else 0}")

            return True

        except KeyboardInterrupt:
            print("\n🔴 用户中断")
        except Exception as e:
            print(f"\n❌ 重建失败: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()

        return False

    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")

        try:
            if self.camera1_sdk:
                self.camera1_sdk.cleanup()
            if self.camera2_sdk:
                self.camera2_sdk.cleanup()

            if self.cv_camera1:
                self.cv_camera1.release()
            if self.cv_camera2:
                self.cv_camera2.release()

            print("✅ 资源已清理")

        except Exception as e:
            print(f"⚠️ 清理失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("精确双相机彩色3D重建系统")
    print("基于图像匹配计算实际相机间距，解决分层问题")
    print("=" * 70)

    # 检查依赖
    try:
        print("✅ OpenCV:", cv2.__version__)
        print("✅ NumPy:", np.__version__)
        print("✅ Open3D:", o3d.__version__)
    except Exception as e:
        print(f"❌ 依赖检查失败: {e}")
        return

    # 获取SDK路径
    sdk_path = get_default_sdk_path()
    if not os.path.exists(sdk_path):
        sdk_path = ""

    # 创建重建器
    reconstructor = AccurateDualCameraReconstructor(sdk_path)

    # 运行重建
    print("\n🚀 开始精确双相机重建...")
    print("🎯 目标: 通过图像匹配计算实际相机间距")
    print("=" * 70)

    try:
        success = reconstructor.run_reconstruction()

        if success:
            print("\n🎉 精确重建成功！")
            print("\n📋 技术原理:")
            print("  1. 提取两个图像的SIFT特征点")
            print("  2. 匹配特征点计算像素视差")
            print("  3. 根据深度和视差计算实际相机间距")
            print("  4. 使用计算得到的间距进行配准")

        else:
            print("\n⚠️ 重建失败")

    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("💡 重要提示:")
    print("  1. 系统会自动计算相机间距，不再假设16cm")
    print("  2. 检查calibration_result.json中的计算距离")
    print("  3. 如果计算距离不合理，调整相机位置重新运行")
    print("=" * 70)

    input("\n按Enter键退出...")


if __name__ == "__main__":
    # 设置环境变量
    default_sdk_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"
    if os.path.exists(default_sdk_path):
        drivers_dir = os.path.join(default_sdk_path, "OpenNI2", "Drivers")
        if os.path.exists(drivers_dir):
            os.environ['PATH'] = drivers_dir + ';' + os.environ['PATH']

    main()