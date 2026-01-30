"""
实用工具函数集 - 针对多相机深度系统
"""

"""
实用工具函数集 - 针对多相机深度系统
"""

import cv2
import numpy as np
import open3d as o3d
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any, Union  # 添加类型导入


class MultiCameraUtils:
    """多相机系统工具类"""

    @staticmethod
    def depth_to_pointcloud(depth_image: np.ndarray,
                           intrinsic_matrix: np.ndarray,
                           depth_scale: float = 0.001,
                           max_depth: float = 2.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        深度图转换为点云

        Args:
            depth_image: 深度图像 (H, W)
            intrinsic_matrix: 相机内参矩阵 (3x3)
            depth_scale: 深度缩放因子
            max_depth: 最大有效深度

        Returns:
            points: 点云坐标 (N, 3)
            valid_mask: 有效点掩码
        """
        if depth_image is None or depth_image.size == 0:
            return None, None

        height, width = depth_image.shape

        # 创建网格
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        # 转换为齐次坐标
        uv_hom = np.stack([xx.flatten(), yy.flatten(), np.ones_like(xx.flatten())], axis=1)

        # 应用内参逆矩阵
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # 计算归一化坐标
        x_normalized = (xx.flatten() - cx) / fx
        y_normalized = (yy.flatten() - cy) / fy

        # 获取深度
        depths = depth_image.flatten() * depth_scale

        # 创建有效掩码
        valid_mask = (depths > 0) & (depths < max_depth) & ~np.isinf(depths) & ~np.isnan(depths)

        if not np.any(valid_mask):
            return None, None

        # 计算3D坐标
        x_3d = x_normalized[valid_mask] * depths[valid_mask]
        y_3d = y_normalized[valid_mask] * depths[valid_mask]
        z_3d = depths[valid_mask]

        points = np.stack([x_3d, y_3d, z_3d], axis=1)

        return points, valid_mask.reshape(height, width)

    @staticmethod
    def create_open3d_pointcloud(points: np.ndarray,
                                colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        创建Open3D点云

        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3) [0, 1]范围

        Returns:
            Open3D点云对象
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def visualize_pointclouds(pointclouds: List[o3d.geometry.PointCloud],
                             window_name: str = "点云可视化"):
        """
        可视化点云列表

        Args:
            pointclouds: 点云列表
            window_name: 窗口名称
        """
        if not pointclouds:
            print("无点云可显示")
            return

        # 为每个点云分配不同颜色（如果无颜色）
        colors = [
            [1, 0, 0],  # 红色
            [0, 1, 0],  # 绿色
            [0, 0, 1],  # 蓝色
        ]

        for i, pcd in enumerate(pointclouds):
            if len(pcd.colors) == 0:
                pcd.paint_uniform_color(colors[i % len(colors)])

        # 可视化
        o3d.visualization.draw_geometries(
            pointclouds,
            window_name=window_name,
            width=1024,
            height=768,
            left=50,
            top=50
        )

    @staticmethod
    def visualize_frames(frames_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
                        window_name: str = "多相机视图"):
        """
        可视化多相机帧

        Args:
            frames_dict: 相机ID到(彩色图, 深度图)的映射
            window_name: 窗口名称
        """
        if not frames_dict:
            print("无帧数据可显示")
            return

        displays = []
        max_width = 0

        for cam_id, (color, depth) in frames_dict.items():
            display = np.zeros((300, 400, 3), dtype=np.uint8)

            if color is not None:
                color_resized = cv2.resize(color, (400, 300))
                display = color_resized

                if depth is not None:
                    # 添加深度信息
                    valid_pixels = np.sum(depth > 0)
                    depth_text = f"Depth: {valid_pixels}"
                    cv2.putText(display, depth_text, (10, 280),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elif depth is not None:
                # 显示深度图
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_normalized = depth_normalized.astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                display = cv2.resize(depth_colored, (400, 300))

            # 添加相机ID标签
            cv2.putText(display, f"Cam {cam_id}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            displays.append(display)
            max_width = max(max_width, display.shape[1])

        # 组合显示
        if displays:
            if len(displays) == 1:
                cv2.imshow(window_name, displays[0])

            elif len(displays) == 2:
                combined = np.hstack(displays)
                cv2.imshow(window_name, combined)

            else:
                # 对于3个相机，前2个在上排，第3个在下排居中
                row1 = np.hstack(displays[:2])
                row2 = displays[2]

                # 调整第二行居中
                padding = (row1.shape[1] - row2.shape[1]) // 2
                if padding > 0:
                    row2_padded = np.zeros((row2.shape[0], row1.shape[1], 3), dtype=np.uint8)
                    row2_padded[:, padding:padding+row2.shape[1]] = row2
                    row2 = row2_padded

                combined = np.vstack([row1, row2])
                cv2.imshow(window_name, combined)

    @staticmethod
    def save_pointcloud(pcd: o3d.geometry.PointCloud,
                       filename: str,
                       binary: bool = True):
        """
        保存点云到文件

        Args:
            pcd: Open3D点云
            filename: 文件名
            binary: 是否保存为二进制格式
        """
        if pcd and len(pcd.points) > 0:
            o3d.io.write_point_cloud(filename, pcd, write_ascii=not binary)
            print(f"点云已保存: {filename}")
            return True
        return False

    @staticmethod
    def save_mesh(mesh: o3d.geometry.TriangleMesh,
                 filename: str,
                 binary: bool = True):
        """
        保存网格到文件

        Args:
            mesh: Open3D网格
            filename: 文件名
            binary: 是否保存为二进制格式
        """
        if mesh and len(mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(filename, mesh, write_ascii=not binary)
            print(f"网格已保存: {filename}")
            return True
        return False

    @staticmethod
    def create_checkerboard_points(board_size: Tuple[int, int],
                                  square_size: float) -> np.ndarray:
        """
        创建棋盘格3D点（世界坐标系）

        Args:
            board_size: 棋盘格角点数 (列, 行)
            square_size: 方格大小（米）

        Returns:
            棋盘格角点的3D坐标
        """
        points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        points *= square_size
        return points

    @staticmethod
    def save_calibration_results(calibration_data: Dict,
                               output_dir: str):
        """
        保存标定结果

        Args:
            calibration_data: 标定数据字典
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存为JSON
        json_file = os.path.join(output_dir, "calibration_results.json")

        # 转换NumPy数组为列表以便JSON序列化
        json_data = {}
        for key, value in calibration_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, dict):
                json_data[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                json_data[key] = value

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)

        print(f"标定结果已保存: {json_file}")
        return json_file

    @staticmethod
    def load_calibration_results(calibration_file: str) -> Optional[Dict]:
        """
        加载标定结果

        Args:
            calibration_file: 标定文件路径

        Returns:
            标定数据字典
        """
        if not os.path.exists(calibration_file):
            print(f"标定文件不存在: {calibration_file}")
            return None

        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)

            # 转换列表为NumPy数组
            for key in calibration_data:
                if isinstance(calibration_data[key], list):
                    # 检查是否为嵌套列表（矩阵）
                    if all(isinstance(item, list) for item in calibration_data[key]):
                        calibration_data[key] = np.array(calibration_data[key])
                    else:
                        calibration_data[key] = np.array(calibration_data[key])

            print(f"标定结果已加载: {calibration_file}")
            return calibration_data

        except Exception as e:
            print(f"加载标定结果失败: {e}")
            return None

    @staticmethod
    def calculate_reprojection_error(obj_points: List[np.ndarray],
                                   img_points: List[np.ndarray],
                                   rvecs: List[np.ndarray],
                                   tvecs: List[np.ndarray],
                                   camera_matrix: np.ndarray,
                                   dist_coeffs: np.ndarray) -> float:
        """
        计算重投影误差

        Args:
            obj_points: 对象点列表
            img_points: 图像点列表
            rvecs: 旋转向量列表
            tvecs: 平移向量列表
            camera_matrix: 相机矩阵
            dist_coeffs: 畸变系数

        Returns:
            平均重投影误差
        """
        total_error = 0
        total_points = 0

        for i in range(len(obj_points)):
            # 投影3D点到2D
            img_points2, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )

            # 计算误差
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            total_error += error * len(img_points2)
            total_points += len(img_points2)

        if total_points > 0:
            mean_error = total_error / total_points
            return mean_error

        return 0.0