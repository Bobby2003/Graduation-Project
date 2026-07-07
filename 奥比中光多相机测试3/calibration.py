"""
相机标定模块 - 支持多相机内外参标定
"""

import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime
from config import MultiCameraConfig
from utils import MultiCameraUtils


class MultiCameraCalibrator:
    """多相机标定器"""

    def __init__(self, config=None):
        self.config = config or MultiCameraConfig.CALIBRATION_CONFIG
        self.camera_data = {}  # 存储每个相机的标定数据
        self.stereo_calibration = {}  # 立体标定结果

    def capture_calibration_images(self, camera_system, output_dir):
        """采集标定图像"""
        print("=" * 60)
        print("开始采集标定图像")
        print("=" * 60)

        calib_dir = os.path.join(output_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)

        # 创建各相机图像目录
        for cam_id in range(MultiCameraConfig.NUM_CAMERAS):
            cam_dir = os.path.join(calib_dir, f"camera_{cam_id}")
            os.makedirs(cam_dir, exist_ok=True)

        checkerboard_size = self.config['checkerboard_size']
        square_size = self.config['square_size']

        print(f"棋盘格: {checkerboard_size} 角点")
        print(f"方格大小: {square_size} 米")
        print(f"需要采集: {self.config['calibration_images']} 张图像")
        print("请确保棋盘格在所有相机视野内")
        print("按空格键采集图像，ESC键退出")

        # 准备棋盘格角点
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0],
                      0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size

        image_count = 0
        cv2.namedWindow("标定图像采集", cv2.WINDOW_NORMAL)

        while image_count < self.config['calibration_images']:
            # 同步采集所有相机
            frames = camera_system.capture_sync_frames()
            if frames is None:
                print("采集失败，跳过")
                continue

            # 显示图像并检测棋盘格
            all_good = True
            display_images = []

            for cam_id, (color, depth) in frames.items():
                if color is None:
                    all_good = False
                    break

                # 查找棋盘格角点
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, checkerboard_size,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )

                # 绘制角点
                if ret:
                    cv2.drawChessboardCorners(color, checkerboard_size, corners, ret)
                    cv2.putText(color, "OK", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(color, "NO BOARD", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    all_good = False

                # 调整大小用于显示
                display = cv2.resize(color, (320, 240))
                cv2.putText(display, f"Cam{cam_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_images.append(display)

            # 组合显示
            if len(display_images) == 3:
                top_row = np.hstack(display_images[:2])
                bottom_row = display_images[2]

                # 居中显示
                padding = (top_row.shape[1] - bottom_row.shape[1]) // 2
                if padding > 0:
                    bottom_padded = np.zeros((bottom_row.shape[0], top_row.shape[1], 3), dtype=np.uint8)
                    bottom_padded[:, padding:padding + bottom_row.shape[1]] = bottom_row
                    bottom_row = bottom_padded

                combined = np.vstack([top_row, bottom_row])
                cv2.imshow("标定图像采集", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == 32 and all_good:  # 空格键保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                for cam_id, (color, depth) in frames.items():
                    # 保存图像
                    color_path = os.path.join(calib_dir, f"camera_{cam_id}",
                                              f"color_{image_count:03d}_{timestamp}.jpg")
                    depth_path = os.path.join(calib_dir, f"camera_{cam_id}",
                                              f"depth_{image_count:03d}_{timestamp}.png")

                    cv2.imwrite(color_path, color)
                    if depth is not None:
                        depth_colored = MultiCameraUtils.colorize_depth(depth)
                        if depth_colored is not None:
                            cv2.imwrite(depth_path, depth_colored)

                    # 保存角点数据
                    if cam_id not in self.camera_data:
                        self.camera_data[cam_id] = {
                            'object_points': [],
                            'image_points': [],
                            'image_size': color.shape[:2]
                        }

                    # 检测角点
                    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray, checkerboard_size,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                    )

                    if ret:
                        # 亚像素优化
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                        self.camera_data[cam_id]['object_points'].append(objp)
                        self.camera_data[cam_id]['image_points'].append(corners_refined)

                image_count += 1
                print(f"已采集 {image_count}/{self.config['calibration_images']} 张图像")

            elif key == 27:  # ESC键退出
                print("用户中断采集")
                break

        cv2.destroyAllWindows()
        return image_count > 0

    def calibrate_cameras(self):
        """标定相机内参和外参"""
        print("\n开始相机标定...")

        calibration_results = {}

        for cam_id, data in self.camera_data.items():
            if len(data['object_points']) < 10:
                print(f"相机 {cam_id} 标定图像不足，跳过")
                continue

            print(f"标定相机 {cam_id}...")

            # 内参标定
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                data['object_points'],
                data['image_points'],
                data['image_size'][::-1],  # (width, height)
                None, None
            )

            if ret:
                # 计算重投影误差
                mean_error = 0
                for i in range(len(data['object_points'])):
                    imgpoints2, _ = cv2.projectPoints(
                        data['object_points'][i],
                        rvecs[i], tvecs[i],
                        camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(data['image_points'][i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error

                mean_error /= len(data['object_points'])

                calibration_results[cam_id] = {
                    'intrinsic': camera_matrix.tolist(),
                    'distortion': dist_coeffs.tolist(),
                    'reprojection_error': float(mean_error),
                    'image_size': data['image_size'],
                    'num_images': len(data['object_points'])
                }

                print(f"  重投影误差: {mean_error:.4f} 像素")
                print(f"  内参矩阵:\n{camera_matrix}")
                print(f"  畸变系数: {dist_coeffs.flatten()}")
            else:
                print(f"  标定失败")

        # 立体标定（相机对之间的标定）
        if len(calibration_results) >= 2:
            self._stereo_calibrate(calibration_results)

        return calibration_results

    def _stereo_calibrate(self, single_calib_results):
        """立体标定"""
        print("\n进行立体标定...")

        # 获取所有相机对
        camera_pairs = [(0, 1), (1, 2), (0, 2)]

        for pair in camera_pairs:
            cam1, cam2 = pair

            if cam1 not in self.camera_data or cam2 not in self.camera_data:
                continue

            # 找到共同采集的图像
            common_indices = []
            min_len = min(len(self.camera_data[cam1]['object_points']),
                          len(self.camera_data[cam2]['object_points']))

            # 简单假设：相同索引的图像是同时采集的
            for i in range(min_len):
                common_indices.append(i)

            if len(common_indices) < 5:
                print(f"相机对 {pair} 共同图像不足，跳过立体标定")
                continue

            # 提取共同的特征点
            objpoints = [self.camera_data[cam1]['object_points'][i] for i in common_indices]
            imgpoints1 = [self.camera_data[cam1]['image_points'][i] for i in common_indices]
            imgpoints2 = [self.camera_data[cam2]['image_points'][i] for i in common_indices]

            # 获取内参
            camera_matrix1 = np.array(single_calib_results[cam1]['intrinsic'])
            dist_coeffs1 = np.array(single_calib_results[cam1]['distortion'])
            camera_matrix2 = np.array(single_calib_results[cam2]['intrinsic'])
            dist_coeffs2 = np.array(single_calib_results[cam2]['distortion'])
            image_size = tuple(single_calib_results[cam1]['image_size'][::-1])

            # 立体标定
            flags = cv2.CALIB_FIX_INTRINSIC
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints1, imgpoints2,
                camera_matrix1, dist_coeffs1,
                camera_matrix2, dist_coeffs2,
                image_size,
                flags=flags
            )

            if ret:
                self.stereo_calibration[pair] = {
                    'rotation': R.tolist(),
                    'translation': T.tolist(),
                    'essential': E.tolist(),
                    'fundamental': F.tolist()
                }

                print(f"相机对 {pair} 立体标定完成")
                print(f"  旋转矩阵 R:\n{R}")
                print(f"  平移向量 T: {T.flatten()}")

    def save_calibration(self, output_dir):
        """保存标定结果"""
        calib_file = os.path.join(output_dir, "calibration_results.json")

        results = {
            'timestamp': datetime.now().isoformat(),
            'single_camera': {},
            'stereo_calibration': self.stereo_calibration
        }

        # 保存每个相机的标定结果
        for cam_id, data in self.camera_data.items():
            if len(data['object_points']) > 0:
                results['single_camera'][cam_id] = {
                    'num_images': len(data['object_points']),
                    'image_size': data['image_size']
                }

        with open(calib_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"标定结果已保存: {calib_file}")
        return calib_file

    def load_calibration(self, calib_file):
        """加载标定结果"""
        if not os.path.exists(calib_file):
            return False

        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 这里可以根据需要加载标定数据
            print(f"加载标定结果: {calib_file}")
            return True
        except Exception as e:
            print(f"加载标定失败: {e}")
            return False