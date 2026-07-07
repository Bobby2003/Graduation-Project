"""
Python视觉里程计模块
基于特征点匹配的简单视觉里程计实现
"""

import cv2
import numpy as np
import time
from collections import deque


class VisualOdometry:
    def __init__(self, config=None):
        """初始化视觉里程计"""
        self.config = config or {}

        # 算法参数
        self.max_features = self.config.get('max_features', 1000)
        self.min_matches = self.config.get('min_matches', 20)
        self.ransac_threshold = self.config.get('ransac_threshold', 1.0)
        self.scale_factor = 1.0  # 估计的尺度因子
        self.min_parallax = 10  # 最小视差（像素）

        # 特征检测器
        self.detector = cv2.ORB_create(nfeatures=self.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 状态变量
        self.initialized = False
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.frame_count = 0

        # 轨迹和位姿
        self.trajectory = []
        self.current_pose = None
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)

        # 相机内参（默认值，应在初始化时设置）
        self.fx = 475.0
        self.fy = 475.0
        self.cx = 320.0
        self.cy = 240.0

        # 相机矩阵
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]])

        # 历史数据
        self.pose_history = deque(maxlen=100)
        self.motion_history = deque(maxlen=10)

        print("🤖 视觉里程计初始化完成")

    def initialize(self, camera_params=None):
        """初始化视觉里程计"""
        try:
            if camera_params:
                self.fx = camera_params.get('fx', self.fx)
                self.fy = camera_params.get('fy', self.fy)
                self.cx = camera_params.get('cx', self.cx)
                self.cy = camera_params.get('cy', self.cy)

                self.K = np.array([[self.fx, 0, self.cx],
                                   [0, self.fy, self.cy],
                                   [0, 0, 1]])

            self.initialized = True
            print(f"✅ 视觉里程计已初始化 (fx={self.fx}, fy={self.fy})")
            return True

        except Exception as e:
            print(f"❌ 视觉里程计初始化失败: {e}")
            return False

    def process_frame(self, color_image):
        """处理一帧图像并估计位姿"""
        if not self.initialized or color_image is None:
            return None

        try:
            # 转换为灰度图
            if len(color_image.shape) == 3:
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = color_image

            # 检测特征点
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            # 第一帧，初始化
            if self.last_frame is None:
                self.last_frame = gray
                self.last_keypoints = keypoints
                self.last_descriptors = descriptors

                # 初始位姿
                self.current_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                self.trajectory.append([0.0, 0.0, 0.0])

                self.frame_count += 1
                return self.current_pose

            # 特征匹配
            if descriptors is not None and self.last_descriptors is not None:
                matches = self.matcher.match(descriptors, self.last_descriptors)
                matches = sorted(matches, key=lambda x: x.distance)[:100]

                if len(matches) >= self.min_matches:
                    # 提取匹配点
                    src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([self.last_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    # 计算基础矩阵
                    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC,
                                                     self.ransac_threshold, 0.99)

                    if mask is not None and mask.sum() > self.min_matches:
                        # 使用内点重新计算
                        src_inliers = src_pts[mask.ravel() == 1]
                        dst_inliers = dst_pts[mask.ravel() == 1]

                        # 计算本质矩阵
                        E = self.K.T @ F @ self.K

                        # 从本质矩阵恢复位姿
                        R1, R2, t = cv2.decomposeEssentialMat(E)

                        # 选择正确的位姿
                        R, t = self._select_correct_pose(R1, R2, t, src_inliers, dst_inliers)

                        # 计算尺度（使用平均深度假设）
                        scale = self._estimate_scale()

                        # 更新位姿
                        self.translation = self.translation + self.rotation @ (t.ravel() * scale)
                        self.rotation = R @ self.rotation

                        # 转换为四元数
                        q = self._rotation_matrix_to_quaternion(self.rotation)

                        # 构造位姿 [x, y, z, qx, qy, qz, qw]
                        self.current_pose = [
                            float(self.translation[0]),
                            float(self.translation[1]),
                            float(self.translation[2]),
                            float(q[0]),  # qx
                            float(q[1]),  # qy
                            float(q[2]),  # qz
                            float(q[3])  # qw
                        ]

                        # 记录轨迹
                        self.trajectory.append([self.translation[0], self.translation[1], self.translation[2]])

                        # 记录运动历史（用于尺度估计）
                        motion = np.linalg.norm(t.ravel())
                        if motion > 0.01:  # 过滤微小运动
                            self.motion_history.append(motion)

            # 更新上一帧
            self.last_frame = gray
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors

            self.frame_count += 1

            return self.current_pose

        except Exception as e:
            print(f"[VO] 处理帧失败: {e}")
            return None

    def _select_correct_pose(self, R1, R2, t, pts1, pts2):
        """选择正确的位姿（通过三角化点在前方的数量判断）"""
        # 四种可能的位姿组合
        poses = [
            (R1, t), (R1, -t), (R2, t), (R2, -t)
        ]

        # 相机投影矩阵
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])

        best_score = -1
        best_pose = (R1, t)

        for R, t_vec in poses:
            P2 = self.K @ np.hstack([R, t_vec.reshape(3, 1)])

            # 三角化点
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]

            # 计算分数（点在两个相机前方的数量）
            # 相机1前方
            z1 = points_3d[2]
            # 相机2前方
            points_cam2 = R @ points_3d + t_vec.ravel().reshape(3, 1)
            z2 = points_cam2[2]

            positive_count = np.sum((z1 > 0) & (z2 > 0))

            if positive_count > best_score:
                best_score = positive_count
                best_pose = (R, t_vec)

        return best_pose

    def _estimate_scale(self):
        """估计运动尺度（简化版本）"""
        if len(self.motion_history) < 2:
            return 1.0  # 默认尺度

        # 使用平均运动作为参考
        avg_motion = np.mean(list(self.motion_history))

        # 简单尺度估计（可根据实际调整）
        scale = 0.1 / avg_motion if avg_motion > 0 else 1.0
        scale = np.clip(scale, 0.5, 2.0)  # 限制尺度范围

        return scale

    def _rotation_matrix_to_quaternion(self, R):
        """旋转矩阵转四元数"""
        # 确保是旋转矩阵
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt

        trace = np.trace(R)

        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        # 归一化
        q = np.array([qx, qy, qz, qw])
        q = q / np.linalg.norm(q)

        return q

    def get_trajectory(self):
        """获取轨迹"""
        return self.trajectory.copy()

    def get_current_pose(self):
        """获取当前位姿"""
        return self.current_pose.copy() if self.current_pose else None

    def get_stats(self):
        """获取统计信息"""
        return {
            'frames_processed': self.frame_count,
            'trajectory_length': len(self.trajectory),
            'initialized': self.initialized
        }

    def reset(self):
        """重置视觉里程计"""
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.trajectory = []
        self.current_pose = None
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.motion_history.clear()
        self.frame_count = 0

        print("✅ 视觉里程计已重置")

    def visualize_matches(self, frame1, frame2, matches, keypoints1, keypoints2):
        """可视化匹配结果"""
        if frame1 is None or frame2 is None:
            return None

        # 绘制匹配
        match_img = cv2.drawMatches(
            frame1, keypoints1,
            frame2, keypoints2,
            matches[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 添加文本
        cv2.putText(match_img, f"Matches: {len(matches)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return match_img