"""
数据集导出工具模块
将采集的数据导出为COLMAP/NeRF/3DGS等格式
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime


class DatasetExporter:
    def __init__(self):
        """初始化数据集导出器"""
        self.export_formats = ['colmap', 'nerf', '3dgs', 'custom']
        self.is_exported = False

    def export_dataset(self, frames, output_dir, format='colmap'):
        """导出数据集"""
        if not frames or len(frames) == 0:
            print("❌ 无数据可导出")
            return False

        print(f"📦 导出数据集到 {output_dir} (格式: {format})")

        try:
            # 创建目录
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "poses"), exist_ok=True)

            # 根据格式导出
            if format == 'colmap':
                success = self._export_colmap_format(frames, output_dir)
            elif format == 'nerf':
                success = self._export_nerf_format(frames, output_dir)
            elif format == '3dgs':
                success = self._export_3dgs_format(frames, output_dir)
            else:
                success = self._export_custom_format(frames, output_dir)

            if success:
                self.is_exported = True
                print(f"✅ 数据集导出完成: {output_dir}")
                print(f"   总帧数: {len(frames)}")
                print(f"   图像尺寸: {self._get_image_size(frames)}")

                # 保存元数据
                self._save_metadata(frames, output_dir)

            return success

        except Exception as e:
            print(f"❌ 导出数据集失败: {e}")
            return False

    def _export_colmap_format(self, frames, output_dir):
        """导出为COLMAP格式"""
        try:
            images_file = os.path.join(output_dir, "images.txt")
            cameras_file = os.path.join(output_dir, "cameras.txt")

            with open(cameras_file, 'w') as f:
                f.write("# Camera list with one line of data per camera:\n")
                f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                f.write("# Number of cameras: 1\n")
                f.write("1 PINHOLE 640 480 475.0 475.0 320.0 240.0\n")

            with open(images_file, 'w') as f:
                f.write("# Image list with two lines of data per image:\n")
                f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

                for i, frame in enumerate(frames):
                    if i >= 100:  # 限制导出数量
                        break

                    # 保存图像
                    if frame.get('color') is not None:
                        img_path = os.path.join(output_dir, "images", f"image_{i:06d}.jpg")
                        cv2.imwrite(img_path, frame['color'])

                    # 保存深度图
                    if frame.get('depth') is not None:
                        depth_path = os.path.join(output_dir, "depths", f"depth_{i:06d}.png")
                        depth_normalized = cv2.normalize(frame['depth'], None, 0, 65535, cv2.NORM_MINMAX)
                        cv2.imwrite(depth_path, depth_normalized.astype(np.uint16))

                    # 写入位姿
                    pose = frame.get('pose')
                    if pose is not None and len(pose) == 7:
                        # 转换为COLMAP格式: QW, QX, QY, QZ, TX, TY, TZ
                        x, y, z, qx, qy, qz, qw = pose

                        f.write(f"{i + 1} {qw} {qx} {qy} {qz} {x} {y} {z} 1 image_{i:06d}.jpg\n")
                        f.write("\n")  # 空行（无2D点）

            return True

        except Exception as e:
            print(f"❌ 导出COLMAP格式失败: {e}")
            return False

    def _export_nerf_format(self, frames, output_dir):
        """导出为NeRF格式"""
        try:
            transforms_file = os.path.join(output_dir, "transforms.json")

            frames_data = []

            for i, frame in enumerate(frames):
                if i >= 100:  # 限制数量
                    break

                # 保存图像
                if frame.get('color') is not None:
                    img_path = os.path.join(output_dir, "images", f"image_{i:06d}.png")
                    cv2.imwrite(img_path, frame['color'])

                    # 保存深度图
                    if frame.get('depth') is not None:
                        depth_path = os.path.join(output_dir, "depths", f"depth_{i:06d}.npy")
                        np.save(depth_path, frame['depth'])

                    # 构建帧数据
                    pose = frame.get('pose')
                    if pose is not None and len(pose) == 7:
                        x, y, z, qx, qy, qz, qw = pose

                        # 四元数转旋转矩阵
                        R = np.array([
                            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
                        ])

                        # 构建变换矩阵
                        transform = np.eye(4)
                        transform[:3, :3] = R
                        transform[:3, 3] = [x, y, z]

                        frame_data = {
                            "file_path": f"./images/image_{i:06d}.png",
                            "transform_matrix": transform.tolist()
                        }

                        frames_data.append(frame_data)

            # 构建完整的transforms.json
            transforms = {
                "camera_angle_x": 1.0472,  # 60度FOV
                "frames": frames_data
            }

            with open(transforms_file, 'w') as f:
                json.dump(transforms, f, indent=2)

            return True

        except Exception as e:
            print(f"❌ 导出NeRF格式失败: {e}")
            return False

    def _export_3dgs_format(self, frames, output_dir):
        """导出为3D Gaussian Splatting格式"""
        try:
            # 3DGS需要COLMAP格式作为输入
            # 所以我们先导出为COLMAP格式
            success = self._export_colmap_format(frames, output_dir)

            if success:
                # 创建额外的配置文件
                config = {
                    "dataset_type": "colmap",
                    "source_path": output_dir,
                    "model_path": output_dir,
                    "images_path": os.path.join(output_dir, "images"),
                    "resolution": -1,
                    "white_background": False,
                    "data_device": "cuda"
                }

                config_file = os.path.join(output_dir, "config.json")
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)

            return success

        except Exception as e:
            print(f"❌ 导出3DGS格式失败: {e}")
            return False

    def _export_custom_format(self, frames, output_dir):
        """导出为自定义格式"""
        try:
            # 保存所有数据为.npz格式（简单直接）
            data_dict = {}

            for i, frame in enumerate(frames):
                if i >= 100:  # 限制数量
                    break

                if frame.get('color') is not None:
                    data_dict[f'color_{i}'] = frame['color']

                if frame.get('depth') is not None:
                    data_dict[f'depth_{i}'] = frame['depth']

                if frame.get('pose') is not None:
                    data_dict[f'pose_{i}'] = frame['pose']

                if frame.get('timestamp') is not None:
                    data_dict[f'time_{i}'] = frame['timestamp']

            # 保存为压缩的.npz文件
            npz_file = os.path.join(output_dir, "dataset.npz")
            np.savez_compressed(npz_file, **data_dict)

            return True

        except Exception as e:
            print(f"❌ 导出自定义格式失败: {e}")
            return False

    def _save_metadata(self, frames, output_dir):
        """保存元数据"""
        metadata = {
            "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": len(frames),
            "image_size": self._get_image_size(frames),
            "has_depth": any(f.get('depth') is not None for f in frames),
            "has_pose": any(f.get('pose') is not None for f in frames),
            "description": "Real-time 3D扫描数据集"
        }

        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _get_image_size(self, frames):
        """获取图像尺寸"""
        for frame in frames:
            if frame.get('color') is not None:
                return frame['color'].shape
        return "Unknown"