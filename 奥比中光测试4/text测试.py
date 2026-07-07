"""
稳定版多帧TSDF融合模型修复系统 - 完全修复版
修复相机连接、内存管理和导入问题
"""

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import json
import os
import gc
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 动态导入OpenNI2，处理可能的导入错误
try:
    from primesense import openni2
    OPENNI2_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ OpenNI2导入失败: {e}")
    print("请确保已安装primesense: pip install primesense")
    OPENNI2_AVAILABLE = False

class StableMultiFrameReconstructor:
    """稳定版多帧重建器 - 完全修复版"""

    def __init__(self, driver_path="", use_gpu=False):
        self.driver_path = driver_path
        self.use_gpu = use_gpu

        if not OPENNI2_AVAILABLE:
            print("❌ OpenNI2不可用，无法使用相机功能")
            return

        # 相机相关资源
        self.device = None
        self.depth_stream = None
        self.ir_stream = None
        self.camera_initialized = False
        self.frame_counter = 0
        self.last_frame_time = 0

        # 数据存储
        self.depth_frames = []
        self.ir_frames = []

        # TSDF体积
        self.tsdf_volume = None

        # 收敛检测
        self.convergence_history = []

        # 相机参数
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525.0, 525.0, 319.5, 239.5
        )

        # 优化后的参数 - 针对稳定性优化
        self.params = {
            'depth_scale': 0.001,
            'depth_clip_min': 0.3,
            'depth_clip_max': 8.0,
            'median_filter_size': 1,
            'bilateral_sigma_color': 5,
            'bilateral_sigma_space': 5,
            'tsdf_voxel_length': 0.008,  # 稍微增大，减少内存压力
            'tsdf_sdf_trunc': 0.04,
            'convergence_threshold': 0.001,
            'max_iterations': 20,  # 减少最大迭代次数，测试稳定性
            'min_iterations': 3,
            'stable_frames': 3,
            'frame_interval': 0.2,  # 增加帧采集间隔
            'memory_cleanup_interval': 5,  # 每5帧清理一次内存
        }

        self.output_dir = None
        self.iteration = 0

        print("🚀 稳定版多帧重建器初始化完成")
        print(f"   最大迭代: {self.params['max_iterations']}帧")
        print(f"   帧间隔: {self.params['frame_interval']}秒")
        print(f"   OpenNI2状态: {'可用' if OPENNI2_AVAILABLE else '不可用'}")

    def safe_camera_initialization(self):
        """安全的相机初始化，带重试机制"""
        if not OPENNI2_AVAILABLE:
            print("❌ OpenNI2不可用，跳过相机初始化")
            return False

        print("=" * 60)
        print("安全初始化相机")
        print("=" * 60)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"尝试 {attempt + 1}/{max_retries}...")

                # 清理之前的资源
                self.safe_camera_cleanup()

                # 等待一段时间
                time.sleep(0.5)

                # 初始化OpenNI2
                if self.driver_path and os.path.exists(self.driver_path):
                    print(f"使用驱动路径: {self.driver_path}")
                    openni2.initialize(self.driver_path)
                else:
                    print("使用默认驱动路径")
                    openni2.initialize()

                # 打开设备
                self.device = openni2.Device.open_any()

                # 创建并启动深度流
                self.depth_stream = self.device.create_depth_stream()
                self.depth_stream.start()

                # 创建并启动红外流
                self.ir_stream = self.device.create_ir_stream()
                self.ir_stream.start()

                # 注意：移除了c_api相关代码，避免导入问题

                self.camera_initialized = True
                print("✅ 相机初始化成功")
                return True

            except Exception as e:
                print(f"❌ 初始化尝试 {attempt + 1} 失败: {str(e)[:100]}...")
                self.safe_camera_cleanup()

                if attempt < max_retries - 1:
                    print("等待2秒后重试...")
                    time.sleep(2)
                else:
                    print("❌ 相机初始化完全失败")
                    return False

    def safe_camera_cleanup(self):
        """安全的相机资源清理"""
        try:
            if hasattr(self, 'depth_stream') and self.depth_stream:
                try:
                    self.depth_stream.stop()
                except:
                    pass
                finally:
                    self.depth_stream = None

            if hasattr(self, 'ir_stream') and self.ir_stream:
                try:
                    self.ir_stream.stop()
                except:
                    pass
                finally:
                    self.ir_stream = None

            if hasattr(self, 'device') and self.device:
                try:
                    self.device.close()
                except:
                    pass
                finally:
                    self.device = None

            self.camera_initialized = False

        except Exception as e:
            print(f"⚠️ 资源清理错误: {e}")

    def safe_frame_capture(self):
        """安全的帧采集，带错误恢复"""
        if not self.camera_initialized:
            print("❌ 相机未初始化")
            return None, None, None

        try:
            # 控制采集频率
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time
            if time_since_last < self.params['frame_interval']:
                sleep_time = self.params['frame_interval'] - time_since_last
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 采集深度帧
            depth_frame = self.depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)
            depth = depth.astype(np.float32) * self.params['depth_scale']

            # 采集红外帧
            ir_frame = self.ir_stream.read_frame()
            ir_data = ir_frame.get_buffer_as_uint16()
            ir = np.frombuffer(ir_data, dtype=np.uint16).reshape(480, 640)

            self.last_frame_time = time.time()
            self.frame_counter += 1

            return depth, ir, self.last_frame_time

        except Exception as e:
            print(f"❌ 帧采集失败: {e}")

            # 尝试恢复相机连接
            print("尝试恢复相机连接...")
            self.camera_initialized = False
            time.sleep(1)

            if self.safe_camera_initialization():
                print("✅ 相机恢复成功")
                # 限制递归深度
                if self.frame_counter < 10:
                    return self.safe_frame_capture()

            return None, None, None

    def preprocess_depth(self, depth):
        """轻量级深度图预处理"""
        if depth is None:
            return None

        depth_processed = depth.copy()

        # 范围裁剪
        mask = (depth_processed >= self.params['depth_clip_min']) & \
               (depth_processed <= self.params['depth_clip_max'])
        depth_processed[~mask] = 0

        # 跳过复杂的滤波，直接使用中值滤波（可选）
        if self.params['median_filter_size'] > 1 and np.sum(mask) > 100:
            try:
                # 只对有效区域进行滤波
                depth_filtered = depth_processed.copy()
                depth_filtered[mask] = np.median(depth_filtered[mask])
                depth_processed = depth_filtered
            except:
                pass  # 如果滤波失败，使用原始数据

        return depth_processed

    def preprocess_ir(self, ir):
        """轻量级红外图预处理"""
        if ir is None:
            return None

        ir_processed = ir.copy()

        # 简单归一化
        ir_nonzero = ir_processed[ir_processed > 0]
        if len(ir_nonzero) > 0:
            min_val, max_val = ir_nonzero.min(), ir_nonzero.max()
            if max_val > min_val:
                ir_normalized = ((ir_processed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                ir_normalized = ir_processed.astype(np.uint8)
        else:
            ir_normalized = ir_processed.astype(np.uint8)

        # 轻微平滑
        try:
            ir_smoothed = cv2.GaussianBlur(ir_normalized, (3, 3), 0.5)
        except:
            ir_smoothed = ir_normalized

        return ir_smoothed

    def initialize_tsdf(self):
        """初始化TSDF体积"""
        print("初始化TSDF体积...")

        try:
            self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.params['tsdf_voxel_length'],
                sdf_trunc=self.params['tsdf_sdf_trunc'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            print(f"✅ TSDF体积初始化完成，体素大小: {self.params['tsdf_voxel_length']}米")
            return True
        except Exception as e:
            print(f"❌ TSDF初始化失败: {e}")
            return False

    def integrate_frame(self, depth, ir, iteration):
        """将帧融合到TSDF"""
        if depth is None or ir is None or self.tsdf_volume is None:
            return False

        try:
            # 检查深度图有效性
            valid_pixels = np.sum(depth > 0)
            if valid_pixels < 100:
                print(f"⚠️ 第{iteration}帧有效像素过少: {valid_pixels}")
                return False

            # 转换为Open3D格式
            depth_image = o3d.geometry.Image((depth * 1000).astype(np.uint16))

            # 将红外图转换为彩色
            try:
                ir_normalized = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX)
                ir_colored = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                color_image = o3d.geometry.Image(ir_colored)
            except:
                # 如果彩色转换失败，使用灰度图
                color_image = o3d.geometry.Image(ir.astype(np.uint8))

            # 创建RGBD图像
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1000.0,
                depth_trunc=self.params['depth_clip_max'],
                convert_rgb_to_intensity=False
            )

            # 融合（静态相机，使用单位矩阵作为位姿）
            extrinsic = np.eye(4)
            self.tsdf_volume.integrate(rgbd_image, self.intrinsic, extrinsic)

            print(f"✅ 第{iteration}帧融合成功，有效点: {valid_pixels}")
            return True

        except Exception as e:
            print(f"❌ 第{iteration}帧融合失败: {e}")
            return False

    def periodic_memory_cleanup(self, iteration):
        """定期内存清理"""
        if iteration % self.params['memory_cleanup_interval'] == 0:
            print("🔄 执行定期内存清理...")

            # 强制垃圾回收
            gc.collect()

            # 清理不必要的列表数据
            if len(self.depth_frames) > 5:
                self.depth_frames = self.depth_frames[-3:]  # 只保留最近3帧

            if len(self.ir_frames) > 5:
                self.ir_frames = self.ir_frames[-3:]

            print("✅ 内存清理完成")

    def run_stable_reconstruction(self):
        """稳定版重建主循环"""
        print("=" * 70)
        print("稳定版多帧重建开始")
        print("=" * 70)

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"stable_reconstruction_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")

        # 检查OpenNI2可用性
        if not OPENNI2_AVAILABLE:
            print("❌ OpenNI2不可用，无法进行相机采集")
            print("💡 将使用模拟数据测试流程...")
            # 这里可以添加模拟数据模式
            return False

        # 初始化相机
        if not self.safe_camera_initialization():
            print("❌ 无法初始化相机，程序退出")
            return False

        # 初始化TSDF
        if not self.initialize_tsdf():
            print("❌ 无法初始化TSDF，程序退出")
            self.safe_camera_cleanup()
            return False

        print("\n" + "=" * 60)
        print("开始多帧融合")
        print("=" * 60)

        prev_depth = None
        self.iteration = 0
        converged = False

        try:
            while not converged and self.iteration < self.params['max_iterations']:
                self.iteration += 1
                print(f"\n📸 第 {self.iteration}/{self.params['max_iterations']} 帧")

                # 1. 采集帧
                depth, ir, timestamp = self.safe_frame_capture()
                if depth is None:
                    print("❌ 采集失败，跳过本帧")
                    # 如果连续失败多次，提前退出
                    if self.iteration > 5 and self.frame_counter == 0:
                        print("❌ 连续采集失败，提前退出")
                        break
                    continue

                # 2. 预处理
                processed_depth = self.preprocess_depth(depth)
                processed_ir = self.preprocess_ir(ir)

                if processed_depth is None:
                    print("❌ 预处理失败，跳过本帧")
                    continue

                # 3. 计算收敛度量
                convergence_metric = 1.0  # 默认值
                if prev_depth is not None:
                    valid_mask = (processed_depth > 0) & (prev_depth > 0)
                    if np.sum(valid_mask) > 100:
                        diff = processed_depth[valid_mask] - prev_depth[valid_mask]
                        convergence_metric = np.mean(diff ** 2)
                        # 归一化
                        depth_range = self.params['depth_clip_max'] - self.params['depth_clip_min']
                        if depth_range > 0:
                            convergence_metric = convergence_metric / (depth_range ** 2)

                self.convergence_history.append(convergence_metric)
                print(f"   收敛度量: {convergence_metric:.6f}")
                print(f"   有效深度点: {np.sum(processed_depth > 0)}")

                # 4. 融合到TSDF
                if self.integrate_frame(processed_depth, processed_ir, self.iteration):
                    # 5. 保存结果（每3帧保存一次，减少I/O压力）
                    if self.iteration % 3 == 0 or self.iteration == 1:
                        self.save_iteration_results(
                            self.iteration, depth, ir, processed_depth, processed_ir
                        )

                    # 6. 检查收敛
                    if self.iteration >= self.params['min_iterations']:
                        # 检查连续稳定帧
                        if len(self.convergence_history) >= self.params['stable_frames']:
                            recent = self.convergence_history[-self.params['stable_frames']:]
                            if all(m < self.params['convergence_threshold'] for m in recent):
                                print(f"🎯 模型已收敛！连续{self.params['stable_frames']}帧稳定")
                                converged = True

                    # 7. 更新前一帧
                    prev_depth = processed_depth

                    # 8. 定期内存清理
                    self.periodic_memory_cleanup(self.iteration)

                    # 9. 短暂暂停，控制节奏
                    time.sleep(0.05)
                else:
                    print("⚠️ 融合失败，继续尝试")

        except KeyboardInterrupt:
            print("\n🔴 用户中断重建")
        except Exception as e:
            print(f"\n❌ 重建过程错误: {e}")
            traceback.print_exc()

        # 提取并保存最终模型
        print("\n" + "=" * 60)
        print("提取最终模型")
        print("=" * 60)

        if self.tsdf_volume is not None:
            try:
                mesh = self.tsdf_volume.extract_triangle_mesh()
                if mesh is not None and len(mesh.vertices) > 0:
                    mesh.compute_vertex_normals()

                    # 保存最终模型
                    final_dir = os.path.join(self.output_dir, "final_model")
                    os.makedirs(final_dir, exist_ok=True)

                    obj_path = os.path.join(final_dir, "final_model.obj")
                    stl_path = os.path.join(final_dir, "final_model.stl")
                    ply_path = os.path.join(final_dir, "final_model.ply")

                    # 简化网格，避免文件过大
                    if len(mesh.triangles) > 50000:
                        print("简化网格...")
                        mesh = mesh.simplify_quadric_decimation(50000)
                        mesh.compute_vertex_normals()

                    try:
                        o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=True)
                        print(f"✅ 保存OBJ: {obj_path}")
                    except:
                        print("⚠️ 保存OBJ失败")

                    try:
                        o3d.io.write_triangle_mesh(stl_path, mesh)
                        print(f"✅ 保存STL: {stl_path}")
                    except:
                        print("⚠️ 保存STL失败")

                    try:
                        o3d.io.write_triangle_mesh(ply_path, mesh)
                        print(f"✅ 保存PLY: {ply_path}")
                    except:
                        print("⚠️ 保存PLY失败")

                    print(f"   顶点数: {len(mesh.vertices):,}")
                    print(f"   面片数: {len(mesh.triangles):,}")

                    # 显示模型（可选）
                    try:
                        print("正在显示3D模型...")
                        o3d.visualization.draw_geometries(
                            [mesh],
                            window_name="最终模型",
                            width=1024,
                            height=768
                        )
                    except:
                        print("⚠️ 3D可视化失败，模型文件已保存")
                else:
                    print("❌ 提取的网格为空")
            except Exception as e:
                print(f"❌ 网格提取失败: {e}")

        # 生成报告
        self.generate_report()

        # 清理资源
        self.safe_camera_cleanup()

        print("\n" + "=" * 70)
        print("稳定版重建完成")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print("=" * 70)

        return True

    def save_iteration_results(self, iteration, raw_depth, raw_ir, proc_depth, proc_ir):
        """保存迭代结果"""
        if raw_depth is None or raw_ir is None:
            return

        # 创建迭代目录
        iter_dir = os.path.join(self.output_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_dir, exist_ok=True)

        try:
            # 1. 保存深度图处理步骤
            self.save_depth_steps(iter_dir, raw_depth, proc_depth)

            # 2. 保存红外图处理步骤
            self.save_ir_steps(iter_dir, raw_ir, proc_ir)

            # 3. 提取并保存当前网格（每5次迭代或最后一次）
            if iteration % 5 == 0 or iteration == 1:
                if self.tsdf_volume is not None:
                    try:
                        mesh = self.tsdf_volume.extract_triangle_mesh()
                        if mesh is not None and len(mesh.vertices) > 0:
                            mesh.compute_vertex_normals()

                            # 保存简化版网格
                            if len(mesh.triangles) > 50000:
                                mesh = mesh.simplify_quadric_decimation(50000)
                                mesh.compute_vertex_normals()

                            obj_path = os.path.join(iter_dir, f"mesh_iter_{iteration:04d}.obj")
                            stl_path = os.path.join(iter_dir, f"mesh_iter_{iteration:04d}.stl")

                            o3d.io.write_triangle_mesh(obj_path, mesh, write_vertex_normals=True)
                            o3d.io.write_triangle_mesh(stl_path, mesh)

                            print(f"📊 网格已保存: {len(mesh.vertices)}顶点, {len(mesh.triangles)}面片")
                    except Exception as e:
                        print(f"⚠️ 保存网格失败: {e}")

            print(f"📁 第{iteration}次迭代结果已保存: {iter_dir}")

        except Exception as e:
            print(f"❌ 保存迭代结果失败: {e}")

    def save_depth_steps(self, save_dir, raw_depth, proc_depth):
        """保存深度图处理步骤"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            # 原始深度图
            depth_colored = self.depth_to_colormap(raw_depth)
            axes[0, 0].imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("1. Raw Depth")
            axes[0, 0].axis('off')

            # 范围裁剪后
            depth_clipped = raw_depth.copy()
            depth_clipped[(depth_clipped < self.params['depth_clip_min']) |
                         (depth_clipped > self.params['depth_clip_max'])] = 0
            clipped_colored = self.depth_to_colormap(depth_clipped)
            axes[0, 1].imshow(cv2.cvtColor(clipped_colored, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("2. Range Clipped")
            axes[0, 1].axis('off')

            # 有效区域显示
            valid_mask = raw_depth > 0
            mask_display = np.zeros((*raw_depth.shape, 3), dtype=np.uint8)
            mask_display[valid_mask] = [0, 255, 0]  # 绿色表示有效
            axes[1, 0].imshow(mask_display)
            axes[1, 0].set_title("3. Valid Pixels")
            axes[1, 0].axis('off')

            # 最终处理结果
            proc_colored = self.depth_to_colormap(proc_depth)
            axes[1, 1].imshow(cv2.cvtColor(proc_colored, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title("4. Final Processed")
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "depth_steps.png"), dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 保存深度图步骤失败: {e}")

    def save_ir_steps(self, save_dir, raw_ir, proc_ir):
        """保存红外图处理步骤"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            # 原始红外图
            axes[0, 0].imshow(raw_ir, cmap='gray')
            axes[0, 0].set_title("1. Raw IR")
            axes[0, 0].axis('off')

            # 直方图均衡化
            if raw_ir.max() > raw_ir.min():
                ir_normalized = cv2.normalize(raw_ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ir_equalized = cv2.equalizeHist(ir_normalized)
            else:
                ir_equalized = raw_ir.astype(np.uint8)
            axes[0, 1].imshow(ir_equalized, cmap='gray')
            axes[0, 1].set_title("2. Histogram Equalized")
            axes[0, 1].axis('off')

            # 伪彩色显示
            ir_colored = cv2.applyColorMap(ir_equalized, cv2.COLORMAP_JET)
            axes[1, 0].imshow(cv2.cvtColor(ir_colored, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("3. Color Mapped")
            axes[1, 0].axis('off')

            # 最终处理结果
            axes[1, 1].imshow(proc_ir, cmap='gray')
            axes[1, 1].set_title("4. Final Processed")
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "ir_steps.png"), dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ 保存红外图步骤失败: {e}")

    def depth_to_colormap(self, depth):
        """深度图转伪彩色"""
        if depth is None or depth.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            depth_mm = depth * 1000
            valid_depth = depth_mm[depth_mm > 0]

            if len(valid_depth) == 0:
                return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

            min_val, max_val = valid_depth.min(), valid_depth.max()
            if max_val > min_val:
                depth_normalized = np.zeros_like(depth_mm, dtype=np.uint8)
                mask = depth_mm > 0
                depth_normalized[mask] = ((depth_mm[mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_mm, dtype=np.uint8)

            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            depth_colored[depth_mm == 0] = [0, 0, 0]

            return depth_colored
        except:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def generate_report(self):
        """生成报告"""
        report_path = os.path.join(self.output_dir, "reconstruction_report.txt")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("稳定版多帧重建报告\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"重建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总迭代次数: {self.iteration}\n")
                f.write(f"成功采集帧数: {self.frame_counter}\n")
                f.write(f"输出目录: {self.output_dir}\n\n")

                f.write("参数设置:\n")
                for key, value in self.params.items():
                    f.write(f"  {key}: {value}\n")

                f.write("\n收敛历史:\n")
                for i, metric in enumerate(self.convergence_history, 1):
                    f.write(f"  第{i:03d}帧: {metric:.6f}\n")

            print(f"📄 报告已保存: {report_path}")
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")

def main():
    """主函数"""
    print("=" * 70)
    print("稳定版多帧TSDF融合模型修复系统 - 完全修复版")
    print("修复相机连接、内存管理和导入问题")
    print("=" * 70)

    # 检查主要依赖
    try:
        import cv2
        import numpy as np
        import open3d as o3d
        print("✅ 主要依赖库加载成功")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install opencv-python numpy open3d matplotlib")
        return

    # 初始化重建器
    driver_path = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"

    # 检查驱动路径是否存在
    if not os.path.exists(driver_path):
        print(f"⚠️ 驱动路径不存在: {driver_path}")
        print("将尝试使用默认路径...")
        driver_path = ""

    reconstructor = StableMultiFrameReconstructor(driver_path, use_gpu=False)

    # 运行重建
    print("\n🚀 开始稳定版重建...")
    print("注意：此版本优化了相机连接和内存管理")
    print("=" * 70)

    success = reconstructor.run_stable_reconstruction()

    if success:
        print("\n🎉 重建成功完成！")
        print("💡 查看输出目录中的完整结果")
    else:
        print("\n❌ 重建失败或部分完成")
        print("💡 请检查相机连接和驱动路径")

    print("=" * 70)

if __name__ == "__main__":
    main()