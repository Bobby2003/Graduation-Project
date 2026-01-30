"""
奥比中光Astra相机 - 彩色3D重建系统
使用独立的SDK接口层，代码更清晰
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
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 导入独立的SDK接口层
try:
    # 确保当前目录在Python路径中
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

    SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SDK接口层导入失败: {e}")
    SDK_AVAILABLE = False


class Color3DReconstructor:
    """彩色3D重建器 - 使用独立的SDK接口层"""

    def __init__(self, sdk_path=None):
        """初始化重建器

        Args:
            sdk_path: SDK路径，如果为None则使用默认路径
        """
        if not SDK_AVAILABLE:
            print("❌ SDK接口层不可用，无法使用相机功能")
            return

        # 使用独立的SDK接口层
        self.sdk_path = sdk_path if sdk_path else get_default_sdk_path()
        self.camera_sdk = None
        self.camera_initialized = False
        self.frame_counter = 0

        # OpenCV RGB摄像头
        self.cv_camera = None
        self.cv_camera_initialized = False

        # TSDF体积
        self.tsdf_volume = None

        # 相机参数
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, 525.0, 525.0, 319.5, 239.5
        )

        # 优化的彩色重建参数
        self.params = {
            # 深度参数
            'depth_scale': 0.001,
            'depth_clip_min': 0.6,  # 30cm
            'depth_clip_max': 8.0,  # 2米，更好的彩色效果

            # TSDF参数 - 优化彩色重建
            'tsdf_voxel_length': 0.008,  # 更小的体素，更精细的颜色
            'tsdf_sdf_trunc': 0.04,

            # 重建参数
            'max_iterations': 10,  # 减少迭代次数，提高稳定性
            'frame_interval': 0.5,  # 增加间隔，减少负载

            # 摄像头参数
            'camera_index': 0,
            'camera_width': 640,
            'camera_height': 480,
            'camera_fps': 15,

            # 颜色增强参数
            'color_enhance': True,
            'color_contrast': 1.2,
            'color_brightness': 10,

            # 网格优化
            'mesh_simplify': True,
            'target_vertices': 20000,
            'smooth_mesh': True,
            'smooth_iterations': 1,

            # 导出设置 - 确保所有格式都有颜色
            'export_ply': True,
            'export_obj': True,
            'export_stl': True,
            'export_glb': False,  # GLB格式支持颜色和纹理

            # 保存设置
            'save_every_frame': True,
            'save_intermediate': True,
        }

        self.output_dir = None
        self.iteration = 0
        self.mesh_count = 0

        print("🎨 彩色3D重建器初始化完成")
        print("   使用独立的SDK接口层")
        print("   专为彩色模型导出优化")

    def setup_camera(self):
        """设置相机 - 使用独立的SDK接口层"""
        if not SDK_AVAILABLE:
            return False

        print("=" * 50)
        print("设置相机 (使用独立SDK接口层)")
        print("=" * 50)

        try:
            # 1. 初始化SDK接口层
            print("初始化相机SDK...")
            self.camera_sdk = OrbbecCameraSDK(self.sdk_path)

            if not self.camera_sdk.initialize():
                print("❌ SDK初始化失败")
                return False

            # 2. 获取设备列表
            print("扫描设备...")
            devices = self.camera_sdk.get_device_list()
            if not devices:
                print("⚠️  未检测到奥比中光设备")

            # 3. 打开设备
            if not self.camera_sdk.open_device(device_index=0):
                print("❌ 打开设备失败")
                self.camera_sdk.cleanup()
                return False

            # 4. 创建深度流
            if not self.camera_sdk.create_stream(self.camera_sdk.ONI_SENSOR_DEPTH):
                print("❌ 创建深度流失败")
                self.camera_sdk.cleanup()
                return False

            # 5. 启动深度流
            if not self.camera_sdk.start_stream():
                print("❌ 启动深度流失败")
                self.camera_sdk.destroy_stream()
                self.camera_sdk.close_device()
                return False

            # 6. 设置RGB摄像头
            print("设置RGB摄像头...")
            self.cv_camera = cv2.VideoCapture(self.params['camera_index'])

            if not self.cv_camera.isOpened():
                print("尝试备用摄像头索引...")
                for idx in [1, 2, 3]:
                    self.cv_camera = cv2.VideoCapture(idx)
                    if self.cv_camera.isOpened():
                        print(f"✅ 使用摄像头索引 {idx}")
                        break

            if self.cv_camera.isOpened():
                self.cv_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.params['camera_width'])
                self.cv_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.params['camera_height'])
                self.cv_camera.set(cv2.CAP_PROP_FPS, self.params['camera_fps'])

                # 测试读取
                ret, frame = self.cv_camera.read()
                if ret and frame is not None:
                    self.cv_camera_initialized = True
                    print(f"✅ RGB摄像头就绪 ({frame.shape[1]}x{frame.shape[0]})")
                else:
                    print("❌ RGB摄像头读取失败")
            else:
                print("⚠️ 未找到可用的RGB摄像头")

            self.camera_initialized = True
            return True

        except Exception as e:
            print(f"❌ 相机设置失败: {e}")
            traceback.print_exc()
            return False

    def cleanup(self):
        """清理资源"""
        print("\n清理所有资源...")
        try:
            # 清理SDK资源
            if self.camera_sdk:
                self.camera_sdk.cleanup()
                self.camera_sdk = None
                print("✅ SDK资源已清理")

            # 清理OpenCV摄像头
            if self.cv_camera:
                self.cv_camera.release()
                print("✅ OpenCV摄像头已释放")

            # 清理Open3D资源
            self.tsdf_volume = None

            # 强制垃圾回收
            gc.collect()

            self.camera_initialized = False
            print("✅ 所有资源已清理")

        except Exception as e:
            print(f"⚠️ 清理资源时出错: {e}")

    def capture_frames(self):
        """采集深度和颜色帧 - 使用SDK接口层"""
        try:
            # 使用SDK接口层捕获深度帧
            result = self.camera_sdk.capture_depth_frame(timeout=1000)
            if not result:
                print("⚠️  深度帧读取失败")
                return None, None

            depth_array, frame_info = result

            # 深度数据单位转换: mm -> m
            depth_mm = depth_array.astype(np.float32)
            depth = depth_mm * 0.001  # 转换为米

            # 采集颜色帧
            color = None
            if self.cv_camera_initialized:
                ret, frame = self.cv_camera.read()
                if ret and frame is not None:
                    # 调整大小以匹配深度
                    if frame.shape[:2] != (480, 640):
                        frame = cv2.resize(frame, (640, 480))
                    color = frame.copy()

                    # 颜色增强
                    if self.params['color_enhance']:
                        color = self.enhance_color(color)

            self.frame_counter += 1
            return depth, color

        except Exception as e:
            print(f"❌ 帧采集失败: {e}")
            return None, None

    def enhance_color(self, image):
        """增强颜色"""
        try:
            # 转换为浮点数进行计算
            img_float = image.astype(np.float32) / 255.0

            # 对比度调整
            img_float = np.clip((img_float - 0.5) * self.params['color_contrast'] + 0.5, 0, 1)

            # 亮度调整
            img_float = np.clip(img_float + self.params['color_brightness'] / 255.0, 0, 1)

            # 转回8位
            enhanced = (img_float * 255).astype(np.uint8)
            return enhanced

        except:
            return image

    def process_depth(self, depth):
        """处理深度图"""
        if depth is None:
            return None

        # 范围裁剪
        mask = (depth >= self.params['depth_clip_min']) & (depth <= self.params['depth_clip_max'])
        depth_processed = depth.copy()
        depth_processed[~mask] = 0

        # 简单滤波
        if np.sum(mask) > 100:
            try:
                depth_processed = cv2.medianBlur(depth_processed, 3)
            except:
                pass

        return depth_processed

    def init_tsdf(self):
        """初始化TSDF体积"""
        print("初始化TSDF体积...")

        try:
            # 使用支持颜色的TSDF
            self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.params['tsdf_voxel_length'],
                sdf_trunc=self.params['tsdf_sdf_trunc'],
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )

            print(f"✅ TSDF体积初始化完成")
            print(f"   体素大小: {self.params['tsdf_voxel_length']}米")
            return True

        except Exception as e:
            print(f"❌ TSDF初始化失败: {e}")
            return False

    def fuse_frame(self, depth, color):
        """融合一帧"""
        if depth is None:
            return False

        try:
            # 创建深度图像 (单位: mm)
            depth_mm = (depth * 1000).astype(np.uint16)
            depth_image = o3d.geometry.Image(depth_mm)

            # 创建颜色图像
            if color is not None:
                # BGR转RGB
                color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                color_image = o3d.geometry.Image(color_rgb)
            else:
                # 使用默认颜色
                default_color = np.full((480, 640, 3), 200, dtype=np.uint8)
                color_image = o3d.geometry.Image(default_color)

            # 创建RGBD图像
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1000.0,
                depth_trunc=self.params['depth_clip_max'],
                convert_rgb_to_intensity=False
            )

            # 融合到TSDF
            extrinsic = np.eye(4)
            self.tsdf_volume.integrate(rgbd_image, self.intrinsic, extrinsic)

            return True

        except Exception as e:
            print(f"❌ 融合失败: {e}")
            return False

    def extract_color_mesh(self):
        """提取带颜色的网格"""
        if self.tsdf_volume is None:
            return None

        try:
            # 提取原始网格
            mesh = self.tsdf_volume.extract_triangle_mesh()

            if mesh is None or len(mesh.vertices) == 0:
                return None

            # 镜像修正（Astra相机）
            vertices = np.asarray(mesh.vertices)
            vertices[:, 0] = -vertices[:, 0]  # 左右镜像
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            # 确保有顶点颜色
            if not mesh.has_vertex_colors():
                print("⚠️ 网格没有颜色，添加默认颜色")
                mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色

            # 计算法线
            mesh.compute_vertex_normals()

            # 网格优化
            if self.params['mesh_simplify'] and len(mesh.vertices) > self.params['target_vertices']:
                target_triangles = int(len(mesh.triangles) * 0.5)
                mesh = mesh.simplify_quadric_decimation(target_triangles)
                mesh.compute_vertex_normals()

            # 网格平滑
            if self.params['smooth_mesh']:
                mesh = mesh.filter_smooth_simple(number_of_iterations=self.params['smooth_iterations'])
                mesh.compute_vertex_normals()

            return mesh

        except Exception as e:
            print(f"❌ 网格提取失败: {e}")
            return None

    def save_color_mesh(self, mesh, iteration, is_final=False):
        """保存彩色网格"""
        if mesh is None or len(mesh.vertices) == 0:
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if is_final:
                save_dir = os.path.join(self.output_dir, "final_models")
            else:
                save_dir = os.path.join(self.output_dir, "intermediate_models")

            os.makedirs(save_dir, exist_ok=True)

            saved_files = []

            # 保存PLY格式 - 最佳彩色支持
            if self.params['export_ply']:
                try:
                    ply_filename = f"color_model_f{iteration:03d}_{timestamp}.ply" if not is_final else f"final_color_model_{timestamp}.ply"
                    ply_path = os.path.join(save_dir, ply_filename)

                    # 使用二进制格式保存，支持顶点颜色
                    o3d.io.write_triangle_mesh(
                        ply_path,
                        mesh,
                        write_ascii=False,
                        compressed=True
                    )
                    saved_files.append(('PLY', ply_path))
                    print(f"   ✅ PLY格式已保存: {os.path.basename(ply_path)}")
                except Exception as e:
                    print(f"   ❌ PLY保存失败: {e}")

            # 保存OBJ格式 - 带顶点颜色
            if self.params['export_obj']:
                try:
                    obj_filename = f"color_model_f{iteration:03d}_{timestamp}.obj" if not is_final else f"final_color_model_{timestamp}.obj"
                    obj_path = os.path.join(save_dir, obj_filename)

                    # OBJ格式保存顶点颜色
                    o3d.io.write_triangle_mesh(
                        obj_path,
                        mesh,
                        write_vertex_normals=True,
                        write_vertex_colors=True
                    )
                    saved_files.append(('OBJ', obj_path))
                    print(f"   ✅ OBJ格式已保存: {os.path.basename(obj_path)}")
                except Exception as e:
                    print(f"   ❌ OBJ保存失败: {e}")

            # 保存STL格式 - 不带颜色
            if self.params['export_stl']:
                try:
                    stl_filename = f"model_f{iteration:03d}_{timestamp}.stl" if not is_final else f"final_model_{timestamp}.stl"
                    stl_path = os.path.join(save_dir, stl_filename)

                    # STL不支持颜色，保存几何
                    o3d.io.write_triangle_mesh(stl_path, mesh)
                    saved_files.append(('STL', stl_path))
                    print(f"   ✅ STL格式已保存: {os.path.basename(stl_path)}")
                except Exception as e:
                    print(f"   ❌ STL保存失败: {e}")

            # 保存GLB格式 - 更好的颜色支持
            if self.params['export_glb']:
                try:
                    import trimesh
                    glb_filename = f"color_model_f{iteration:03d}_{timestamp}.glb" if not is_final else f"final_color_model_{timestamp}.glb"
                    glb_path = os.path.join(save_dir, glb_filename)

                    # 转换为trimesh保存GLB
                    tri_mesh = trimesh.Trimesh(
                        vertices=np.asarray(mesh.vertices),
                        faces=np.asarray(mesh.triangles),
                        vertex_colors=np.asarray(mesh.vertex_colors)
                    )
                    tri_mesh.export(glb_path)
                    saved_files.append(('GLB', glb_path))
                    print(f"   ✅ GLB格式已保存: {os.path.basename(glb_path)}")
                except ImportError:
                    print("   ⚠️ trimesh未安装，跳过GLB格式")
                except Exception as e:
                    print(f"   ❌ GLB保存失败: {e}")

            self.mesh_count += 1
            return True

        except Exception as e:
            print(f"❌ 保存网格失败: {e}")
            return False

    def save_frame_images(self, iteration, depth, color):
        """保存帧图像"""
        try:
            images_dir = os.path.join(self.output_dir, "captured_images")
            os.makedirs(images_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存深度图
            if depth is not None:
                depth_colored = self.depth_to_colormap(depth)
                depth_path = os.path.join(images_dir, f"depth_{iteration:03d}_{timestamp}.png")
                cv2.imwrite(depth_path, depth_colored)

            # 保存颜色图
            if color is not None:
                color_path = os.path.join(images_dir, f"color_{iteration:03d}_{timestamp}.png")
                cv2.imwrite(color_path, color)

            return True

        except Exception as e:
            print(f"⚠️ 保存图像失败: {e}")
            return False

    def depth_to_colormap(self, depth):
        """深度图转伪彩色"""
        if depth is None:
            return None

        try:
            depth_mm = depth * 1000
            depth_normalized = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            return depth_colored
        except:
            return None

    def run_color_reconstruction(self):
        """运行彩色重建"""
        print("=" * 70)
        print("奥比中光Astra相机 - 彩色3D重建")
        print("使用独立SDK接口层 - 解决0xC0000374堆损坏问题")
        print("=" * 70)

        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"color_3d_reconstruction_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"📁 输出目录: {os.path.abspath(self.output_dir)}")

        # 保存配置
        config_file = os.path.join(self.output_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)

        # 检查SDK可用性
        if not SDK_AVAILABLE:
            print("❌ SDK接口层不可用")
            return False

        # 设置相机
        if not self.setup_camera():
            print("❌ 相机设置失败")
            return False

        # 初始化TSDF
        if not self.init_tsdf():
            print("❌ TSDF初始化失败")
            self.cleanup()
            return False

        print("\n" + "=" * 50)
        print("开始彩色3D重建")
        print("=" * 50)

        self.iteration = 0
        start_time = time.time()

        try:
            for i in range(self.params['max_iterations']):
                self.iteration = i + 1
                print(f"\n📸 第 {self.iteration}/{self.params['max_iterations']} 帧")

                # 等待帧间隔
                if i > 0:
                    time.sleep(self.params['frame_interval'])

                # 采集帧
                depth_raw, color_raw = self.capture_frames()
                if depth_raw is None:
                    print("❌ 深度采集失败，跳过")
                    continue

                # 处理深度
                depth = self.process_depth(depth_raw)
                if depth is None or np.sum(depth > 0) < 1000:
                    print(f"❌ 有效深度点不足: {np.sum(depth > 0)}")
                    continue

                print(f"   深度点: {np.sum(depth > 0)}个")
                print(f"   颜色: {'✓' if color_raw is not None else '✗'}")

                # 融合帧
                if self.fuse_frame(depth, color_raw):
                    print("✅ 帧融合成功")

                    # 保存中间结果
                    if self.params['save_intermediate'] and (i % 2 == 0 or i == self.params['max_iterations'] - 1):
                        mesh = self.extract_color_mesh()
                        if mesh is not None:
                            print(f"💾 保存中间模型 ({len(mesh.vertices)}顶点)...")
                            self.save_color_mesh(mesh, self.iteration, is_final=False)
                            del mesh

                    # 保存图像
                    if self.params['save_every_frame']:
                        self.save_frame_images(self.iteration, depth, color_raw)

                    # 进度显示
                    elapsed = time.time() - start_time
                    avg_time = elapsed / self.iteration
                    remaining = (self.params['max_iterations'] - self.iteration) * avg_time
                    print(f"⏱️  预计剩余: {remaining:.1f}秒")

                else:
                    print("❌ 帧融合失败")

                # 清理内存
                gc.collect()

        except KeyboardInterrupt:
            print("\n🔴 用户中断重建")
        except Exception as e:
            print(f"\n❌ 重建过程中出错: {e}")
            traceback.print_exc()

        # 最终处理
        print("\n" + "=" * 50)
        print("导出最终彩色模型")
        print("=" * 50)

        if self.tsdf_volume is not None and self.iteration > 0:
            print("🎨 提取最终彩色网格...")
            final_mesh = self.extract_color_mesh()

            if final_mesh is not None:
                print(f"   顶点数: {len(final_mesh.vertices)}")
                print(f"   面片数: {len(final_mesh.triangles)}")
                print(f"   有颜色: {final_mesh.has_vertex_colors()}")

                # 保存最终模型
                print("💾 保存最终彩色模型...")
                success = self.save_color_mesh(final_mesh, self.iteration, is_final=True)

                if success:
                    print("✅ 最终彩色模型保存成功")

                    # 显示模型信息
                    self.display_model_info(final_mesh)

                    # 可选：保存模型预览图
                    self.save_model_preview(final_mesh)

                del final_mesh
            else:
                print("❌ 无法提取最终网格")

        # 生成报告
        self.generate_report()

        # 清理资源
        self.cleanup()

        print("\n" + "=" * 70)
        print("彩色3D重建完成")
        print(f"处理帧数: {self.iteration}")
        print(f"模型数量: {self.mesh_count}")
        print(f"输出目录: {os.path.abspath(self.output_dir)}")
        print("=" * 70)

        return True

    def display_model_info(self, mesh):
        """显示模型信息"""
        try:
            vertices = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)

            print("\n📊 模型统计信息:")
            print(f"   顶点范围: X [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
            print(f"             Y [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
            print(f"             Z [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")

            if colors.shape[0] > 0:
                print(f"   颜色范围: R [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
                print(f"             G [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
                print(f"             B [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")

            # 计算模型尺寸
            bbox = mesh.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            print(f"   模型尺寸: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} 米")

        except Exception as e:
            print(f"⚠️ 显示模型信息失败: {e}")

    def save_model_preview(self, mesh):
        """保存模型预览图"""
        try:
            preview_dir = os.path.join(self.output_dir, "previews")
            os.makedirs(preview_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)  # 不显示窗口

            # 添加网格
            vis.add_geometry(mesh)

            # 设置视图
            vis.get_render_option().mesh_show_back_face = True
            vis.get_render_option().light_on = True
            vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])

            # 从不同角度保存图像
            angles = [0, 45, 90, 135, 180, 225, 270, 315]

            for i, angle in enumerate(angles):
                # 设置相机位置
                ctr = vis.get_view_control()
                ctr.set_zoom(0.8)
                ctr.rotate(angle * 10.0, 0.0)  # 10度步长

                # 捕获图像
                image_path = os.path.join(preview_dir, f"preview_angle_{angle:03d}_{timestamp}.png")
                vis.capture_screen_image(image_path, do_render=True)

            vis.destroy_window()
            print(f"📸 模型预览图已保存: {preview_dir}")

        except Exception as e:
            print(f"⚠️ 保存预览图失败: {e}")

    def generate_report(self):
        """生成重建报告"""
        try:
            report_path = os.path.join(self.output_dir, "彩色重建报告.txt")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("奥比中光Astra相机 - 彩色3D重建报告\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"重建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理帧数: {self.iteration}\n")
                f.write(f"输出模型: {self.mesh_count}个\n")
                f.write(f"输出目录: {self.output_dir}\n\n")

                f.write("重建参数:\n")
                for key, value in self.params.items():
                    f.write(f"  {key}: {value}\n")

                f.write("\n导出格式:\n")
                f.write(f"  PLY格式: {'✓' if self.params['export_ply'] else '✗'} (推荐，带颜色)\n")
                f.write(f"  OBJ格式: {'✓' if self.params['export_obj'] else '✗'} (带顶点颜色)\n")
                f.write(f"  STL格式: {'✓' if self.params['export_stl'] else '✗'} (仅几何)\n")
                f.write(f"  GLB格式: {'✓' if self.params['export_glb'] else '✗'} (带颜色和纹理)\n")

                f.write("\n📋 使用说明:\n")
                f.write("1. PLY文件: 在MeshLab、Blender、3D Viewer中打开查看颜色\n")
                f.write("2. OBJ文件: 在支持顶点颜色的软件中查看颜色\n")
                f.write("3. STL文件: 仅包含几何，适用于3D打印\n")
                f.write("4. 所有文件保存在 'final_models' 文件夹中\n")

            print(f"📄 重建报告已保存: {report_path}")

        except Exception as e:
            print(f"⚠️ 生成报告失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("奥比中光Astra相机彩色3D重建系统")
    print("基于独立SDK接口层 - 解决0xC0000374堆损坏问题")
    print("专门导出带颜色的3D模型 - PLY/OBJ/STL格式")
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
        print(f"⚠️ 默认SDK路径不存在，尝试自动查找...")
        sdk_path = ""

    # 创建重建器
    reconstructor = Color3DReconstructor(sdk_path)

    # 运行重建
    print("\n🚀 开始彩色3D重建...")
    print("🎯 目标: 导出带颜色的3D模型")
    print("=" * 70)

    try:
        success = reconstructor.run_color_reconstruction()

        if success:
            print("\n🎉 彩色3D重建成功！")
            print("💡 所有带颜色的模型已保存在输出目录中")

            # 显示重要文件
            print("\n📁 重要文件:")
            print("  final_models/       - 最终彩色模型")
            print("  intermediate_models/ - 中间模型")
            print("  captured_images/    - 采集的图像")
            print("  previews/          - 模型预览图")
            print("  彩色重建报告.txt   - 详细报告")

        else:
            print("\n⚠️ 重建过程中出现问题")
            print("💡 部分数据已保存，请检查输出目录")

    except Exception as e:
        print(f"\n❌ 运行重建时出错: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("📋 提示:")
    print("1. PLY格式在MeshLab中打开可查看颜色")
    print("2. 确保重建对象光照充足")
    print("3. 相机与对象距离建议0.3-2米")
    print("4. 此版本使用独立的SDK接口层，已解决堆损坏问题")
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