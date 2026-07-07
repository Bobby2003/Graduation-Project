"""
三相机深度3D重建系统 - 新SDK版本
专注于核心功能：电压优化采集和3D重建
"""

import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any  # 添加了 Any 导入

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import open3d as o3d

from config import MultiCameraConfig
from voltage_safe_camera import VoltageOptimizedSystem
from utils import MultiCameraUtils


def print_header():
    """打印程序标题"""
    print("=" * 80)
    print("三相机深度3D重建系统 - 新SDK版本")
    print("针对奥比中光Astra相机电压不足问题")
    print("=" * 80)
    print()


def check_dependencies():
    """检查依赖"""
    print("检查依赖...")

    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV")
        return False

    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy")
        return False

    try:
        import open3d as o3d
        print(f"✅ Open3D: {o3d.__version__}")
    except ImportError:
        print("❌ Open3D")
        return False

    # 检查新的SDK接口
    try:
        from orbbec_sdk import OrbbecCameraSDK
        print("✅ Orbbec SDK接口")
        return True
    except ImportError:
        print("⚠️ Orbbec SDK接口未找到，请确保orbbec_sdk.py在正确位置")
        return False


def main_menu():
    """主菜单"""
    print_header()

    if not check_dependencies():
        print("\n请先安装缺少的依赖包")
        return

    print("\n请选择功能:")
    print("1. 电压优化相机视图 (两两交替)")
    print("2. 测试电压优化采集")
    print("3. 采集数据并3D重建")
    print("4. 可视化点云")
    print("5. 测试新SDK接口")
    print("6. 退出")
    print()

    choice = input("请输入选项 (1-6): ").strip()

    if choice == "1":
        run_voltage_optimized_view()
    elif choice == "2":
        run_voltage_test()
    elif choice == "3":
        run_capture_and_reconstruct()
    elif choice == "4":
        run_pointcloud_visualization()
    elif choice == "5":
        test_new_sdk()
    elif choice == "6":
        print("退出程序")
        return
    else:
        print("无效选项")

    input("\n按Enter键返回主菜单...")
    main_menu()


def test_new_sdk():
    """测试新的SDK接口"""
    print("\n" + "=" * 60)
    print("测试新的SDK接口")
    print("=" * 60)

    try:
        from orbbec_sdk import OrbbecCameraSDK

        # 创建SDK实例
        sdk = OrbbecCameraSDK()

        print("1. 初始化SDK...")
        if not sdk.initialize():
            print("❌ SDK初始化失败")
            return

        print("2. 获取设备列表...")
        devices = sdk.get_device_list()
        if not devices:
            print("❌ 未找到设备")
            return

        print(f"找到 {len(devices)} 个设备:")
        for idx, info in devices.items():
            print(f"  设备 {idx}: {info['name']} (URI: {info['uri']})")

        if len(devices) > 0:
            print("\n3. 打开第一个设备...")
            if not sdk.open_device(0):
                print("❌ 打开设备失败")
                return

            print("4. 创建深度流...")
            if not sdk.create_stream(sdk.ONI_SENSOR_DEPTH):
                print("❌ 创建深度流失败")
                return

            print("5. 启动深度流...")
            if not sdk.start_stream():
                print("❌ 启动深度流失败")
                return

            print("\n6. 测试采集5帧深度数据...")
            for i in range(5):
                depth_data, frame_info = sdk.capture_depth_frame(timeout=1000)
                if depth_data is not None:
                    print(f"  帧 {i + 1}: {depth_data.shape}, 最大值: {np.max(depth_data)} mm")
                else:
                    print(f"  帧 {i + 1}: 采集失败")
                time.sleep(0.1)

            print("\n7. 清理资源...")
            sdk.cleanup()

        print("✅ 新SDK接口测试完成")

    except Exception as e:
        print(f"❌ 测试新SDK接口失败: {e}")
        traceback.print_exc()


def run_voltage_optimized_view():
    """运行电压优化视图"""
    print("\n" + "=" * 60)
    print("电压优化相机视图 - 新SDK版本")
    print("=" * 60)

    # 指定SDK路径（如果需要）
    sdk_path = None  # 自动查找
    # sdk_path = "C:/OpenNI2/Redist/OpenNI2.dll"  # 或手动指定

    system = VoltageOptimizedSystem(sdk_path)

    if not system.initialize():
        print("❌ 系统初始化失败")
        return

    try:
        system.visualize_alternating_view()
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
    finally:
        system.cleanup()


def run_voltage_test():
    """运行电压测试"""
    print("\n" + "=" * 60)
    print("电压优化测试 - 新SDK版本")
    print("=" * 60)

    system = VoltageOptimizedSystem()

    if not system.initialize():
        print("❌ 系统初始化失败")
        return

    print("\n开始测试，采集20秒...")
    print("按Ctrl+C中断")

    try:
        system.start_capture()

        start_time = time.time()
        last_print_time = start_time

        while time.time() - start_time < 20:
            current_time = time.time()

            # 每秒打印一次统计
            if current_time - last_print_time >= 1.0:
                stats = system.get_statistics()
                print(f"  时间: {current_time - start_time:.1f}s, "
                      f"总帧: {stats['total_frames']}, "
                      f"FPS: {stats['average_fps']:.1f}, "
                      f"激活组合: {stats['active_combo']}")
                last_print_time = current_time

            time.sleep(0.1)

        system.stop_capture()

        # 最终统计
        stats = system.get_statistics()
        print("\n测试完成:")
        print(f"  总时间: {stats['elapsed_time']:.1f}秒")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  平均FPS: {stats['average_fps']:.1f}")
        print(f"  轮换次数: {stats['rotation_count']}")

        # 每个相机的统计
        for i in range(MultiCameraConfig.NUM_CAMERAS):
            fps = stats['fps_per_camera'][i]
            frames = stats['frames_per_camera'][i]
            activations = stats['activations'][i]
            print(f"  相机{i}: {activations}次激活, {frames}帧, {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n测试中断")
    except Exception as e:
        print(f"测试错误: {e}")
        traceback.print_exc()
    finally:
        system.cleanup()


def run_capture_and_reconstruct():
    """采集数据并3D重建"""
    print("\n" + "=" * 60)
    print("采集数据并3D重建 - 新SDK版本")
    print("=" * 60)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"reconstruction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 初始化系统
    system = VoltageOptimizedSystem()

    if not system.initialize():
        print("❌ 系统初始化失败")
        return

    print("\n开始采集，目标: 30秒")
    print("按ESC键停止")

    cv2.namedWindow("数据采集", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("数据采集", 800, 600)

    try:
        system.start_capture()

        start_time = time.time()
        captured_data = []
        frame_count = 0
        save_interval = 5  # 每5帧保存一次点云

        while time.time() - start_time < 30:
            # 获取帧
            frames, timestamp = system.get_frames(timeout=0.1)

            if frames:
                # 显示
                displays = []
                for cam_id, (color, depth) in frames.items():
                    if color is not None:
                        display = cv2.resize(color, (320, 240))
                        cv2.putText(display, f"Cam{cam_id}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        displays.append(display)

                if displays:
                    if len(displays) == 2:
                        combined = np.hstack(displays)
                        cv2.imshow("数据采集", combined)

                # 保存数据
                frame_count += 1
                if frame_count % save_interval == 0:
                    # 获取点云
                    pointclouds = system.get_pointclouds()
                    if pointclouds:
                        captured_data.append(pointclouds)
                        print(f"已保存第 {frame_count} 帧点云数据")

            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("用户停止")
                break

        # 最终统计
        system.stop_capture()
        stats = system.get_statistics()

        print(f"\n采集完成:")
        print(f"  总时间: {stats['elapsed_time']:.1f}秒")
        print(f"  总帧数: {stats['total_frames']}")
        print(f"  平均FPS: {stats['average_fps']:.1f}")
        print(f"  采集到 {len(captured_data)} 组点云数据")

        # 3D重建
        if captured_data:
            print("\n开始3D重建...")
            reconstructed_pcd = reconstruct_3d_model(captured_data, output_dir)

            if reconstructed_pcd and len(reconstructed_pcd.points) > 0:
                print(f"\n✅ 3D重建成功!")
                print(f"  点云点数: {len(reconstructed_pcd.points)}")

                # 保存点云
                pcd_file = os.path.join(output_dir, "reconstructed_model.ply")
                MultiCameraUtils.save_pointcloud(reconstructed_pcd, pcd_file)

                # 可视化
                print("可视化重建结果...")
                o3d.visualization.draw_geometries(
                    [reconstructed_pcd],
                    window_name="重建的3D模型",
                    width=1024,
                    height=768,
                    left=50,
                    top=50
                )
            else:
                print("❌ 3D重建失败")
        else:
            print("❌ 未采集到有效数据")

    except KeyboardInterrupt:
        print("\n采集中断")
    except Exception as e:
        print(f"采集错误: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        system.cleanup()

    print(f"\n重建结果保存在: {output_dir}")


def reconstruct_3d_model(captured_data: List[Dict[int, Any]],
                         output_dir: str) -> Optional[o3d.geometry.PointCloud]:
    """3D重建模型"""
    if not captured_data:
        print("❌ 没有采集到数据")
        return None

    try:
        print("合并点云数据...")

        # 合并所有点云
        all_points = []
        all_colors = []

        for frame_data in captured_data:
            for cam_id, pcd in frame_data.items():
                if pcd and len(pcd.points) > 0:
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                    all_points.append(points)
                    if colors is not None:
                        all_colors.append(colors)

        if not all_points:
            print("❌ 无有效点云数据")
            return None

        # 合并所有点
        merged_points = np.vstack(all_points)

        # 合并颜色（如果有）
        merged_colors = None
        if all_colors and len(all_colors) > 0:
            merged_colors = np.vstack(all_colors)

        # 创建点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

        if merged_colors is not None:
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        print(f"✅ 合并完成，原始点数: {len(merged_points)}")

        # 下采样
        print("下采样点云...")
        voxel_size = MultiCameraConfig.RECONSTRUCTION_CONFIG['voxel_size']
        downsampled_pcd = merged_pcd.voxel_down_sample(voxel_size)
        print(f"✅ 下采样完成，点数: {len(downsampled_pcd.points)}")

        # 移除离群点
        print("移除离群点...")
        cl, ind = downsampled_pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
        filtered_pcd = downsampled_pcd.select_by_index(ind)
        print(f"✅ 离群点移除完成，点数: {len(filtered_pcd.points)}")

        if len(filtered_pcd.points) == 0:
            print("❌ 移除离群点后点云为空")
            return None

        # 法线估计
        print("估计法线...")
        filtered_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1,
                max_nn=30
            )
        )
        filtered_pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0, 0, 0])
        )
        print("✅ 法线估计完成")

        # 保存中间结果
        intermediate_pcd = filtered_pcd
        intermediate_file = os.path.join(output_dir, "intermediate_pointcloud.ply")
        o3d.io.write_point_cloud(intermediate_file, intermediate_pcd)
        print(f"中间结果已保存: {intermediate_file}")

        return filtered_pcd

    except Exception as e:
        print(f"❌ 3D重建错误: {e}")
        traceback.print_exc()
        return None


def run_pointcloud_visualization():
    """可视化点云"""
    print("\n" + "=" * 60)
    print("可视化点云")
    print("=" * 60)

    # 检查是否有已保存的点云文件
    if not os.path.exists("outputs"):
        print("❌ 没有找到输出目录")
        return

    # 查找最新的点云文件
    reconstruction_dirs = []
    for dir_name in os.listdir("outputs"):
        if dir_name.startswith("reconstruction_"):
            reconstruction_dirs.append(dir_name)

    if not reconstruction_dirs:
        print("❌ 没有找到重建数据")
        return

    # 按时间排序
    reconstruction_dirs.sort(reverse=True)
    latest_dir = os.path.join("outputs", reconstruction_dirs[0])

    print(f"使用最新的重建目录: {latest_dir}")

    # 查找点云文件
    pcd_files = []
    for file_name in os.listdir(latest_dir):
        if file_name.endswith(".ply") or file_name.endswith(".pcd"):
            pcd_files.append(os.path.join(latest_dir, file_name))

    if not pcd_files:
        print("❌ 没有找到点云文件")
        return

    print(f"找到 {len(pcd_files)} 个点云文件")

    # 加载并显示点云
    for pcd_file in pcd_files:
        try:
            print(f"加载点云: {os.path.basename(pcd_file)}")
            pcd = o3d.io.read_point_cloud(pcd_file)

            if pcd and len(pcd.points) > 0:
                print(f"  点数: {len(pcd.points)}")

                # 可视化
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name=f"点云: {os.path.basename(pcd_file)}",
                    width=1024,
                    height=768,
                    left=50,
                    top=50
                )
            else:
                print(f"  点云为空或无效")

        except Exception as e:
            print(f"加载点云失败: {e}")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        traceback.print_exc()

    print("\n程序结束")
    input("按Enter键退出...")