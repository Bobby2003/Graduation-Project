"""
三相机深度3D重建系统 - 电压优化版本
基于两两交替工作，解决电压不足问题
"""

import os
import sys
import time
import traceback
import cv2
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MultiCameraConfig
from voltage_optimized_camera import VoltageOptimizedSystem
from utils import MultiCameraUtils


def print_header():
    """打印程序标题"""
    print("=" * 80)
    print("三相机深度3D重建系统 - 电压优化版")
    print("模式: 两两交替工作 (解决电压不足问题)")
    print("=" * 80)
    print()


def main_menu():
    """主菜单"""
    print_header()

    print("\n请选择功能:")
    print("1. 电压优化相机视图 (两两交替)")
    print("2. 测试电压优化采集")
    print("3. 采集数据用于3D重建")
    print("4. 退出")
    print()

    choice = input("请输入选项 (1-4): ").strip()

    if choice == "1":
        run_voltage_optimized_view()
    elif choice == "2":
        run_voltage_test()
    elif choice == "3":
        run_data_capture()
    elif choice == "4":
        print("退出程序")
        return
    else:
        print("无效选项")

    input("\n按Enter键返回主菜单...")
    main_menu()


def run_voltage_optimized_view():
    """运行电压优化视图"""
    print("\n" + "=" * 60)
    print("电压优化相机视图")
    print("=" * 60)

    system = VoltageOptimizedSystem()

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
    print("电压优化测试")
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
            frames = system.stats['frames_captured'][i]
            activations = system.stats['activations'][i]
            print(f"  相机{i}: {activations}次激活, {frames}帧, {fps:.1f} FPS")

    except KeyboardInterrupt:
        print("\n测试中断")
    except Exception as e:
        print(f"测试错误: {e}")
        traceback.print_exc()
    finally:
        system.cleanup()


def run_data_capture():
    """采集数据用于3D重建"""
    print("\n" + "=" * 60)
    print("数据采集模式 (用于3D重建)")
    print("=" * 60)

    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"capture_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 子目录
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

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
        frame_count = 0
        save_interval = 5  # 每5帧保存一次

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
                    for cam_id, (color, depth) in frames.items():
                        if color is not None:
                            color_file = os.path.join(color_dir,
                                                      f"cam{cam_id}_f{frame_count:04d}.jpg")
                            cv2.imwrite(color_file, color)

                        if depth is not None:
                            depth_file = os.path.join(depth_dir,
                                                      f"cam{cam_id}_f{frame_count:04d}.npy")
                            np.save(depth_file, depth)

                    print(f"已保存帧 {frame_count}")

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

        # 保存配置文件
        config_file = os.path.join(output_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write(f"采集时间: {timestamp}\n")
            f.write(f"采集时长: {stats['elapsed_time']:.1f}秒\n")
            f.write(f"总帧数: {stats['total_frames']}\n")
            f.write(f"平均FPS: {stats['average_fps']:.1f}\n")
            f.write(f"轮换次数: {stats['rotation_count']}\n")
            f.write(f"相机内参:\n")
            f.write(str(MultiCameraConfig.get_intrinsic_matrix()) + "\n")

        print(f"\n数据已保存到: {output_dir}")
        print(f"  彩色图像: {len(os.listdir(color_dir))}张")
        print(f"  深度图像: {len(os.listdir(depth_dir))}张")
        print(f"  配置文件: config.txt")

    except KeyboardInterrupt:
        print("\n采集中断")
    except Exception as e:
        print(f"采集错误: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        system.cleanup()

    print("\n下一步: 使用采集的数据进行3D重建")


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