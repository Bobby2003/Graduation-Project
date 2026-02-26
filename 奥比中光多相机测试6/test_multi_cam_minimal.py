"""
三相机最小化测试程序 - 诊断SDK多相机阻塞问题
"""

import sys
import os
import time

# 导入独立的SDK接口层
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

    SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SDK导入失败: {e}")
    SDK_AVAILABLE = False


def test_multi_cam_capture():
    """最小化测试：只测试三相机能否依次成功采集一帧"""
    if not SDK_AVAILABLE:
        print("❌ SDK不可用")
        return

    sdk_path = get_default_sdk_path()
    cameras = []

    print("=" * 60)
    print("三相机最小化采集测试")
    print("=" * 60)

    # 第一阶段：逐一初始化并启动相机（串行，间隔加大）
    for i in range(3):  # 测试三个相机
        print(f"\n[1/3] 正在初始化相机 {i}...")
        cam = OrbbecCameraSDK(sdk_path)

        if not cam.initialize():
            print(f"  ❌ 相机 {i} SDK初始化失败")
            continue

        # 尝试打开第i个设备
        if not cam.open_device(device_index=i):
            print(f"  ❌ 相机 {i} 打开设备失败")
            cam.cleanup()
            continue

        # 创建深度流
        if not cam.create_stream(cam.ONI_SENSOR_DEPTH):
            print(f"  ❌ 相机 {i} 创建深度流失败")
            cam.close_device()
            cam.cleanup()
            continue

        # 启动流
        if not cam.start_stream():
            print(f"  ❌ 相机 {i} 启动流失败")
            cam.destroy_stream()
            cam.close_device()
            cam.cleanup()
            continue

        cameras.append(cam)
        print(f"  ✅ 相机 {i} 就绪")

        # 关键：启动后等待更长时间，让设备稳定
        time.sleep(1.0)  # 增加等待时间到1秒

    if len(cameras) < 3:
        print(f"\n⚠️ 警告：只成功初始化了 {len(cameras)} 个相机，继续测试...")

    # 第二阶段：逐一尝试采集（使用更长的超时）
    print("\n[2/3] 开始逐一采集测试...")
    for i, cam in enumerate(cameras):
        print(f"\n  尝试采集相机 {i}...")
        try:
            # 使用更长的超时，并捕获可能的异常
            result = cam.capture_depth_frame(timeout=5000)  # 5秒超时
            if result:
                depth_array, frame_info = result
                print(f"  ✅ 相机 {i} 采集成功")
                print(f"     深度图尺寸: {depth_array.shape}")
                print(f"     有效点数量: {np.sum(depth_array > 0)}")
            else:
                print(f"  ❌ 相机 {i} 采集返回空结果")
        except Exception as e:
            print(f"  ❌ 相机 {i} 采集异常: {e}")

        # 采集间隔
        time.sleep(0.5)

    # 第三阶段：尝试同步采集（模拟真实使用场景）
    print("\n[3/3] 尝试快速连续采集（模拟重建场景）...")
    for frame_num in range(3):  # 只采集3帧测试
        print(f"\n  模拟第 {frame_num + 1} 帧采集:")
        for i, cam in enumerate(cameras):
            start_time = time.time()
            try:
                result = cam.capture_depth_frame(timeout=3000)
                elapsed = time.time() - start_time
                if result:
                    print(f"    相机 {i}: ✅ 成功 ({elapsed:.2f}秒)")
                else:
                    print(f"    相机 {i}: ❌ 失败 ({elapsed:.2f}秒)")
            except Exception as e:
                print(f"    相机 {i}: ❌ 异常 ({str(e)[:50]})")
        time.sleep(0.1)  # 帧间隔

    # 清理
    print("\n[4/4] 清理资源...")
    for i, cam in enumerate(cameras):
        try:
            cam.cleanup()
            print(f"  ✅ 相机 {i} 已清理")
        except:
            print(f"  ⚠️ 相机 {i} 清理异常")

    print("\n" + "=" * 60)
    print("最小化测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 这里需要导入numpy用于显示形状（仅测试用）
    import numpy as np

    test_multi_cam_capture()