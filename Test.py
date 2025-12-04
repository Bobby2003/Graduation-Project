"""
R200摄像头 - DirectShow强制独占访问方案
使用DSHOW后端并保持摄像头独占，避免NVIDIA干扰
"""

import cv2
import numpy as np
import time
import threading
from datetime import datetime

class R200_DirectShow_Capture:
    """
    专门针对R200的DirectShow捕获方案
    强制使用DSHOW后端并保持摄像头独占
    """

    def __init__(self):
        self.cap = None
        self.camera_index = 19  # 根据测试，索引0有效
        self.is_capturing = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

    def initialize_camera(self):
        """初始化摄像头 - 强制使用DirectShow"""
        print("初始化R200摄像头（强制DirectShow）...")

        # 关键：使用cv2.CAP_DSHOW并设置独占模式
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            print("❌ 无法打开摄像头")
            return False

        print("✅ 摄像头已打开")

        # 尝试设置独占访问模式（某些摄像头支持）
        try:
            # 尝试设置更高的优先级
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # 关闭自动对焦
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 关闭自动曝光
        except:
            pass

        # 获取当前配置
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"摄像头配置:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.1f} FPS")

        return True

    def start_capture(self):
        """开始捕获线程"""
        if not self.cap:
            print("摄像头未初始化")
            return False

        self.is_capturing = True

        # 创建捕获线程
        self.capture_thread = threading.Thread(target=self._capture_worker)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        print("开始摄像头捕获...")
        return True

    def _capture_worker(self):
        """捕获工作线程"""
        frame_count = 0
        error_count = 0

        while self.is_capturing and error_count < 10:
            try:
                ret, frame = self.cap.read()

                if ret:
                    with self.frame_lock:
                        self.current_frame = frame.copy()

                    frame_count += 1
                    error_count = 0

                    if frame_count % 30 == 0:
                        print(f"捕获帧数: {frame_count}")
                else:
                    error_count += 1
                    print(f"读取帧失败 ({error_count}/10)")

                    # 尝试重新初始化
                    if error_count > 5:
                        self._reinitialize_camera()

                # 控制帧率
                time.sleep(0.01)

            except Exception as e:
                error_count += 1
                print(f"捕获错误: {e}")

        if error_count >= 10:
            print("❌ 捕获线程因多次错误停止")

    def _reinitialize_camera(self):
        """重新初始化摄像头"""
        print("尝试重新初始化摄像头...")

        if self.cap:
            self.cap.release()
            time.sleep(0.5)

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        time.sleep(0.5)

    def get_frame(self):
        """获取当前帧"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def display_stream(self, duration_seconds=30):
        """显示视频流"""
        print("\n" + "=" * 60)
        print("开始显示R200视频流")
        print("按 's' 保存当前帧，按 'q' 退出")
        print("=" * 60)

        if not self.start_capture():
            return

        start_time = time.time()
        frame_count = 0
        save_count = 0

        try:
            while time.time() - start_time < duration_seconds:
                frame = self.get_frame()

                if frame is not None:
                    frame_count += 1

                    # 显示帧
                    display = self._prepare_display_frame(frame, frame_count)
                    cv2.imshow('R200摄像头 - DirectShow独占模式', display)

                    # 按键处理
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户退出")
                        break
                    elif key == ord('s'):
                        self._save_frame(frame, save_count)
                        save_count += 1
                else:
                    # 没有帧，短暂等待
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("捕获被中断")

        finally:
            self.stop_capture()

            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0

            print(f"\n捕获统计:")
            print(f"  总帧数: {frame_count}")
            print(f"  保存帧数: {save_count}")
            print(f"  运行时间: {elapsed:.1f}秒")
            print(f"  实际FPS: {actual_fps:.1f}")

    def _prepare_display_frame(self, frame, frame_count):
        """准备显示帧"""
        h, w = frame.shape[:2]

        # 调整显示大小
        if w > 800 or h > 600:
            display_w = 800
            display_h = int(800 * h / w)
            if display_h > 600:
                display_h = 600
                display_w = int(600 * w / h)
        else:
            display_w, display_h = w, h

        display_frame = cv2.resize(frame, (display_w, display_h))

        # 添加信息
        info_text = f"R200: {w}x{h}"
        cv2.putText(display_frame, info_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps_text = f"帧: {frame_count}"
        cv2.putText(display_frame, fps_text,
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        control_text = "按 's' 保存, 'q' 退出"
        cv2.putText(display_frame, control_text,
                   (10, display_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        return display_frame

    def _save_frame(self, frame, count):
        """保存帧"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"r200_frame_{timestamp}_{count:03d}.png"

        cv2.imwrite(filename, frame)
        print(f"✅ 保存: {filename}")

    def stop_capture(self):
        """停止捕获"""
        self.is_capturing = False

        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()
        print("摄像头已释放")

# ==================== 3D重建部分 ====================

def estimate_depth_from_ir(ir_frame):
    """
    从IR图像估计深度（简化版）
    注意：这是基于边缘的简化方法，精度有限
    """
    # 转换为灰度
    if len(ir_frame.shape) == 3:
        gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = ir_frame

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 距离变换
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

    # 归一化为深度图
    depth = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)

    return depth

def create_pointcloud(ir_frame, depth_map):
    """
    从IR图像和深度图创建点云
    """
    import open3d as o3d

    height, width = ir_frame.shape[:2]

    # 创建网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 深度转换为距离（假设0-255对应0.5-3.5米）
    depth_normalized = depth_map.astype(np.float32) / 255.0
    z = 0.5 + depth_normalized * 3.0

    # 计算3D坐标（使用近似焦距）
    fx, fy = 300.0, 300.0  # 近似焦距
    cx, cy = width / 2, height / 2

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 重塑为点云
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # 颜色
    if len(ir_frame.shape) == 3:
        colors = ir_frame.reshape(-1, 3) / 255.0
    else:
        colors = np.stack([ir_frame, ir_frame, ir_frame], axis=-1).reshape(-1, 3) / 255.0

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 下采样
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    return pcd

def main():
    """主函数"""
    print("=" * 60)
    print("R200 DirectShow独占访问与3D重建")
    print("=" * 60)

    # 创建捕获器
    capture = R200_DirectShow_Capture()

    if not capture.initialize_camera():
        print("❌ 摄像头初始化失败")
        return

    try:
        # 选项：选择运行模式
        print("\n选择运行模式:")
        print("  1. 实时视频流显示")
        print("  2. 3D点云重建（实时）")
        print("  3. 采集数据供后期处理")

        choice = input("请输入选择 (1/2/3): ").strip()

        if choice == "1":
            # 模式1：实时视频流
            capture.display_stream(duration_seconds=30)

        elif choice == "2":
            # 模式2：3D点云重建
            print("\n3D点云重建模式")
            print("按 's' 生成并保存点云，按 'q' 退出")

            capture.start_capture()

            pointcloud_count = 0
            last_pointcloud_time = 0

            try:
                while True:
                    frame = capture.get_frame()

                    if frame is not None:
                        # 显示视频流
                        display = capture._prepare_display_frame(frame, 0)
                        cv2.imshow('R200 IR流', display)

                        # 按键处理
                        key = cv2.waitKey(1) & 0xFF

                        if key == ord('q'):
                            break
                        elif key == ord('s'):
                            current_time = time.time()
                            if current_time - last_pointcloud_time > 2.0:
                                print("生成点云中...")

                                # 估计深度
                                depth_map = estimate_depth_from_ir(frame)

                                # 显示深度图
                                depth_display = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                                cv2.imshow('估计深度图', depth_display)

                                # 生成点云
                                pcd = create_pointcloud(frame, depth_map)

                                # 保存点云
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"r200_pointcloud_{timestamp}.ply"

                                import open3d as o3d
                                o3d.io.write_point_cloud(filename, pcd)

                                print(f"✅ 点云已保存: {filename}")

                                # 显示点云
                                o3d.visualization.draw_geometries(
                                    [pcd],
                                    window_name=f"R200点云 - {len(pcd.points)}点",
                                    width=800,
                                    height=600
                                )

                                last_pointcloud_time = current_time
                                pointcloud_count += 1

            finally:
                capture.stop_capture()

        elif choice == "3":
            # 模式3：数据采集
            print("\n数据采集模式")
            print("将采集IR图像和估计的深度图")

            capture.start_capture()

            save_count = 0
            try:
                while save_count < 50:  # 采集50帧
                    frame = capture.get_frame()

                    if frame is not None:
                        # 每5帧保存一次
                        if save_count % 5 == 0:
                            # 保存IR图像
                            ir_filename = f"r200_ir_{save_count:04d}.png"
                            cv2.imwrite(ir_filename, frame)

                            # 估计并保存深度图
                            depth_map = estimate_depth_from_ir(frame)
                            depth_filename = f"r200_depth_{save_count:04d}.png"
                            cv2.imwrite(depth_filename, depth_map)

                            print(f"保存: {ir_filename}, {depth_filename}")

                            # 显示
                            display = capture._prepare_display_frame(frame, save_count)
                            cv2.imshow('数据采集', display)

                            if cv2.waitKey(100) & 0xFF == ord('q'):
                                break

                        save_count += 1

                    time.sleep(0.05)

            finally:
                capture.stop_capture()
                print(f"\n采集完成: {save_count}帧")

        else:
            print("无效选择，运行默认模式...")
            capture.display_stream(duration_seconds=15)

    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        capture.stop_capture()
        print("\n程序结束")

if __name__ == "__main__":
    main()