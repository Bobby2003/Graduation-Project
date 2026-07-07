import cv2
import numpy as np
import time
from primesense import openni2


class AstraDepthIRCamera:
    def __init__(self, driver_path):
        self.driver_path = driver_path
        self.dev = None
        self.depth_stream = None
        self.ir_stream = None
        self.is_running = False

    def initialize(self):
        """初始化相机"""
        try:
            openni2.initialize(self.driver_path)
            self.dev = openni2.Device.open_any()
            print(f"相机型号: {self.dev.get_device_info().name}")

            # 创建深度流
            self.depth_stream = self.dev.create_depth_stream()
            self.depth_stream.start()

            # 创建红外流
            self.ir_stream = self.dev.create_ir_stream()
            self.ir_stream.start()

            self.is_running = True
            print("深度+红外相机初始化成功")
            return True

        except Exception as e:
            print(f"相机初始化失败: {e}")
            return False

    def get_frames(self):
        """获取一帧深度和红外图像"""
        if not self.is_running:
            return None, None

        try:
            # 读取深度帧
            depth_frame = self.depth_stream.read_frame()
            depth_data = depth_frame.get_buffer_as_uint16()
            depth_image = np.frombuffer(depth_data, dtype=np.uint16).reshape(480, 640)

            # 读取红外帧
            ir_frame = self.ir_stream.read_frame()
            ir_data = ir_frame.get_buffer_as_uint16()
            ir_image = np.frombuffer(ir_data, dtype=np.uint16).reshape(480, 640)

            return depth_image, ir_image

        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None

    def process_frames(self, depth_image, ir_image):
        """处理深度和红外图像"""
        # 将深度图像转换为可视化的伪彩色
        depth_vis = self.depth_to_colormap(depth_image)

        # 将红外图像转换为8位并增强对比度
        ir_normalized = self.normalize_ir_image(ir_image)
        ir_colored = cv2.applyColorMap(ir_normalized, cv2.COLORMAP_JET)

        return depth_vis, ir_colored, ir_normalized

    def depth_to_colormap(self, depth_image):
        """深度图转伪彩色图"""
        # 移除无效深度值（0通常表示无效测量）
        depth_valid = depth_image.copy()
        depth_valid[depth_valid == 0] = depth_valid[depth_valid > 0].min() if np.any(depth_valid > 0) else 1

        # 归一化到0-255
        depth_normalized = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 将无效深度标记为黑色
        depth_colored[depth_image == 0] = [0, 0, 0]

        return depth_colored

    def normalize_ir_image(self, ir_image):
        """归一化红外图像"""
        # 移除极端值
        ir_clipped = np.clip(ir_image, np.percentile(ir_image, 5), np.percentile(ir_image, 95))

        # 归一化到0-255
        ir_normalized = cv2.normalize(ir_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return ir_normalized

    def calculate_point_cloud(self, depth_image, intrinsic_matrix=None):
        """从深度图计算点云（简化版）"""
        if intrinsic_matrix is None:
            # Astra相机的近似内参（需要实际标定）
            fx, fy = 525.0, 525.0  # 焦距
            cx, cy = 320.0, 240.0  # 主点

            h, w = depth_image.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # 将深度从毫米转换为米
            z = depth_image.astype(float) / 1000.0

            # 计算3D坐标
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # 创建点云
            points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

            # 移除无效点（深度为0）
            valid_mask = depth_image.flatten() > 0
            points = points[valid_mask]

            return points
        else:
            # 使用提供的相机内参
            pass

    def shutdown(self):
        """关闭相机"""
        if self.depth_stream:
            self.depth_stream.stop()
        if self.ir_stream:
            self.ir_stream.stop()
        openni2.unload()
        self.is_running = False
        print("相机已关闭")


# ================ 使用示例 ================
if __name__ == "__main__":
    DRIVER_PATH = r"C:\Users\Bobby2003\Desktop\相机驱动\奥比中光Win64-Release\sdk\libs"

    # 创建相机对象
    camera = AstraDepthIRCamera(DRIVER_PATH)

    if camera.initialize():
        print("\n按 'q' 退出，'s' 保存当前帧")

        frame_count = 0
        while True:
            # 获取原始帧
            depth_raw, ir_raw = camera.get_frames()

            if depth_raw is None or ir_raw is None:
                print("无法获取帧，退出...")
                break

            # 处理帧
            depth_vis, ir_colored, ir_gray = camera.process_frames(depth_raw, ir_raw)

            # 显示结果
            cv2.imshow('Depth (Colored)', depth_vis)
            cv2.imshow('IR (Colored)', ir_colored)
            cv2.imshow('IR (Gray)', ir_gray)

            # 显示叠加效果
            overlay = cv2.addWeighted(depth_vis, 0.5, ir_colored, 0.5, 0)
            cv2.imshow('Depth+IR Overlay', overlay)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                cv2.imwrite(f'depth_frame_{frame_count}.png', depth_vis)
                cv2.imwrite(f'ir_frame_{frame_count}.png', ir_colored)
                cv2.imwrite(f'ir_gray_{frame_count}.png', ir_gray)
                print(f"保存帧 {frame_count}")
                frame_count += 1

            # 显示帧率
            time.sleep(0.03)  # 约30FPS

        # 关闭相机
        camera.shutdown()
        cv2.destroyAllWindows()
    else:
        print("相机初始化失败")