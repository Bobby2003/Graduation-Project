"""
R200摄像头智能识别与选择工具
结合PyCameraList解决摄像头索引问题
"""

import cv2
import time
from PyCameraList.camera_device import list_video_devices, list_audio_devices
import numpy as np
import sys

class SmartCameraSelector:
    def __init__(self):
        self.camera_list = []
        self.r200_cameras = []

    def scan_cameras(self):
        """扫描所有摄像头"""
        print("="*60)
        print("扫描系统中的所有摄像头")
        print("="*60)

        try:
            # 使用PyCameraList获取摄像头列表
            cameras = list_video_devices()
            self.camera_list = dict(cameras)

            if not self.camera_list:
                print("❌ 未检测到任何摄像头")
                return False

            print(f"✅ 找到 {len(self.camera_list)} 个摄像头设备:")
            print("-"*50)

            for idx, name in self.camera_list.items():
                status = "✅ 可用"

                # 测试摄像头是否真正可用
                if not self.test_camera_availability(idx):
                    status = "⚠️ 可能不可用"

                # 检查是否是R200
                is_r200 = self.is_r200_camera(name)
                r200_mark = "🎯 R200" if is_r200 else ""

                print(f"索引 {idx:2d}: {name}")
                print(f"      状态: {status} {r200_mark}")
                print()

                if is_r200:
                    self.r200_cameras.append({
                        'index': idx,
                        'name': name,
                        'mode': self.detect_r200_mode(name)
                    })

            return True

        except Exception as e:
            print(f"扫描摄像头时出错: {e}")
            # 如果PyCameraList失败，使用备用方法
            return self.scan_cameras_backup()

    def scan_cameras_backup(self):
        """备用扫描方法（不使用PyCameraList）"""
        print("使用备用方法扫描摄像头...")

        found_cameras = {}

        # 扫描索引0-20
        for idx in range(21):
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                # 尝试读取帧
                for attempt in range(3):
                    ret, frame = cap.read()
                    if ret:
                        # 获取可能的设备信息
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # 生成描述性名称
                        name = f"Camera_{idx}_{width}x{height}"
                        if width == 640 and height == 480:
                            name = f"IR_Camera_{idx}"
                        elif width == 1920 and height == 1080:
                            name = f"RGB_Camera_{idx}"

                        found_cameras[idx] = name
                        break

                cap.release()

        self.camera_list = found_cameras
        return len(found_cameras) > 0

    def test_camera_availability(self, index, timeout=3):
        """测试摄像头是否真正可用"""
        cap = None
        try:
            # 尝试多种后端
            backends = [cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW]

            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    # 尝试读取帧
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        ret, frame = cap.read()
                        if ret:
                            cap.release()
                            return True
                        time.sleep(0.1)

                if cap:
                    cap.release()

            return False

        except:
            if cap:
                cap.release()
            return False

    def is_r200_camera(self, name):
        """判断是否是R200摄像头"""
        r200_keywords = [
            'R200', 'RealSense', 'LR200',
            'VID_8086', 'Intel', '3D Camera'
        ]

        name_lower = name.lower()
        for keyword in r200_keywords:
            if keyword.lower() in name_lower:
                return True
        return False

    def detect_r200_mode(self, name):
        """检测R200的工作模式"""
        if 'RGB' in name:
            return "RGB彩色模式"
        elif 'Depth' in name:
            return "深度模式"
        elif 'Left-Right' in name or 'Stereo' in name:
            return "双目IR模式"
        elif 'IR' in name or 'Infrared' in name:
            return "IR模式"
        else:
            return "未知模式"

    def select_camera_interactive(self):
        """交互式选择摄像头"""
        if not self.camera_list:
            print("❌ 没有可用的摄像头")
            return None

        print("\n" + "="*60)
        print("选择要使用的摄像头")
        print("="*60)

        # 如果有R200摄像头，优先显示
        if self.r200_cameras:
            print("\n🎯 检测到的R200摄像头:")
            for i, cam in enumerate(self.r200_cameras, 1):
                print(f"{i}. 索引 {cam['index']} - {cam['name']}")
                print(f"   模式: {cam['mode']}")

        print("\n📷 所有摄像头:")
        for idx, name in self.camera_list.items():
            is_r200 = self.is_r200_camera(name)
            marker = " (R200)" if is_r200 else ""
            print(f"  索引 {idx}: {name}{marker}")

        while True:
            try:
                selection = input("\n请输入摄像头索引号 (或输入 'q' 退出): ").strip()

                if selection.lower() == 'q':
                    return None

                index = int(selection)

                if index in self.camera_list:
                    return index
                else:
                    print(f"❌ 索引 {index} 无效，请重新选择")

            except ValueError:
                print("❌ 请输入有效的数字")

    def preview_camera(self, index, camera_name="", duration=30):
        """预览摄像头"""
        print(f"\n正在预览摄像头 (索引: {index})...")
        print("按 'q' 退出预览")
        print("按 's' 保存截图")
        print("按 'i' 显示信息")

        # 尝试找到最佳的后端
        backends = [
            (cv2.CAP_ANY, "AUTO"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_DSHOW, "DSHOW")
        ]

        cap = None
        backend_name = "AUTO"

        for backend, name in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # 测试是否能读取帧
                for attempt in range(5):
                    ret, frame = cap.read()
                    if ret:
                        backend_name = name
                        print(f"✅ 使用 {backend_name} 后端")
                        break

                if ret:
                    break
                else:
                    cap.release()
            else:
                if cap:
                    cap.release()

        if not cap or not cap.isOpened():
            print("❌ 无法打开摄像头")
            return False

        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        # 解码FourCC
        if fourcc > 0:
            codec = (chr(fourcc & 0xFF) +
                    chr((fourcc >> 8) & 0xFF) +
                    chr((fourcc >> 16) & 0xFF) +
                    chr((fourcc >> 24) & 0xFF))
        else:
            codec = "Unknown"

        print(f"摄像头信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.1f} FPS")
        print(f"  编码: {codec}")

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("读取帧失败")
                time.sleep(0.1)
                continue

            frame_count += 1
            elapsed = time.time() - start_time

            # 计算实时FPS
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # 显示信息
            display_frame = frame.copy()

            # 添加文字信息
            info_line1 = f"Index: {index} | {width}x{height}"
            info_line2 = f"FPS: {current_fps:.1f} | Backend: {backend_name}"

            cv2.putText(display_frame, info_line1, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, info_line2, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            if camera_name:
                cv2.putText(display_frame, camera_name[:40], (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 如果是R200，添加标记
            if self.is_r200_camera(camera_name):
                cv2.putText(display_frame, "R200 Camera", (width-200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(f"摄像头预览 - 索引 {index}", display_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存截图
                filename = f"camera_{index}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"截图已保存: {filename}")
            elif key == ord('i'):
                # 显示详细信息
                print(f"\n当前帧信息:")
                print(f"  帧大小: {frame.shape}")
                print(f"  数据类型: {frame.dtype}")
                print(f"  总帧数: {frame_count}")
                print(f"  运行时间: {elapsed:.1f}秒")

            # 检查预览时间
            if duration > 0 and elapsed >= duration:
                print(f"\n预览时间到 ({duration}秒)")
                break

        cap.release()
        cv2.destroyAllWindows()

        # 显示统计信息
        print(f"\n预览统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  平均FPS: {frame_count/elapsed:.1f}")
        print(f"  运行时间: {elapsed:.1f}秒")

        return True

def generate_camera_usage_example(camera_index, camera_name=""):
    """生成摄像头使用示例代码"""

    example_code = f'''"""
摄像头使用示例代码
摄像头索引: {camera_index}
摄像头名称: {camera_name}
"""

import cv2
import time

def main():
    # 方法1: 使用索引直接打开
    print("方法1: 使用索引直接打开")
    cap = cv2.VideoCapture({camera_index}, cv2.CAP_ANY)
    
    if not cap.isOpened():
        # 方法2: 尝试其他后端
        print("方法1失败，尝试方法2...")
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW]
        
        for backend in backends:
            cap = cv2.VideoCapture({camera_index}, backend)
            if cap.isOpened():
                print(f"使用后端: {{'MSMF' if backend == cv2.CAP_MSMF else 'DSHOW'}}")
                break
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("✅ 摄像头打开成功!")
    
    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"分辨率: {{width}}x{{height}}")
    print(f"帧率: {{fps:.1f}} FPS")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("读取帧失败")
                continue
            
            frame_count += 1
            
            # 计算实时FPS
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                print(f"\\r实时FPS: {{current_fps:.1f}}", end="")
                frame_count = 0
                start_time = time.time()
            
            # 显示图像
            cv2.imshow('摄像头预览', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\\n程序被中断")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\\n程序结束")

if __name__ == "__main__":
    main()
'''

    return example_code

def main():
    print("R200摄像头智能选择工具")
    print("使用PyCameraList解决摄像头索引问题")
    print("="*60)

    selector = SmartCameraSelector()

    # 1. 扫描摄像头
    if not selector.scan_cameras():
        print("❌ 摄像头扫描失败")
        return

    # 2. 交互式选择
    selected_index = selector.select_camera_interactive()

    if selected_index is None:
        print("操作取消")
        return

    selected_name = selector.camera_list.get(selected_index, "Unknown Camera")

    # 3. 预览摄像头
    preview = input(f"\n是否预览摄像头 '{selected_name}'？(y/n): ").strip().lower()

    if preview == 'y':
        selector.preview_camera(selected_index, selected_name, duration=20)

    # 4. 生成示例代码
    generate = input(f"\n是否生成使用示例代码？(y/n): ").strip().lower()

    if generate == 'y':
        code = generate_camera_usage_example(selected_index, selected_name)

        print("\n" + "="*60)
        print("示例代码:")
        print("="*60)
        print(code)

        # 保存到文件
        save = input("\n是否保存到文件？(y/n): ").strip().lower()
        if save == 'y':
            filename = f"camera_example_{selected_index}.py"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            print(f"已保存到: {filename}")

    print("\n" + "="*60)
    print("操作完成!")
    print("="*60)

    # 显示总结
    print(f"\n选中的摄像头:")
    print(f"  索引: {selected_index}")
    print(f"  名称: {selected_name}")

    if selector.is_r200_camera(selected_name):
        print("  🎯 检测为R200摄像头")
        mode = selector.detect_r200_mode(selected_name)
        print(f"  模式: {mode}")

def simple_demo():
    """简单的演示程序"""
    print("简单的摄像头列表演示")
    print("="*60)

    try:
        # 使用PyCameraList获取摄像头列表
        cameras = list_video_devices()
        camera_dict = dict(cameras)

        if not camera_dict:
            print("没有检测到摄像头")
            return

        print(f"检测到 {len(camera_dict)} 个摄像头:")
        print("-"*50)

        for idx, name in camera_dict.items():
            print(f"[{idx}] {name}")

        print("\n使用示例:")
        for idx, name in camera_dict.items():
            if 'R200' in name or 'RealSense' in name:
                print(f"\n# R200摄像头示例 (索引: {idx})")
                print(f"import cv2")
                print(f"cap = cv2.VideoCapture({idx}, cv2.CAP_ANY)")
                print(f"print(f'摄像头: {name}')")
                break

    except Exception as e:
        print(f"错误: {e}")
        print("尝试安装PyCameraList: pip install pycameralist")

if __name__ == "__main__":
    try:
        # 简单演示
        # simple_demo()

        # 完整功能
        main()

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\n程序结束")