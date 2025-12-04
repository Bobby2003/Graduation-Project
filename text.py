"""
R200摄像头索引发现器 - 系统级枚举
找到R200在系统中的实际索引号
"""

import cv2
import numpy as np
import time
import subprocess
import re
from datetime import datetime

class CameraEnumerator:
    """摄像头设备枚举器 - 查找R200的实际索引"""

    def __init__(self):
        self.available_cameras = []
        self.r200_cameras = []

    def enumerate_windows_cameras(self, max_index=10):
        """
        在Windows上枚举所有可用的摄像头索引
        返回所有可打开摄像头的详细信息
        """
        print(f"\n正在枚举摄像头索引 0-{max_index}...")
        print("-" * 60)

        self.available_cameras = []

        for index in range(max_index + 1):
            print(f"\n尝试索引 {index}:")

            camera_info = {
                'index': index,
                'backends': {},
                'successful': False
            }

            # 尝试不同的后端
            backends = [
                ('DSHOW', cv2.CAP_DSHOW),
                ('MSMF', cv2.CAP_MSMF),
                ('VFW', cv2.CAP_VFW),
                ('AUTO', cv2.CAP_ANY),
            ]

            for backend_name, backend_id in backends:
                print(f"  尝试{backend_name}后端...", end=" ")

                try:
                    cap = cv2.VideoCapture(index, backend_id)

                    if cap.isOpened():
                        # 尝试读取帧（最多尝试3次）
                        frame = None
                        for attempt in range(3):
                            ret, test_frame = cap.read()
                            if ret:
                                frame = test_frame
                                break
                            time.sleep(0.1)

                        if frame is not None:
                            h, w = frame.shape[:2]
                            channels = frame.shape[2] if len(frame.shape) == 3 else 1

                            # 获取更多信息
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

                            camera_info['backends'][backend_name] = {
                                'success': True,
                                'resolution': (w, h),
                                'channels': channels,
                                'fps': fps,
                                'frame_sample': frame.copy() if attempt == 2 else None
                            }

                            print(f"✅ {w}x{h}, {channels}通道")

                            # 检查是否是R200
                            if self._is_likely_r200(w, h, channels):
                                camera_info['likely_r200'] = True
                                camera_info['r200_mode'] = self._identify_r200_mode(w, h, channels)
                        else:
                            camera_info['backends'][backend_name] = {
                                'success': True,
                                'resolution': None,
                                'channels': None,
                                'fps': None,
                                'frame_sample': None
                            }
                            print("⚠️ 可打开但无法读取帧")

                        cap.release()
                        camera_info['successful'] = True

                    else:
                        camera_info['backends'][backend_name] = {'success': False}
                        print("❌ 无法打开")

                except Exception as e:
                    camera_info['backends'][backend_name] = {'success': False, 'error': str(e)}
                    print(f"❌ 错误: {e}")

            if camera_info['successful']:
                self.available_cameras.append(camera_info)

            # 短暂延迟，避免设备冲突
            time.sleep(0.3)

        return self.available_cameras

    def _is_likely_r200(self, width, height, channels):
        """判断是否是R200摄像头（基于分辨率特征）"""
        # R200的典型分辨率
        r200_resolutions = [
            (640, 480),      # 深度/IR模式
            (1920, 1080),    # RGB模式
            (1280, 480),     # 双目IR模式
        ]

        for res_w, res_h in r200_resolutions:
            if width == res_w and height == res_h:
                return True

        return False

    def _identify_r200_mode(self, width, height, channels):
        """识别R200的工作模式"""
        if width == 1920 and height == 1080 and channels == 3:
            return "RGB模式（彩色相机）"
        elif width == 640 and height == 480:
            if channels == 1:
                return "深度/IR模式（单通道）"
            else:
                return "IR模式（三通道）"
        elif width == 1280 and height == 480:
            return "双目IR模式"
        else:
            return f"未知模式 ({width}x{height})"

    def analyze_results(self):
        """分析枚举结果"""
        print("\n" + "=" * 60)
        print("枚举结果分析")
        print("=" * 60)

        if not self.available_cameras:
            print("❌ 未找到任何可用的摄像头")
            return

        print(f"找到 {len(self.available_cameras)} 个可用摄像头:\n")

        for cam in self.available_cameras:
            print(f"摄像头索引 {cam['index']}:")

            # 找出最佳的后端
            best_backend = None
            for backend_name, backend_info in cam['backends'].items():
                if backend_info.get('success') and backend_info.get('resolution'):
                    best_backend = (backend_name, backend_info)
                    break

            if best_backend:
                backend_name, backend_info = best_backend
                w, h = backend_info['resolution']
                channels = backend_info['channels']

                print(f"  最佳后端: {backend_name}")
                print(f"  分辨率: {w}x{h}")
                print(f"  通道数: {channels}")

                if cam.get('likely_r200'):
                    print(f"  🎯 可能是R200: {cam.get('r200_mode', '未知模式')}")

                    # 显示预览
                    if backend_info.get('frame_sample') is not None:
                        self._show_camera_preview(cam['index'], backend_info['frame_sample'],
                                                 cam.get('r200_mode', '未知'))

                print()

    def _show_camera_preview(self, index, frame, mode_name):
        """显示摄像头预览"""
        h, w = frame.shape[:2]

        # 调整显示大小
        display_h = 400
        display_w = int(display_h * w / h)
        if display_w > 600:
            display_w = 600
            display_h = int(display_w * h / w)

        display_frame = cv2.resize(frame, (display_w, display_h))

        # 添加文字
        info_text = f"索引 {index}: {w}x{h}"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        mode_text = f"模式: {mode_name}"
        cv2.putText(display_frame, mode_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        window_name = f"摄像头索引 {index}"
        cv2.imshow(window_name, display_frame)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

def find_r200_with_powershell():
    """
    使用PowerShell命令查找R200摄像头
    这是最可靠的方法，因为PowerShell能访问系统级设备信息
    """
    print("\n" + "=" * 60)
    print("使用PowerShell查找R200摄像头")
    print("=" * 60)

    # PowerShell命令：查找所有摄像头设备
    ps_command = '''
    $cameras = Get-PnpDevice -Class Camera
    $usbDevices = Get-PnpDevice | Where-Object {$_.DeviceID -like "*VID_8086*"}  # Intel VID
    
    Write-Host "=== 摄像头设备 ==="
    foreach ($cam in $cameras) {
        Write-Host "设备: $($cam.FriendlyName)"
        Write-Host "状态: $($cam.Status)"
        Write-Host "设备ID: $($cam.DeviceID)"
        Write-Host "---"
    }
    
    Write-Host "`n=== Intel设备 (可能包含R200) ==="
    foreach ($dev in $usbDevices) {
        Write-Host "设备: $($dev.FriendlyName)"
        Write-Host "设备ID: $($dev.DeviceID)"
        Write-Host "---"
    }
    
    # 尝试获取DirectShow设备
    try {
        $source = [System.Management.ManagementObjectSearcher]::new("SELECT * FROM Win32_PnPEntity WHERE (PNPClass = 'Image' OR PNPClass = 'Camera')")
        $allCams = $source.Get()
        Write-Host "`n=== 所有图像/摄像头设备 ==="
        foreach ($c in $allCams) {
            Write-Host "$($c.Name) | $($c.DeviceID)"
        }
    } catch {
        Write-Host "无法获取详细设备信息"
    }
    '''

    try:
        # 执行PowerShell命令
        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            encoding='gbk',
            errors='ignore'
        )

        print("PowerShell输出:")
        print(result.stdout)

        if result.stderr:
            print("错误信息:", result.stderr)

        # 分析输出，查找R200
        lines = result.stdout.split('\n')
        r200_lines = []

        for line in lines:
            if 'R200' in line or 'RealSense' in line or 'VID_8086' in line:
                r200_lines.append(line)

        if r200_lines:
            print("\n🎯 找到可能的R200设备:")
            for line in r200_lines:
                print(f"  {line}")
        else:
            print("\n⚠️ 未找到明确的R200设备信息")

    except Exception as e:
        print(f"执行PowerShell时出错: {e}")

def try_specific_indices():
    """
    针对性地尝试常见索引
    基于经验，R200可能有多个索引对应不同流
    """
    print("\n" + "=" * 60)
    print("针对性索引测试")
    print("=" * 60)

    # 常见的R200索引组合（根据用户报告和经验）
    test_indices = [
        (0, "常见主索引"),
        (1, "常见副索引"),
        (2, "常见RGB流索引"),
        (3, "常见双目IR流索引"),
        (4, "常见深度流索引"),
        (700, "特殊大索引（某些系统）"),
        (701, "大索引+1"),
        (702, "大索引+2"),
    ]

    successful_indices = []

    for index, description in test_indices:
        print(f"\n测试索引 {index} ({description}):")

        # 只尝试MSMF后端（因为DSHOW失败）
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

        if not cap.isOpened():
            print(f"  ❌ 无法打开索引 {index}")
            continue

        # 尝试读取帧
        frames = []
        for attempt in range(5):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

            # 显示读取尝试
            print(f"  尝试 {attempt+1}: {'成功' if ret else '失败'}")
            time.sleep(0.1)

        if frames:
            frame = frames[-1]
            h, w = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1

            print(f"  ✅ 成功! {w}x{h}, {channels}通道")

            successful_indices.append({
                'index': index,
                'description': description,
                'resolution': (w, h),
                'channels': channels,
                'frame': frame
            })

            # 显示预览
            display_h = 300
            display_w = int(display_h * w / h)
            display_frame = cv2.resize(frame, (display_w, display_h))

            cv2.putText(display_frame, f"索引 {index}: {w}x{h}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, description,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            cv2.imshow(f"索引 {index}", display_frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print(f"  ⚠️ 可打开但无法读取帧")

        cap.release()
        time.sleep(0.5)

    return successful_indices

def main():
    """主函数"""
    print("=" * 60)
    print("R200摄像头索引发现器")
    print("=" * 60)
    print("目标: 找到R200在系统中的实际索引号")
    print()

    # 方法1: 使用PowerShell查找设备信息
    find_r200_with_powershell()

    # 方法2: 系统枚举
    print("\n" + "=" * 60)
    print("开始系统级摄像头枚举...")

    enumerator = CameraEnumerator()
    all_cameras = enumerator.enumerate_windows_cameras(max_index=15)  # 检查更多索引

    # 分析结果
    enumerator.analyze_results()

    # 方法3: 针对性测试
    print("\n" + "=" * 60)
    print("开始针对性索引测试...")

    specific_results = try_specific_indices()

    # 总结
    print("\n" + "=" * 60)
    print("索引发现总结")
    print("=" * 60)

    if specific_results:
        print(f"✅ 找到 {len(specific_results)} 个可用的摄像头索引:")

        for result in specific_results:
            print(f"\n索引 {result['index']} ({result['description']}):")
            print(f"  分辨率: {result['resolution'][0]}x{result['resolution'][1]}")
            print(f"  通道数: {result['channels']}")

            # 判断可能是什么流
            w, h = result['resolution']
            if w == 1920 and h == 1080:
                print("  🎯 可能是: R200 RGB彩色流")
            elif w == 640 and h == 480:
                if result['channels'] == 1:
                    print("  🎯 可能是: R200 深度/IR流（单通道）")
                else:
                    print("  🎯 可能是: R200 IR流（三通道）")
            elif w == 1280 and h == 480:
                print("  🎯 可能是: R200 双目IR流")
    else:
        print("❌ 未找到任何可用的摄像头索引")

        print("\n🔧 故障排除建议:")
        print("1. 确保R200已正确连接到USB 3.0端口")
        print("2. 关闭所有可能使用摄像头的程序（OBS、Zoom、微信等）")
        print("3. 重启电脑后重试")
        print("4. 尝试在另一台电脑上测试R200")
        print("5. 考虑重新安装摄像头驱动程序")

    # 如果找到了可用索引，提供使用示例
    if specific_results:
        print("\n" + "=" * 60)
        print("使用示例代码")
        print("=" * 60)

        best_index = specific_results[0]['index']
        print(f"使用索引 {best_index} 的示例:")

        example_code = f"""
import cv2
import time

# 使用发现的索引
cap = cv2.VideoCapture({best_index}, cv2.CAP_MSMF)

if cap.isOpened():
    print("摄像头打开成功")
    
    # 读取并显示
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('R200摄像头', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("读取帧失败")
            break
    
    cap.release()
    cv2.destroyAllWindows()
else:
    print("无法打开摄像头")
"""

        print(example_code)

if __name__ == "__main__":
    try:
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