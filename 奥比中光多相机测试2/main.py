"""
三相机智能交替采集系统
参数可调，根据实际结果优化
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

class SmartAlternatingSystem:
    """智能交替采集系统"""

    def __init__(self):
        self.cameras = [None, None, None]
        self.rotation_index = 0
        self.running = False

        # 可调参数 - 基于测试结果优化
        self.config = {
            'resolution': (320, 240),   # 平衡速度和质量
            'fps': 30,                   # 目标帧率
            'codec': 'MJPG',            # 压缩格式
            'pair_duration': 0.8,       # 每个组合0.8秒（基于测试优化）
            'switch_delay': 0.03,       # 切换延迟30ms
            'warmup_delay': 0.08,       # 预热80ms
            'total_duration': 45,       # 总时间45秒
            'save_interval': 6,         # 每6帧保存一次
            'camera_combinations': [[0, 1], [1, 2], [0, 2]],
            'output_dir': f"smart_alternating_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'min_brightness': 25,       # 最小亮度
            'enable_debug': False,      # 调试模式
        }

        print("🎯 智能交替采集系统")
        print("=" * 60)
        print(f"基于测试结果优化的参数:")
        print(f"  每个组合: {self.config['pair_duration']}秒")
        print(f"  总时间: {self.config['total_duration']}秒")
        print(f"  保存间隔: 每{self.config['save_interval']}帧")
        print(f"  预期总FPS: 15-20")

    def run_optimized_capture(self):
        """运行优化后的采集"""
        print("\n🚀 开始智能交替采集")
        print("=" * 60)

        # 创建目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        image_dir = os.path.join(self.config['output_dir'], "images")
        os.makedirs(image_dir, exist_ok=True)

        print(f"📁 输出目录: {self.config['output_dir']}")

        # 初始化统计
        stats = {
            'frames': [0, 0, 0],
            'saved': [0, 0, 0],
            'combos_used': [0, 0, 0],
            'combo_times': [],
            'combo_frames': []
        }

        self.running = True
        start_time = time.time()

        # 创建显示窗口
        cv2.namedWindow("智能交替采集", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("智能交替采集", 800, 600)

        print("\n⚡ 开始采集... 按ESC停止")
        print("=" * 60)

        try:
            cycle = 0
            while self.running and time.time() - start_time < self.config['total_duration']:
                cycle += 1
                combo = self.config['camera_combinations'][self.rotation_index]
                cam1, cam2 = combo
                stats['combos_used'][self.rotation_index] += 1

                print(f"\n🔄 周期 {cycle}: 相机{combo}")

                # 确保相机就绪
                active_caps = []
                for cam_idx in combo:
                    if self.cameras[cam_idx] is None:
                        cap = self.quick_init_camera(cam_idx)
                        if cap:
                            self.cameras[cam_idx] = cap
                            active_caps.append((cam_idx, cap))
                    else:
                        active_caps.append((cam_idx, self.cameras[cam_idx]))

                if len(active_caps) < 2:
                    print(f"  ⚠️ 组合{combo}失败，跳过")
                    self.rotation_index = (self.rotation_index + 1) % 3
                    continue

                # 预热
                time.sleep(self.config['warmup_delay'])

                # 采集这个组合
                combo_start = time.time()
                combo_frames = {cam1: 0, cam2: 0}
                last_frames = [None, None, None]

                while self.running and time.time() - combo_start < self.config['pair_duration']:
                    frame_time = time.time()

                    # 采集两个相机
                    for cam_idx, cap in active_caps:
                        ret, frame = cap.read()

                        if ret and frame is not None:
                            stats['frames'][cam_idx] += 1
                            combo_frames[cam_idx] += 1
                            last_frames[cam_idx] = frame.copy()

                            # 保存图像
                            if stats['frames'][cam_idx] % self.config['save_interval'] == 0:
                                if self.is_good_frame(frame):
                                    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                                    filename = f"cam{cam_idx}_f{stats['frames'][cam_idx]:04d}_{timestamp}.jpg"
                                    cv2.imwrite(os.path.join(image_dir, filename),
                                               frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                    stats['saved'][cam_idx] += 1

                    # 显示（每3帧更新一次）
                    if sum(combo_frames.values()) % 3 == 0:
                        self.smart_display(last_frames, stats, start_time, combo, combo_frames)

                    # 检查ESC
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("\n🔴 用户停止")
                        self.running = False
                        break

                # 记录组合性能
                combo_elapsed = time.time() - combo_start
                stats['combo_times'].append(combo_elapsed)
                stats['combo_frames'].append(sum(combo_frames.values()))

                print(f"  完成: {combo_elapsed:.2f}秒, 帧数: {combo_frames}")

                # 切换到下一个组合
                self.rotation_index = (self.rotation_index + 1) % 3
                time.sleep(self.config['switch_delay'])

                # 显示进度
                if cycle % 3 == 0:
                    elapsed = time.time() - start_time
                    total_frames = sum(stats['frames'])
                    avg_fps = total_frames / elapsed if elapsed > 0 else 0
                    progress = elapsed / self.config['total_duration'] * 100

                    print(f"⏱️  进度: {elapsed:.1f}/{self.config['total_duration']}秒 ({progress:.1f}%)")
                    print(f"   总帧: {total_frames}, FPS: {avg_fps:.1f}")

        except KeyboardInterrupt:
            print("\n\n🔴 用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            elapsed = time.time() - start_time
            self.generate_smart_report(stats, elapsed)

            print("\n" + "=" * 60)
            print("智能采集完成")
            print("=" * 60)

            return True

    def quick_init_camera(self, index):
        """快速初始化相机"""
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution'][1])
            cap.set(cv2.CAP_PROP_FPS, self.config['fps'])

            # 测试读取
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap

            cap.release()
            return None

        except Exception as e:
            if self.config['enable_debug']:
                print(f"  相机#{index}初始化错误: {e}")
            return None

    def is_good_frame(self, frame):
        """检查帧质量"""
        if frame is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_bright = np.mean(gray)
        std_dev = np.std(gray)

        # 检查亮度和对比度
        if mean_bright < self.config['min_brightness']:
            return False
        if std_dev < 10:
            return False

        return True

    def smart_display(self, frames, stats, start_time, current_combo, combo_frames):
        """智能显示"""
        display_images = []

        for i in range(3):
            # 创建显示块
            display = np.zeros((180, 240, 3), dtype=np.uint8)

            if frames[i] is not None:
                # 有帧数据
                small = cv2.resize(frames[i], (240, 180))
                display = small

                # 状态标识
                if i in current_combo:
                    status_color = (0, 0, 255)  # 红色，激活
                    status_text = "ACT"
                else:
                    status_color = (100, 100, 100)  # 灰色，非激活
                    status_text = "INACT"

                cv2.putText(display, f"Cam{i} {status_text}", (5, 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                cv2.putText(display, f"F:{stats['frames'][i]}", (5, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(display, f"S:{stats['saved'][i]}", (5, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            else:
                # 无帧数据
                cv2.putText(display, f"Cam{i}", (80, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                cv2.putText(display, "NO DATA", (70, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            display_images.append(display)

        # 组合显示
        if len(display_images) >= 3:
            row1 = np.hstack(display_images[:2])
            row2 = display_images[2]

            # 调整第二行居中
            padding = (row1.shape[1] - row2.shape[1]) // 2
            if padding > 0:
                row2_padded = np.zeros((row2.shape[0], row1.shape[1], 3), dtype=np.uint8)
                row2_padded[:, padding:padding+row2.shape[1]] = row2
                row2 = row2_padded

            combined = np.vstack([row1, row2])

            # 全局信息
            elapsed = time.time() - start_time
            total_frames = sum(stats['frames'])
            fps = total_frames / elapsed if elapsed > 0 else 0

            info = f"Time: {elapsed:.1f}s | Frames: {total_frames} | FPS: {fps:.1f}"
            cv2.putText(combined, info, (5, combined.shape[0] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("智能交替采集", combined)

    def cleanup(self):
        """清理资源"""
        for i in range(3):
            if self.cameras[i] is not None:
                try:
                    self.cameras[i].release()
                except:
                    pass
                self.cameras[i] = None

        cv2.destroyAllWindows()

    def generate_smart_report(self, stats, elapsed):
        """生成智能报告"""
        report_file = os.path.join(self.config['output_dir'], "smart_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("智能交替采集报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"采集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"采集时长: {elapsed:.1f}秒\n")
            f.write(f"输出目录: {self.config['output_dir']}\n\n")

            f.write("采集结果:\n")
            total_frames = sum(stats['frames'])
            total_saved = sum(stats['saved'])

            for i in range(3):
                fps = stats['frames'][i] / elapsed if elapsed > 0 else 0
                f.write(f"  相机 #{i}:\n")
                f.write(f"      采集帧数: {stats['frames'][i]}\n")
                f.write(f"      保存图像: {stats['saved'][i]}\n")
                f.write(f"      帧率: {fps:.1f} FPS\n")

            f.write(f"\n组合使用统计:\n")
            for i, combo in enumerate(self.config['camera_combinations']):
                f.write(f"  组合 {combo}: {stats['combos_used'][i]}次\n")

            avg_fps = total_frames / elapsed if elapsed > 0 else 0
            f.write(f"\n总计:\n")
            f.write(f"  总采集帧数: {total_frames}\n")
            f.write(f"  总保存图像: {total_saved}\n")
            f.write(f"  平均FPS: {avg_fps:.1f}\n")

            # 性能分析
            if stats['combo_times'] and stats['combo_frames']:
                avg_combo_time = np.mean(stats['combo_times'])
                avg_combo_frames = np.mean(stats['combo_frames'])
                avg_combo_fps = avg_combo_frames / avg_combo_time if avg_combo_time > 0 else 0

                f.write(f"\n📊 组合性能分析:\n")
                f.write(f"  平均组合时间: {avg_combo_time:.3f}秒\n")
                f.write(f"  平均组合帧数: {avg_combo_frames:.1f}帧\n")
                f.write(f"  组合内FPS: {avg_combo_fps:.1f}\n")

                # 计算效率
                total_active_time = sum(stats['combo_times'])
                efficiency = total_active_time / elapsed * 100 if elapsed > 0 else 0
                f.write(f"  采集效率: {efficiency:.1f}%\n")

            f.write(f"\n🎯 3D重建评估:\n")
            if total_saved >= 30:
                f.write(f"  ✅ 数据量充足 ({total_saved}张图像)\n")
                f.write(f"     可以进行多视角3D重建\n")
                f.write(f"     建议使用MeshLab或COLMAP\n")
            elif total_saved >= 15:
                f.write(f"  ⚠️  数据量一般 ({total_saved}张图像)\n")
                f.write(f"     可以尝试3D重建，但可能不完整\n")
                f.write(f"     建议增加采集时间\n")
            else:
                f.write(f"  ❌ 数据量不足 ({total_saved}张图像)\n")
                f.write(f"     需要增加采集时间或调整参数\n")

            f.write(f"\n🔧 优化建议:\n")
            if avg_fps < 10:
                f.write(f"  1. 增加每个组合的采集时间\n")
                f.write(f"  2. 减少切换延迟和预热时间\n")
                f.write(f"  3. 考虑硬件解决方案（带电源USB集线器）\n")
            elif avg_fps < 15:
                f.write(f"  1. 微调参数以获得最佳性能\n")
                f.write(f"  2. 增加总采集时间以获得更多数据\n")
            else:
                f.write(f"  1. 当前参数工作良好，可以开始3D重建\n")
                f.write(f"  2. 如果重建质量不佳，可以增加分辨率\n")

        print(f"\n📄 报告已保存: {report_file}")

        # 显示简要结果
        print("\n📊 采集结果:")
        print("=" * 40)
        for i in range(3):
            fps = stats['frames'][i] / elapsed if elapsed > 0 else 0
            print(f"相机 #{i}: {stats['frames'][i]}帧, {stats['saved'][i]}张图像, {fps:.1f} FPS")

        print(f"\n组合使用:")
        for i, combo in enumerate(self.config['camera_combinations']):
            print(f"  {combo}: {stats['combos_used'][i]}次")

        print(f"\n总计: {total_frames}帧, {total_saved}张图像")
        print(f"平均FPS: {total_frames/elapsed:.1f}" if elapsed > 0 else "平均FPS: N/A")

    def run(self):
        """运行智能采集系统"""
        try:
            success = self.run_optimized_capture()

            if success:
                print("\n🎉 智能采集完成!")

                # 检查图像
                image_dir = os.path.join(self.config['output_dir'], "images")
                if os.path.exists(image_dir):
                    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
                    print(f"📸 保存的图像: {len(images)}张")

                    if len(images) >= 30:
                        print("✅ 足够进行3D重建!")
                    elif len(images) >= 15:
                        print("⚠️  可以尝试3D重建")
                    else:
                        print("❌ 需要更多数据")

        except Exception as e:
            print(f"\n❌ 程序错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("=" * 60)
    print("三相机智能交替采集系统")
    print("基于测试结果优化参数")
    print("=" * 60)

    system = SmartAlternatingSystem()
    system.run()

    print("\n📋 下一步:")
    print("1. 如果采集到足够图像，可以开始3D重建")
    print("2. 如果FPS不够，尝试调整参数或硬件")
    print("3. 查看报告获取详细建议")
    print("=" * 60)

    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()