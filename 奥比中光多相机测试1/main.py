"""
三相机高帧率同步采集系统
专门解决低FPS和同步问题
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import threading
import queue

class HighFPSMultiCameraSystem:
    """高帧率多相机同步采集系统"""

    def __init__(self):
        self.cameras = []
        self.running = False
        self.frame_timestamps = []
        self.sync_offset = 0.01  # 10ms同步偏移补偿

        # 基于高带宽测试的优化配置
        self.config = {
            'resolution': (320, 240),   # 使用640x480，平衡质量和速度
            'fps': 60,                   # 目标帧率
            'codec': 'MJPG',            # MJPG压缩
            'capture_duration': 30,     # 采集时长
            'save_mode': 'burst',       # 突发保存模式
            'save_every_n_frames': 30,  # 每30帧保存一次
            'burst_size': 3,            # 突发保存3组同步帧
            'output_dir': f"sync_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'enable_exposure_control': True,  # 启用曝光控制
            'target_brightness': 120,         # 目标亮度值（0-255）
        }

        print("⚡ 高帧率三相机同步采集系统")
        print("=" * 60)
        print("配置优化：")
        print(f"  分辨率: {self.config['resolution'][0]}x{self.config['resolution'][1]}")
        print(f"  目标帧率: {self.config['fps']} FPS")
        print(f"  编码: {self.config['codec']}")
        print(f"  采集模式: 突发保存")
        print(f"  同步补偿: {self.sync_offset*1000:.0f}ms")

    def setup_camera_advanced(self, index):
        """高级相机设置，包含曝光控制"""
        print(f"\n设置相机 #{index} (高级模式)...")

        # 尝试不同后端，优先DSHOW（Windows最佳）
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)

        if not cap.isOpened():
            print(f"  ❌ 无法打开相机 #{index}")
            return None

        try:
            # 1. 设置MJPG压缩格式
            fourcc = cv2.VideoWriter_fourcc(*self.config['codec'])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            # 2. 设置分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution'][1])

            # 3. 设置帧率
            cap.set(cv2.CAP_PROP_FPS, self.config['fps'])

            # 4. 曝光控制（如果启用）
            '''
            if self.config['enable_exposure_control']:
                # 尝试设置自动曝光
                try:
                    # 不同相机的属性ID可能不同
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=自动，3=手动
                    cap.set(cv2.CAP_PROP_EXPOSURE, -6)      # 负值=自动，正值=手动值
                except:
                    print(f"  ⚠️ 相机 #{index} 曝光控制不支持")
            '''
            # 5. 其他优化设置
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区大小，降低延迟
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # 关闭自动对焦（如果支持）
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)     # 自动白平衡
            except:
                pass

            # 验证设置
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"  ✅ 设置成功:")
            print(f"     分辨率: {actual_width}x{actual_height}")
            print(f"     帧率: {actual_fps}")

            # 清空缓冲区
            for _ in range(5):
                cap.read()

            return cap

        except Exception as e:
            print(f"  ❌ 设置失败: {e}")
            cap.release()
            return None

    def capture_synchronized_burst(self):
        """同步突发采集模式"""
        print("\n" + "=" * 60)
        print("同步突发采集模式")
        print("=" * 60)
        print("说明: 快速连续采集三个相机，尽可能实现同步")
        print("      使用突发保存减少I/O开销")

        # 创建输出目录
        image_dir = os.path.join(self.config['output_dir'], "sync_images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"📁 图像保存到: {image_dir}")

        # 设置相机
        print("\n🔧 初始化相机...")
        self.cameras = []
        for i in range(3):
            cap = self.setup_camera_advanced(i)
            if cap is not None:
                self.cameras.append(cap)
                print(f"✅ 相机 #{i} 初始化成功")
            else:
                self.cameras.append(None)
                print(f"❌ 相机 #{i} 初始化失败")

        working_cameras = [c for c in self.cameras if c is not None]
        if len(working_cameras) < 2:
            print(f"\n❌ 只有 {len(working_cameras)} 个相机工作，需要至少2个")
            return False

        print(f"\n✅ 可用的相机: {len(working_cameras)}/3")

        # 保存配置
        config_file = os.path.join(self.config['output_dir'], "sync_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"📝 配置文件已保存: {config_file}")

        print("\n⚡ 开始同步采集...")
        print("   按ESC键可以提前停止")
        print(f"   采集将持续 {self.config['capture_duration']} 秒")
        print("=" * 60)

        # 初始化统计
        frame_counts = [0, 0, 0]
        saved_counts = [0, 0, 0]
        self.frame_timestamps = []
        burst_counter = 0
        self.running = True

        # 创建显示窗口
        cv2.namedWindow("三相机同步采集", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("三相机同步采集", 1200, 600)

        start_time = time.time()
        last_save_time = start_time

        try:
            while self.running and time.time() - start_time < self.config['capture_duration']:
                burst_counter += 1
                burst_timestamp = time.time()
                burst_frames = []

                # 突发采集：快速连续采集三个相机
                for cam_idx, cap in enumerate(self.cameras):
                    if cap is None:
                        burst_frames.append(None)
                        continue

                    # 记录采集开始时间
                    capture_start = time.time()

                    # 采集一帧
                    ret, frame = cap.read()

                    if ret and frame is not None:
                        frame_counts[cam_idx] += 1

                        # 检查图像质量（防止黑屏）
                        if self.check_frame_quality(frame):
                            burst_frames.append({
                                'camera': cam_idx,
                                'frame': frame.copy(),
                                'timestamp': capture_start,
                                'frame_number': frame_counts[cam_idx]
                            })

                            # 存储时间戳用于同步分析
                            self.frame_timestamps.append({
                                'camera': cam_idx,
                                'timestamp': capture_start,
                                'frame_number': frame_counts[cam_idx],
                                'burst_id': burst_counter
                            })
                        else:
                            print(f"⚠️ 相机 #{cam_idx} 第{frame_counts[cam_idx]}帧质量不佳")
                            burst_frames.append(None)
                    else:
                        print(f"❌ 相机 #{cam_idx} 采集失败")
                        burst_frames.append(None)

                # 同步分析：计算三个相机的时间差
                valid_timestamps = [f['timestamp'] for f in burst_frames if f is not None]
                if len(valid_timestamps) >= 2:
                    max_diff = max(valid_timestamps) - min(valid_timestamps)
                    if max_diff > 0.05:  # 如果时间差大于50ms
                        print(f"⚠️ 同步警告: 时间差 {max_diff*1000:.1f}ms")

                # 突发保存：每N帧保存一次，一次保存三个相机的同步帧
                current_time = time.time()
                if (burst_counter % self.config['save_every_n_frames'] == 0 or
                    current_time - last_save_time > 1.0):

                    for frame_data in burst_frames:
                        if frame_data is not None:
                            # 保存图像
                            timestamp_str = datetime.now().strftime("%H%M%S_%f")[:-3]
                            filename = os.path.join(
                                image_dir,
                                f"burst{burst_counter:04d}_cam{frame_data['camera']}_"
                                f"frame{frame_data['frame_number']:06d}_{timestamp_str}.jpg"
                            )

                            # 高质量JPEG保存
                            cv2.imwrite(filename, frame_data['frame'],
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])

                            saved_counts[frame_data['camera']] += 1

                    last_save_time = current_time
                    print(f"💾 突发保存 #{burst_counter}: 保存了{len([f for f in burst_frames if f is not None])}张图像")

                # 实时显示（简化显示以减少开销）
                display_images = []
                for frame_data in burst_frames:
                    if frame_data is not None:
                        # 缩小显示
                        small_frame = cv2.resize(frame_data['frame'], (400, 300))

                        # 添加简单信息
                        cv2.putText(small_frame, f"Cam{frame_data['camera']}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(small_frame, f"Frame: {frame_data['frame_number']}", (10, 60),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # 显示同步状态
                        time_diff = frame_data['timestamp'] - burst_timestamp
                        cv2.putText(small_frame, f"Sync: {time_diff*1000:+.1f}ms", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        display_images.append(small_frame)
                    else:
                        # 占位图像
                        placeholder = np.zeros((300, 400, 3), dtype=np.uint8)
                        cv2.putText(placeholder, f"No Signal", (120, 150),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        display_images.append(placeholder)

                # 组合显示
                if len(display_images) >= 3:
                    row1 = np.hstack(display_images[:2])

                    # 第二行居中显示
                    padding = (row1.shape[1] - display_images[2].shape[1]) // 2
                    if padding > 0:
                        row2 = np.zeros((display_images[2].shape[0], row1.shape[1], 3), dtype=np.uint8)
                        row2[:, padding:padding+display_images[2].shape[1]] = display_images[2]
                    else:
                        row2 = display_images[2]

                    combined = np.vstack([row1, row2])

                    # 添加全局信息
                    elapsed = time.time() - start_time
                    progress = elapsed / self.config['capture_duration'] * 100
                    total_frames = sum(frame_counts)
                    avg_fps = total_frames / elapsed if elapsed > 0 else 0

                    info_text = f"Time: {elapsed:.1f}s ({progress:.1f}%) | "
                    info_text += f"Frames: {total_frames} | FPS: {avg_fps:.1f}"
                    cv2.putText(combined, info_text, (10, combined.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow("三相机同步采集", combined)

                # 检查ESC键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("\n🔴 用户按ESC键停止")
                    self.running = False
                    break

                # 显示进度
                elapsed = time.time() - start_time
                if int(elapsed) % 2 == 0:
                    fps_info = []
                    for i, count in enumerate(frame_counts):
                        if self.cameras[i] is not None:
                            fps = count / elapsed if elapsed > 0 else 0
                            fps_info.append(f"Cam{i}:{fps:.1f}")

                    print(f"进度: {elapsed:.1f}s | " + " | ".join(fps_info) +
                          f" | 突发: {burst_counter}", end='\r')

        except KeyboardInterrupt:
            print("\n\n🔴 用户中断采集")
        except Exception as e:
            print(f"\n❌ 采集过程中出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.running = False
            elapsed = time.time() - start_time

            # 关闭相机
            for cam in self.cameras:
                if cam is not None:
                    cam.release()

            cv2.destroyAllWindows()

            # 生成报告和同步分析
            self.generate_sync_report(frame_counts, saved_counts, elapsed)

            print("\n" + "=" * 60)
            print("同步采集完成")
            print("=" * 60)

            return True

    def check_frame_quality(self, frame):
        """检查帧质量，防止黑屏"""
        if frame is None:
            return False

        # 计算图像的平均亮度
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        mean_brightness = np.mean(gray)

        # 检查是否为全黑或接近全黑
        if mean_brightness < 10:  # 阈值可根据需要调整
            return False

        # 检查是否为全白或接近全白（过曝）
        if mean_brightness > 240:
            return False

        # 检查图像对比度（标准差）
        std_dev = np.std(gray)
        if std_dev < 5:  # 图像过于平坦
            return False

        return True

    def analyze_synchronization(self):
        """分析同步性能"""
        if not self.frame_timestamps:
            return None

        # 按突发分组
        bursts = {}
        for record in self.frame_timestamps:
            burst_id = record['burst_id']
            if burst_id not in bursts:
                bursts[burst_id] = []
            bursts[burst_id].append(record)

        # 计算每个突发的同步误差
        sync_errors = []
        valid_bursts = 0

        for burst_id, records in bursts.items():
            if len(records) >= 2:  # 至少两个相机有数据
                timestamps = [r['timestamp'] for r in records]
                max_diff = max(timestamps) - min(timestamps)
                sync_errors.append(max_diff)
                valid_bursts += 1

        if sync_errors:
            avg_error = np.mean(sync_errors) * 1000  # 转换为毫秒
            max_error = max(sync_errors) * 1000
            min_error = min(sync_errors) * 1000
            std_error = np.std(sync_errors) * 1000

            return {
                'valid_bursts': valid_bursts,
                'avg_error_ms': avg_error,
                'max_error_ms': max_error,
                'min_error_ms': min_error,
                'std_error_ms': std_error
            }

        return None

    def generate_sync_report(self, frame_counts, saved_counts, elapsed):
        """生成同步采集报告"""
        report_file = os.path.join(self.config['output_dir'], "sync_report.txt")

        # 分析同步性能
        sync_analysis = self.analyze_synchronization()

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("三相机同步采集报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"采集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"采集时长: {elapsed:.1f}秒\n")
            f.write(f"输出目录: {self.config['output_dir']}\n\n")

            f.write("采集统计:\n")
            total_frames = 0
            total_saved = 0

            for i in range(3):
                if self.cameras[i] is not None:
                    fps = frame_counts[i] / elapsed if elapsed > 0 else 0
                    f.write(f"  相机 #{i}:\n")
                    f.write(f"      采集帧数: {frame_counts[i]}\n")
                    f.write(f"      保存图像: {saved_counts[i]}\n")
                    f.write(f"      帧率: {fps:.1f} FPS\n")

                    total_frames += frame_counts[i]
                    total_saved += saved_counts[i]
                else:
                    f.write(f"  相机 #{i}: ❌ 未工作\n")

            avg_fps = total_frames / elapsed if elapsed > 0 else 0
            f.write(f"\n总计:\n")
            f.write(f"  总采集帧数: {total_frames}\n")
            f.write(f"  总保存图像: {total_saved}\n")
            f.write(f"  平均帧率: {avg_fps:.1f} FPS\n")

            # 同步分析结果
            if sync_analysis:
                f.write(f"\n📊 同步性能分析:\n")
                f.write(f"  有效突发数: {sync_analysis['valid_bursts']}\n")
                f.write(f"  平均同步误差: {sync_analysis['avg_error_ms']:.2f} ms\n")
                f.write(f"  最大同步误差: {sync_analysis['max_error_ms']:.2f} ms\n")
                f.write(f"  最小同步误差: {sync_analysis['min_error_ms']:.2f} ms\n")
                f.write(f"  误差标准差: {sync_analysis['std_error_ms']:.2f} ms\n")

                # 评估同步质量
                if sync_analysis['avg_error_ms'] < 20:
                    f.write(f"\n  ✅ 同步质量: 优秀 (<20ms)\n")
                    f.write(f"     适合动态场景重建\n")
                elif sync_analysis['avg_error_ms'] < 50:
                    f.write(f"\n  ⚠️  同步质量: 良好 (20-50ms)\n")
                    f.write(f"     适合静态或慢速场景重建\n")
                else:
                    f.write(f"\n  ❌ 同步质量: 较差 (>50ms)\n")
                    f.write(f"     只适合完全静态场景重建\n")
            else:
                f.write(f"\n⚠️  无法进行同步分析（数据不足）\n")

            f.write("\n💡 3D重建建议:\n")
            if total_saved >= 30:
                f.write("1. ✅ 数据量足够进行3D重建\n")
                f.write("2. 根据同步质量选择重建策略:\n")
                if sync_analysis and sync_analysis['avg_error_ms'] < 30:
                    f.write("   - 可以直接进行多视角立体匹配\n")
                else:
                    f.write("   - 建议使用基于特征点匹配的方法\n")
                    f.write("   - 或对图像进行时间对齐处理\n")
                f.write("3. 推荐软件: MeshLab, COLMAP, OpenMVG\n")
            else:
                f.write("1. ❌ 数据量不足，建议重新采集\n")
                f.write("2. 增加采集时间或减少保存间隔\n")
                f.write("3. 确保相机工作正常\n")

            f.write("\n📋 重建步骤:\n")
            f.write("1. 相机标定（获取内参和畸变系数）\n")
            f.write("2. 外参标定（相机间相对位置）\n")
            f.write("3. 特征点提取和匹配\n")
            f.write("4. 稀疏点云重建\n")
            f.write("5. 稠密点云重建\n")
            f.write("6. 网格生成和纹理映射\n")

        print(f"\n📄 同步报告已保存: {report_file}")

        # 打印简要结果
        print("\n📊 采集摘要:")
        print("=" * 40)
        for i in range(3):
            if self.cameras[i] is not None:
                fps = frame_counts[i] / elapsed if elapsed > 0 else 0
                print(f"相机 #{i}: {frame_counts[i]}帧, {saved_counts[i]}张图像, {fps:.1f} FPS")

        print(f"\n总计: {total_frames}帧, {total_saved}张图像")
        print(f"平均FPS: {avg_fps:.1f}")

        if sync_analysis:
            print(f"\n📊 同步分析:")
            print(f"  平均同步误差: {sync_analysis['avg_error_ms']:.1f}ms")
            print(f"  最大同步误差: {sync_analysis['max_error_ms']:.1f}ms")
            print(f"  最小同步误差: {sync_analysis['min_error_ms']:.1f}ms")

    def run(self):
        """运行同步采集系统"""
        print("\n🚀 开始高帧率同步采集")
        print("=" * 60)

        # 创建输出目录
        os.makedirs(self.config['output_dir'], exist_ok=True)
        print(f"📁 输出目录: {os.path.abspath(self.config['output_dir'])}")

        try:
            success = self.capture_synchronized_burst()

            if success:
                print("\n🎉 同步采集完成！")
                print(f"📁 所有数据已保存到: {self.config['output_dir']}")

                # 显示下一步建议
                image_dir = os.path.join(self.config['output_dir'], "sync_images")
                if os.path.exists(image_dir):
                    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
                    print(f"📸 保存的图像数量: {len(image_files)}")

                    if len(image_files) > 0:
                        print("\n💡 下一步:")
                        print("1. 检查图像质量（避免黑屏）")
                        print("2. 查看同步报告评估同步质量")
                        print("3. 根据同步质量选择3D重建方法")

            else:
                print("\n⚠️ 采集过程中出现问题")
                print("💡 请检查错误信息并调整设置")

        except Exception as e:
            print(f"\n❌ 程序出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("\n" + "=" * 60)
            print("程序结束")
            print("=" * 60)

def main():
    """主函数"""
    print("=" * 60)
    print("三相机高帧率同步采集系统")
    print("版本: 3.0 - 同步优化版")
    print("=" * 60)
    print("特点:")
    print("  ✅ 高帧率采集 (目标30 FPS)")
    print("  ✅ 同步性能分析")
    print("  ✅ 突发保存模式减少I/O开销")
    print("  ✅ 图像质量检查避免黑屏")
    print("  ✅ 详细的同步报告")
    print("=" * 60)

    # 创建系统
    system = HighFPSMultiCameraSystem()

    # 运行采集
    system.run()

    print("\n📋 使用提示:")
    print("1. 确保三个相机都连接到USB 3.0接口")
    print("2. 如果出现黑屏，尝试调整曝光设置")
    print("3. 按ESC键可以提前停止采集")
    print("4. 查看同步报告决定重建策略")
    print("=" * 60)

    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()