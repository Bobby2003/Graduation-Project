import open3d as o3d
import numpy as np
import os
import time
import serial
import struct
import threading
from scipy.spatial.transform import Rotation as R

from realtime_scanner import DepthScanner 

class TSDFSensorFusion:
    def __init__(self):
        self.scanner = DepthScanner()
        
        # 1. 配置 TSDF 空间容器
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.005,      # 5mm 精度
            sdf_trunc=0.04,          # 4cm 截断
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # 2. 相机内参
        width, height = 640, 480
        fx, fy = 580.0, 580.0  
        cx, cy = 320.0, 240.0
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # 3. 外部 IMU 配置
        self.serial_port = 'COM7'     # 亚博智能 IMU 所在串口
        self.baud_rate = 115200       # 通讯波特率
        self.ser = None
        self.is_imu_running = False
        
        # 线程锁，防止相机线程和IMU线程同时读写发生冲突
        self.lock = threading.Lock()
        self.current_pose = np.eye(4)
        self.initial_rotation_inv = None # 用于储存“第一帧的逆矩阵”

    def start_imu_thread(self):
        """开启后台服务，监听串口并更新位姿矩阵"""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            self.is_imu_running = True
            
            # 开启守候线程
            self.imu_thread = threading.Thread(target=self._imu_worker)
            self.imu_thread.daemon = True # 主程序退出，此线程跟着死掉
            self.imu_thread.start()
            print(" IMU 串口后台线程已启动!")
            
        except serial.SerialException as e:
            print(f" 打开串口 {self.serial_port} 失败，请检查连线。错误: {e}")
            print(" 警告：无法获取 IMU 数据，将回退到无姿态模式(相机被认为静止)。")


    def _imu_worker(self):
            while self.is_imu_running:
                try:
                    if self.ser.read(1) == b'\x7e':
                        if self.ser.read(1) == b'\x23':
                            length_byte = self.ser.read(1)
                            if not length_byte: continue
                            length = length_byte[0]
                            payload = self.ser.read(length - 3)
                            
                            if len(payload) > 0 and payload[0] == 0x26:
                                data_bytes = payload[1:13]
                                r_rad = struct.unpack('<f', data_bytes[0:4])[0]
                                p_rad = struct.unpack('<f', data_bytes[4:8])[0]
                                y_rad = struct.unpack('<f', data_bytes[8:12])[0]

                                current_R = R.from_euler('xyz', [-p_rad, y_rad, -r_rad], degrees=False).as_matrix()

                                with self.lock:
                                    # 【零点重置核心逻辑】
                                    if self.initial_rotation_inv is None:
                                        self.initial_rotation_inv = np.linalg.inv(current_R)
                                        print("\n IMU 零点标定完成,当前视角已被强行设为正前方/绝对水平。")
                                    
                                    relative_R = self.initial_rotation_inv @ current_R
                                    
                                    # 覆盖到全局共享变量
                                    self.current_pose[:3, :3] = relative_R
                except:
                    pass


    def get_external_sensor_pose(self):
        """主线程调用这个接口获取最新融合位姿"""
        with self.lock:
            # 避免多线程访问同一个对象
            return np.copy(self.current_pose)

    def run(self):
        if not self.scanner.init():
            print(" 相机初始化失败")
            return

        # 启动 IMU 抓取线程
        self.start_imu_thread()

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(" Yahboom IMU 3D Scan", width=1280, height=720)
        
        # 视觉优化：设置背景色为深灰，方便观察白色模型
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2]) # 深灰色背景
        opt.mesh_show_back_face = True  # 显示背面，防止模型半透明消失
        
        # 添加坐标轴：红色-X, 绿色-Y, 蓝色-Z (原点参考)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        vis.add_geometry(axes)

        self.mesh = o3d.geometry.TriangleMesh()
        vis.add_geometry(self.mesh)
        
        print("\n 正在重置视距，请稍候...")
        frame_idx = 0
        initialized_view = False

        try:
            while True:
                result = self.scanner.get_rgb_and_depth()
                if result is None: continue
                color_np, depth_np = result

                color_o3d = o3d.geometry.Image(color_np)
                depth_o3d = o3d.geometry.Image(depth_np)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, 
                    depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
                )

                # 获取 IMU 位姿并积分
                current_pose = self.get_external_sensor_pose()
                self.volume.integrate(rgbd, self.intrinsic, np.linalg.inv(current_pose))

                frame_idx += 1

                # 每 10 帧更新一次 mesh
                if frame_idx % 10 == 0:
                    new_mesh = self.volume.extract_triangle_mesh()
                    if len(new_mesh.vertices) > 50: # 至少有50个点才更新
                        new_mesh.compute_vertex_normals()
                        self.mesh.vertices = new_mesh.vertices
                        self.mesh.triangles = new_mesh.triangles
                        self.mesh.vertex_colors = new_mesh.vertex_colors
                        self.mesh.vertex_normals = new_mesh.vertex_normals
                        vis.update_geometry(self.mesh)
                        
                        # 4. 🌟 第一次长出模型时，自动对焦
                        if not initialized_view:
                            vis.reset_view_point(True)
                            initialized_view = True
                            print(" 已捕获点云，自动对焦模型成功！")

                if not vis.poll_events(): break
                vis.update_renderer()

        finally:
            self.is_imu_running = False
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.scanner.close()
            vis.destroy_window()
            
            # --- 最终导出模型 ---
            print("\n 正在保存带有 IMU 角度融合的模型...")
            final_mesh = self.volume.extract_triangle_mesh()
            if len(final_mesh.vertices) > 0:
                final_mesh.compute_vertex_normals()
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(script_dir, "model")
                if not os.path.exists(model_dir): os.makedirs(model_dir)
                save_path = os.path.join(model_dir, "imu_fusion_model.obj")
                o3d.io.write_triangle_mesh(save_path, final_mesh, write_vertex_normals=True)
                print(f"✅ 模型完美保存！路径: {save_path}")
            else:
                print("⚠️ 无效数据。")

if __name__ == "__main__":
    app = TSDFSensorFusion()
    app.run()