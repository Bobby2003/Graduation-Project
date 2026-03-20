import serial
import serial.tools.list_ports
import struct
import math
import threading
import time
import numpy as np
import open3d as o3d
import copy

HEADER_1 = 0x7E
HEADER_2 = 0x23
FUNC_EULER = 0x26
FUNC_QUAT = 0x16
FUNC_RAW = 0x04

ACCEL_SCALE = 16.0 / 32767.0


def calc_checksum(data: bytes) -> int:
    return sum(data) & 0xFF


class FrameParser:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)
        frames = []
        while True:
            idx = next((i for i in range(len(self.buf) - 1)
                        if self.buf[i] == HEADER_1 and self.buf[i + 1] == HEADER_2), -1)
            if idx == -1:
                self.buf.clear();
                break
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < 3:
                break
            frame_len = self.buf[2]
            if len(self.buf) < frame_len:
                break
            frame = bytes(self.buf[:frame_len])
            del self.buf[:frame_len]
            if calc_checksum(frame[:-1]) != frame[-1]:
                continue
            frames.append((frame[3], frame[4:-1]))
        return frames


def quat_to_matrix(w, x, y, z) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_matrix(roll, pitch, yaw) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def find_ch340_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.upper() for k in ('CH340', 'CH341', 'USB-SERIAL')):
            return p.device
    return None


class IMUReader:
    def __init__(self):
        self._lock = threading.Lock()
        self._R = np.eye(3)
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._last_t = None
        self._running = False
        self.info = "Waiting..."
        self._accel_buf = []

    @property
    def state(self):
        with self._lock:
            return self._R.copy(), self._pos.copy()

    def reset_position(self):
        with self._lock:
            self._pos = np.zeros(3)
            self._vel = np.zeros(3)

    def start(self, port, baudrate=115200):
        self._running = True
        threading.Thread(target=self._loop, args=(port, baudrate), daemon=True).start()

    def stop(self):
        self._running = False

    def _update_position(self, accel_raw: np.ndarray, dt: float):
        accel_world = self._R @ accel_raw
        accel_linear = accel_world - np.array([0.0, 0.0, 9.81])

        self._accel_buf.append(np.linalg.norm(accel_linear))
        if len(self._accel_buf) > 20:
            self._accel_buf.pop(0)
        if len(self._accel_buf) == 20 and np.std(self._accel_buf) < 0.08:
            self._vel[:] = 0.0

        self._vel += accel_linear * dt
        self._pos += self._vel * dt

    def _loop(self, port, baudrate):
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            parser = FrameParser()
            while self._running:
                raw = ser.read(256)
                if not raw:
                    continue
                now = time.time()
                for func, payload in parser.feed(raw):
                    with self._lock:
                        dt = (now - self._last_t) if self._last_t else 0.02
                        dt = min(dt, 0.1)
                        self._last_t = now

                        if func == FUNC_QUAT and len(payload) >= 16:
                            w, x, y, z = struct.unpack_from('<ffff', payload, 0)
                            self._R = quat_to_matrix(w, x, y, z)
                        elif func == FUNC_EULER and len(payload) >= 12:
                            roll, pitch, yaw = struct.unpack_from('<fff', payload, 0)
                            self._R = euler_to_matrix(roll, pitch, yaw)
                        elif func == FUNC_RAW and len(payload) >= 6:
                            ax = struct.unpack_from('<h', payload, 0)[0] * ACCEL_SCALE * 9.81
                            ay = struct.unpack_from('<h', payload, 2)[0] * ACCEL_SCALE * 9.81
                            az = struct.unpack_from('<h', payload, 4)[0] * ACCEL_SCALE * 9.81
                            self._update_position(np.array([ax, ay, az]), dt)

                        self.info = f"Pos X={self._pos[0]:+.3f} Y={self._pos[1]:+.3f} Z={self._pos[2]:+.3f} m"
        except Exception as e:
            self.info = f"Error: {e}"


def make_sensor_mesh():
    box = o3d.geometry.TriangleMesh.create_box(1.2, 0.6, 0.2)
    box.translate([-0.6, -0.3, -0.1])
    box.paint_uniform_color([0.3, 0.6, 1.0])
    box.compute_vertex_normals()
    return box


def apply_pose(mesh, verts0, R, t):
    mesh.vertices = o3d.utility.Vector3dVector((R @ verts0.T).T + t)
    mesh.compute_vertex_normals()


def main():
    port = find_ch340_port() or input("Port: ").strip()
    print(f"Connecting: {port}\n九轴模块，位移由加速度积分，会有漂移")

    imu = IMUReader()
    imu.start(port)
    time.sleep(0.5)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="IMU 9-Axis", width=1100, height=850)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(world_frame)

    sensor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)
    vis.add_geometry(sensor_frame)
    sf_v0 = np.asarray(sensor_frame.vertices).copy()

    box = make_sensor_mesh()
    vis.add_geometry(box)
    box_v0 = np.asarray(box.vertices).copy()

    traj = o3d.geometry.PointCloud()
    vis.add_geometry(traj)
    traj_pts = []

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = [0.08, 0.08, 0.08]

    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.8, -0.5])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])

    prev_info = ""
    t_acc = 0.0

    while vis.poll_events():
        R, pos = imu.state

        apply_pose(sensor_frame, sf_v0, R, pos)
        apply_pose(box, box_v0, R, pos)
        vis.update_geometry(sensor_frame)
        vis.update_geometry(box)

        t_acc += 0.02
        if t_acc >= 0.1:
            t_acc = 0
            traj_pts.append(pos.copy())
            if len(traj_pts) > 500:
                traj_pts.pop(0)
            if len(traj_pts) > 1:
                traj.points = o3d.utility.Vector3dVector(np.array(traj_pts))
                vis.update_geometry(traj)

        vis.update_renderer()

        if imu.info != prev_info:
            print(f"\r{imu.info}", end="", flush=True)
            prev_info = imu.info

        time.sleep(0.02)

    vis.destroy_window()
    imu.stop()


if __name__ == "__main__":
    main()