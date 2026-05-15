import argparse
import struct
import time
from collections import deque

import numpy as np

HEADER = b"\x7e\x23"
CMD_QUATERNION = 0x16

# ============================================================
# Rotation helpers
# ============================================================

def normalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn

def quaternion_wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm([w, x, y, z])
    if n <= 1e-12 or not np.isfinite(n):
        raise ValueError("Invalid quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

def rotation_to_euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    """
    Return roll/pitch/yaw degrees for R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    R = normalize_rotation(R)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])

def quat_angle_diff_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Quaternion angular difference in degrees.
    q and -q are treated as same rotation.
    """
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    d = abs(float(np.dot(q1, q2)))
    d = min(1.0, max(-1.0, d))
    angle = 2.0 * np.arccos(d)
    return float(np.degrees(angle))

def relative_rpy_deg(R_base: np.ndarray, R_now: np.ndarray) -> np.ndarray:
    """
    Relative rotation from base to now, expressed as:
        R_delta = R_base.T @ R_now
    """
    R_delta = normalize_rotation(R_base.T @ R_now)
    return rotation_to_euler_zyx_deg(R_delta)

def fmt_vec(v, nd=4) -> str:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.{nd}f}" for x in arr) + "]"

# ============================================================
# Serial frame reader
# ============================================================

class IMUSerialReader:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.05):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.ser = None
        self.buffer = bytearray()

    def open(self):
        if self.ser is not None:
            return

        try:
            import serial
        except Exception as exc:
            raise RuntimeError(
                "pyserial is required. Install it with: pip install pyserial"
            ) from exc

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            finally:
                self.ser = None
                self.buffer.clear()

    def read_sample(self, timeout_sec: float = 1.0):
        """
        Return:
            {
                "timestamp": perf_counter,
                "quat": np.ndarray shape (4,), wxyz,
                "raw_frame": bytes,
            }
        or None.
        """
        self.open()
        deadline = time.perf_counter() + timeout_sec

        while time.perf_counter() < deadline:
            waiting = getattr(self.ser, "in_waiting", 0) or 1
            chunk = self.ser.read(waiting)
            if chunk:
                self.buffer.extend(chunk)

            sample = self._pop_sample()
            if sample is not None:
                return sample

            time.sleep(0.001)

        return None

    def _pop_sample(self):
        while True:
            start = self.buffer.find(HEADER)
            if start < 0:
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                return None

            if start > 0:
                del self.buffer[:start]

            if len(self.buffer) < 4:
                return None

            frame_len = int(self.buffer[2])

            if frame_len < 5:
                del self.buffer[0]
                continue

            if len(self.buffer) < frame_len:
                return None

            frame = bytes(self.buffer[:frame_len])
            del self.buffer[:frame_len]

            if not self._valid_checksum(frame):
                print("[WARN] checksum mismatch, dropping frame")
                continue

            if frame[3] != CMD_QUATERNION:
                continue

            if len(frame) != 0x15:
                print(f"[WARN] unexpected quaternion frame length: {len(frame)}")
                continue

            quat = np.asarray(struct.unpack("<ffff", frame[4:20]), dtype=np.float64)
            norm = float(np.linalg.norm(quat))

            if norm <= 1e-12 or not np.isfinite(norm):
                print("[WARN] invalid quaternion, dropping frame")
                continue

            quat /= norm

            return {
                "timestamp": time.perf_counter(),
                "quat": quat,
                "raw_frame": frame,
            }

    @staticmethod
    def _valid_checksum(frame: bytes) -> bool:
        if len(frame) < 2:
            return False
        expected = sum(frame[:-1]) & 0xFF
        return expected == frame[-1]

# ============================================================
# Main test logic
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Live IMU quaternion test script for checking serial IMU motion."
    )
    parser.add_argument("--port", default="COM7", help="Serial port, e.g. COM7")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=60.0, help="Test duration in seconds")
    parser.add_argument("--print-interval", type=float, default=0.2, help="Print interval in seconds")
    parser.add_argument("--baseline-samples", type=int, default=10, help="Samples used to create baseline")
    parser.add_argument("--baseline-timeout", type=float, default=3.0)
    args = parser.parse_args()

    reader = IMUSerialReader(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
    )

    print("=" * 90)
    print("Live IMU Test")
    print("=" * 90)
    print(f"port              = {args.port}")
    print(f"baudrate          = {args.baudrate}")
    print(f"duration          = {args.duration}s")
    print(f"print_interval    = {args.print_interval}s")
    print(f"baseline_samples  = {args.baseline_samples}")
    print()
    print("Instructions:")
    print("  1. Keep the IMU/camera still while baseline is captured.")
    print("  2. After baseline, physically rotate the IMU/camera:")
    print("     - roll right")
    print("     - pitch up/down")
    print("     - yaw left/right")
    print("  3. Watch these fields:")
    print("     - quat_diff_from_base_deg")
    print("     - imu_to_world_delta_rpy")
    print("     - world_to_imu_delta_rpy")
    print("  4. If all values stay near 0 while you rotate 20~45 degrees, the IMU data is not changing.")
    print("=" * 90)
    print()

    try:
        reader.open()
    except Exception as exc:
        print(f"[ERROR] Failed to open serial port: {exc}")
        return

    # ------------------------------------------------------------
    # Capture baseline
    # ------------------------------------------------------------
    print("Capturing baseline... keep device still.")

    baseline_quats = []
    baseline_deadline = time.perf_counter() + args.baseline_timeout

    while len(baseline_quats) < args.baseline_samples and time.perf_counter() < baseline_deadline:
        sample = reader.read_sample(timeout_sec=0.2)
        if sample is None:
            continue

        q = sample["quat"]

        # Align signs with first quaternion
        if baseline_quats:
            if float(np.dot(q, baseline_quats[0])) < 0:
                q = -q

        baseline_quats.append(q)

    if not baseline_quats:
        print("[ERROR] No IMU quaternion received. Check port, baudrate, power, and protocol.")
        reader.close()
        return

    q_base = np.mean(np.stack(baseline_quats, axis=0), axis=0)
    q_base /= np.linalg.norm(q_base)

    R_base_imu_to_world = quaternion_wxyz_to_rotation(q_base)
    R_base_world_to_imu = R_base_imu_to_world.T

    print()
    print("Baseline captured:")
    print(f"  samples_used          = {len(baseline_quats)}")
    print(f"  baseline_q_wxyz       = {fmt_vec(q_base, 6)}")
    print(f"  baseline_norm         = {np.linalg.norm(q_base):.8f}")
    print(f"  baseline imu_to_world rpy = {fmt_vec(rotation_to_euler_zyx_deg(R_base_imu_to_world), 4)}")
    print(f"  baseline world_to_imu rpy = {fmt_vec(rotation_to_euler_zyx_deg(R_base_world_to_imu), 4)}")
    print()
    print("Now move/rotate the IMU/camera.")
    print("Press Ctrl+C to stop.")
    print("=" * 90)

    # ------------------------------------------------------------
    # Live loop
    # ------------------------------------------------------------
    start_time = time.perf_counter()
    last_print = 0.0
    last_q = None
    last_t = None

    quat_diff_window = deque(maxlen=100)
    sample_count = 0
    printed_count = 0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - start_time
            if elapsed > args.duration:
                break

            sample = reader.read_sample(timeout_sec=0.2)
            if sample is None:
                continue

            sample_count += 1
            t = sample["timestamp"]
            q = sample["quat"]

            if float(np.dot(q, q_base)) < 0:
                q = -q

            R_imu_to_world = quaternion_wxyz_to_rotation(q)
            R_world_to_imu = R_imu_to_world.T

            qdiff_base = quat_angle_diff_deg(q_base, q)
            quat_diff_window.append(qdiff_base)

            rpy_imu_to_world = rotation_to_euler_zyx_deg(R_imu_to_world)
            rpy_world_to_imu = rotation_to_euler_zyx_deg(R_world_to_imu)

            delta_imu_to_world = relative_rpy_deg(R_base_imu_to_world, R_imu_to_world)
            delta_world_to_imu = relative_rpy_deg(R_base_world_to_imu, R_world_to_imu)

            if last_q is not None:
                qdiff_last = quat_angle_diff_deg(last_q, q)
                dt = max(1e-9, t - last_t)
                approx_ang_speed = qdiff_last / dt
            else:
                qdiff_last = 0.0
                approx_ang_speed = 0.0

            last_q = q.copy()
            last_t = t

            if now - last_print >= args.print_interval:
                last_print = now
                printed_count += 1

                max_window = max(quat_diff_window) if quat_diff_window else 0.0

                print()
                print(f"[{elapsed:8.3f}s] sample={sample_count} printed={printed_count}")
                print(f"  q_wxyz                     = {fmt_vec(q, 6)}")
                print(f"  q_norm                     = {np.linalg.norm(q):.8f}")
                print(f"  quat_diff_from_base_deg    = {qdiff_base:.4f}")
                print(f"  quat_diff_from_prev_deg    = {qdiff_last:.4f}")
                print(f"  approx_ang_speed_deg_s     = {approx_ang_speed:.4f}")
                print(f"  max_diff_recent_deg        = {max_window:.4f}")
                print()
                print(f"  imu_to_world_rpy_deg       = {fmt_vec(rpy_imu_to_world, 4)}")
                print(f"  imu_to_world_delta_rpy     = {fmt_vec(delta_imu_to_world, 4)}")
                print()
                print(f"  world_to_imu_rpy_deg       = {fmt_vec(rpy_world_to_imu, 4)}")
                print(f"  world_to_imu_delta_rpy     = {fmt_vec(delta_world_to_imu, 4)}")
                print()
                print("  Interpretation:")
                if qdiff_base < 1.0:
                    print("    - Rotation from baseline is < 1 deg. IMU is almost unchanged.")
                elif qdiff_base < 5.0:
                    print("    - Rotation from baseline is small. Move more strongly for calibration.")
                else:
                    print("    - Rotation from baseline is significant. IMU is responding.")

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")

    finally:
        reader.close()

    print()
    print("=" * 90)
    print("Test finished.")
    print(f"Total samples read: {sample_count}")
    print("=" * 90)

if __name__ == "__main__":
    main()