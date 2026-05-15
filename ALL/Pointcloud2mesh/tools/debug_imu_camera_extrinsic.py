import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud2mesh.imu.world_initializer import (  # noqa: E402
    _normalize_rotation,
    quaternion_wxyz_to_rotation,
    rotation_to_euler_zyx_deg,
)


DEFAULT_QUATERNION_WXYZ = np.asarray(
    [0.9668, -0.0224, 0.1554, 0.2016],
    dtype=np.float64,
)


CANDIDATES = [
    (
        "identity",
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        "imu_x_forward_y_right_z_down_to_cam_x_right_y_down_z_forward",
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        "imu_x_forward_y_left_z_up_to_cam_x_right_y_down_z_forward",
        np.asarray(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        "imu_x_right_y_forward_z_up_to_cam_x_right_y_down_z_forward",
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        "imu_x_right_y_down_z_forward_to_cam_x_right_y_down_z_forward",
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ),
    (
        "imu_x_forward_y_left_z_up_to_cam_x_right_y_down_z_forward_opencv",
        np.asarray(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Debug candidate fixed rotations R_cam_imu for IMU-to-camera "
            "extrinsic calibration. Quaternion order is w x y z."
        )
    )
    parser.add_argument(
        "--quat",
        nargs=4,
        type=float,
        metavar=("W", "X", "Y", "Z"),
        help="IMU quaternion in w x y z order.",
    )
    return parser.parse_args()


def fmt_vec(v) -> str:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x: .6f}" for x in arr) + "]"


def fmt_mat(R) -> str:
    rows = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return "\n".join("  " + fmt_vec(row) for row in rows)


def print_manual_calibration_notes():
    print("Manual calibration guide:")
    print("  1. Keep the camera lens pointing toward the real-world forward direction.")
    print("  2. Keep the camera level.")
    print("  3. Read one IMU quaternion in w x y z order.")
    print("  4. Run this script with --quat W X Y Z, or omit --quat to use the built-in sample.")
    print("  5. Compare camera_z_axis_in_imu_world with the real-world forward direction.")
    print("  6. Check whether camera_y_axis_in_imu_world matches your camera-down/world-up convention.")
    print()
    print(
        "Note: in align_initial_camera_to_forward mode, every candidate should end with "
        "R_recon_cam close to identity. Use the printed world-axis directions to choose "
        "the physically correct R_cam_imu."
    )
    print()


def main():
    args = parse_args()
    q = DEFAULT_QUATERNION_WXYZ if args.quat is None else np.asarray(args.quat, dtype=np.float64)
    q = q.reshape(4)
    q_norm = np.linalg.norm(q)
    if q_norm <= 1e-12 or not np.isfinite(q_norm):
        raise ValueError("Invalid quaternion")
    q = q / q_norm

    print("IMU camera extrinsic debug")
    print(f"quaternion_wxyz = {fmt_vec(q)}")
    print("quaternion_convention = imu_to_world")
    print()
    print_manual_calibration_notes()

    R_world_imu = quaternion_wxyz_to_rotation(q)
    print(f"R_world_imu rpy_deg = {fmt_vec(rotation_to_euler_zyx_deg(R_world_imu))}")
    print()

    for name, R_cam_imu_raw in CANDIDATES:
        R_cam_imu = _normalize_rotation(R_cam_imu_raw)
        R_imu_cam = R_cam_imu.T
        R_world_cam = _normalize_rotation(R_world_imu @ R_imu_cam)
        R_correction = _normalize_rotation(R_world_cam.T)
        R_recon_cam = _normalize_rotation(R_correction @ R_world_cam)

        print("=" * 88)
        print(f"candidate name: {name}")
        print("R_cam_imu:")
        print(fmt_mat(R_cam_imu))
        print(f"R_world_imu rpy_deg: {fmt_vec(rotation_to_euler_zyx_deg(R_world_imu))}")
        print(f"R_world_cam rpy_deg: {fmt_vec(rotation_to_euler_zyx_deg(R_world_cam))}")
        print(
            "align_initial_camera_to_forward correction rpy_deg: "
            f"{fmt_vec(rotation_to_euler_zyx_deg(R_correction))}"
        )
        print(f"final R_recon_cam rpy_deg: {fmt_vec(rotation_to_euler_zyx_deg(R_recon_cam))}")
        print(f"camera_x_axis_in_imu_world: {fmt_vec(R_world_cam[:, 0])}")
        print(f"camera_y_axis_in_imu_world: {fmt_vec(R_world_cam[:, 1])}")
        print(f"camera_z_axis_in_imu_world: {fmt_vec(R_world_cam[:, 2])}")


if __name__ == "__main__":
    main()
