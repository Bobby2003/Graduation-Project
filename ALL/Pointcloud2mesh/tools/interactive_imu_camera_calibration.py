"""
Interactive IMU-to-camera extrinsic calibration helper.

Run from the project root:

Manual mode:
    python pointcloud2mesh/tools/interactive_imu_camera_calibration.py --manual

Serial mode:
    python pointcloud2mesh/tools/interactive_imu_camera_calibration.py --serial COM7 --baudrate 115200

Serial mode uses live guided capture by default. It prints motion from neutral
until the motion is large and stable enough to capture.

Reuse saved quaternions:
    python pointcloud2mesh/tools/interactive_imu_camera_calibration.py \
        --quat_neutral W X Y Z \
        --quat_roll_right W X Y Z \
        --quat_pitch_up W X Y Z \
        --quat_yaw_right W X Y Z \
        --convention imu_to_world

The script asks you to capture IMU quaternions in w x y z order for:
    Step 0: neutral
    Step 1: roll_right
    Step 2: pitch_up
    Step 3: yaw_right

It compares candidate R_cam_imu matrices by checking whether each known user
motion mainly appears on the expected OpenCV camera-local rotation-vector axis:
roll_right -> z, pitch_up -> x, yaw_right -> y. RPY is still printed for
manual reference.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINTCLOUD2MESH_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud2mesh.imu.world_initializer import (  # noqa: E402
    _normalize_rotation,
    quaternion_wxyz_to_rotation,
    rotation_to_euler_zyx_deg,
)

try:
    from pointcloud2mesh.imu.serial_provider import SerialIMUProvider  # noqa: E402
except Exception:
    SerialIMUProvider = None


@dataclass(frozen=True)
class Candidate:
    name: str
    R_cam_imu: np.ndarray
    alias_of: Optional[str] = None


@dataclass(frozen=True)
class CalibrationStep:
    key: str
    title: str
    instructions: tuple[str, ...]


STEPS = [
    CalibrationStep(
        key="neutral",
        title="Step 0: neutral",
        instructions=(
            "Place the camera level.",
            "Point the lens toward the real-world forward direction.",
            "Keep it still, then press Enter to capture neutral.",
        ),
    ),
    CalibrationStep(
        key="roll_right",
        title="Step 1: roll_right",
        instructions=(
            "Keep the lens roughly pointing forward.",
            "Roll the camera clockwise, so the camera right side moves downward.",
            "About 20 to 45 degrees is enough, then press Enter.",
        ),
    ),
    CalibrationStep(
        key="pitch_up",
        title="Step 2: pitch_up",
        instructions=(
            "Return close to neutral.",
            "Pitch the camera lens upward.",
            "About 20 to 45 degrees is enough, then press Enter.",
        ),
    ),
    CalibrationStep(
        key="yaw_right",
        title="Step 3: yaw_right",
        instructions=(
            "Return close to neutral.",
            "Yaw the camera to the right, so the lens points right.",
            "About 20 to 45 degrees is enough, then press Enter.",
        ),
    ),
]


CANDIDATES = [
    Candidate(
        name="identity",
        R_cam_imu=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ),
    Candidate(
        name="imu_x_forward_y_right_z_down_to_cam_x_right_y_down_z_forward",
        R_cam_imu=np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    Candidate(
        name="imu_x_forward_y_left_z_up_to_cam_x_right_y_down_z_forward",
        R_cam_imu=np.asarray(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    Candidate(
        name="imu_x_right_y_forward_z_up_to_cam_x_right_y_down_z_forward",
        R_cam_imu=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
    Candidate(
        name="imu_x_right_y_down_z_forward_to_cam_x_right_y_down_z_forward",
        alias_of="identity",
        R_cam_imu=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    ),
    Candidate(
        name="imu_x_forward_y_left_z_up_to_cam_x_right_y_down_z_forward_opencv",
        alias_of="imu_x_forward_y_left_z_up_to_cam_x_right_y_down_z_forward",
        R_cam_imu=np.asarray(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ),
]


EXPECTED_ACTION_AXES = {
    "roll_right": 0,
    "pitch_up": 1,
    "yaw_right": 2,
}

ROTVEC_EXPECTED_ACTION_AXES = {
    "roll_right": 2,
    "pitch_up": 0,
    "yaw_right": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively calibrate candidate R_cam_imu matrices using known camera motions."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--manual",
        action="store_true",
        help="Manually enter quaternions as: w x y z.",
    )
    mode_group.add_argument(
        "--serial",
        metavar="PORT",
        help="Read quaternions from the IMU serial port, for example COM7.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Serial baudrate. Default: 115200.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        help="Serial read timeout per capture in seconds. Default: 0.5.",
    )
    parser.add_argument(
        "--convention",
        choices=("imu_to_world", "world_to_imu"),
        default="imu_to_world",
        help="Quaternion convention. Default: imu_to_world.",
    )
    parser.add_argument(
        "--score-method",
        choices=("rotvec", "euler", "both"),
        default="rotvec",
        help="Candidate sorting method. Default: rotvec.",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Enable live guided capture. Serial mode enables this by default.",
    )
    parser.add_argument(
        "--min-motion-deg",
        type=float,
        default=15.0,
        help="Minimum motion angle from neutral required for a valid action. Default: 15.0.",
    )
    parser.add_argument(
        "--capture-samples",
        type=int,
        default=10,
        help="Number of quaternion samples to average per capture. Default: 10.",
    )
    parser.add_argument(
        "--stable-window-sec",
        type=float,
        default=0.5,
        help="Window duration for stability check. Default: 0.5.",
    )
    parser.add_argument(
        "--stable-max-diff-deg",
        type=float,
        default=1.5,
        help="Maximum quaternion change inside the stability window. Default: 1.5.",
    )
    parser.add_argument(
        "--live-print-interval",
        type=float,
        default=0.2,
        help="Interval for live guided status printing. Default: 0.2.",
    )
    parser.add_argument("--quat_neutral", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--quat_roll_right", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--quat_pitch_up", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--quat_yaw_right", nargs=4, type=float, metavar=("W", "X", "Y", "Z"))
    return parser.parse_args()


def fmt_vec(v) -> str:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x: .6f}" for x in arr) + "]"


def fmt_mat(R) -> str:
    rows = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return "\n".join("  " + fmt_vec(row) for row in rows)


def to_list(value):
    return np.asarray(value, dtype=np.float64).tolist()


def average_quaternions_wxyz(quats) -> np.ndarray:
    quats = [normalize_quaternion(q) for q in quats]
    if not quats:
        raise ValueError("No quaternions to average")

    q_ref = quats[0]
    aligned = []
    for q in quats:
        if float(np.dot(q, q_ref)) < 0.0:
            q = -q
        aligned.append(q)

    q_avg = np.mean(np.asarray(aligned, dtype=np.float64), axis=0)
    n = np.linalg.norm(q_avg)
    if n <= 1e-12 or not np.isfinite(n):
        return q_ref.copy()
    return q_avg / n


def quaternion_diff_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = normalize_quaternion(q_a)
    qb = normalize_quaternion(q_b)
    dot = abs(float(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


def rotation_matrix_to_rotvec_deg(R: np.ndarray) -> np.ndarray:
    R = _normalize_rotation(R)
    cos_angle = (float(np.trace(R)) - 1.0) * 0.5
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = float(np.arccos(cos_angle))
    if abs(angle) < 1e-8:
        return np.zeros(3, dtype=np.float64)

    sin_angle = float(np.sin(angle))
    if abs(sin_angle) < 1e-8:
        return np.zeros(3, dtype=np.float64)

    axis = np.asarray(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sin_angle)
    return axis * np.degrees(angle)


def relative_rpy_for_convention(q_base: np.ndarray, q_now: np.ndarray, convention: str) -> np.ndarray:
    R_base = quaternion_to_world_imu(q_base, convention)
    R_now = quaternion_to_world_imu(q_now, convention)
    R_delta = _normalize_rotation(R_base.T @ R_now)
    return rotation_to_euler_zyx_deg(R_delta)


def print_relative_status(q_neutral: np.ndarray, q_now: np.ndarray, prefix: str = "live"):
    print(f"{prefix}:")
    print(f"  q_wxyz                     = {fmt_vec(q_now)}")
    print(f"  quat_diff_from_neutral_deg = {quaternion_diff_deg(q_neutral, q_now):.4f}")
    print(
        "  imu_to_world_delta_rpy     = "
        f"{fmt_vec(relative_rpy_for_convention(q_neutral, q_now, 'imu_to_world'))}"
    )
    print(
        "  world_to_imu_delta_rpy     = "
        f"{fmt_vec(relative_rpy_for_convention(q_neutral, q_now, 'world_to_imu'))}"
    )


def stdin_enter_pressed() -> bool:
    try:
        import msvcrt

        if not msvcrt.kbhit():
            return False
        char = msvcrt.getwch()
        if char == "\r":
            # Consume a following newline if the terminal provides one.
            if msvcrt.kbhit():
                next_char = msvcrt.getwch()
                if next_char not in ("\n", "\r"):
                    try:
                        msvcrt.ungetwch(next_char)
                    except Exception:
                        pass
            return True
        return False
    except Exception:
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            return False
        if readable:
            sys.stdin.readline()
            return True
        return False


def read_manual_quaternion(step: CalibrationStep) -> np.ndarray:
    while True:
        raw = input("Enter quaternion w x y z: ").strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != 4:
            print("Please enter exactly four numbers in w x y z order.")
            continue
        try:
            q = np.asarray([float(x) for x in parts], dtype=np.float64)
        except ValueError:
            print("Invalid number. Please try again.")
            continue
        return normalize_quaternion(q)


def read_serial_quaternion(provider, timeout_sec: float) -> np.ndarray:
    sample = provider.read_quaternion_sample(timeout_sec=timeout_sec)
    if sample is None:
        raise RuntimeError("Timed out while reading IMU quaternion from serial port.")
    return normalize_quaternion(sample.quaternion_wxyz)


def read_serial_average_quaternion(provider, timeout_sec: float, sample_count: int) -> np.ndarray:
    count = max(1, int(sample_count))
    samples = []
    while len(samples) < count:
        samples.append(read_serial_quaternion(provider, timeout_sec))
    return average_quaternions_wxyz(samples)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 1e-12 or not np.isfinite(n):
        raise ValueError("Invalid quaternion")
    return q / n


def quaternion_to_world_imu(q: np.ndarray, convention: str) -> np.ndarray:
    R_from_quat = quaternion_wxyz_to_rotation(q)
    if convention == "imu_to_world":
        return R_from_quat
    if convention == "world_to_imu":
        return R_from_quat.T
    raise ValueError("convention must be 'imu_to_world' or 'world_to_imu'")


def print_step_prompt(step: CalibrationStep):
    print()
    print(step.title)
    for instruction in step.instructions:
        print(f"  - {instruction}")


def stable_diff_deg(window) -> float:
    if len(window) < 2:
        return float("inf")
    samples = [item[1] for item in window]
    return max(quaternion_diff_deg(samples[0], q) for q in samples[1:])


def print_captured_motion(step_key: str, q_neutral: np.ndarray, q_captured: np.ndarray, min_motion_deg: float):
    motion_deg = quaternion_diff_deg(q_neutral, q_captured)
    print(f"Captured {step_key}:")
    print(f"  q_wxyz = {fmt_vec(q_captured)}")
    print(f"  quat_diff_from_neutral_deg = {motion_deg:.4f}")
    print(
        "  imu_to_world_delta_rpy = "
        f"{fmt_vec(relative_rpy_for_convention(q_neutral, q_captured, 'imu_to_world'))}"
    )
    print(
        "  world_to_imu_delta_rpy = "
        f"{fmt_vec(relative_rpy_for_convention(q_neutral, q_captured, 'world_to_imu'))}"
    )
    if step_key != "neutral" and motion_deg < min_motion_deg:
        print("WARNING: captured motion is too small. Calibration result may be invalid.")


def guided_capture_action(provider, step: CalibrationStep, q_neutral: np.ndarray, args) -> np.ndarray:
    print_step_prompt(step)
    print("Live guided mode:")
    print("  Rotate to the target angle, hold still, then press Enter to capture.")
    print("  Press Enter is accepted after warning even if motion is still small.")

    stability_window = deque()
    last_print_time = 0.0
    latest_q = None
    latest_motion = 0.0
    latest_stable_diff = float("inf")

    while True:
        q_now = read_serial_quaternion(provider, args.timeout)
        latest_q = q_now
        now = time.perf_counter()
        latest_motion = quaternion_diff_deg(q_neutral, q_now)
        stability_window.append((now, q_now))
        while stability_window and now - stability_window[0][0] > args.stable_window_sec:
            stability_window.popleft()
        latest_stable_diff = stable_diff_deg(stability_window)

        motion_ok = latest_motion >= args.min_motion_deg
        stable_ok = (
            len(stability_window) >= 2
            and now - stability_window[0][0] >= args.stable_window_sec * 0.8
            and latest_stable_diff <= args.stable_max_diff_deg
        )

        if now - last_print_time >= args.live_print_interval:
            print_relative_status(q_neutral, q_now)
            if not motion_ok:
                print("  status = motion too small, keep rotating")
            elif not stable_ok:
                print("  status = motion OK, waiting stable...")
            else:
                print(f"  status = motion is large and stable, press Enter to capture {step.key}")
            print(f"  stable_window_diff_deg     = {latest_stable_diff:.4f}")
            last_print_time = now

        if stdin_enter_pressed():
            if not motion_ok:
                print("WARNING: motion is still smaller than min-motion-deg.")
                answer = input("Capture anyway? [y/N]: ").strip().lower()
                if answer not in ("y", "yes"):
                    continue
            elif not stable_ok:
                print("WARNING: motion is not stable yet.")
                answer = input("Capture anyway? [y/N]: ").strip().lower()
                if answer not in ("y", "yes"):
                    continue
            break

    q_captured = read_serial_average_quaternion(provider, args.timeout, args.capture_samples)
    print_captured_motion(step.key, q_neutral, q_captured, args.min_motion_deg)
    return q_captured


def capture_quaternions_from_args(args) -> Optional[dict[str, np.ndarray]]:
    provided = {
        "neutral": args.quat_neutral,
        "roll_right": args.quat_roll_right,
        "pitch_up": args.quat_pitch_up,
        "yaw_right": args.quat_yaw_right,
    }
    count = sum(value is not None for value in provided.values())
    if count == 0:
        return None
    if count != len(provided):
        raise ValueError(
            "Provide all of --quat_neutral, --quat_roll_right, --quat_pitch_up, "
            "and --quat_yaw_right, or provide none of them."
        )
    return {key: normalize_quaternion(value) for key, value in provided.items()}


def capture_quaternions(args) -> dict[str, np.ndarray]:
    supplied_quaternions = capture_quaternions_from_args(args)
    if supplied_quaternions is not None:
        print("Using quaternions supplied from command line; skipping capture.")
        for key, q in supplied_quaternions.items():
            print(f"  {key}: {fmt_vec(q)}")
        return supplied_quaternions

    if not args.manual and not args.serial:
        raise ValueError("Use --manual, --serial PORT, or provide all --quat_* arguments.")

    provider = None
    guided = bool(args.guided or args.serial)
    if args.serial:
        if SerialIMUProvider is None:
            raise RuntimeError(
                "Could not import SerialIMUProvider. Use --manual, or run from the project root."
            )
        provider = SerialIMUProvider(
            port=args.serial,
            baudrate=args.baudrate,
            timeout=0.05,
            read_window_sec=args.timeout,
        )

    quaternions = {}
    try:
        if args.serial and guided:
            neutral_step = STEPS[0]
            print_step_prompt(neutral_step)
            input("Press Enter when neutral is steady to capture averaged samples...")
            quaternions["neutral"] = read_serial_average_quaternion(
                provider,
                args.timeout,
                args.capture_samples,
            )
            print(f"Captured neutral: {fmt_vec(quaternions['neutral'])}")
            print(
                "neutral imu_to_world rpy_deg = "
                f"{fmt_vec(rotation_to_euler_zyx_deg(quaternion_wxyz_to_rotation(quaternions['neutral'])))}"
            )
            print(
                "neutral world_to_imu rpy_deg = "
                f"{fmt_vec(rotation_to_euler_zyx_deg(quaternion_wxyz_to_rotation(quaternions['neutral']).T))}"
            )

            for step in STEPS[1:]:
                quaternions[step.key] = guided_capture_action(
                    provider,
                    step,
                    quaternions["neutral"],
                    args,
                )
        else:
            for step in STEPS:
                print_step_prompt(step)
                input("Press Enter when ready to capture...")
                if args.manual:
                    quaternions[step.key] = read_manual_quaternion(step)
                else:
                    quaternions[step.key] = read_serial_average_quaternion(
                        provider,
                        args.timeout,
                        args.capture_samples,
                    )
                print(f"Captured {step.key}: {fmt_vec(quaternions[step.key])}")
                if step.key != "neutral":
                    print_captured_motion(
                        step.key,
                        quaternions["neutral"],
                        quaternions[step.key],
                        args.min_motion_deg,
                    )
    finally:
        if provider is not None:
            provider.close()

    print()
    print("Optional Step 4: move_check")
    print("  After selecting R_cam_imu, run the main program and inspect pose logs.")
    print("  Move the camera forward/right/up and verify translation signs match expectations.")
    return quaternions


def score_delta_euler(delta_rpy: np.ndarray, main_axis: int) -> tuple[float, dict]:
    abs_rpy = np.abs(delta_rpy)
    main_abs = float(abs_rpy[main_axis])
    cross_abs = float(np.sum(abs_rpy) - main_abs)
    score = 0.0
    if main_abs >= 10.0:
        score += 2.0
    if main_abs > cross_abs:
        score += 2.0
    if cross_abs > 30.0:
        score -= 1.0
    if main_abs < 5.0:
        score -= 1.0
    return score, {
        "main_abs": main_abs,
        "cross_abs": cross_abs,
        "score": score,
    }


def score_delta_rotvec(rotvec: np.ndarray, main_axis: int) -> tuple[float, dict]:
    abs_rotvec = np.abs(rotvec)
    main_abs = float(abs_rotvec[main_axis])
    other = np.delete(abs_rotvec, main_axis)
    cross_abs = float(np.sum(other))
    max_other = float(np.max(other)) if other.size else 0.0

    score = 0.0
    if main_abs >= 10.0:
        score += 1.0
    if main_abs >= 15.0:
        score += 1.0
    if main_abs > cross_abs:
        score += 2.0
    if main_abs > 2.0 * max_other:
        score += 1.0
    if cross_abs > main_abs:
        score -= 1.0

    return score, {
        "main_abs": main_abs,
        "cross_abs": cross_abs,
        "max_other": max_other,
        "expected_axis_index": main_axis,
        "score": score,
    }


def evaluate_candidate(candidate: Candidate, world_imu_by_step: dict[str, np.ndarray]) -> dict:
    R_cam_imu = _normalize_rotation(candidate.R_cam_imu)
    R_imu_cam = R_cam_imu.T

    R_world_cam_by_step = {
        key: _normalize_rotation(R_world_imu @ R_imu_cam)
        for key, R_world_imu in world_imu_by_step.items()
    }
    R_world_cam_neutral = R_world_cam_by_step["neutral"]
    neutral_rpy = rotation_to_euler_zyx_deg(R_world_cam_neutral)

    rpy_deltas = {}
    rotvec_deltas = {}
    euler_score_details = {}
    rotvec_score_details = {}
    euler_score = 0.0
    rotvec_score = 0.0
    for action_key in EXPECTED_ACTION_AXES:
        R_delta_cam = _normalize_rotation(
            R_world_cam_neutral.T @ R_world_cam_by_step[action_key]
        )
        delta_rpy = rotation_to_euler_zyx_deg(R_delta_cam)
        delta_rotvec = rotation_matrix_to_rotvec_deg(R_delta_cam)

        action_euler_score, euler_details = score_delta_euler(
            delta_rpy,
            EXPECTED_ACTION_AXES[action_key],
        )
        action_rotvec_score, rotvec_details = score_delta_rotvec(
            delta_rotvec,
            ROTVEC_EXPECTED_ACTION_AXES[action_key],
        )
        euler_score += action_euler_score
        rotvec_score += action_rotvec_score
        rpy_deltas[action_key] = delta_rpy
        rotvec_deltas[action_key] = delta_rotvec
        euler_score_details[action_key] = euler_details
        rotvec_score_details[action_key] = rotvec_details

    return {
        "name": candidate.name,
        "alias_of": candidate.alias_of,
        "R_cam_imu": R_cam_imu,
        "neutral_rpy": neutral_rpy,
        "delta_roll_right_rpy": rpy_deltas["roll_right"],
        "delta_pitch_up_rpy": rpy_deltas["pitch_up"],
        "delta_yaw_right_rpy": rpy_deltas["yaw_right"],
        "delta_roll_right_rotvec_deg": rotvec_deltas["roll_right"],
        "delta_pitch_up_rotvec_deg": rotvec_deltas["pitch_up"],
        "delta_yaw_right_rotvec_deg": rotvec_deltas["yaw_right"],
        "camera_x_axis_in_world": R_world_cam_neutral[:, 0],
        "camera_y_axis_in_world": R_world_cam_neutral[:, 1],
        "camera_z_axis_in_world": R_world_cam_neutral[:, 2],
        "euler_score": euler_score,
        "rotvec_score": rotvec_score,
        "final_score": rotvec_score,
        "euler_score_details": euler_score_details,
        "rotvec_score_details": rotvec_score_details,
    }


def compute_sample_validity(quaternions: dict[str, np.ndarray], min_motion_deg: float) -> dict:
    q_neutral = quaternions["neutral"]
    validity = {
        "neutral_valid": True,
        "min_motion_deg": float(min_motion_deg),
    }
    all_valid = True
    for key in ("roll_right", "pitch_up", "yaw_right"):
        motion_deg = quaternion_diff_deg(q_neutral, quaternions[key])
        valid = motion_deg >= min_motion_deg
        validity[f"{key}_motion_deg"] = motion_deg
        validity[f"{key}_valid"] = valid
        all_valid = all_valid and valid
    validity["all_motions_valid"] = all_valid
    return validity


def print_sample_validity(sample_validity: dict):
    print()
    print("Sample validity:")
    for key in ("roll_right", "pitch_up", "yaw_right"):
        motion = sample_validity[f"{key}_motion_deg"]
        valid = sample_validity[f"{key}_valid"]
        print(f"  {key}: motion_deg={motion:.4f}, valid={valid}")
    if not sample_validity["all_motions_valid"]:
        print("Calibration samples are invalid because one or more motions are too small.")


def print_candidate_result(result: dict):
    alias_text = f" (alias of {result['alias_of']})" if result["alias_of"] else ""
    print("=" * 96)
    print(f"candidate name: {result['name']}{alias_text}")
    print("R_cam_imu:")
    print(fmt_mat(result["R_cam_imu"]))
    print(f"neutral R_world_cam rpy_deg: {fmt_vec(result['neutral_rpy'])}")
    print(f"roll_right delta rpy_deg: {fmt_vec(result['delta_roll_right_rpy'])}")
    print(f"pitch_up delta rpy_deg: {fmt_vec(result['delta_pitch_up_rpy'])}")
    print(f"yaw_right delta rpy_deg: {fmt_vec(result['delta_yaw_right_rpy'])}")
    print(f"roll_right delta rotvec_deg: {fmt_vec(result['delta_roll_right_rotvec_deg'])}")
    print(f"pitch_up delta rotvec_deg: {fmt_vec(result['delta_pitch_up_rotvec_deg'])}")
    print(f"yaw_right delta rotvec_deg: {fmt_vec(result['delta_yaw_right_rotvec_deg'])}")
    print(f"camera_x_axis_in_world: {fmt_vec(result['camera_x_axis_in_world'])}")
    print(f"camera_y_axis_in_world: {fmt_vec(result['camera_y_axis_in_world'])}")
    print(f"camera_z_axis_in_world: {fmt_vec(result['camera_z_axis_in_world'])}")
    print(f"euler_score: {result['euler_score']:.2f}")
    print(f"rotvec_score: {result['rotvec_score']:.2f}")
    print(f"final_score: {result['final_score']:.2f}")


def save_report(
    convention: str,
    quaternions: dict[str, np.ndarray],
    candidate_results: list[dict],
    recommendations: list[dict],
    sample_validity: dict,
    capture_settings: dict,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = POINTCLOUD2MESH_ROOT / "log" / f"imu_camera_calibration_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "quaternion_convention": convention,
        "raw_quaternions": {key: to_list(q) for key, q in quaternions.items()},
        "sample_validity": sample_validity,
        "capture_settings": capture_settings,
        "candidates": [
            {
                "name": result["name"],
                "alias_of": result["alias_of"],
                "R_cam_imu": to_list(result["R_cam_imu"]),
                "neutral_rpy": to_list(result["neutral_rpy"]),
                "delta_roll_right_rpy": to_list(result["delta_roll_right_rpy"]),
                "delta_pitch_up_rpy": to_list(result["delta_pitch_up_rpy"]),
                "delta_yaw_right_rpy": to_list(result["delta_yaw_right_rpy"]),
                "delta_roll_right_rotvec_deg": to_list(result["delta_roll_right_rotvec_deg"]),
                "delta_pitch_up_rotvec_deg": to_list(result["delta_pitch_up_rotvec_deg"]),
                "delta_yaw_right_rotvec_deg": to_list(result["delta_yaw_right_rotvec_deg"]),
                "camera_x_axis_in_world": to_list(result["camera_x_axis_in_world"]),
                "camera_y_axis_in_world": to_list(result["camera_y_axis_in_world"]),
                "camera_z_axis_in_world": to_list(result["camera_z_axis_in_world"]),
                "euler_score": result["euler_score"],
                "rotvec_score": result["rotvec_score"],
                "final_score": result["final_score"],
                "euler_score_details": result["euler_score_details"],
                "rotvec_score_details": result["rotvec_score_details"],
            }
            for result in candidate_results
        ],
        "recommendations": recommendations,
    }

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"WARNING: failed to save calibration report: {exc}")
        return
    print(f"Saved calibration report: {report_path}")


def main():
    args = parse_args()
    args.guided = bool(args.guided or args.serial)
    print("Interactive IMU camera calibration")
    print(f"quaternion_convention = {args.convention}")
    print("Quaternion order is always: w x y z")
    print()
    print("Expected OpenCV camera axes: x=right, y=down, z=forward.")
    print("OpenCV expected local axes:")
    print("  - roll_right should mainly affect rotvec z")
    print("  - pitch_up should mainly affect rotvec x")
    print("  - yaw_right should mainly affect rotvec y")
    print(f"score_method = {args.score_method}")

    quaternions = capture_quaternions(args)
    sample_validity = compute_sample_validity(quaternions, args.min_motion_deg)
    print_sample_validity(sample_validity)

    world_imu_by_step = {
        key: quaternion_to_world_imu(q, args.convention)
        for key, q in quaternions.items()
    }

    score_key = "euler_score" if args.score_method == "euler" else "rotvec_score"
    results = [evaluate_candidate(candidate, world_imu_by_step) for candidate in CANDIDATES]
    results.sort(key=lambda item: item[score_key], reverse=True)

    print()
    print("Candidate results")
    for result in results:
        print_candidate_result(result)

    recommendations = [
        {
            "rank": index + 1,
            "name": result["name"],
            "alias_of": result["alias_of"],
            "euler_score": result["euler_score"],
            "rotvec_score": result["rotvec_score"],
            "final_score": result["final_score"],
            "sort_score": result[score_key],
            "R_cam_imu": to_list(result["R_cam_imu"]),
        }
        for index, result in enumerate(results)
    ]

    print()
    if sample_validity["all_motions_valid"]:
        print("Recommended candidates by local rotation-vector score:")
        for item in recommendations[:3]:
            alias_text = f" (alias of {item['alias_of']})" if item["alias_of"] else ""
            print(
                f"{item['rank']}. name={item['name']}{alias_text}, "
                f"rotvec_score={item['rotvec_score']:.2f}, "
                f"euler_score={item['euler_score']:.2f}, "
                f"sort_score={item['sort_score']:.2f}"
            )
    else:
        print("Recommended candidates: not shown because sample motions are invalid.")

    if sample_validity["all_motions_valid"] and len(recommendations) >= 2:
        gap = recommendations[0]["sort_score"] - recommendations[1]["sort_score"]
        print()
        if gap >= 2.0:
            print(
                "If the top candidate's rotvec_score is clearly ahead and the three main "
                "rotvec components land on the expected local axes, copy its R_cam_imu "
                "into config.py IMUConfig.R_cam_imu."
            )
        else:
            print("Top scores are close. Possible reasons:")
            print("  - Re-capture cleaner motions.")
            print("  - Return close to neutral before each motion.")
            print("  - Keep each motion around 20 to 45 degrees, and avoid exceeding 60 degrees.")
            print("  - roll_right may have included yaw.")
            print("  - pitch_up may have included yaw.")
            print("  - yaw_right may have included roll/pitch.")
            print("  - The quaternion convention may be reversed.")
            print("  - Re-test with --convention world_to_imu.")

    capture_settings = {
        "guided": bool(args.guided),
        "min_motion_deg": float(args.min_motion_deg),
        "capture_samples": int(args.capture_samples),
        "stable_window_sec": float(args.stable_window_sec),
        "stable_max_diff_deg": float(args.stable_max_diff_deg),
        "live_print_interval": float(args.live_print_interval),
        "score_method": args.score_method,
    }
    save_report(
        args.convention,
        quaternions,
        results,
        recommendations,
        sample_validity,
        capture_settings,
    )


if __name__ == "__main__":
    main()
