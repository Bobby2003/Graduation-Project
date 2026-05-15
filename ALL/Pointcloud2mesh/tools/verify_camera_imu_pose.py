"""
Verify live camera orientation after applying IMU convention, R_cam_imu, and
initial alignment from the project config.

Run from the project root:
    python pointcloud2mesh/tools/verify_camera_imu_pose.py --port COM7 --baudrate 115200

Guided automatic validation:
    python pointcloud2mesh/tools/verify_camera_imu_pose.py --port COM7 --baudrate 115200 --guided-validate --enter-to-capture --save-report

In guided validation, each action captures its own pre-neutral reference and
is judged with local delta rotation vector:
    R_delta = R_pre_neutral_ref.T @ R_action_ref

Expected checks:
- Neutral: roll approx 0, pitch approx 0, yaw approx 0
- Roll right: roll should change clearly; pitch/yaw should not dominate
- Pitch up: pitch should change clearly; roll/yaw should not dominate
- Yaw right: yaw should change clearly; roll/pitch should not dominate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointcloud2mesh.config import PipelineConfig  # noqa: E402
from pointcloud2mesh.imu.serial_provider import SerialIMUProvider  # noqa: E402
from pointcloud2mesh.imu.world_initializer import (  # noqa: E402
    _normalize_rotation,
    quaternion_wxyz_to_rotation,
    rotation_to_euler_zyx_deg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify camera RPY after applying configured IMU-camera extrinsic."
    )
    parser.add_argument("--port", default=None, help="Serial port, e.g. COM7. Defaults to config.")
    parser.add_argument("--baudrate", type=int, default=None, help="Serial baudrate. Defaults to config.")
    parser.add_argument("--duration", type=float, default=60.0, help="Verification duration in seconds.")
    parser.add_argument("--print-interval", type=float, default=0.2, help="Print interval in seconds.")
    parser.add_argument("--guided-validate", action="store_true", help="Run guided automatic validation.")
    parser.add_argument("--stage-hold-sec", type=float, default=2.0, help="Sampling duration per stage.")
    parser.add_argument("--prep-sec", type=float, default=2.0, help="Preparation time before each stage.")
    parser.add_argument("--neutral-tol-deg", type=float, default=5.0, help="Neutral axis tolerance.")
    parser.add_argument("--dominance-ratio", type=float, default=1.5, help="Main axis dominance ratio.")
    parser.add_argument("--min-action-deg", type=float, default=10.0, help="Minimum main axis action.")
    parser.add_argument("--max-cross-deg", type=float, default=12.0, help="Maximum cross-axis motion.")
    parser.add_argument("--save-report", action="store_true", help="Save guided validation JSON report.")
    parser.add_argument(
        "--enter-to-capture",
        action="store_true",
        help="Wait for Enter before each guided validation stage starts sampling.",
    )
    parser.add_argument(
        "--pre-neutral-hold-sec",
        type=float,
        default=1.0,
        help="Sampling duration for each action's pre-neutral reference.",
    )
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=0.5,
        help="Delay after pressing Enter before guided sampling starts.",
    )
    return parser.parse_args()


def fmt_vec(v) -> str:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x: .4f}" for x in arr) + "]"


def fmt_mat(R) -> str:
    rows = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return "\n".join("  " + fmt_vec(row) for row in rows)


def to_list(value):
    return np.asarray(value, dtype=np.float64).tolist()


def load_imu_config():
    cfg = PipelineConfig()
    if not hasattr(cfg, "imu"):
        raise RuntimeError("PipelineConfig is missing imu config")

    imu_cfg = cfg.imu
    missing = [
        name
        for name in ("R_cam_imu", "quaternion_convention", "initial_alignment_mode")
        if not hasattr(imu_cfg, name)
    ]
    if missing:
        raise RuntimeError(f"IMU config missing fields: {', '.join(missing)}")

    return imu_cfg


def quaternion_to_world_imu(q: np.ndarray, convention: str) -> np.ndarray:
    R_from_quat = quaternion_wxyz_to_rotation(q)
    if convention == "imu_to_world":
        return R_from_quat
    if convention == "world_to_imu":
        return R_from_quat.T
    raise ValueError("quaternion_convention must be 'imu_to_world' or 'world_to_imu'")


def camera_rotation_from_quaternion(
    q_wxyz: np.ndarray,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
) -> np.ndarray:
    R_world_imu = quaternion_to_world_imu(q_wxyz, quaternion_convention)
    R_imu_cam = R_cam_imu.T
    return _normalize_rotation(R_world_imu @ R_imu_cam)


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


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 1e-12 or not np.isfinite(n):
        raise ValueError("Invalid quaternion")
    return q / n


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


def read_average_quaternion(provider, target_samples: int = 10, timeout_sec: float = 3.0) -> np.ndarray:
    deadline = time.perf_counter() + timeout_sec
    quats = []
    while len(quats) < target_samples and time.perf_counter() < deadline:
        sample = provider.read_quaternion_sample(timeout_sec=0.2)
        if sample is not None:
            quats.append(sample.quaternion_wxyz)
    if not quats:
        raise RuntimeError("No IMU quaternion received")
    return average_quaternions_wxyz(quats)


def make_reconstruction_alignment(
    imu_cfg,
    q_base: np.ndarray,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    initial_alignment_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    R_world_cam_base = camera_rotation_from_quaternion(
        q_base,
        R_cam_imu,
        quaternion_convention,
    )

    if initial_alignment_mode == "align_initial_camera_to_forward":
        R_recon_world = R_world_cam_base.T
    elif initial_alignment_mode == "imu_absolute_world":
        R_recon_world = _normalize_rotation(np.asarray(imu_cfg.R_reconstruction_imu_world, dtype=np.float64))
    elif initial_alignment_mode == "disabled":
        R_recon_world = np.eye(3, dtype=np.float64)
    else:
        raise RuntimeError(f"Unsupported initial_alignment_mode: {initial_alignment_mode}")

    return R_recon_world, R_world_cam_base


def aligned_camera_rpy_from_quaternion(
    q_wxyz: np.ndarray,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    R_recon_world: np.ndarray,
) -> np.ndarray:
    R_world_cam = camera_rotation_from_quaternion(q_wxyz, R_cam_imu, quaternion_convention)
    R_camera_aligned = _normalize_rotation(R_recon_world @ R_world_cam)
    return rotation_to_euler_zyx_deg(R_camera_aligned)


def aligned_camera_rotation_from_quaternion(
    q_wxyz: np.ndarray,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    R_recon_world: np.ndarray,
) -> np.ndarray:
    R_world_cam = camera_rotation_from_quaternion(q_wxyz, R_cam_imu, quaternion_convention)
    return _normalize_rotation(R_recon_world @ R_world_cam)


def print_countdown(seconds: float, message: str):
    print(message)
    remaining = float(seconds)
    while remaining > 0:
        print(f"  starting in {remaining:.1f}s")
        sleep_time = min(1.0, remaining)
        time.sleep(sleep_time)
        remaining -= sleep_time


def collect_stage_samples(
    provider,
    stage_name: str,
    hold_sec: float,
    print_interval: float,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    R_recon_world: np.ndarray,
) -> list[dict]:
    samples = []
    start_time = time.perf_counter()
    last_print = 0.0

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed > hold_sec:
            break

        sample = provider.read_quaternion_sample(timeout_sec=0.2)
        if sample is None:
            continue

        R_aligned = aligned_camera_rotation_from_quaternion(
            sample.quaternion_wxyz,
            R_cam_imu,
            quaternion_convention,
            R_recon_world,
        )
        rpy = rotation_to_euler_zyx_deg(R_aligned)
        samples.append({
            "rpy_deg": to_list(rpy),
            "R_aligned": to_list(R_aligned),
        })

        if now - last_print >= print_interval:
            last_print = now
            print(f"  {stage_name} camera_rpy_aligned_deg = {fmt_vec(rpy)}")

    if not samples:
        print(f"[ERROR] No samples captured during stage {stage_name}.")
    return samples


def maybe_check_pre_stage_neutral(
    provider,
    args,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    R_recon_world: np.ndarray,
):
    print()
    print("Return to neutral before the next action.")
    if getattr(args, "enter_to_capture", False):
        print("Enter-to-capture mode: neutral check is only a prompt and will not sample before the action.")
        return
    samples = collect_stage_samples(
        provider,
        "pre_neutral_check",
        args.prep_sec,
        args.print_interval,
        R_cam_imu,
        quaternion_convention,
        R_recon_world,
    )
    if not samples:
        print("WARNING: could not check neutral before stage.")
        return

    mean_rpy = mean_rpy_from_samples(samples)
    if np.max(np.abs(mean_rpy)) > 10.0:
        print(
            "WARNING: detected that camera may not have returned to neutral; "
            "current stage result may be affected."
        )
        print(f"  pre_stage_neutral_mean_rpy_deg = {fmt_vec(mean_rpy)}")


def evaluated_samples(samples: list[dict]) -> list[dict]:
    if not samples:
        return []
    start = len(samples) // 2
    return samples[start:]


def mean_rpy_from_samples(samples: list[dict]) -> np.ndarray:
    arr = np.asarray([sample["rpy_deg"] for sample in samples], dtype=np.float64)
    return np.mean(arr, axis=0)


def std_rpy_from_samples(samples: list[dict]) -> np.ndarray:
    arr = np.asarray([sample["rpy_deg"] for sample in samples], dtype=np.float64)
    return np.std(arr, axis=0)


def mean_rotation_from_samples(samples: list[dict]) -> np.ndarray:
    mats = np.asarray([sample["R_aligned"] for sample in samples], dtype=np.float64)
    return _normalize_rotation(np.mean(mats, axis=0))


def evaluate_neutral_stage(samples: list[dict], neutral_tol_deg: float) -> dict:
    if not samples:
        return {
            "samples": [],
            "mean_rpy_deg": [0.0, 0.0, 0.0],
            "std_rpy_deg": [0.0, 0.0, 0.0],
            "neutral_ref_matrix": np.eye(3, dtype=np.float64).tolist(),
            "evaluated_sample_count": 0,
            "evaluated_sample_range": "last_half",
            "pass": False,
            "reasons": ["no samples captured"],
        }

    eval_samples = evaluated_samples(samples)
    mean_rpy = mean_rpy_from_samples(eval_samples)
    std_rpy = std_rpy_from_samples(eval_samples)
    neutral_reference_R = mean_rotation_from_samples(eval_samples)
    reasons = []
    for idx, axis_name in enumerate(("roll", "pitch", "yaw")):
        if abs(mean_rpy[idx]) > neutral_tol_deg:
            reasons.append(f"{axis_name} offset too large")
    if np.max(std_rpy) > 3.0:
        reasons.append("unstable during neutral capture")

    return {
        "samples": samples,
        "mean_rpy_deg": to_list(mean_rpy),
        "std_rpy_deg": to_list(std_rpy),
        "neutral_ref_matrix": to_list(neutral_reference_R),
        "evaluated_sample_count": len(eval_samples),
        "evaluated_sample_range": "last_half",
        "pass": not reasons,
        "reasons": reasons,
    }


def evaluate_motion_stage(
    stage_name: str,
    pre_neutral_samples: list[dict],
    action_samples: list[dict],
    expected_axis: int,
    min_action_deg: float,
    max_cross_deg: float,
    dominance_ratio: float,
) -> dict:
    axis_names = ("x", "y", "z")
    expected_axis_name = axis_names[expected_axis]

    if not pre_neutral_samples or not action_samples:
        return {
            "pre_neutral_samples": pre_neutral_samples,
            "pre_neutral_mean_rpy_deg": [0.0, 0.0, 0.0],
            "pre_neutral_ref_matrix": np.eye(3, dtype=np.float64).tolist(),
            "action_samples": action_samples,
            "action_mean_rpy_deg": [0.0, 0.0, 0.0],
            "action_ref_matrix": np.eye(3, dtype=np.float64).tolist(),
            "mean_rpy_deg": [0.0, 0.0, 0.0],
            "delta_rotvec_deg": [0.0, 0.0, 0.0],
            "expected_rotvec_axis": expected_axis_name,
            "main_abs": 0.0,
            "cross_max": 0.0,
            "dominance_ratio_actual": 0.0,
            "evaluated_sample_count": 0,
            "evaluated_sample_range": "last_half",
            "sign_warning": "no samples captured",
            "pass": False,
            "reasons": ["pre-neutral or action samples missing"],
        }

    pre_eval_samples = evaluated_samples(pre_neutral_samples)
    action_eval_samples = evaluated_samples(action_samples)
    pre_neutral_mean_rpy = mean_rpy_from_samples(pre_eval_samples)
    action_mean_rpy = mean_rpy_from_samples(action_eval_samples)
    pre_neutral_reference_R = mean_rotation_from_samples(pre_eval_samples)
    action_reference_R = mean_rotation_from_samples(action_eval_samples)
    R_delta = _normalize_rotation(pre_neutral_reference_R.T @ action_reference_R)
    delta_rotvec = rotation_matrix_to_rotvec_deg(R_delta)
    abs_rotvec = np.abs(delta_rotvec)
    main_abs = float(abs_rotvec[expected_axis])
    other_abs = np.delete(abs_rotvec, expected_axis)
    cross_max = float(np.max(other_abs)) if other_abs.size else 0.0
    dominance_ratio_actual = main_abs / max(cross_max, 1e-6)

    reasons = []
    if main_abs < min_action_deg:
        reasons.append("main motion too small")
    if cross_max > max_cross_deg:
        reasons.append("cross-axis motion too large")
    if main_abs < dominance_ratio * cross_max:
        other_names = [name for idx, name in enumerate(axis_names) if idx != expected_axis]
        reasons.append(f"expected {expected_axis_name} but {'/'.join(other_names)} dominated")

    expected_signs = {
        "roll_right": 1.0,
        "pitch_up": 1.0,
        "yaw_right": 1.0,
    }
    sign_warning = None
    expected_sign = expected_signs.get(stage_name)
    main_value = float(delta_rotvec[expected_axis])
    if expected_sign is not None and main_abs >= min_action_deg and main_value * expected_sign < 0.0:
        sign_warning = (
            "main axis is correct but sign may be opposite to the display convention; "
            "confirm with the main visualization."
        )

    return {
        "pre_neutral_samples": pre_neutral_samples,
        "pre_neutral_mean_rpy_deg": to_list(pre_neutral_mean_rpy),
        "pre_neutral_ref_matrix": to_list(pre_neutral_reference_R),
        "action_samples": action_samples,
        "action_mean_rpy_deg": to_list(action_mean_rpy),
        "action_ref_matrix": to_list(action_reference_R),
        "mean_rpy_deg": to_list(action_mean_rpy),
        "delta_rotvec_deg": to_list(delta_rotvec),
        "expected_rotvec_axis": expected_axis_name,
        "main_abs": main_abs,
        "cross_max": cross_max,
        "dominance_ratio_actual": dominance_ratio_actual,
        "evaluated_sample_count": len(action_eval_samples),
        "evaluated_sample_range": "last_half",
        "sign_warning": sign_warning,
        "pass": not reasons,
        "reasons": reasons,
    }


def print_validation_summary(
    configuration: dict,
    stages: dict,
    overall_motion_axis_pass: bool,
    overall_pass: bool,
):
    print()
    print("=" * 90)
    print("Auto validation summary")
    print("=" * 90)
    print("Configuration:")
    print("- R_cam_imu:")
    print(fmt_mat(configuration["R_cam_imu"]))
    print(f"- quaternion_convention: {configuration['quaternion_convention']}")
    print(f"- initial_alignment_mode: {configuration['initial_alignment_mode']}")

    neutral = stages["neutral"]
    print()
    print("Stage neutral:")
    print(f"- mean_rpy_deg = {fmt_vec(neutral['mean_rpy_deg'])}")
    print(f"- std_rpy_deg = {fmt_vec(neutral['std_rpy_deg'])}")
    print(f"- result = {'PASS' if neutral['pass'] else 'FAIL'}")
    if neutral["reasons"]:
        print(f"- reasons = {neutral['reasons']}")

    for stage_name in ("roll_right", "pitch_up", "yaw_right"):
        stage = stages[stage_name]
        print()
        print(f"Stage {stage_name}:")
        print(f"- pre_neutral_mean_rpy_deg = {fmt_vec(stage['pre_neutral_mean_rpy_deg'])}")
        print(f"- action_mean_rpy_deg = {fmt_vec(stage['action_mean_rpy_deg'])}")
        print(f"- mean_rpy_deg = {fmt_vec(stage['mean_rpy_deg'])}")
        print(f"- delta_rotvec_deg = {fmt_vec(stage['delta_rotvec_deg'])}")
        print(f"- expected_rotvec_axis = {stage['expected_rotvec_axis']}")
        print(f"- main_abs = {stage['main_abs']:.4f}")
        print(f"- cross_max = {stage['cross_max']:.4f}")
        print(f"- dominance_ratio_actual = {stage['dominance_ratio_actual']:.4f}")
        print(f"- evaluated_sample_count = {stage['evaluated_sample_count']}")
        print(f"- evaluated_sample_range = {stage['evaluated_sample_range']}")
        if stage["sign_warning"]:
            print(f"- sign_check = warning: {stage['sign_warning']}")
        else:
            print("- sign_check = ok")
        print(f"- result = {'PASS' if stage['pass'] else 'FAIL'}")
        if stage["reasons"]:
            print(f"- reasons = {stage['reasons']}")

    print()
    print(f"Overall motion axis result: {'PASS' if overall_motion_axis_pass else 'FAIL'}")
    print(f"Overall result: {'PASS' if overall_pass else 'FAIL'}")
    if overall_motion_axis_pass and not stages["neutral"]["pass"]:
        print("Overall result: FAIL due to neutral offset")
    failed = [name for name, stage in stages.items() if not stage["pass"]]
    if failed:
        print(f"Failed stages: {failed}")


def save_validation_report(
    configuration: dict,
    validation_parameters: dict,
    stages: dict,
    overall_motion_axis_pass: bool,
    overall_pass: bool,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = PROJECT_ROOT / "pointcloud2mesh" / "log" / f"imu_camera_pose_validation_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "configuration": {
            "R_cam_imu": to_list(configuration["R_cam_imu"]),
            "quaternion_convention": configuration["quaternion_convention"],
            "initial_alignment_mode": configuration["initial_alignment_mode"],
        },
        "validation_parameters": validation_parameters,
        "stages": stages,
        "overall_motion_axis_pass": overall_motion_axis_pass,
        "overall_pass": overall_pass,
    }

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        print(f"WARNING: failed to save validation report: {exc}")
        return

    print(f"Saved validation report: {report_path}")


def run_guided_validation(
    provider,
    args,
    imu_cfg,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    initial_alignment_mode: str,
    configuration: dict,
):
    print()
    print("Guided auto validation mode")
    if not args.enter_to_capture:
        print("Tip: use --enter-to-capture for cleaner validation after each pose is already held.")
    print("Keep the camera neutral and still while the initial alignment baseline is captured.")
    if args.enter_to_capture:
        input("Keep neutral still, then press Enter to capture initial alignment baseline...")
        time.sleep(args.settle_sec)
    else:
        print_countdown(args.prep_sec, "Preparing to capture initial alignment baseline...")

    try:
        q_base = read_average_quaternion(provider, target_samples=10, timeout_sec=max(3.0, args.prep_sec + 1.0))
        R_recon_world, R_world_cam_base = make_reconstruction_alignment(
            imu_cfg,
            q_base,
            R_cam_imu,
            quaternion_convention,
            initial_alignment_mode,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to capture initial alignment baseline: {exc}")
        print("Check that the IMU is outputting quaternion frames and the serial port is not occupied.")
        return

    print(f"baseline_q_wxyz = {fmt_vec(q_base)}")
    print(f"baseline camera_rpy_raw_deg = {fmt_vec(rotation_to_euler_zyx_deg(R_world_cam_base))}")

    print()
    print("Stage 1: global neutral")
    if args.enter_to_capture:
        input("请保持相机水平朝前并静止，按 Enter 捕获 global neutral...")
        time.sleep(args.settle_sec)
    else:
        print("请保持相机水平朝前并静止")
        print_countdown(args.prep_sec, "Neutral stage preparation...")
    neutral_samples = collect_stage_samples(
        provider,
        "neutral",
        args.stage_hold_sec,
        args.print_interval,
        R_cam_imu,
        quaternion_convention,
        R_recon_world,
    )

    stage_specs = [
        (
            "roll_right",
            "Stage 2: roll_right",
            "请做 roll_right：相机右侧向下滚转约 20~30 度并保持，不要向右转头，不要抬头/低头，按 Enter 捕获动作",
            "请回到 neutral：相机水平朝前并静止，按 Enter 捕获 roll_right 的 pre-neutral",
            2,
        ),
        (
            "pitch_up",
            "Stage 3: pitch_up",
            "请做 pitch_up：镜头向上抬约 20~30 度，左右不要歪，不要向左/右转头，按 Enter 捕获动作",
            "请回到 neutral：相机水平朝前并静止，按 Enter 捕获 pitch_up 的 pre-neutral",
            0,
        ),
        (
            "yaw_right",
            "Stage 4: yaw_right",
            "请做 yaw_right：保持相机水平，镜头向右转约 20~30 度，不要抬头/低头，不要滚转，按 Enter 捕获动作",
            "请回到 neutral：相机水平朝前并静止，按 Enter 捕获 yaw_right 的 pre-neutral",
            1,
        ),
    ]

    raw_stage_samples = {"neutral": neutral_samples}
    for stage_name, title, action_prompt, pre_neutral_prompt, _expected_axis in stage_specs:
        print()
        print(title)
        if args.enter_to_capture:
            input(pre_neutral_prompt + "...")
            time.sleep(args.settle_sec)
        else:
            print(pre_neutral_prompt)
            print_countdown(args.prep_sec, f"{stage_name} pre-neutral preparation...")
        pre_neutral_samples = collect_stage_samples(
            provider,
            f"{stage_name}_pre_neutral",
            args.pre_neutral_hold_sec,
            args.print_interval,
            R_cam_imu,
            quaternion_convention,
            R_recon_world,
        )

        if args.enter_to_capture:
            input(action_prompt + "...")
            time.sleep(args.settle_sec)
        else:
            print(action_prompt)
            print_countdown(args.prep_sec, f"{stage_name} action preparation...")
        action_samples = collect_stage_samples(
            provider,
            stage_name,
            args.stage_hold_sec,
            args.print_interval,
            R_cam_imu,
            quaternion_convention,
            R_recon_world,
        )
        raw_stage_samples[stage_name] = {
            "pre_neutral": pre_neutral_samples,
            "action": action_samples,
        }

    stages = {
        "neutral": evaluate_neutral_stage(neutral_samples, args.neutral_tol_deg),
    }
    for stage_name, _title, _action_prompt, _pre_neutral_prompt, expected_axis in stage_specs:
        stages[stage_name] = evaluate_motion_stage(
            stage_name,
            raw_stage_samples[stage_name]["pre_neutral"],
            raw_stage_samples[stage_name]["action"],
            expected_axis,
            args.min_action_deg,
            args.max_cross_deg,
            args.dominance_ratio,
        )

    overall_motion_axis_pass = all(stages[name]["pass"] for name in ("roll_right", "pitch_up", "yaw_right"))
    overall_pass = stages["neutral"]["pass"] and overall_motion_axis_pass
    print_validation_summary(configuration, stages, overall_motion_axis_pass, overall_pass)

    if args.save_report:
        validation_parameters = {
            "prep_sec": float(args.prep_sec),
            "stage_hold_sec": float(args.stage_hold_sec),
            "neutral_tol_deg": float(args.neutral_tol_deg),
            "neutral_std_tol_deg": 3.0,
            "dominance_ratio": float(args.dominance_ratio),
            "min_action_deg": float(args.min_action_deg),
            "max_cross_deg": float(args.max_cross_deg),
            "enter_to_capture": bool(args.enter_to_capture),
            "pre_neutral_hold_sec": float(args.pre_neutral_hold_sec),
            "settle_sec": float(args.settle_sec),
            "motion_evaluation": "local_delta_rotvec",
            "evaluated_sample_range": "last_half",
        }
        save_validation_report(
            configuration,
            validation_parameters,
            stages,
            overall_motion_axis_pass,
            overall_pass,
        )


def run_live_display(
    provider,
    args,
    imu_cfg,
    R_cam_imu: np.ndarray,
    quaternion_convention: str,
    initial_alignment_mode: str,
):
    print()
    print("Capturing baseline from first IMU quaternion. Keep camera neutral and still...")
    baseline_sample = provider.read_quaternion_sample(timeout_sec=3.0)
    if baseline_sample is None:
        print("[ERROR] No IMU quaternion received.")
        print("Check that the IMU is outputting quaternion frames and the serial port is not occupied.")
        return

    q_base = baseline_sample.quaternion_wxyz
    try:
        R_recon_world, R_world_cam_base = make_reconstruction_alignment(
            imu_cfg,
            q_base,
            R_cam_imu,
            quaternion_convention,
            initial_alignment_mode,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"baseline_q_wxyz = {fmt_vec(q_base)}")
    print(f"baseline camera_rpy_raw_deg = {fmt_vec(rotation_to_euler_zyx_deg(R_world_cam_base))}")
    print("Now rotate the camera. Press Ctrl+C to stop.")

    start_time = time.perf_counter()
    last_print = 0.0
    sample_count = 0

    while True:
        now = time.perf_counter()
        elapsed = now - start_time
        if elapsed > args.duration:
            break

        sample = provider.read_quaternion_sample(timeout_sec=0.2)
        if sample is None:
            continue

        sample_count += 1
        camera_rpy_aligned = aligned_camera_rpy_from_quaternion(
            sample.quaternion_wxyz,
            R_cam_imu,
            quaternion_convention,
            R_recon_world,
        )

        if now - last_print >= args.print_interval:
            last_print = now
            print()
            print(f"[{elapsed:8.3f}s] sample={sample_count}")
            print(f"  q_wxyz                 = {fmt_vec(sample.quaternion_wxyz)}")
            print(f"  camera_rpy_aligned_deg = {fmt_vec(camera_rpy_aligned)}")


def main():
    args = parse_args()

    try:
        imu_cfg = load_imu_config()
    except Exception as exc:
        print(f"[ERROR] Failed to load IMU config: {exc}")
        return

    port = args.port if args.port is not None else imu_cfg.serial_port
    baudrate = int(args.baudrate if args.baudrate is not None else imu_cfg.serial_baudrate)
    R_cam_imu = _normalize_rotation(np.asarray(imu_cfg.R_cam_imu, dtype=np.float64))
    quaternion_convention = str(imu_cfg.quaternion_convention).lower().strip()
    initial_alignment_mode = str(imu_cfg.initial_alignment_mode).lower().strip()

    print("=" * 90)
    print("Camera IMU Pose Verification")
    print("=" * 90)
    print(f"port = {port}")
    print(f"baudrate = {baudrate}")
    print(f"duration = {args.duration}s")
    print(f"print_interval = {args.print_interval}s")
    print()
    print("Configured R_cam_imu:")
    print(fmt_mat(R_cam_imu))
    print(f"quaternion_convention = {quaternion_convention}")
    print(f"initial_alignment_mode = {initial_alignment_mode}")
    print()
    print("Expected checks:")
    print("- Neutral: roll approx 0, pitch approx 0, yaw approx 0")
    print("- Roll right: roll should change clearly; pitch/yaw should not dominate")
    print("- Pitch up: pitch should change clearly; roll/yaw should not dominate")
    print("- Yaw right: yaw should change clearly; roll/pitch should not dominate")
    print("=" * 90)

    provider = SerialIMUProvider(
        port=port,
        baudrate=baudrate,
        timeout=float(getattr(imu_cfg, "serial_timeout_sec", 0.05)),
        read_window_sec=0.5,
    )

    try:
        provider.open()
    except Exception as exc:
        print(f"[ERROR] Failed to open serial port {port}: {exc}")
        print("Check COM port, baudrate, USB connection, and whether another process is using it.")
        return

    try:
        configuration = {
            "R_cam_imu": R_cam_imu,
            "quaternion_convention": quaternion_convention,
            "initial_alignment_mode": initial_alignment_mode,
        }
        if args.guided_validate:
            run_guided_validation(
                provider,
                args,
                imu_cfg,
                R_cam_imu,
                quaternion_convention,
                initial_alignment_mode,
                configuration,
            )
        else:
            run_live_display(
                provider,
                args,
                imu_cfg,
                R_cam_imu,
                quaternion_convention,
                initial_alignment_mode,
            )

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")
    finally:
        provider.close()


if __name__ == "__main__":
    main()
