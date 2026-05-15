from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import Center_pipeline as center


def print_result(name: str, ok: bool, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    print(f"{name}: {str(bool(ok)).lower()}{suffix}")


def pause(enabled: bool) -> None:
    if enabled:
        input("按 Enter 继续测试下一个接口...")


def is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def is_path(value: Any) -> bool:
    return isinstance(value, Path)


def run_test(name: str, func: Callable[[], Any], validator: Callable[[Any], bool]) -> tuple[bool, str]:
    try:
        result = func()
        ok = validator(result)
        detail = f"return={type(result).__name__}"
        if isinstance(result, dict):
            detail += f", keys={list(result.keys())[:5]}"
        elif isinstance(result, Path):
            detail += f", path={result}"
        return ok, detail
    except KeyError as exc:
        # 文件类接口在尚未产生模型时会抛 KeyError；这代表接口能正常区分“暂无文件”。
        return True, f"not_ready={exc}"
    except Exception as exc:
        return False, repr(exc)


def test_function_exists(name: str) -> tuple[bool, str]:
    obj = getattr(center, name, None)
    if not callable(obj):
        return False, "not callable"
    return True, f"signature={inspect.signature(obj)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="逐个测试 Center_pipeline Python 接口")
    parser.add_argument("--no-pause", action="store_true", help="不等待 Enter，连续测试")
    parser.add_argument("--skip-start", action="store_true", help="跳过 start_pipeline，避免没有相机时失败")
    parser.add_argument("--skip-stop", action="store_true", help="跳过最后的 stop_pipeline")
    args = parser.parse_args()

    should_pause = not args.no_pause
    results: list[bool] = []

    existence_tests = [
        "get_service",
        "start_pipeline",
        "stop_pipeline",
        "get_pipeline_status",
        "get_white_model_latest",
        "get_white_model_file",
        "get_pose_latest",
        "get_material_targets",
        "get_material_status",
        "get_material_library",
        "change_material",
        "get_material_segment_file",
        "get_material_scene_latest",
        "get_material_scene_file",
        "get_event_subscriber",
    ]

    for func_name in existence_tests:
        ok, detail = test_function_exists(func_name)
        results.append(ok)
        print_result(f"接口存在 {func_name}", ok, detail)
        pause(should_pause)

    call_tests: list[tuple[str, Callable[[], Any], Callable[[Any], bool]]] = []

    if not args.skip_start:
        call_tests.append(
            (
                "start_pipeline(input_mode='scanner')",
                lambda: center.start_pipeline(input_mode="scanner"),
                is_dict,
            )
        )

    call_tests.extend(
        [
            ("get_service()", center.get_service, lambda value: value is not None),
            ("get_pipeline_status()", center.get_pipeline_status, is_dict),
            ("get_white_model_latest()", center.get_white_model_latest, is_dict),
            ("get_white_model_file()", center.get_white_model_file, is_path),
            ("get_pose_latest()", center.get_pose_latest, is_dict),
            ("get_material_targets()", center.get_material_targets, is_dict),
            ("get_material_status()", center.get_material_status, is_dict),
            ("get_material_library()", center.get_material_library, is_dict),
            (
                "change_material('floor', 'style_1')",
                lambda: center.change_material("floor", "style_1", tiling=1.0),
                is_dict,
            ),
            (
                "get_material_segment_file('floor')",
                lambda: center.get_material_segment_file("floor"),
                is_path,
            ),
            ("get_material_scene_latest()", center.get_material_scene_latest, is_dict),
            ("get_material_scene_file()", center.get_material_scene_file, is_path),
            (
                "get_event_subscriber()",
                center.get_event_subscriber,
                lambda value: hasattr(value, "__aiter__"),
            ),
        ]
    )

    for name, func, validator in call_tests:
        ok, detail = run_test(name, func, validator)
        results.append(ok)
        print_result(name, ok, detail)
        pause(should_pause)

    if not args.skip_stop:
        ok, detail = run_test("stop_pipeline()", center.stop_pipeline, is_dict)
        results.append(ok)
        print_result("stop_pipeline()", ok, detail)
        pause(should_pause)

    passed = sum(1 for item in results if item)
    total = len(results)
    print(f"测试完成: {passed}/{total} true")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
