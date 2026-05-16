导入总控模块                     import Center_pipeline as center
获取总控实例                     center.get_service()

启动建图                         center.start_pipeline(input_mode="scanner")
停止建图                         center.stop_pipeline()
查询整体状态                     center.get_pipeline_status()

查询最新白模 metadata             center.get_white_model_latest()
获取白模文件路径                  center.get_white_model_file(version=None)

查询最新位姿                     center.get_pose_latest()

查询可换材质目标                  center.get_material_targets()
查询材质状态                     center.get_material_status()
查询材质库                       center.get_material_library()
切换 segment 材质                 center.change_material(segment_key, material_id, tiling=1.0)
获取 segment 文件路径             center.get_material_segment_file(segment_key, version=None)
查询最新 Material scene metadata  center.get_material_scene_latest()
获取 Material scene 文件路径      center.get_material_scene_file(version=None)

订阅后台事件                     center.get_event_subscriber()

Django 使用示例：
import json
from django.http import JsonResponse, FileResponse
import Center_pipeline as center

def pipeline_status(request):
    return JsonResponse(center.get_pipeline_status())

def start_pipeline(request):
    return JsonResponse(center.start_pipeline())

def latest_white_model(request):
    return JsonResponse(center.get_white_model_latest())

def download_white_model(request):
    path = center.get_white_model_file()
    return FileResponse(open(path, "rb"), filename=path.name)

def material_targets(request):
    return JsonResponse(center.get_material_targets())

def change_material(request):
    data = json.loads(request.body.decode("utf-8"))
    result = center.change_material(
        data["segment_key"],
        data["material_id"],
        tiling=float(data.get("tiling", 1.0)),
    )
    return JsonResponse(result)

输出目录：
Pointcloud2mesh 白模              pointcloud2mesh/model/
Meshfix 实时修复结果              Meshfix/outputs_realtime/
Material 分区与换材质结果         Material/outputs/

坐标系：
白模接口使用 reconstruction_world。
pose 接口中的 world_to_camera 和 camera_to_world 也使用 reconstruction_world。
Django/前端做 AR/MR 对齐时，应让模型保持在 reconstruction_world，通过 center.get_pose_latest() 更新虚拟 camera，不要让模型粘在相机前方移动。

测试脚本：
python test_center_api.py             逐个测试所有 Python 接口，每个接口输出 true/false，并按 Enter 继续
python test_center_api.py --no-pause  连续测试，不等待 Enter
python test_center_api.py --skip-start 如果没有相机或不想启动建图，可跳过 start_pipeline
python test_center_api.py --skip-stop  测试结束后不调用 stop_pipeline

HTTP（Django，`login_required`，POST 需 CSRF token，与 `/api/center/start/` 一致）：
- `POST /api/center/resume/`：从 `hard_paused_lost` 进入 `recovering`，清零 CLF / GpuICP lost 计数，恢复跟踪与建图写入。
- `POST /api/center/reset-reconstruction/`：调用管线 `reset_reconstruction` 清空体积并重启 worker；必要时自动 `start_background`；成功后恢复状态为 `normal` 并清零 CLF。

`GET /api/center/status/` 扩展字段（不影响原有字段）：
- `center_recovery_state`: `normal` | `hard_paused_lost` | `recovering` | `recovery_failed`
- `consecutive_lost_frames`, `consecutive_success_frames`
- `clf_pause_threshold`, `clf_fail_threshold`, `recover_success_frames_threshold`
- `tracking_enabled`, `mapping_enabled`, `capture_enabled`
- `hint` / `message`（如恢复失败提示）
- `recommended_view_distance`：由当前 TSDF mesh 顶点包围估算的建议视距（米）

环境变量（可选）：
- `CENTER_PAUSE_CLF_THRESHOLD`（默认 30）：连续跟踪失败帧上限，触发 `hard_paused_lost`
- `CENTER_FAIL_AFTER_RESUME_CLF`（默认 200）：resume 后仍处于 `recovering` 时的失败帧上限 → `recovery_failed`
- `CENTER_RECOVER_SUCCESS_FRAMES`（默认 5）：`recovering` 下连续成功帧达到后回到 `normal`
