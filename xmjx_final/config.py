from dataclasses import dataclass
from pathlib import Path

# 当前 config.py 所在目录。
BASE_DIR = Path(__file__).resolve().parent

@dataclass
class PipelineConfig:
    # ============================================================
    # 基础线程 / 队列参数
    # ============================================================

    # Tracking 线程每次循环末尾的休眠时间，单位毫秒。
    # 数值越小，tracking 轮询越频繁，实时性更强，但 CPU 占用也可能更高。
    # 一般保持 1~5ms 即可。
    tracking_sleep_ms: int = 1

    # Mapping 输入队列的最大缓存长度。
    # 当 tracking 产出过快、mapping 消费不过来时，旧数据会被丢弃或覆盖
    # （取决于你的 BoundedDropQueue 实现），用于防止系统延迟不断累积。
    # 值太小可能丢帧更多，值太大可能增加时延。
    mapping_queue_size: int = 4

    # 渲染线程目标刷新率，单位 FPS。
    # 仅在 enable_render=True 时生效。
    # 提高该值会让显示更流畅，但会增加渲染线程负担。
    render_fps: float = 20.0

    # ============================================================
    # 输入时间戳参数
    # ============================================================

    # 输入设备时间戳单位。
    # 可选值通常为：
    # - "auto" : 自动推断
    # - "sec"  : 秒
    # - "ms"   : 毫秒
    # - "us"   : 微秒
    device_timestamp_unit: str = "us"

    # 是否对输入帧做额外合法性检查。
    # 开启后通常会检查：
    # - RGB / depth 是否为空
    # - 分辨率是否匹配
    # - 数据类型是否合理
    # - 时间戳是否有效
    # 建议：联调阶段可开，稳定运行时关闭。
    validate_input_frame: bool = False

    # ============================================================
    # 模块功能开关
    # ============================================================

    # 是否启用 IMU 融合。
    # True  时会创建 IMUManager，并在 tracker 中参与旋转估计/初始化/回退。
    # False 时只走视觉相关流程。
    use_imu: bool = False

    # 是否启用实时渲染窗口。
    # True  时会启动 RenderWorker，并弹出 Open3D 可视化窗口。
    # False 时仅做 tracking / mapping，不显示三维窗口。
    # 调试算法时可以先关掉，减少干扰和资源占用。
    enable_render: bool = False

    # 是否启用控制台日志输出。
    # 这个开关主要传给 tracker / mapper / renderer 等模块，
    # 控制它们是否打印运行过程中的状态信息。
    enable_console_log: bool = True

    # ============================================================
    # 相机内参
    # ============================================================

    # 输入图像宽度，单位像素。
    width: int = 640

    # 输入图像高度，单位像素。
    height: int = 480

    # 相机焦距 fx（x 方向），单位像素。
    fx: float = 525.0

    # 相机焦距 fy（y 方向），单位像素。
    fy: float = 525.0

    # 相机主点 cx（x 坐标），单位像素。
    cx: float = 319.5

    # 相机主点 cy（y 坐标），单位像素。
    cy: float = 239.5

    # ============================================================
    # RGBD 输入预处理参数
    # ============================================================

    # 输入彩色图是否为 BGR 排列。
    # - True  : 输入是 OpenCV 常见的 BGR
    # - False : 输入已经是 RGB
    input_color_is_bgr: bool = False

    # 深度缩放因子。
    # 用于把原始 depth 数据转换为“米”或算法内部需要的尺度。
    # 当前为 None，表示交由下游逻辑自行判断或使用默认行为。
    depth_scale: float | None = None

    # 最大有效深度截断距离，单位米。
    # 超过该距离的深度值会被视为无效或不参与建图。
    # 值越大，远处场景保留更多，但噪声和误匹配也可能增加。
    depth_trunc: float = 3.0

    # ============================================================
    # Tracking 参数
    # ============================================================

    # tracking 输出到 mapping 的抽样步长。
    # 例如：
    # - 1 表示每帧都尝试送入 mapping
    # - 2 表示每 2 帧送 1 帧
    # 当 mapping 压力大时，可以适当增大，减少积分频率。
    mapping_stride: int = 1

    # 最少有效深度像素数量。
    # 当一帧中有效深度点太少时，通常说明画面信息不足、遮挡严重或数据异常，
    # 可直接跳过该帧，避免位姿估计不稳定。
    min_valid_pixels: int = 500

    # 单帧允许的最大旋转角度，单位度。
    # 如果视觉估计结果超过这个阈值，通常会被视为异常运动或错误匹配。
    # 值太小可能误杀快速运动，值太大可能放过错误估计。
    max_rotation_deg_per_frame: float = 20.0

    # 判断“本帧是否以旋转为主”的角度阈值，单位度。
    # 当旋转超过该值时，某些平移约束或融合逻辑可能会切换到更保守模式。
    rotation_dominant_angle_deg: float = 0.25

    # 当检测到“旋转占主导”时，允许的最大平移量，单位米。
    # 用来抑制“纯旋转时却估计出较大平移”的常见 VO 漂移问题。
    max_translation_when_rotating: float = 0.015

    # 位姿估计信息矩阵的最小 trace 阈值。
    # 可理解为“本次配准结果可信度”的一个下限。
    # 小于该值可能说明匹配不稳定、约束不足，结果应谨慎使用或直接拒绝。
    min_info_trace: float = 1e5

    # 单帧最小平移量，单位米。
    # 小于该值时，可认为平移几乎为 0，用于抑制噪声导致的微小抖动。
    min_translation_per_frame: float = 0.0008

    # 单帧最大平移量，单位米。
    # 超过该值通常视为异常估计，避免瞬间跳变。
    max_translation_per_frame: float = 0.05

    # 平移平滑系数，范围通常在 0~1。
    # 越接近 0：更平滑、更稳，但响应更慢。
    # 越接近 1：更灵敏，但更容易抖动。
    trans_smooth_alpha: float = 0.3

    # IMU 判定“存在足够旋转”的最小角度阈值，单位度。
    # 当 IMU 检测到旋转低于该阈值时，可能认为当前旋转信息不足，不参与某些修正。
    min_imu_rot_deg_per_frame: float = 0.30

    # 是否仅在 VO 初始化阶段使用 IMU。
    # True  : IMU 主要用于给视觉里程计提供初值，后续尽量依赖 VO
    # False : IMU 可持续参与后续估计
    imu_as_vo_init_only: bool = True

    # 当视觉里程计失败时，是否允许退回到 IMU 结果。
    # True  : VO 失败时可使用 IMU 维持姿态连续性
    # False : VO 一旦失败就不使用 IMU 兜底
    allow_imu_fallback_when_vo_fails: bool = True

    # 是否打印 tracking / odometry 的详细调试信息。
    # 开启后便于分析每帧位姿变化、判定逻辑、融合效果。
    # 正式长时间运行时日志会比较多。
    debug_print_odom: bool = True

    # ============================================================
    # Mapping / TSDF 参数
    # ============================================================

    # TSDF 体素边长，单位米。
    # 值越小，重建越精细，但内存和计算开销更大。
    # 值越大，模型更粗糙，但更省资源。
    voxel_length: float = 0.01

    # TSDF 截断距离，单位米。
    # 决定表面附近多远范围内的 SDF 会参与融合。
    # 通常与 voxel_length 成比例，过小容易缺面，过大容易糊。
    sdf_trunc: float = 0.05

    # 是否仅在“相机确实发生运动”时才积分当前帧。
    # True  : 静止帧不重复融合，可减少冗余和噪声叠加
    # False : 所有符合条件的帧都参与融合
    integrate_only_when_motion: bool = False

    # 允许进行积分的最小旋转量，单位度。
    # 如果旋转变化小于该值，可视为“基本没动”，不一定值得积分。
    min_integrate_rot_deg: float = 0.30

    # 允许进行积分的最大旋转量，单位度。
    # 若当前帧相对上一关键状态旋转过大，可能说明运动太快或估计不稳定，
    # 此时跳过积分可避免把错误姿态写入地图。
    max_integrate_rot_deg: float = 8.0

    # 允许进行积分的最小平移量，单位米。
    # 用于避免相机几乎没动时重复向 TSDF 写入高度相似的数据。
    min_integrate_trans_m: float = 0.0008

    # 每隔多少次积分/更新后重新提取一次 mesh。
    # 值越小，显示更新越及时，但 mesh 提取开销越高。
    mesh_update_interval: int = 5

    # mesh 顶点数少于该值时，不显示或不认为结果有效。
    # 用于过滤初始化阶段尚未形成有效模型的空壳 mesh。
    mesh_min_vertices_to_show: int = 10

    # 模型输出目录。
    # 最终导出的 obj / ply 等模型文件会默认保存到这里。
    # 当前设置为项目目录下的 model 文件夹。
    model_dir: str = str(BASE_DIR / "model")

    # ============================================================
    # Render 参数
    # ============================================================

    # 渲染窗口标题。
    render_window_name: str = "TSDF Fusion (Pipeline)"

    # 渲染窗口宽度，单位像素。
    render_width: int = 1280

    # 渲染窗口高度，单位像素。
    render_height: int = 720

    # 是否打印渲染线程状态日志。
    # 开启后会定期输出渲染循环状态、mesh 更新情况等调试信息。
    render_enable_status_log: bool = False

    # 渲染状态日志打印间隔。
    # 单位取决于 RenderWorker 内部实现，通常是“多少帧打印一次”或“多少次循环打印一次”。
    # 仅在 render_enable_status_log=True 时有意义。
    render_status_log_interval: int = 30

    # ============================================================
    # 日志参数
    # ============================================================

    # 是否将控制台输出额外保存到日志文件。
    # True  : 终端输出同时写入 log 文件
    # False : 只在终端显示，不写文件
    save_log: bool = True

    # 日志输出目录。
    # 默认保存在项目目录下的 log 文件夹。
    log_dir: str = str(BASE_DIR / "log")

    # 日志文件名前缀。
    # 实际生成的文件名通常类似：
    # log_20260414_192015.txt
    log_prefix: str = "log"