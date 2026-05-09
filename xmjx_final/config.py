from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# 通用相机内参
# ============================================================
@dataclass
class CameraIntrinsics:
    """
    相机内参定义

    含义：
    - width, height: 图像分辨率
    - fx, fy: 焦距（像素单位）
    - cx, cy: 主点（像素坐标）

    注意：
    1. width / height 必须和实际输入图像尺寸一致
    2. 如果图像分辨率发生缩放，fx/fy/cx/cy 也应按比例缩放
    3. tracking_camera / mapping_camera 可以与 input_camera 不同，
       但其内参与对应分辨率必须匹配，否则会直接影响位姿估计或建图结果
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

# ============================================================
# 输入参数
# ============================================================
@dataclass
class InputConfig:
    """
    输入与预处理相关配置

    这些参数主要影响：
    - 输入帧时间戳解释方式
    - RGB/BGR 颜色通道解释方式
    - 深度图单位换算
    - 深度有效范围截断
    """

    # 输入设备时间戳单位
    device_timestamp_unit: str = "us"

    # 是否对输入帧做额外合法性校验
    # True:
    #   更安全，能更早发现输入异常；
    # False:
    #   开销更低，适合输入源稳定时使用
    validate_input_frame: bool = False

    # 输入彩色图是否为 BGR 排列
    # - OpenCV 默认常见为 BGR
    # - 某些设备或库输出可能本来就是 RGB
    # 如果设置错误，会导致颜色看起来不对，并可能影响某些依赖颜色的后续流程
    input_color_is_bgr: bool = False

    # 深度缩放因子
    # 用于把原始 depth 转成米（m）
    # 常见情况：
    # - 若深度本身已经是 float32 米单位，可保持 None
    # - 若深度是毫米整数，通常需要 scale=1000.0 或对应换算方式
    depth_scale: float | None = None

    # 深度截断距离（米）
    # 超过该距离的深度将被视为无效或被裁掉
    # 调大：
    # - 保留更远处深度
    # - 远距离噪声更多
    # 调小：
    # - 更关注近距离稳定区域
    # - 可减少远处噪声影响
    depth_trunc: float = 3.0

    # 是否只在启动初期打印一次时间戳调试信息
    # 用于确认输入时间戳单位是否正确
    print_timestamp_debug_once: bool = True

# ============================================================
# Tracking 参数
# ============================================================
@dataclass
class TrackingConfig:
    """
    跟踪模块（位姿估计）参数

    主要控制：
    - 跟踪线程轮询节奏
    - VO/ICP 对输入质量的要求
    - 单帧旋转/平移的接受范围
    - IMU 是否参与初始化 / 失败回退
    - profiling 与调试输出
    - 向 mapping 发送数据的频率
    """

    # tracking 线程空转时的 sleep 时间（毫秒）
    # 调大：
    # - 降低 CPU 占用
    # - 会增加一点响应延迟
    # 调小：
    # - 响应更及时
    # - CPU 占用可能上升
    sleep_ms: int = 1

    # 跟踪后端
    # 可选：
    # - "cpu_rgbd": 使用旧的 Open3D legacy RGBD odometry
    # - "gpu_icp" : 使用新增 CUDA Tensor ICP tracker
    tracking_backend: str = "gpu_icp"

    # 认为当前帧“可用于跟踪”的最少有效像素数
    # 调大：
    # - 对输入质量要求更高
    # - 能减少坏帧误跟踪
    # - 可能更容易丢帧
    # 调小：
    # - 更容易接受弱纹理/弱深度帧
    # - 误匹配风险上升
    min_valid_pixels: int = 500

    # 单帧允许的最大旋转角（度）
    # 若估计结果超过该值，通常会被判定为不可信或过猛运动
    # 调大：
    # - 对快速转动更宽容
    # - 更容易接纳异常旋转
    # 调小：
    # - 更保守，更稳
    # - 快速转动时更可能拒绝更新
    max_rotation_deg_per_frame: float = 15.0

    # 认为“旋转主导”的阈值角度（度）
    # 超过这个角度时，系统可能会用更保守的平移约束
    # 调小：
    # - 更容易进入“旋转主导”判定
    # - 有助于抑制纯旋转时的虚假平移
    # 调大：
    # - 只有较明显旋转才触发特殊处理
    rotation_dominant_angle_deg: float = 8.0

    # 在“旋转主导”情况下允许的最大平移（米）
    # 目的：
    # - 当相机主要在转动时，限制估计出的平移过大
    # - 抑制 VO/ICP 在纯旋转场景下产生的伪位移
    # 调小：
    # - 更保守，能更强抑制假平移
    # 调大：
    # - 对边转边移更宽容，更可能放过错误平移
    max_translation_when_rotating: float = 0.08

    # 信息矩阵 / Hessian / 对齐质量相关的最小阈值
    # 通常用于判定当前配准是否“信息量足够”
    # 调大：
    # - 只接受更高质量匹配
    # - 更稳，但更容易拒绝更新
    # 调小：
    # - 接受更多帧
    min_info_trace: float = 1e5

    # 单帧最小平移阈值（米）
    # 小于此值时，可能被认为“几乎没动”
    # 调大：
    # - 更不容易认定为有效位移，可抑制小噪声抖动
    # 调小：
    # - 对细微运动更敏感，更容易受到噪声影响
    min_translation_per_frame: float = 0.0001

    # 单帧最大平移阈值（米）
    # 超过该值时，可能被视为异常大跳变
    # 调大：
    # - 对快速移动更宽容，异常跳变更容易混入
    # 调小：
    # - 更保守，快速手持移动时可能频繁拒绝
    max_translation_per_frame: float = 0.10

    # 平移平滑系数，通常范围建议在 [0, 1]
    # 越接近 0：更平滑、更保守
    # 越接近 1：更相信当前帧估计、响应更快
    # 如果轨迹抖动明显，可适当减小
    # 如果跟踪反应太钝，可适当增大
    trans_smooth_alpha: float = 1.0

    # ========================================================
    # GPU ICP 后端参数
    # ========================================================

    # Open3D CUDA device
    gpu_device: str = "CUDA:0"

    # tracking 点云 voxel 下采样体素大小（米）
    # 调大：
    # - 点更少，速度更快
    # - 位姿精度下降
    # 调小：
    # - 点更多，位姿更细
    # - 速度变慢
    gpu_tracking_voxel_size: float = 0.04

    # tracking 点云从 depth 图生成时的 stride
    gpu_tracking_pcd_stride: int = 2

    # ICP 最大对应距离（米）
    # 过小：
    # - 快速运动时找不到对应
    # 过大：
    # - 错误对应变多
    gpu_icp_max_correspondence_distance: float = 0.08

    # ICP 迭代次数
    gpu_icp_max_iteration: int = 8

    # 法线估计半径（米）
    gpu_normal_radius: float = 0.08

    # 法线估计最多邻居数
    gpu_normal_max_nn: int = 12

    # ICP fitness 最小阈值
    # GPU ICP 没有旧 odometry 的 info matrix，所以用 fitness/rmse 判断质量
    gpu_min_icp_fitness: float = 0.12

    # ICP RMSE 最大阈值（米）
    gpu_max_icp_rmse: float = 0.055

    # point-to-plane 失败时，是否回退到 point-to-point
    gpu_allow_point_to_point_fallback: bool = True

    # ========================================================
    # Frame-to-model / LocalMap tracking 参数
    # ========================================================

    # 是否启用 current frame -> local map 的跟踪模式
    use_map_tracking: bool = True

    # local map 至少多少点以后才启用 frame-to-model
    map_tracking_min_points: int = 3000

    # frame-to-model ICP 最大对应距离
    map_tracking_max_correspondence_distance: float = 0.15

    # frame-to-model ICP 的质量门限
    map_min_icp_fitness: float = 0.22
    map_max_icp_rmse: float = 0.045

    # frame-to-model 单帧允许的最大相对位姿跳变
    map_max_delta_trans: float = 0.06
    map_max_delta_rot_deg: float = 10.0

    # local map ICP 失败后的冷却帧数，避免每帧重复尝试和刷屏
    map_icp_cooldown_after_fail: int = 10

    # lost 状态下，每隔多少帧尝试一次 local map relocalization
    relocalize_try_interval: int = 5

    # tracking lost 状态日志打印间隔
    lost_print_interval: int = 30

    # local map 至少多少点才允许作为 tracking target
    local_map_min_points_for_tracking: int = 1500

    # local map 下采样体素
    local_map_voxel_size: float = 0.025

    # local map 最大点数，太大会拖慢 ICP
    local_map_max_points: int = 100000

    # 只取相机附近多少米的局部地图参与 ICP
    local_map_radius: float = 3.0

    # tracking -> mapping 的送包步长，单位：帧
    # 调大：
    # - 降低 mapping 压力
    # - 建图更新更稀疏
    # 调小：
    # - mapping 更连续
    # - 计算压力更大
    # ----------------------------
    # 显著影响建图频率和队列压力
    # ----------------------------
    output_mapping_stride: int = 2

# ============================================================
# Mapping 参数
# ============================================================
@dataclass
class MappingConfig:
    """
    建图模块参数

    主要控制：
    - mapping 队列大小
    - TSDF 体素分辨率
    - 深度融合截断距离
    - 什么情况下允许 integrate
    - mesh 抽取频率与展示门槛
    - 模型输出目录
    """

    # mapping 输入队列大小
    # 调大：
    # - 更不容易因短时积压而丢包
    # 调小：
    # - 更实时
    queue_size: int = 4

    # TSDF 体素边长（米）
    # 调小：
    # - 模型更精细，内存与计算开销更高
    # 调大：
    # - 模型更粗，但速度更快、占用更低
    # 范围：0.005~0.02
    voxel_length: float = 0.02

    # TSDF 截断距离（米）
    # 一般建议与 voxel_length 保持合理比例
    # 截断太小：融合范围过窄，噪声敏感
    # 截断太大：边界会更厚、更糊
    sdf_trunc: float = 0.05

    # 是否仅在“检测到有足够运动”时才 integrate
    # True:
    # - 可减少重复融合、静止帧堆积
    # False:
    # - 每次都尽量融合
    integrate_only_when_motion: bool = True

    # 允许 integrate 的最小旋转角阈值（度）
    # 若旋转或平移至少有一个达到阈值，就认为有运动
    min_integrate_rot_deg: float = 0.3

    # 允许 integrate 的最大旋转角阈值（度）
    # 太大的旋转通常意味着当前帧变化过猛，可能不适合直接融合
    # 调大：
    # - 更允许快速转动时融合
    # 调小：
    # - 更保守，快速运动下融合会更少
    max_integrate_rot_deg: float = 6.0

    # 允许 integrate 的最小平移阈值（米）
    # 平移或旋转任一达到条件即触发融合
    min_integrate_trans_m: float = 0.0015

    # 抽取 mesh 的间隔
    # 调小：
    # - mesh 更新更频繁，显示更及时
    # 调大：
    # - 可视化模型刷新更慢
    mesh_update_interval: int = 5

    # 只有 mesh 顶点数超过该阈值时，才认为值得展示/更新
    # 用于过滤极小、无意义的初期 mesh
    mesh_min_vertices_to_show: int = 10

    # tracking lost 时 mapping 暂停日志打印间隔
    mapping_lost_print_interval: int = 30

    # Mapper 自己的位姿连续性保护。
    max_mapping_pose_jump_trans: float = 0.08
    max_mapping_pose_jump_rot_deg: float = 12.0

    # local_map 点云生成下采样体素
    local_map_frame_pcd_voxel_size: float = 0.03

    # local_map 点云生成 stride
    local_map_frame_pcd_stride: int = 4

    # 模型保存目录
    # 支持相对路径，主流程中会统一解析为绝对路径
    model_dir: str = str(BASE_DIR / "model")

# ============================================================
# Render 参数
# ============================================================
@dataclass
class RenderConfig:
    """
    渲染与显示相关参数
    """

    # 是否启用渲染线程
    # False:
    # - 不创建渲染流程
    # True:
    # - 启用可视化窗口与状态展示
    enabled: bool = True

    # 渲染帧率
    fps: float = 20.0

    # 窗口标题
    window_name: str = "TSDF Fusion (Pipeline)"

    # 渲染窗口尺寸
    # 只影响显示窗口，不影响实际 tracking/mapping 分辨率
    width: int = 1280
    height: int = 720

# ============================================================
# 日志参数
# ============================================================
@dataclass
class LogCategoryConfig:
    """
    单类日志配置。

    enabled:
        是否启用该类日志。
    interval:
        按 frame_id 节流打印时的间隔。
        例如 interval=30 表示每 30 帧打印一次。
    """
    enabled: bool = True
    interval: int = 30

@dataclass
class LoggingConfig:
    """
    统一日志配置。

    设计：
    - enable_console 控制是否允许 ModuleLogger 输出到控制台
    - save_log 控制是否通过 LogRedirectManager 保存 stdout/stderr 到文件
    - profile/debug/status/warning/save 按日志类型单独开关
    """

    # 是否允许 ModuleLogger 输出到控制台
    enable_console: bool = True

    # 是否把 stdout/stderr 同时保存到文件
    # 这个配合 utils/log_redirect.py 使用
    save_log: bool = False

    # 日志目录
    log_dir: str = str(BASE_DIR / "log")

    # 日志文件名前缀
    log_prefix: str = "log"

    # 性能日志，例如 TRACK_GPU_PROFILE / MAPPER_PROFILE
    profile: LogCategoryConfig = field(default_factory=lambda: LogCategoryConfig(
        enabled=True,
        interval=30,
    ))

    # 调试日志，例如 odom、ICP reject 详细信息
    debug: LogCategoryConfig = field(default_factory=lambda: LogCategoryConfig(
        enabled=False,
        interval=30,
    ))

    # 状态日志，例如 initialized、tracking lost、mapping paused/resumed
    status: LogCategoryConfig = field(default_factory=lambda: LogCategoryConfig(
        enabled=True,
        interval=30,
    ))

    # 警告日志，例如 pose jump reject、local map failed
    warning: LogCategoryConfig = field(default_factory=lambda: LogCategoryConfig(
        enabled=True,
        interval=1,
    ))

    # 保存类日志，例如模型保存成功/失败
    save: LogCategoryConfig = field(default_factory=lambda: LogCategoryConfig(
        enabled=True,
        interval=1,
    ))

# ============================================================
# 总配置
# ============================================================
@dataclass
class PipelineConfig:
    """
    总配置入口

    设计原则：
    - 所有运行参数尽量从这里集中管理
    - main_pipeline.py 作为唯一配置分发入口

    配置分层：
    - input_camera   : 输入设备原始分辨率对应内参
    - tracking_camera: 跟踪使用的内参，通常可低分辨率以提升速度
    - mapping_camera : 建图使用的内参，通常与输入一致或较高
    - input          : 输入预处理参数
    - tracking       : 跟踪参数
    - mapping        : 建图参数
    - render         : 渲染参数
    - logging        : 日志参数
    """

    # --------------------------------------------------------
    # 三套相机参数
    # --------------------------------------------------------

    # 输入设备原始内参
    # 应与 UnifiedDepthScanner 实际输出图像完全匹配
    input_camera: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
    ))

    # tracking 使用的相机内参
    tracking_camera: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        width=320,
        height=240,
        fx=262.5,
        fy=262.5,
        cx=159.75,
        cy=119.75,
    ))

    # mapping 使用的相机内参
    mapping_camera: CameraIntrinsics = field(default_factory=lambda: CameraIntrinsics(
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
    ))

    # --------------------------------------------------------
    # 分模块配置
    # --------------------------------------------------------
    input: InputConfig = field(default_factory=InputConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)