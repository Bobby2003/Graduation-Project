# RGB-D / IMU Fusion Reconstruction Pipeline

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <project-root>
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

激活环境：

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. 安装依赖

如果项目已提供依赖文件：

```bash
pip install -r requirements.txt
```

如果尚未整理 `requirements.txt`，请根据实际代码安装所需依赖，例如：

- `numpy`
- `open3d`
- `opencv-python`
- 设备 SDK / 扫描器相关绑定库

### 4. 运行项目

```bash
python main_pipeline.py
```

### 5. 初次联调建议

第一次运行建议使用较保守配置：

```python
use_imu = False
enable_render = False
validate_input_frame = True
debug_print_odom = True
```

优先确认以下内容：

- 输入帧是否连续
- 时间戳单位是否正确
- 相机内参是否正确
- Tracking 是否稳定输出
- Mapping 是否正常生成 mesh

---

## 配置说明

项目主要配置集中在：

```text
config.py
```

### 1. 基础线程参数

用于控制各线程运行节奏和队列容量，例如：

- `tracking_sleep_ms`
- `mapping_queue_size`
- `render_fps`

### 2. 输入与时间戳参数

用于控制输入帧合法性检查和时间戳解释方式，例如：

- `device_timestamp_unit`
- `validate_input_frame`

### 3. 模块功能开关

用于控制是否启用某些模块，例如：

- `use_imu`
- `enable_render`
- `enable_console_log`

### 4. 相机内参与输入预处理

用于深度反投影和 RGB-D 数据预处理，例如：

- `width`, `height`
- `fx`, `fy`, `cx`, `cy`
- `input_color_is_bgr`
- `depth_scale`
- `depth_trunc`

### 5. Tracking 参数

用于控制位姿估计和结果过滤，例如：

- `mapping_stride`
- `min_valid_pixels`
- `max_rotation_deg_per_frame`
- `min_info_trace`
- `min_translation_per_frame`
- `max_translation_per_frame`
- `trans_smooth_alpha`

如果启用了 IMU，还会涉及：

- `min_imu_rot_deg_per_frame`
- `imu_as_vo_init_only`
- `allow_imu_fallback_when_vo_fails`

### 6. Mapping / TSDF 参数

用于控制体素地图融合与 mesh 更新，例如：

- `voxel_length`
- `sdf_trunc`
- `integrate_only_when_motion`
- `min_integrate_rot_deg`
- `max_integrate_rot_deg`
- `min_integrate_trans_m`
- `mesh_update_interval`
- `mesh_min_vertices_to_show`

### 7. Render 参数

用于控制渲染窗口行为，例如：

- `render_window_name`
- `render_width`
- `render_height`
- `render_enable_status_log`

### 8. 日志参数

用于控制日志保存，例如：

- `save_log`
- `log_dir`
- `log_prefix`

### 9. 配置修改原则

协同开发时建议遵守以下原则：

1. 新参数统一加入 `config.py`
2. 不要把阈值直接写死在模块内部
3. 修改配置含义时同步更新注释
4. 若参数会显著影响运行结果，建议同步更新 README

---

## 输出说明

### 1. 日志输出

当以下配置启用时：

```python
save_log = True
```

控制台输出会同时写入日志文件。默认日志目录为：

```text
log/
```

日志文件名通常带时间戳，便于区分不同运行记录。

### 2. 模型输出

重建过程中或结束后生成的 mesh / 模型文件默认输出到：

```text
model/
```

该目录用于保存：

- 最终 mesh
- 导出模型
- 可能的中间结果（视实现而定）

### 3. 调试输出

如果开启详细调试选项，例如：

```python
debug_print_odom = True
```

则 Tracking / Odometry 过程会输出更详细的状态信息，便于问题定位。

### 4. 输出目录建议

建议所有运行生成内容统一放入以下目录：

- `log/`：日志
- `model/`：模型与重建结果

避免临时文件散落到项目根目录。

---

## 主要模块职责说明

### `main_pipeline.py`

主入口文件，负责组装整个 pipeline。

主要职责：

- 加载配置
- 初始化日志
- 初始化 scanner / 输入适配器
- 创建共享状态、队列、停止事件
- 构建 tracker / mapper / renderer / imu manager
- 启动并协调各 worker 线程
- 在程序退出时执行资源清理和最终输出

### `config.py`

全局配置中心。

主要职责：

- 管理所有关键参数
- 作为实验配置和联调配置的统一入口
- 避免参数散落在不同模块中

### `input/`

输入层模块。

典型职责：

- 接收底层设备 RGB-D 数据
- 转换成项目内部统一帧格式
- 处理颜色格式、深度格式、时间戳单位
- 可选进行输入合法性检查

### `tracking/`

位姿跟踪层。

典型职责：

- 根据输入帧估计相机位姿
- 执行配准、阈值判断与结果过滤
- 在需要时结合 IMU 信息
- 将带位姿的帧送入 Mapping 队列

### `mapping/`

建图层。

典型职责：

- 根据 Tracking 输出的位姿进行 TSDF 融合
- 更新体素地图
- 周期性提取 mesh
- 保存最终模型结果

### `rendering/`

渲染层。

典型职责：

- 读取共享状态中的当前 mesh
- 刷新可视化窗口
- 输出必要的渲染状态信息

说明：

- 该模块可关闭
- 关闭后不影响 Tracking 和 Mapping 主流程

### `imu/`

IMU 相关模块。

典型职责：

- 接入 IMU 数据
- 为 Tracking 提供姿态初始化或失败回退支持
- 管理 IMU 数据读取和状态更新

### `common/`

公共基础模块。

典型职责：

- 维护共享状态
- 提供线程间通信需要的通用结构
- 提供队列、缓冲区等基础工具

### `utils/`

工具模块。

典型职责：

- 日志重定向
- 文件输出辅助
- 与核心业务弱相关但会被多处复用的功能

---

## 多人协作开发建议

### 1. 保持模块边界清晰

建议遵循以下原则：

- `main_pipeline.py` 只负责组装和调度
- 算法逻辑放在 `tracker.py` / `mapper.py`
- 线程逻辑放在各自的 `*_worker.py`
- 输入适配放在 `input/`
- 公共结构放在 `common/`
- 工具函数放在 `utils/`

避免把不同层的逻辑混在一起。

### 2. 新增参数统一进 `config.py`

新增阈值、开关、路径、频率等参数时：

- 优先放进 `config.py`
- 补充注释
- 若影响较大，同步更新 README

不要在模块内部直接写死魔法数字。

### 3. 修改公共接口前先确认影响范围

以下对象通常会影响多个模块：

- 输入帧数据结构
- Tracking 输出格式
- Mapping 输入格式
- `SharedState` 内部字段
- Queue 中传递的数据对象

修改前建议先确认依赖方，避免联动崩坏。

### 4. 调试代码尽量可开关

建议：

- 调试输出放在开关控制下
- 临时实验逻辑加注释
- 不要把一次性调试 hack 长期留在主流程

### 5. 提交前建议自查

提交前建议至少检查：

- 项目是否可运行
- 没有误提交日志和模型输出
- 本地临时路径没有写进代码
- 调试打印没有污染主流程
- 配置修改是否同步到 `config.py`

### 6. 推荐阅读顺序

对于新接手该模块的开发者，建议阅读顺序：

1. `config.py`
2. `main_pipeline.py`
3. `input/`
4. `tracking/`
5. `mapping/`
6. `rendering/`
7. `imu/`
8. `common/`
9. `utils/`

这样更容易建立整体理解。

---

## Git 建议

### 1. `.gitignore` 放置位置

`.gitignore` 应放在 **Git 仓库根目录**，即与 `.git/` 同级的位置。

### 2. 建议忽略的内容

建议忽略以下内容：

- Python 缓存：`__pycache__/`, `*.pyc`
- 虚拟环境：`.venv/`, `venv/`
- IDE 配置：`.vscode/`, `.idea/`
- 运行日志：`log/`
- 模型输出：`model/`
- 临时导出文件：`*.ply`, `*.obj`, `*.pcd`, `*.npy` 等

### 3. 建议保留的内容

建议提交到仓库中的内容包括：

- 源码
- `config.py`
- `README.md`
- `requirements.txt`（如果有）
- 空的 `__init__.py`

### 4. 关于输出目录

如果希望保留目录结构但不提交运行结果，可以采用：

```gitignore
log/*
!log/.gitkeep

model/*
!model/.gitkeep
```

并在目录中放置：

- `log/.gitkeep`
- `model/.gitkeep`

### 5. 分支与提交建议

协作开发时建议：

- 功能开发使用独立分支
- 提交信息尽量清晰描述改动模块
- 涉及公共接口修改时在提交说明中注明影响范围

例如：

```bash
git commit -m "tracking: refine pose validation thresholds"
git commit -m "mapping: adjust tsdf integration conditions"
git commit -m "config: add imu fallback switch"
```