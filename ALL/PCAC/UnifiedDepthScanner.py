import time
import cv2
import numpy as np
import open3d as o3d

from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

from .orbbec_sdk import OrbbecCameraSDK, get_default_sdk_path

# ─────────────────────────────────────────────
# 引导滤波
# 作用：
#   在保边缘的前提下平滑深度图，减少噪声
# 参数：
#   guide_u8 : 引导图（uint8）
#   src      : 输入待滤波图（float32）
#   r        : 局部窗口大小
#   eps      : 正则项，越小越保边缘
# ─────────────────────────────────────────────
def guided_filter(guide_u8, src, r=4, eps=50.0):
    I = guide_u8.astype(np.float32)
    p = src.astype(np.float32)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    return mean_a * I + mean_b

# ─────────────────────────────────────────────
# 最近邻填洞
# 作用：
#   对小空洞区域做最近邻有效值填充
#   不做插值，不制造新的中间深度层
# 参数：
#   roi       : 局部深度图
#   hole_mask : 该局部区域中的空洞布尔掩码
# 返回：
#   填充后的局部深度图
# ─────────────────────────────────────────────
def nearest_neighbor_fill(roi, hole_mask):
    result = roi.copy()

    # 找到有效像素：深度 > 0 且不在空洞中
    valid_mask = (roi > 0) & (~hole_mask)
    if not valid_mask.any():
        return result

    valid_coords = np.argwhere(valid_mask)
    hole_coords = np.argwhere(hole_mask)

    tree = cKDTree(valid_coords)
    _, idx = tree.query(hole_coords)

    nearest_vals = roi[valid_coords[idx, 0], valid_coords[idx, 1]]
    result[hole_coords[:, 0], hole_coords[:, 1]] = nearest_vals

    return result

# ─────────────────────────────────────────────
# 仅填充小空洞
# 作用：
#   只修补面积较小的空洞，大空洞保持为0，防止边缘被污染
# 参数：
#   depth       : 输入深度图
#   max_hole_px : 最大允许填充的空洞面积
# 返回：
#   填充后的深度图
# ─────────────────────────────────────────────
def fill_small_holes(depth, max_hole_px=100):
    mask_invalid = (depth == 0).astype(np.uint8)
    if mask_invalid.sum() == 0:
        return depth

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_invalid, connectivity=8
    )

    filled = depth.copy()

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_hole_px:
            continue

        hole_mask = (labels == label_id)

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        pad = 4
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, depth.shape[1])
        y1 = min(y + h + pad, depth.shape[0])

        roi_depth = filled[y0:y1, x0:x1].copy()
        roi_hole = hole_mask[y0:y1, x0:x1]

        roi_filled = nearest_neighbor_fill(roi_depth, roi_hole)
        roi_depth[roi_hole] = roi_filled[roi_hole]
        filled[y0:y1, x0:x1] = roi_depth

    return filled

# ─────────────────────────────────────────────
# 为引导滤波临时补全所有0值
# 作用：
#   boxFilter 遇到0值会污染均值，因此在引导滤波前，
#   先把所有0值用最近邻有效值临时补上。
# 注意：
#   这里只是“临时补全”，滤波后还会再用 final_mask 把无效区域抠掉
# ─────────────────────────────────────────────
def fill_all_holes_for_filter(depth):
    if (depth > 0).all():
        return depth

    result = depth.copy()
    invalid = result == 0

    _, idx = distance_transform_edt(
        invalid, return_distances=True, return_indices=True
    )

    result[invalid] = depth[idx[0][invalid], idx[1][invalid]]
    return result

class UnifiedDepthScanner:
    """
    统一版深度扫描器

    功能包含：
    1. 设备初始化 / 关闭
    2. 深度采集
    3. 深度处理
    4. 输出最终 final_mask
    5. 输出点云
    6. 输出伪彩色深度图 + 深度图
    7. Open3D 实时显示
    8. 保存点云
    9. 每帧附带 timestamp 和 frame_id
    """

    # ─────────────────────────────────────────
    # 默认参数区
    # ─────────────────────────────────────────
    MIN_DEPTH = 100            # mm，最小有效深度
    MAX_DEPTH = 3000           # mm，最大有效深度
    MAX_HOLE_PX = 100          # px，仅填充小于该面积的空洞
    GF_RADIUS = 4              # 引导滤波窗口大小
    GF_EPS = 50.0              # 引导滤波正则项
    EMA_ALPHA = 0.5            # EMA 时间滤波系数，1.0=不平滑
    POINT_SIZE = 2.0           # Open3D 点大小

    def __init__(
        self,
        width=640,
        height=480,
        fx=525.0,
        fy=525.0,
        cx=319.5,
        cy=239.5,
        min_depth=100,
        max_depth=3000,
        max_hole_px=100,
        gf_radius=4,
        gf_eps=50.0,
        ema_alpha=0.5,
        point_size=2.0
    ):
        """
        初始化扫描器对象，但此时还不打开设备

        参数说明：
        - width, height : 深度图尺寸
        - fx, fy, cx, cy: 相机内参
        - min_depth     : 最小深度阈值
        - max_depth     : 最大深度阈值
        - max_hole_px   : 最大填洞面积
        - gf_radius     : 引导滤波半径
        - gf_eps        : 引导滤波正则项
        - ema_alpha     : EMA 平滑系数
        - point_size    : 可视化点大小
        """
        self.sdk = OrbbecCameraSDK(get_default_sdk_path())

        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.MIN_DEPTH = min_depth
        self.MAX_DEPTH = max_depth
        self.MAX_HOLE_PX = max_hole_px
        self.GF_RADIUS = gf_radius
        self.GF_EPS = gf_eps
        self.EMA_ALPHA = ema_alpha
        self.POINT_SIZE = point_size

        xx = np.arange(self.width)
        yy = np.arange(self.height)
        self.u, self.v = np.meshgrid(xx, yy)

        # EMA缓存
        self.ema_depth = None

        # 帧计数器
        self.frame_id = 0

        # 首帧标志，给可视化用
        self.first_frame = True

        # 设备是否初始化成功
        self.initialized = False

    # ─────────────────────────────────────────
    # 设备相关接口
    # ─────────────────────────────────────────
    def init(self):
        """
        初始化并启动深度设备

        返回：
        - True  : 初始化成功
        - False : 初始化失败
        """
        ok = (
            self.sdk.initialize() and
            self.sdk.open_device() and
            self.sdk.create_stream(self.sdk.ONI_SENSOR_DEPTH) and
            self.sdk.start_stream()
        )
        self.initialized = ok
        return ok

    def close(self):
        """
        关闭设备并释放资源
        """
        try:
            self.sdk.cleanup()
        except Exception:
            pass
        self.initialized = False

    def reset_temporal_state(self):
        """
        重置时间相关滤波状态

        说明：
        - 这里只重置本类内部的EMA缓存和可视化首帧状态
        - frame_id 由SDK返回，不在这里手动维护
        """
        self.ema_depth = None
        self.first_frame = True

    # ─────────────────────────────────────────
    # 核心预处理接口
    # ─────────────────────────────────────────
    def remove_flying_pixels(self, depth, depth_thresh=100.0, erode_px=2):
        """
        去除飞点 / 边缘毛刺

        原理：
        - 计算相邻像素的深度差
        - 深度突变大于阈值时，视为边缘不稳定区域
        - 将该区域适当膨胀后清零

        参数：
        - depth        : 输入深度图
        - depth_thresh : 深度突变阈值（mm）
        - erode_px     : 膨胀半径（像素）

        返回：
        - 去飞点后的深度图
        """
        diff_x = np.abs(np.diff(depth, axis=1, append=0))
        diff_y = np.abs(np.diff(depth, axis=0, append=0))

        edge = ((diff_x > depth_thresh) | (diff_y > depth_thresh)).astype(np.uint8)

        edge = cv2.dilate(
            edge,
            np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        )

        depth_out = depth.copy()
        depth_out[edge > 0] = 0.0
        return depth_out

    def process_depth(self, raw_depth):
        """
        处理一帧原始深度图

        处理流程：
        1. 深度范围裁剪
        2. 形态学开运算去孤立点
        3. 去飞点
        4. 小洞填充
        5. EMA时间平滑
        6. 引导滤波
        7. 生成 final_mask

        注意：
        - 这里最终只输出 final_mask
        - 不再额外返回中间mask

        参数：
        - raw_depth : SDK返回的原始深度图

        返回：
        - depth_processed : 处理后的深度图(float32, mm)
        - final_mask      : 最终有效区域掩码(uint8, 0/1)
        """
        d = raw_depth.astype(np.float32)

        # 1. 深度范围裁剪
        d[(d < self.MIN_DEPTH) | (d > self.MAX_DEPTH)] = 0.0

        # 2. 去孤立噪点
        valid = (d > 0).astype(np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        d[valid == 0] = 0.0

        # 3. 去飞点
        d = self.remove_flying_pixels(d)

        # 4. 只填小空洞
        d_filled = fill_small_holes(d, self.MAX_HOLE_PX)

        # 5. EMA 时间平滑
        if self.ema_depth is None:
            self.ema_depth = d_filled.copy()
        else:
            both_valid = (d_filled > 0) & (self.ema_depth > 0)

            self.ema_depth[both_valid] = (
                self.EMA_ALPHA * d_filled[both_valid]
                + (1.0 - self.EMA_ALPHA) * self.ema_depth[both_valid]
            )

            new_valid = (d_filled > 0) & (self.ema_depth == 0)
            self.ema_depth[new_valid] = d_filled[new_valid]

            # 当前帧无效的位置直接清零，避免拖尾
            self.ema_depth[d_filled == 0] = 0.0

        d_ema = self.ema_depth.copy()

        # 6. 引导滤波前，临时填满所有0，避免均值污染
        d_temp = fill_all_holes_for_filter(d_ema)
        guide = cv2.normalize(d_temp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        d_gf = guided_filter(
            guide,
            d_temp,
            r=self.GF_RADIUS,
            eps=self.GF_EPS
        )

        # 7. 生成最终 final_mask
        final_mask = (
            (d_gf > self.MIN_DEPTH) &
            (d_gf < self.MAX_DEPTH) &
            (d > 0)  # 用原始有效区域限制，避免大空洞被误恢复
        ).astype(np.uint8)

        # 把无效区域清零
        d_gf[final_mask == 0] = 0.0

        return d_gf.astype(np.float32), final_mask.astype(np.uint8)

    # ─────────────────────────────────────────
    # 点云转换接口
    # ─────────────────────────────────────────
    def depth_to_pointcloud(self, depth, final_mask):
        """
        将深度图 + final_mask 转换为点云

        参数：
        - depth      : 深度图，单位 mm
        - final_mask : 最终有效掩码

        返回：
        - pts : 点云坐标，shape=(N,3)
        - col : 点云颜色，shape=(N,3)，近红远蓝
        """
        sel = (depth > 0) & (final_mask > 0)
        if not sel.any():
            return None, None

        z = depth[sel] / 1000.0
        u = self.u[sel]
        v = self.v[sel]

        x = (u - self.cx) * z / self.fx
        y = -(v - self.cy) * z / self.fy

        pts = np.stack([x, y, z], axis=-1)

        # 颜色映射：近红远蓝
        t = np.clip(z / (self.MAX_DEPTH / 1000.0), 0.0, 1.0)
        col = np.zeros_like(pts)
        col[:, 0] = 1.0 - t
        col[:, 2] = t

        return pts, col

    # ─────────────────────────────────────────
    # 伪彩深度图接口
    # ─────────────────────────────────────────
    def depth_to_colormap(self, depth, final_mask):
        """
        将处理后的深度图转换为伪彩色RGB图

        参数：
        - depth      : 处理后的深度图
        - final_mask : 最终有效mask

        返回：
        - color_rgb : RGB伪彩图
        """
        depth_clean = depth.copy()
        depth_clean[final_mask == 0] = 0.0

        # 近处红，远处蓝
        t = np.clip(1.0 - depth_clean / self.MAX_DEPTH, 0.0, 1.0)
        color_8u = (t * 255).astype(np.uint8)

        color_bgr = cv2.applyColorMap(color_8u, cv2.COLORMAP_JET)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # 无效区域设黑
        color_rgb[final_mask == 0] = [0, 0, 0]

        return color_rgb

    # ─────────────────────────────────────────
    # 原始采集接口
    # ─────────────────────────────────────────
    def capture_raw_depth(self):
        """
        采集一帧原始深度图

        返回：
        - raw_depth : 原始深度图
        - timestamp : SDK 原始时间戳
        - frame_id  : SDK 原始帧号
        失败时返回 None
        """
        res = self.sdk.capture_depth_frame()
        if res is None:
            return None

        raw_depth, frame_info = res

        return {
            "frame_id": frame_info["frame_index"],  # 直接使用SDK帧号
            "timestamp": frame_info["timestamp"],  # 直接使用SDK时间戳
            "raw_depth": raw_depth,
            "frame_info": frame_info  # 如果后面你还想用原始信息，顺手保留
        }

    # ─────────────────────────────────────────
    # 标准帧接口：返回最完整的一帧处理结果
    # ─────────────────────────────────────────
    def get_frame(self):
        """
        获取完整处理后的一帧数据
        """
        cap = self.capture_raw_depth()
        if cap is None:
            return None

        raw_depth = cap["raw_depth"]
        depth, final_mask = self.process_depth(raw_depth)
        color_rgb = self.depth_to_colormap(depth, final_mask)
        points, colors = self.depth_to_pointcloud(depth, final_mask)

        return {
            "frame_id": cap["frame_id"],
            "timestamp": cap["timestamp"],
            "raw_depth": raw_depth,
            "depth": depth,
            "final_mask": final_mask,
            "color_rgb": color_rgb,
            "points": points,
            "colors": colors,
            "frame_info": cap["frame_info"],  # 可选保留
        }

    # ─────────────────────────────────────────
    # 点云接口
    # ─────────────────────────────────────────
    def get_pointcloud(self):
        """
        获取一帧点云数据

        返回：
        - frame_id
        - timestamp
        - points
        - colors
        - depth
        - final_mask

        失败时返回 None
        """
        frame = self.get_frame()
        if frame is None:
            return None

        return {
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "points": frame["points"],
            "colors": frame["colors"],
            "depth": frame["depth"],
            "final_mask": frame["final_mask"]
        }

    # ─────────────────────────────────────────
    # RGB + Depth 接口
    # ─────────────────────────────────────────
    def get_rgb_and_depth(self):
        """
        获取伪彩RGB图与深度图

        返回：
        - frame_id
        - timestamp
        - color_rgb
        - depth
        - final_mask
        - raw_depth
        - frame_info

        说明：
        - color_rgb 为伪彩深度图，不是真实RGB相机图
        - depth 为 float32 毫米深度图
        """
        frame = self.get_frame()
        if frame is None:
            return None

        return {
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "color_rgb": frame["color_rgb"],
            "depth": frame["depth"],
            "final_mask": frame["final_mask"],
            "raw_depth": frame["raw_depth"],
            "frame_info": frame["frame_info"],
        }

    # ─────────────────────────────────────────
    # 深度 + mask 接口
    # ─────────────────────────────────────────
    def get_depth_and_mask(self):
        """
        获取处理后的深度图与最终mask

        返回：
        - frame_id
        - timestamp
        - depth
        - final_mask
        """
        frame = self.get_frame()
        if frame is None:
            return None

        return {
            "frame_id": frame["frame_id"],
            "timestamp": frame["timestamp"],
            "depth": frame["depth"],
            "final_mask": frame["final_mask"]
        }

    # ─────────────────────────────────────────
    # 仅时间戳与帧编号接口
    # ─────────────────────────────────────────
    def get_frame_meta(self):
        """
        获取一帧的元信息 + 原始深度图

        返回：
        - frame_id
        - timestamp
        - raw_depth
        """
        return self.capture_raw_depth()

    # ─────────────────────────────────────────
    # 保存点云接口
    # ─────────────────────────────────────────
    def save_pointcloud(self, points, colors, filename=None):
        """
        保存点云为 ply 文件

        参数：
        - points   : 点云坐标
        - colors   : 点云颜色
        - filename : 文件名，为空时自动生成

        返回：
        - 保存后的文件名
        """
        if points is None or colors is None:
            return None

        if filename is None:
            filename = "pointcloud_{}.ply".format(int(time.time()))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

        return filename

    # ─────────────────────────────────────────
    # OpenCV 预览图接口
    # ─────────────────────────────────────────
    def make_preview_bgr(self, depth):
        """
        生成用于 OpenCV 显示的 BGR 预览图

        参数：
        - depth : 处理后的深度图

        返回：
        - OpenCV可直接imshow的BGR图
        """
        disp = np.clip(depth / self.MAX_DEPTH * 255, 0, 255).astype(np.uint8)
        preview_bgr = cv2.applyColorMap(cv2.flip(disp, 1), cv2.COLORMAP_JET)
        return preview_bgr

    # ─────────────────────────────────────────
    # 运行可视化主循环
    # 保留 main.py 的完整功能
    # ─────────────────────────────────────────

    def run_viewer(self, window_name="Unified 3D Scanner"):
        """
        启动实时可视化界面

        注意：
        - 该接口不会自动执行
        - 只有主动调用时才会创建 Open3D / OpenCV 窗口
        - 用于调试、演示、人工观察点云效果
        """
        if not self.initialized:
            if not self.init():
                print("[Error] 设备初始化失败")
                return

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name, width=1280, height=800)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        vis.add_geometry(pcd)

        ro = vis.get_render_option()
        ro.background_color = np.array([0.08, 0.08, 0.08])
        ro.point_size = self.POINT_SIZE
        ro.show_coordinate_frame = True

        print("运行中 | OpenCV窗口聚焦后：'q'退出，'s'保存点云")

        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    continue

                pts = frame["points"]
                col = frame["colors"]
                depth = frame["depth"]

                if pts is None:
                    pts = np.zeros((1, 3))
                    col = np.zeros((1, 3))

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(col)
                vis.update_geometry(pcd)

                if self.first_frame and len(pts) > 1:
                    vis.reset_view_point(True)
                    self.first_frame = False

                if not vis.poll_events():
                    break
                vis.update_renderer()

                preview_bgr = self.make_preview_bgr(depth)
                cv2.imshow("Depth Preview", preview_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("退出")
                    break
                elif key == ord('s'):
                    fname = self.save_pointcloud(frame["points"], frame["colors"])
                    print(f"[已保存] {fname}")

        finally:
            vis.destroy_window()
            cv2.destroyAllWindows()

