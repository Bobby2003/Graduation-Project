import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- 页面配置 ---
st.set_page_config(page_title="双相机点云还原模拟", layout="wide")
st.title("📷 双相机视觉重叠与墙面还原模拟")
st.sidebar.header("控制参数")


# --- 1. 模拟环境生成 ---
def generate_wall(seed):
    np.random.seed(seed)
    # 墙面的 X 范围更宽一点
    x_base = np.linspace(-30, 50, 400)
    # 使用多个正弦波叠加生成平滑但不规则的墙
    y_base = 12 + 4 * np.sin(x_base * 0.2) + 2 * np.cos(x_base * 0.5) + np.random.normal(0, 0.15, len(x_base))
    return interp1d(x_base, y_base, fill_value="extrapolate")


# --- 2. 相机扫描逻辑 ---
def scan_wall(cam_pos, wall_func, fov_deg=40, num_rays=150):
    # 转换视角范围：朝向正上方(90度)，左右各分 FOV/2
    angles = np.linspace(90 - fov_deg / 2, 90 + fov_deg / 2, num_rays)
    angles_rad = np.deg2rad(angles)

    depths = []
    points = []

    for ang in angles_rad:
        # 简单的射线投射：在射线上步进寻找碰撞点
        # 射线方程: P(r) = Cam + r * (cos(ang), sin(ang))
        r_range = np.linspace(0.1, 40, 300)
        found = False
        for r in r_range:
            px = cam_pos[0] + r * np.cos(ang)
            py = cam_pos[1] + r * np.sin(ang)
            # 如果射线的 y 超过了墙的 y (在当前 x 位置)
            if py >= wall_func(px):
                depths.append(r)
                points.append([px, py])
                found = True
                break
        if not found:
            depths.append(np.nan)
            points.append([np.nan, np.nan])

    return np.array(depths), np.array(points), angles_rad


# --- 3. 核心算法：估计相机距离 (计算逻辑) ---
def estimate_baseline(depths1, angles1, depths2, angles2):
    # 1. 将各自的点云转换到本地直角坐标系 (Local Coordinate)
    # Cam 1 假设在 (0,0)
    x1_l = depths1 * np.cos(angles1)
    y1_l = depths1 * np.sin(angles1)

    # Cam 2 也假设自己在 (0,0)
    x2_l = depths2 * np.cos(angles2)
    y2_l = depths2 * np.sin(angles2)

    # 过滤无效点
    mask1 = ~np.isnan(x1_l)
    mask2 = ~np.isnan(x2_l)
    x1_f, y1_f = x1_l[mask1], y1_l[mask1]
    x2_f, y2_f = x2_l[mask2], y2_l[mask2]

    # 2. 寻找最佳基线 B
    # 我们尝试不同的 B 值，使得 Cam2 的点平移 B 后与 Cam1 的重合度最高
    search_range = np.linspace(0, 20, 200)
    min_err = float('inf')
    best_b = 0

    for b in search_range:
        # 将 Cam2 本地坐标平移 B
        x2_shifted = x2_f + b
        # 对 Cam2 的形状进行插值，以便在 Cam1 的点位上进行比较
        if len(x2_shifted) < 2: continue

        interp_cam2 = interp1d(x2_shifted, y2_f, bounds_error=False, fill_value=np.nan)
        y2_at_x1 = interp_cam2(x1_f)

        # 计算在重叠区域的均方误差 (MSE)
        overlap = ~np.isnan(y2_at_x1)
        if np.sum(overlap) > 10:  # 至少有10个点重合才计算
            err = np.mean((y1_f[overlap] - y2_at_x1[overlap]) ** 2)
            if err < min_err:
                min_err = err
                best_b = b

    return best_b


# --- 4. Streamlit UI 布局 ---
if 'seed' not in st.session_state:
    st.session_state.seed = 42

if st.sidebar.button("随机生成新墙面"):
    st.session_state.seed = np.random.randint(0, 10000)

real_b = st.sidebar.slider("设置相机实际间距 (Ground Truth)", 2.0, 15.0, 6.0)
fov = st.sidebar.slider("相机视场角 (FOV)", 20, 60, 40)

# 计算过程
wall_f = generate_wall(st.session_state.seed)
cam1_pos = np.array([0, 0])
cam2_pos = np.array([real_b, 0])

# 两个相机分别获取“自己的”点云数据
d1, p1, a1 = scan_wall(cam1_pos, wall_f, fov)
d2, p2, a2 = scan_wall(cam2_pos, wall_f, fov)

# 仅利用各自的 d (距离) 和 a (角度) 还原 B
est_b = estimate_baseline(d1, a1, d2, a2)

# --- 5. 绘图 ---
fig = plt.figure(figsize=(12, 10))

# 图1: 相机各自的原始深度数据 (本地视角)
ax_l = plt.subplot(2, 2, 3, projection='polar')
ax_r = plt.subplot(2, 2, 4, projection='polar')

for ax, d, a, title, c in zip([ax_l, ax_r], [d1, d2], [a1, a2], ["Cam 1 Local", "Cam 2 Local"], ['blue', 'green']):
    ax.set_thetamin(90 - fov / 2)
    ax.set_thetamax(90 + fov / 2)
    ax.scatter(a, d, c=c, s=5)
    ax.set_title(title)

# 图2: 物理世界真实情况
ax_top = plt.subplot(2, 2, 1)
test_x = np.linspace(-10, real_b + 10, 500)
ax_top.plot(test_x, wall_f(test_x), 'k--', alpha=0.3, label="Real Wall")
ax_top.scatter(p1[:, 0], p1[:, 1], color='blue', s=2, label="Cam 1 Rays")
ax_top.scatter(p2[:, 0], p2[:, 1], color='green', s=2, label="Cam 2 Rays")
ax_top.plot(0, 0, 'ro', label="C1")
ax_top.plot(real_b, 0, 'mo', label="C2")
ax_top.legend()
ax_top.set_title("Physical Reality (Truth)")

# 图3: 利用估算的 B 还原生成的墙面
ax_res = plt.subplot(2, 2, 2)
# 还原坐标
x1_rec = d1 * np.cos(a1)
y1_rec = d1 * np.sin(a1)
# 这里的 est_b 是算出来的！
x2_rec = (d2 * np.cos(a2)) + est_b
y2_rec = d2 * np.sin(a2)

ax_res.scatter(x1_rec, y1_rec, c='blue', s=5, alpha=0.5)
ax_res.scatter(x2_rec, y2_rec, c='green', s=5, alpha=0.5)

# 高亮重合区域 (在还原坐标系中寻找 x 靠近的点)
interp_check = interp1d(x2_rec[~np.isnan(x2_rec)], y2_rec[~np.isnan(y2_rec)], bounds_error=False)
y_check = interp_check(x1_rec)
overlap_mask = np.abs(y1_rec - y_check) < 0.2
ax_res.scatter(x1_rec[overlap_mask], y1_rec[overlap_mask], c='red', s=15, label="Overlap Highlight")

ax_res.set_title(f"Reconstructed Wall (Estimated Baseline: {est_b:.3f})")
ax_res.legend()

st.pyplot(fig)

# 结果显示
c1, c2, c3 = st.columns(3)
c1.metric("实际相机距离", f"{real_b} m")
c2.metric("计算所得距离", f"{est_b:.3f} m")
c3.metric("计算误差", f"{abs(real_b - est_b):.4f} m")

st.write("### 原理解析")
st.markdown("""
1. **获取数据**：两个投影相机分别获取各自相对于中心点的极坐标（角度+深度）。
2. **坐标转换**：将极坐标转为本地 Cartesian 坐标 $(x, y)$。此时由于不知道基线，两个点云是重叠在原点的。
3. **滑窗匹配**：算法尝试改变相机2的水平位移 $B$，计算在这两个点云在重叠 $X$ 轴区间内的垂直高度 $Y$ 的均方误差。
4. **还原**：寻找误差最小时的 $B$ 即为推算的相机距离，进而拼接出完整的墙面轮廓。
""")