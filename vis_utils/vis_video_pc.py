import os
import zarr
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio

# =========================
# 配置
# =========================
pc_zarr_path = "/home/zhiyuan/Project/TVB/data/screwing_valve/data/pointcloud"
output_video = "/home/zhiyuan/Project/TVB/pointcloud_vis_zoomed.mp4"

num_frames = 100
fps = 10

point_size = 40

# 点云整体旋转角度
rotate_xyz = True
rx_deg = 60
ry_deg = 120
rz_deg = 0

# 相机视角
elev = 20
azim = -225

auto_rotate_view = False
rotate_view_speed = 1.0

use_global_axis = True
black_background = True

# 去掉离群点的显示范围估计
use_percentile_lims = True
percentile = 2.0

# zoom < 1 会放大主体点云
zoom = 1.0

# 是否隐藏坐标轴
hide_axis = True


# =========================
# 工具函数：旋转点云坐标
# =========================
def rotate_pointcloud_xyz(xyz, rx_deg=0, ry_deg=0, rz_deg=0):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float32)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx

    valid = np.isfinite(xyz).all(axis=-1)
    center = np.nanmean(xyz[valid], axis=0)

    xyz_rot = xyz - center
    xyz_rot = xyz_rot @ R.T
    xyz_rot = xyz_rot + center

    return xyz_rot


def compute_global_lims(xyz, use_percentile=True, percentile=2.0, zoom=0.55):
    valid_xyz = xyz[np.isfinite(xyz).all(axis=-1)]

    if use_percentile:
        xyz_min = np.percentile(valid_xyz, percentile, axis=0)
        xyz_max = np.percentile(valid_xyz, 100.0 - percentile, axis=0)
    else:
        xyz_min = np.nanmin(valid_xyz, axis=0)
        xyz_max = np.nanmax(valid_xyz, axis=0)

    center = (xyz_min + xyz_max) / 2.0
    max_range = np.max(xyz_max - xyz_min) / 2.0

    # zoom 越小，主体越大
    max_range = max_range * zoom

    x_lim = (center[0] - max_range, center[0] + max_range)
    y_lim = (center[1] - max_range, center[1] + max_range)
    z_lim = (center[2] - max_range, center[2] + max_range)

    return x_lim, y_lim, z_lim


# =========================
# 读取点云
# =========================
arr = zarr.open(pc_zarr_path, mode="r")

print("zarr shape:", arr.shape)
print("zarr dtype:", arr.dtype)

assert arr.ndim == 3 and arr.shape[-1] in [3, 6], \
    f"Expected [T,N,3] or [T,N,6], got {arr.shape}"

T, N, C = arr.shape
num_frames = min(num_frames, T)

data = np.asarray(arr[:num_frames], dtype=np.float32)

xyz = data[..., :3]

if rotate_xyz:
    xyz = rotate_pointcloud_xyz(
        xyz,
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
    )

if C == 6:
    rgb = data[..., 3:6]

    if np.nanmax(rgb) <= 1.0:
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)
else:
    rgb = None


# =========================
# 全局坐标范围
# =========================
if use_global_axis:
    x_lim, y_lim, z_lim = compute_global_lims(
        xyz,
        use_percentile=use_percentile_lims,
        percentile=percentile,
        zoom=zoom,
    )

print("XYZ min:", np.nanmin(xyz.reshape(-1, 3), axis=0))
print("XYZ max:", np.nanmax(xyz.reshape(-1, 3), axis=0))

os.makedirs(os.path.dirname(output_video), exist_ok=True)


# =========================
# 渲染视频
# =========================
frames = []

for t in range(num_frames):
    pts = xyz[t]

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]

    if rgb is not None:
        colors = rgb[t][valid]
    else:
        colors = "white"

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    if black_background:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=colors,
        s=point_size,
        depthshade=False,
        linewidths=0,
    )

    if auto_rotate_view:
        cur_azim = azim + t * rotate_view_speed
    else:
        cur_azim = azim

    ax.view_init(elev=elev, azim=cur_azim)

    if use_global_axis:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_zlim(*z_lim)

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    if hide_axis:
        ax.set_axis_off()
        ax.grid(False)
    else:
        ax.set_title(
            f"Point Cloud t = {t}",
            color="white" if black_background else "black",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if black_background:
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")
            ax.tick_params(colors="white")

            ax.xaxis.set_pane_color((0, 0, 0, 1))
            ax.yaxis.set_pane_color((0, 0, 0, 1))
            ax.zaxis.set_pane_color((0, 0, 0, 1))

            ax.xaxis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.5)
            ax.yaxis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.5)
            ax.zaxis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.5)

    # 尽量铺满画面
    try:
        ax.dist = 6
    except Exception:
        pass

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[..., :3]

    frames.append(img)
    plt.close(fig)


# =========================
# 保存视频
# =========================
imageio.mimsave(
    output_video,
    frames,
    fps=fps,
    codec="libx264",
)

print(f"Saved point cloud video to: {output_video}")