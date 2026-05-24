import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# =========================
# 配置
# =========================
zarr_path = "data/exploration_sorting_normal/data/tactile_force_field_right"
output_video = "/home/zhiyuan/Project/TVB/tactile_force_field_right_vis.mp4"

num_frames = 200
fps = 10
upscale = 40

# quiver 的 scale 越大，箭头越短
arrow_scale = 0.001

use_global_scale = True
show_colorbar = True

# =========================
# 读取 zarr
# =========================
arr = zarr.open(zarr_path, mode="r")

print("zarr shape:", arr.shape)
print("zarr dtype:", arr.dtype)

assert arr.ndim == 4 and arr.shape[-1] == 3, f"Expected [T, H, W, 3], got {arr.shape}"

T, H, W, C = arr.shape
num_frames = min(num_frames, T)

data = np.asarray(arr[:num_frames], dtype=np.float32)

fx = data[..., 0]
fy = data[..., 1]
fz = data[..., 2]

mag = np.linalg.norm(data, axis=-1)

if use_global_scale:
    mag_max_global = float(np.max(mag)) + 1e-8
else:
    mag_max_global = None

print("num_frames:", num_frames)
print("force magnitude min/max:", float(np.min(mag)), float(np.max(mag)))

# =========================
# 网格坐标
# =========================
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)

os.makedirs(os.path.dirname(output_video), exist_ok=True)

# =========================
# 逐帧渲染
# =========================
frames = []

for t in range(num_frames):
    frame_fx = fx[t]
    frame_fy = fy[t]
    frame_mag = mag[t]

    if use_global_scale:
        mag_max = mag_max_global
    else:
        mag_max = float(np.max(frame_mag)) + 1e-8

    norm_mag = np.clip(frame_mag / mag_max, 0.0, 1.0)

    fig_w = W * upscale / 100
    fig_h = H * upscale / 100

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    # 黑底
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 箭头：颜色由力幅值控制，小=绿，中=黄，大=红
    q = ax.quiver(
        X, Y,
        frame_fx, -frame_fy,
        norm_mag,
        cmap="RdYlGn_r",
        clim=(0.0, 1.0),
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        width=0.01,
        pivot="middle"
    )

    ax.set_title(f"t = {t}", fontsize=12, color="white")

    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.grid(
        color="gray",
        linestyle="--",
        linewidth=0.4,
        alpha=0.35
    )

    ax.tick_params(colors="white", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("white")

    if show_colorbar:
        cbar = plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Force magnitude (normalized)", rotation=90, color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
        cbar.outline.set_edgecolor("white")

    plt.tight_layout()

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
    codec="libx264"
)

print(f"Saved video to: {output_video}")