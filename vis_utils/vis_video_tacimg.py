import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# =========================
# 配置
# =========================
data_root = "data/insertion_usb/data"

rgb_zarr_path = f"{data_root}/tactile_rgb_right"
depth_zarr_path = f"{data_root}/tactile_depth_right"

rgb_output_video = f"{data_root}/tactile_rgb_right_vis.mp4"
depth_output_video = f"{data_root}/tactile_depth_right_vis.mp4"

num_frames = 200
fps = 10

use_global_depth_scale = True


# =========================
# 工具函数
# =========================
def normalize_rgb(rgb):
    rgb = np.asarray(rgb)

    if rgb.dtype == np.uint8:
        return rgb

    rgb = rgb.astype(np.float32)

    # 如果是 [0, 1]
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def depth_to_uint8(depth, vmin=None, vmax=None):
    depth = np.asarray(depth, dtype=np.float32)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 2:
        raise ValueError(f"Expected depth frame shape [H,W] or [H,W,1], got {depth.shape}")

    if vmin is None:
        vmin = float(np.nanmin(depth))
    if vmax is None:
        vmax = float(np.nanmax(depth))

    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return depth_uint8


def save_rgb_video(zarr_path, output_video, num_frames=200, fps=20):
    arr = zarr.open(zarr_path, mode="r")

    print("\n================ RGB ================")
    print("zarr path:", zarr_path)
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

    assert arr.ndim == 4 and arr.shape[-1] == 3, \
        f"Expected RGB shape [T,H,W,3], got {arr.shape}"

    T = arr.shape[0]
    n = min(num_frames, T)

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    frames = []

    for t in range(n):
        frame = arr[t]
        frame = normalize_rgb(frame)
        frames.append(frame)

    imageio.mimsave(
        output_video,
        frames,
        fps=fps,
        codec="libx264"
    )

    print(f"Saved RGB video to: {output_video}")


def save_depth_video(
    zarr_path,
    output_video,
    num_frames=200,
    fps=20,
    use_global_scale=True,
):
    arr = zarr.open(zarr_path, mode="r")

    print("\n================ DEPTH ================")
    print("zarr path:", zarr_path)
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

    T = arr.shape[0]
    n = min(num_frames, T)

    assert arr.ndim in [3, 4], \
        f"Expected depth shape [T,H,W] or [T,H,W,1], got {arr.shape}"

    if arr.ndim == 4:
        assert arr.shape[-1] == 1, \
            f"Expected depth last dim = 1, got {arr.shape}"

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    data = np.asarray(arr[:n], dtype=np.float32)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if use_global_scale:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
    else:
        vmin = None
        vmax = None

    print("depth min/max:", float(np.nanmin(data)), float(np.nanmax(data)))

    frames = []

    for t in range(n):
        depth = data[t]

        depth_uint8 = depth_to_uint8(depth, vmin=vmin, vmax=vmax)

        # 用 colormap 转成 RGB，方便看深度变化
        cmap = plt.get_cmap("viridis")
        depth_rgb = cmap(depth_uint8 / 255.0)[..., :3]
        depth_rgb = (depth_rgb * 255).astype(np.uint8)

        frames.append(depth_rgb)

    imageio.mimsave(
        output_video,
        frames,
        fps=fps,
        codec="libx264"
    )

    print(f"Saved depth video to: {output_video}")


# =========================
# 主程序
# =========================
save_rgb_video(
    rgb_zarr_path,
    rgb_output_video,
    num_frames=num_frames,
    fps=fps
)

save_depth_video(
    depth_zarr_path,
    depth_output_video,
    num_frames=num_frames,
    fps=fps,
    use_global_scale=use_global_depth_scale
)