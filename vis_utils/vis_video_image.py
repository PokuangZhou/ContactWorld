import os
import zarr
import numpy as np
import imageio.v2 as imageio

# =========================
# 配置
# =========================
data_root = "data/exploration_sorting_normal/data"

front_zarr_path = f"{data_root}/front"
wrist_zarr_path = f"{data_root}/wrist"

front_output_video = f"{data_root}/front_vis.mp4"
wrist_output_video = f"{data_root}/wrist_vis.mp4"

num_frames = 200
fps = 10


# =========================
# 工具函数
# =========================
def normalize_rgb(frame):
    frame = np.asarray(frame)

    if frame.dtype == np.uint8:
        return frame

    frame = frame.astype(np.float32)

    # 如果是 [0, 1]，转成 [0, 255]
    if np.nanmax(frame) <= 1.0:
        frame = frame * 255.0

    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def save_rgb_video(zarr_path, output_video, num_frames=200, fps=20):
    arr = zarr.open(zarr_path, mode="r")

    print("\n================ RGB VIDEO ================")
    print("zarr path:", zarr_path)
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

    assert arr.ndim == 4 and arr.shape[-1] == 3, \
        f"Expected shape [T,H,W,3], got {arr.shape}"

    T = arr.shape[0]
    n = min(num_frames, T)

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    frames = []

    for t in range(n):
        frame = normalize_rgb(arr[t])
        frames.append(frame)

    imageio.mimsave(
        output_video,
        frames,
        fps=fps,
        codec="libx264"
    )

    print(f"Saved video to: {output_video}")


# =========================
# 主程序
# =========================
save_rgb_video(
    front_zarr_path,
    front_output_video,
    num_frames=num_frames,
    fps=fps
)

save_rgb_video(
    wrist_zarr_path,
    wrist_output_video,
    num_frames=num_frames,
    fps=fps
)