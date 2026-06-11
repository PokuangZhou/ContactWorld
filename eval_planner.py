from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
_HYDRA_TEMP_DIR = None


def _setup_runtime_cache_dirs() -> None:
    cache_root = REPO_ROOT / ".cache" / "eval_planner"
    python_bin = Path(sys.prefix) / "bin"
    if python_bin.exists():
        os.environ["PATH"] = f"{python_bin}:{os.environ.get('PATH', '')}"

    defaults = {
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
    }
    for env_name, path in defaults.items():
        os.environ.setdefault(env_name, str(path))
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)


def _prepend_sys_path(*paths: Path) -> None:
    for path in reversed(paths):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _setup_local_thirdparty_paths() -> None:
    manifeel_dir = Path(
        os.environ.get(
            "MANIFEEL_DIR",
            REPO_ROOT / "thirdparty" / "manifeel" / "manifeel",
        )
    )
    manifeel_isaacgym_root = Path(
        os.environ.get(
            "MANIFEEL_ISAACGYM_ROOT",
            REPO_ROOT / "thirdparty" / "manifeel-isaacgymenvs",
        )
    )
    isaacgym_python = Path(
        os.environ.get(
            "ISAACGYM_PYTHON_ROOT",
            REPO_ROOT / "thirdparty" / "IsaacGym_Preview_TacSL_Package" / "isaacgym" / "python",
        )
    )

    _prepend_sys_path(REPO_ROOT, manifeel_dir, manifeel_isaacgym_root, isaacgym_python)


def _preload_python_shared_library() -> None:
    candidates = [
        Path(sys.prefix) / "lib" / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0",
        Path(sys.prefix) / "lib" / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            pass


_setup_runtime_cache_dirs()
_setup_local_thirdparty_paths()
_preload_python_shared_library()

# IsaacGym MUST be imported before torch.
import isaacgym

import argparse
import json

import yaml
from typing import Optional

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dataset.zarr_dataset import ZarrDataset
from envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper
from planner_utils import (
    BatchedCEMPlanner,
    PlannerConfig,
    build_model,
    init_history_buffers,
    load_lightning_ckpt,
    update_history_buffers,
)



def _link_path(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _get_hydra_config_root() -> str:
    global _HYDRA_TEMP_DIR

    if _HYDRA_TEMP_DIR is not None:
        return _HYDRA_TEMP_DIR.name

    repo_config_dir = REPO_ROOT / "config"
    manifeel_dir = Path(
        os.environ.get(
            "MANIFEEL_DIR",
            REPO_ROOT / "thirdparty" / "manifeel" / "manifeel",
        )
    )
    manifeel_config_dir = Path(
        os.environ.get("MANIFEEL_CONFIG_DIR", manifeel_dir / "config")
    )
    manifeel_assets_dir = Path(
        os.environ.get(
            "MANIFEEL_ASSETS_DIR",
            REPO_ROOT / "thirdparty" / "manifeel" / "assets",
        )
    )

    if not manifeel_config_dir.exists():
        raise FileNotFoundError(f"ManiFeel config directory not found: {manifeel_config_dir}")
    if not manifeel_assets_dir.exists():
        raise FileNotFoundError(f"ManiFeel assets directory not found: {manifeel_assets_dir}")

    _HYDRA_TEMP_DIR = tempfile.TemporaryDirectory(prefix="contactworld_hydra_")
    atexit.register(_HYDRA_TEMP_DIR.cleanup)
    hydra_root = Path(_HYDRA_TEMP_DIR.name)

    if repo_config_dir.exists():
        for yaml_file in repo_config_dir.glob("*.yaml"):
            _link_path(yaml_file, hydra_root / yaml_file.name)
    for yaml_file in manifeel_config_dir.glob("*.yaml"):
        _link_path(yaml_file, hydra_root / yaml_file.name)
    _link_path(manifeel_config_dir / "task", hydra_root / "task")
    _link_path(manifeel_assets_dir, hydra_root / "assets")

    return str(hydra_root)

def is_rgb_like_key(key: str) -> bool:
    key = key.lower()
    rgb_tokens = ["image", "camera", "rgb", "taxim", "vision"]
    return any(tok in key for tok in rgb_tokens)


def sample_init_goal_segments(
    dataset: ZarrDataset,
    num_goals: int,
    goal_offset_steps: int,
    seed: int,
    vision_key: str,
    tactile_key: Optional[str] = None,
):
    rng = np.random.default_rng(seed)

    valid = []
    for ep_idx, ep_len in enumerate(dataset.lengths):
        max_start = int(ep_len - goal_offset_steps - 1)
        if max_start >= 0:
            for s in range(max_start + 1):
                valid.append((ep_idx, s, s + goal_offset_steps))

    if len(valid) == 0:
        raise RuntimeError("No valid (init, goal) segments found in dataset.")

    choice_idx = rng.choice(len(valid), size=num_goals, replace=len(valid) < num_goals)
    chosen = [valid[i] for i in choice_idx]

    ep_ids = np.array([x[0] for x in chosen], dtype=np.int64)
    start_steps = np.array([x[1] for x in chosen], dtype=np.int64)
    goal_steps = np.array([x[2] for x in chosen], dtype=np.int64)

    init_rows = dataset.offsets[ep_ids] + start_steps
    goal_rows = dataset.offsets[ep_ids] + goal_steps

    result = {
        "episode_idx": ep_ids,
        "start_step": start_steps,
        "goal_step": goal_steps,
        "init_row_idx": init_rows.astype(np.int64),
        "goal_row_idx": goal_rows.astype(np.int64),
        "init_dof_pos": dataset.get_col_data("dof_pos")[init_rows],
        "init_dof_vel": dataset.get_col_data("dof_vel")[init_rows],
        "init_plug_pos": dataset.get_col_data("plug_pos")[init_rows],
        "init_plug_quat": dataset.get_col_data("plug_quat")[init_rows],
        "init_socket_pos": dataset.get_col_data("socket_pos_gt")[init_rows],
        "init_socket_quat": dataset.get_col_data("socket_quat")[init_rows],
        "init_ee_pos": dataset.get_col_data("ee_pos")[init_rows],
        "init_ee_quat": dataset.get_col_data("ee_quat")[init_rows],
        "goal_plug_pos": dataset.get_col_data("plug_pos")[goal_rows],
        "goal_plug_quat": dataset.get_col_data("plug_quat")[goal_rows],
        "goal_socket_pos": dataset.get_col_data("socket_pos_gt")[goal_rows],
        "goal_socket_quat": dataset.get_col_data("socket_quat")[goal_rows],
        "goal_ee_pos": dataset.get_col_data("ee_pos")[goal_rows],
        "goal_ee_quat": dataset.get_col_data("ee_quat")[goal_rows],
    }

    result[vision_key] = dataset.get_col_data(vision_key)[goal_rows]

    if tactile_key is not None and tactile_key in dataset.column_names:
        result[tactile_key] = dataset.get_col_data(tactile_key)[goal_rows]

    return result


def make_env(args) -> MultipleIsaacEnvWrapper:
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=_get_hydra_config_root(), version_base="1.1")

    cfg = compose(config_name=args.isaacgym_cfg_name)
    cfg.num_envs = args.num_envs

    # Important: disable IsaacGym viewer / virtual display / render recording.
    # The rollout-vs-GT videos below are saved from obs arrays, not env.render().
    cfg.headless = True
    cfg.capture_video = False
    cfg.force_render = False

    obs_meta = {
        "plug_pos": {"type": "low_dim"},
        "plug_quat": {"type": "low_dim"},
        "socket_pos_gt": {"type": "low_dim"},
        "socket_quat": {"type": "low_dim"},
        "dof_pos": {"type": "low_dim"},
        "dof_vel": {"type": "low_dim"},
        "ee_pos": {"type": "low_dim"},
        "ee_quat": {"type": "low_dim"},
    }

    # Keep front available because some TacSL envs expect at least one visual key.
    obs_meta["front"] = {"type": "rgb"}

    if args.vision_type == "image":
        obs_meta[args.vision_key] = {"type": "rgb"}
    else:
        obs_meta[args.vision_key] = {"type": "low_dim"}

    if args.use_tactile:
        if is_rgb_like_key(args.tactile_key):
            obs_meta[args.tactile_key] = {"type": "rgb"}
        else:
            obs_meta[args.tactile_key] = {"type": "low_dim"}

    cfg["shape_meta"] = OmegaConf.create({"obs": obs_meta})
    return MultipleIsaacEnvWrapper(cfg)


def safe_env_seed(seed: int) -> int:
    # Some USB/disassembly seeds may fail to initialize in TacSL.
    if seed == 95:
        return 96
    return seed


def np_quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    ang_rad = 2.0 * np.arccos(dot)
    return np.rad2deg(ang_rad)


def end_pose_metrics_joint(
    current_plug_pos: np.ndarray,
    current_plug_quat: np.ndarray,
    goal_plug_pos: np.ndarray,
    goal_plug_quat: np.ndarray,
    current_ee_pos: np.ndarray,
    current_ee_quat: np.ndarray,
    goal_ee_pos: np.ndarray,
    goal_ee_quat: np.ndarray,
    plug_pos_thresh: float,
    plug_quat_thresh_deg: float,
    ee_pos_thresh: float,
    ee_quat_thresh_deg: float,
):
    plug_pos_err = np.linalg.norm(current_plug_pos - goal_plug_pos, axis=-1)
    plug_quat_err_deg = np_quat_angle_deg(current_plug_quat, goal_plug_quat)

    ee_pos_err = np.linalg.norm(current_ee_pos - goal_ee_pos, axis=-1)
    ee_quat_err_deg = np_quat_angle_deg(current_ee_quat, goal_ee_quat)

    plug_success = (plug_pos_err < plug_pos_thresh) & (plug_quat_err_deg < plug_quat_thresh_deg)
    ee_success = (ee_pos_err < ee_pos_thresh) & (ee_quat_err_deg < ee_quat_thresh_deg)
    joint_success = plug_success & ee_success

    return {
        "success": joint_success.astype(np.int32),
        "plug_success": plug_success.astype(np.int32),
        "ee_success": ee_success.astype(np.int32),
        "plug_pos_err": plug_pos_err,
        "plug_quat_err_deg": plug_quat_err_deg,
        "ee_pos_err": ee_pos_err,
        "ee_quat_err_deg": ee_quat_err_deg,
    }


def reset_env_to_dataset_state(env: MultipleIsaacEnvWrapper, batch: dict):
    env.reset_to_dataset_state(
        dof_pos=batch["init_dof_pos"],
        dof_vel=batch["init_dof_vel"],
        plug_pos=batch["init_plug_pos"],
        plug_quat=batch["init_plug_quat"],
        socket_pos=batch["init_socket_pos"],
        socket_quat=batch["init_socket_quat"],
    )


def build_goal_info_numpy(batch: dict, args) -> dict:
    goal_info = {args.vision_key: batch[args.vision_key][:, None]}
    if args.use_tactile:
        goal_info[args.tactile_key] = batch[args.tactile_key][:, None]
    return goal_info


def _to_hwc_rgb(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.moveaxis(frame, 0, -1)
    frame = frame.astype(np.float32)
    if np.nanmax(frame) <= 1.0:
        frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _depth_to_rgb(
    frame: np.ndarray,
    vmin=None,
    vmax=None,
    invert: bool = False,
    bg_value: float = 0.0,
    smooth_sigma: float = 1.0,
    alpha_sigma=None,
    gamma: float = 0.0,
    alpha_gamma: float = 0.5,
    cool_tone: bool = True,
) -> np.ndarray:
    """
    Same tactile-depth visualization as vis_modalities.py:
        black background + soft white/cool-white deformation.

    Default parameters match the paper export setting:
        smooth_sigma=1.0, gamma=0.0, alpha_gamma=0.5, cool_tone=True
    """
    import cv2

    frame = np.asarray(frame, dtype=np.float32)

    if frame.ndim == 3:
        if frame.shape[0] == 1:
            frame = frame[0]
        elif frame.shape[-1] == 1:
            frame = frame[..., 0]

    depth = np.clip(frame, 0.0, 1.0)

    if alpha_sigma is None:
        alpha_sigma = smooth_sigma * 1.5

    if smooth_sigma is not None and smooth_sigma > 0:
        depth_smooth = cv2.GaussianBlur(
            depth,
            ksize=(0, 0),
            sigmaX=smooth_sigma,
            sigmaY=smooth_sigma,
        )
    else:
        depth_smooth = depth

    if vmin is None or vmax is None:
        valid = depth[depth > bg_value + 1e-6]
        if valid.size > 0:
            if vmin is None:
                vmin = float(np.percentile(valid, 2))
            if vmax is None:
                vmax = float(np.percentile(valid, 98))
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

    if vmax <= vmin + 1e-8:
        vmin, vmax = 0.0, 1.0

    norm = (depth_smooth - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)

    if invert:
        norm = 1.0 - norm

    intensity_norm = norm ** gamma

    alpha = norm.copy()
    if alpha_sigma is not None and alpha_sigma > 0:
        alpha = cv2.GaussianBlur(
            alpha,
            ksize=(0, 0),
            sigmaX=alpha_sigma,
            sigmaY=alpha_sigma,
        )
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha ** alpha_gamma

    intensity = intensity_norm * 255.0

    if cool_tone:
        r = intensity * 0.92
        g = intensity * 0.95
        b = intensity * 1.00
    else:
        r = intensity
        g = intensity
        b = intensity

    vis = np.stack([r, g, b], axis=-1)
    vis = vis * alpha[..., None]

    return np.clip(vis, 0, 255).astype(np.uint8)


    if vmin is None:
        vmin = float(np.nanmin(frame))
    if vmax is None:
        vmax = float(np.nanmax(frame))

    x = (frame - vmin) / (vmax - vmin + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    rgb = plt.get_cmap("viridis")(x)[..., :3]
    return (rgb * 255).astype(np.uint8)


def _tacff_to_rgb(
    frame: np.ndarray,
    mag_max=None,
    normal_max=None,
    transpose_hw: bool = True,
    arrow_scale: float = 0.0008,
    resolution: int = 128,
    contact_percentile: float = 40,
) -> np.ndarray:
    """
    Same tactile-force-field visualization as vis_modalities.py.

    Correct channel convention:
        frame[..., 0] = fz / indentation / normal force
        frame[..., 1] = fx
        frame[..., 2] = fy

    Visualization:
        color = fz, green -> red
        arrow = (fx, fy)
        layout = transpose H/W for display by default
    """
    frame = np.asarray(frame, dtype=np.float32)

    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.moveaxis(frame, 0, -1)

    assert frame.ndim == 3 and frame.shape[-1] == 3, (
        f"Expected TacFF frame [H,W,3] or [3,H,W], got {frame.shape}"
    )

    if transpose_hw:
        frame = np.transpose(frame, (1, 0, 2))

    h, w = frame.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # verified convention: [fz, fx, fy]
    fz = np.abs(frame[..., 0])
    fx = frame[..., 1]
    fy = frame[..., 2]

    # only draw arrows in contact-ish area, as in vis_modalities.py
    contact_thr = np.nanpercentile(fz, contact_percentile)
    mask = fz >= contact_thr

    fx_vis = fx.copy()
    fy_vis = fy.copy()
    fx_vis[~mask] = 0.0
    fy_vis[~mask] = 0.0

    if normal_max is None:
        normal_max = float(np.nanpercentile(fz, 98)) + 1e-8

    contact = np.clip(fz / normal_max, 0.0, 1.0)

    colors = np.zeros((h, w, 4), dtype=np.float32)
    colors[..., 0] = contact
    colors[..., 1] = 1.0 - contact
    colors[..., 2] = 0.0
    colors[..., 3] = np.where(mask, 1.0, 0.25)

    # Keep the same paper/export aspect from vis_modalities.py.
    target_aspect = 4.0 / 3.0  # width / height
    fig_h = h * resolution / 100
    fig_w = fig_h * target_aspect

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # background taxel dots
    dot_colors = np.zeros((h, w, 4), dtype=np.float32)
    dot_colors[..., 0] = contact * 0.7
    dot_colors[..., 1] = (1.0 - contact) * 0.95
    dot_colors[..., 2] = 0.0
    dot_colors[..., 3] = 0.75

    ax.scatter(
        x_grid,
        y_grid,
        s=30,
        c=dot_colors.reshape(-1, 4),
        linewidths=0,
    )

    ax.quiver(
        x_grid,
        y_grid,
        fx_vis,
        fy_vis,
        color=colors.reshape(-1, 4),
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        width=0.008,
        headwidth=5.5,
        headlength=7.5,
        headaxislength=6.0,
        pivot="tail",
    )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)

    return img


    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.moveaxis(frame, 0, -1)

    assert frame.ndim == 3 and frame.shape[-1] == 3, (
        f"Expected TacFF frame [H,W,3] or [3,H,W], got {frame.shape}"
    )

    # [H,W,3] -> [W,H,3], 让 10x14 变成 14x10 竖版
    if transpose_hw:
        frame = np.transpose(frame, (1, 0, 2))

    h, w, _ = frame.shape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    fx = frame[..., 0]
    fy = frame[..., 1]
    fz = np.abs(frame[..., 2])

    shear_mag = np.sqrt(fx ** 2 + fy ** 2)

    if mag_max is None:
        mag_max = float(np.nanmax(shear_mag)) + 1e-8
    if normal_max is None:
        normal_max = float(np.nanmax(fz)) + 1e-8

    shear_norm = np.clip(shear_mag / mag_max, 0.0, 1.0)

    # 用 percentile 避免 normal_max 被离群值拉大
    normal_ref = np.percentile(fz, 95) + 1e-8
    contact = np.clip(fz / normal_ref, 0.0, 1.0)

    colors = np.zeros((h, w, 4), dtype=np.float32)
    colors[..., 0] = contact
    colors[..., 1] = 1.0 - 0.4 * contact
    colors[..., 2] = 0.0
    colors[..., 3] = 1.0

    fig_w = w * resolution / 100
    fig_h = h * resolution / 100

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.quiver(
        x_grid,
        y_grid,
        fx,
        -fy,
        color=colors.reshape(-1, 4),
        angles="xy",
        scale_units="xy",
        scale=0.001,
        width=0.01,
        headwidth=3.0,
        headlength=5.0,
        headaxislength=4.5,
        pivot="middle",
    )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)

    return img


def rotate_pointcloud_xyz(xyz, rx_deg=60, ry_deg=120, rz_deg=0):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    rx_mat = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
        dtype=np.float32,
    )
    ry_mat = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
        dtype=np.float32,
    )
    rz_mat = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]],
        dtype=np.float32,
    )

    rot = rz_mat @ ry_mat @ rx_mat
    valid = np.isfinite(xyz).all(axis=-1)
    center = np.nanmean(xyz[valid], axis=0)
    return (xyz - center) @ rot.T + center


def _compute_pc_lims(seq: np.ndarray, percentile: float = 2.0, zoom: float = 1.0):
    xyz = np.asarray(seq)[..., :3]
    xyz = rotate_pointcloud_xyz(xyz, rx_deg=60, ry_deg=120, rz_deg=0)
    valid_xyz = xyz[np.isfinite(xyz).all(axis=-1)]

    xyz_min = np.percentile(valid_xyz, percentile, axis=0)
    xyz_max = np.percentile(valid_xyz, 100.0 - percentile, axis=0)
    center = (xyz_min + xyz_max) / 2.0
    max_range = np.max(xyz_max - xyz_min) / 2.0 * zoom

    return [
        (center[0] - max_range, center[0] + max_range),
        (center[1] - max_range, center[1] + max_range),
        (center[2] - max_range, center[2] + max_range),
    ]


def _pc_to_rgb(frame: np.ndarray, xyz_lims=None) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    assert frame.ndim == 2 and frame.shape[-1] in [3, 6], (
        f"Expected point cloud [N,3] or [N,6], got {frame.shape}"
    )

    xyz = rotate_pointcloud_xyz(frame[:, :3], rx_deg=60, ry_deg=120, rz_deg=0)
    valid = np.isfinite(xyz).all(axis=-1)
    xyz = xyz[valid]

    if frame.shape[-1] == 6:
        rgb = frame[:, 3:6][valid]
        if np.nanmax(rgb) > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = "white"

    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=40, depthshade=False, linewidths=0)
    ax.view_init(elev=20, azim=-225)

    if xyz_lims is not None:
        ax.set_xlim(*xyz_lims[0])
        ax.set_ylim(*xyz_lims[1])
        ax.set_zlim(*xyz_lims[2])

    try:
        ax.set_box_aspect([1, 1, 1])
        ax.dist = 6
    except Exception:
        pass

    ax.set_axis_off()
    ax.grid(False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    plt.close(fig)
    return img


def infer_modality_vis_type(key: str, args) -> str:
    key_l = key.lower()
    if key == args.vision_key and args.vision_type == "pc":
        return "pc"
    if "pointcloud" in key_l or key_l in ["pc", "point_cloud"]:
        return "pc"
    if "force_field" in key_l or "tacff" in key_l:
        return "tacff"
    if "depth" in key_l:
        return "depth"
    return "rgb"


def get_gt_modality_sequence(dataset: ZarrDataset, batch: dict, key: str, env_i: int, num_frames: int):
    ep = int(batch["episode_idx"][env_i])
    start_step = int(batch["start_step"][env_i])
    start_row = int(dataset.offsets[ep] + start_step)
    rows = np.arange(start_row, start_row + num_frames)
    return np.asarray(dataset.get_col_data(key)[rows])


def render_modality_frames(seq: np.ndarray, vis_type: str):
    seq = np.asarray(seq)
    frames = []

    if vis_type == "depth":
        data = seq.astype(np.float32)

        # Same global normalization as vis_modalities.py: use valid non-background depth.
        valid_vals = []
        for t in range(data.shape[0]):
            d = np.asarray(data[t], dtype=np.float32)
            if d.ndim == 3:
                if d.shape[0] == 1:
                    d = d[0]
                elif d.shape[-1] == 1:
                    d = d[..., 0]
            d = np.clip(d, 0.0, 1.0)
            valid = d[d > 1e-6]
            if valid.size > 0:
                valid_vals.append(valid)

        if len(valid_vals) > 0:
            all_valid = np.concatenate(valid_vals, axis=0)
            vmin = float(np.percentile(all_valid, 2))
            vmax = float(np.percentile(all_valid, 98))
        else:
            vmin, vmax = 0.0, 1.0

        if vmax <= vmin + 1e-8:
            vmin, vmax = 0.0, 1.0

        for t in range(seq.shape[0]):
            frames.append(
                _depth_to_rgb(
                    seq[t],
                    vmin=vmin,
                    vmax=vmax,
                    invert=False,
                    smooth_sigma=1.0,
                    gamma=0.0,
                    alpha_gamma=0.5,
                    cool_tone=True,
                )
            )

    elif vis_type == "tacff":
        data = seq.astype(np.float32)

        if data.ndim == 4 and data.shape[1] == 3:
            data_hwc = np.moveaxis(data, 1, -1)
        else:
            data_hwc = data

        if data_hwc.shape[-1] != 3:
            raise ValueError(f"Expected tacff seq [...,3], got {data_hwc.shape}")

        # Same global normalization as vis_modalities.py, with correct [fz, fx, fy].
        transpose_hw = True
        data_for_scale = np.transpose(data_hwc, (0, 2, 1, 3)) if transpose_hw else data_hwc
        fz_all = np.abs(data_for_scale[..., 0])
        normal_max = float(np.nanpercentile(fz_all, 98)) + 1e-8

        for t in range(seq.shape[0]):
            frames.append(
                _tacff_to_rgb(
                    seq[t],
                    normal_max=normal_max,
                    transpose_hw=transpose_hw,
                    arrow_scale=0.0008,
                    resolution=128,
                    contact_percentile=40,
                )
            )

    elif vis_type == "pc":
        xyz_lims = _compute_pc_lims(seq, percentile=2.0, zoom=1.0)
        for t in range(seq.shape[0]):
            frames.append(_pc_to_rgb(seq[t], xyz_lims=xyz_lims))

    else:
        for t in range(seq.shape[0]):
            frames.append(_to_hwc_rgb(seq[t]))

    return frames


    if vis_type == "depth":
        data = seq.astype(np.float32)
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        for t in range(seq.shape[0]):
            frames.append(_depth_to_rgb(seq[t], vmin=vmin, vmax=vmax))

    elif vis_type == "tacff":
        data = seq.astype(np.float32)

        if data.ndim == 4 and data.shape[1] == 3:
            data_hwc = np.moveaxis(data, 1, -1)
        else:
            data_hwc = data

        if data_hwc.shape[-1] != 3:
            raise ValueError(f"Expected tacff seq [...,3], got {data_hwc.shape}")

        # 对应 _tacff_to_rgb 里的 transpose
        data_for_scale = np.transpose(data_hwc, (0, 2, 1, 3))

        shear_mag_all = np.linalg.norm(data_for_scale[..., :2], axis=-1)
        normal_all = np.abs(data_for_scale[..., 2])

        mag_max = float(np.nanmax(shear_mag_all)) + 1e-8
        normal_max = float(np.nanmax(normal_all)) + 1e-8

        for t in range(seq.shape[0]):
            frames.append(
                _tacff_to_rgb(
                    seq[t],
                    mag_max=mag_max,
                    normal_max=normal_max,
                    transpose_hw=True,
                    arrow_scale=0.001,
                    resolution=32,
                )
            )

    elif vis_type == "pc":
        xyz_lims = _compute_pc_lims(seq, percentile=2.0, zoom=1.0)
        for t in range(seq.shape[0]):
            frames.append(_pc_to_rgb(seq[t], xyz_lims=xyz_lims))

    else:
        for t in range(seq.shape[0]):
            frames.append(_to_hwc_rgb(seq[t]))

    return frames


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(h, w) / 500.0)
    thickness = max(1, int(round(font_scale * 2)))

    x, y = 12, 32
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(out, (x - 6, y - text_h - 8), (x + text_w + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _concat_rollout_gt_frames(rollout_frames, gt_frames):
    n = min(len(rollout_frames), len(gt_frames))
    out_frames = []
    for i in range(n):
        left = _put_label(rollout_frames[i], "ROLLOUT")
        right = _put_label(gt_frames[i], "GT")
        target_h = max(left.shape[0], right.shape[0])
        left = _resize_to_height(left, target_h)
        right = _resize_to_height(right, target_h)
        out_frames.append(np.concatenate([left, right], axis=1))
    return out_frames


def save_matched_rollout_and_gt_videos(
    dataset: ZarrDataset,
    batch: dict,
    rollout_frames: dict,
    args,
):
    out_dir = args.output_dir / "media"
    out_dir.mkdir(parents=True, exist_ok=True)

    record_envs = min(args.num_record, args.num_envs)
    target_num_frames = int(args.goal_offset_steps) + 1

    for key, frames in rollout_frames.items():
        vis_type = infer_modality_vis_type(key, args)
        rollout_seq_all = np.stack(frames, axis=0)

        for env_i in range(record_envs):
            rollout_seq = rollout_seq_all[:, env_i]
            rollout_seq = rollout_seq[:target_num_frames]

            if rollout_seq.shape[0] < target_num_frames:
                pad_num = target_num_frames - rollout_seq.shape[0]
                last_frame = rollout_seq[-1:]
                rollout_seq = np.concatenate([rollout_seq, np.repeat(last_frame, pad_num, axis=0)], axis=0)

            gt_seq = get_gt_modality_sequence(
                dataset=dataset,
                batch=batch,
                key=key,
                env_i=env_i,
                num_frames=target_num_frames,
            )

            rollout_vis_frames = render_modality_frames(seq=rollout_seq, vis_type=vis_type)
            gt_vis_frames = render_modality_frames(seq=gt_seq, vis_type=vis_type)
            combined_frames = _concat_rollout_gt_frames(rollout_vis_frames, gt_vis_frames)

            output_path = out_dir / f"{key}_rollout_vs_gt_env{env_i:03d}.mp4"
            imageio.mimsave(str(output_path), combined_frames, fps=args.fps, codec="libx264")
            print(f"Saved rollout-vs-GT video to: {output_path}", flush=True)



def _load_yaml_config() -> dict:
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            with open(sys.argv[idx + 1]) as f:
                return yaml.safe_load(f) or {}
    return {}

def main() -> None:
    cfg = _load_yaml_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data-root", type=str, required="data_root" not in cfg)
    parser.add_argument("--ckpt-path", type=str, required="ckpt_path" not in cfg)
    parser.add_argument("--isaacgym-cfg-name", type=str, required="isaacgym_cfg_name" not in cfg)
    parser.add_argument("--output-dir", type=str, required="output_dir" not in cfg)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vision-key", type=str, default="front")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])

    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="tactile_force_field_right")

    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--num-record", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)

    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--goal-offset-steps", type=int, default=20)

    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)

    parser.add_argument("--use-action-prior", action="store_true")
    parser.set_defaults(use_action_prior=True)
    parser.add_argument("--action-prior-std-scale", type=float, default=1.0)
    parser.add_argument("--action-prior-min-std", type=float, default=0.05)
    parser.add_argument("--warm-start-mode", type=str, default="prev_action", choices=["none", "prev_action"])
    parser.add_argument("--warm-start-std", type=float, default=0.15)
    parser.add_argument("--warm-start-mix", type=float, default=0.5)
    parser.add_argument("--cem-min-std", type=float, default=0.03)
    parser.add_argument("--action-smooth-weight", type=float, default=0.05)
    parser.add_argument("--action-magnitude-weight", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=22)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)
    parser.add_argument("--pc-in-channels", type=int, default=3)

    parser.add_argument("--sum-all-diffs", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)

    parser.add_argument("--tactile-dim", type=int, default=512)

    parser.add_argument("--pos-thresh", type=float, default=0.01)
    parser.add_argument("--quat-thresh-deg", type=float, default=15.0)
    parser.add_argument("--ee-pos-thresh", type=float, default=0.01)
    parser.add_argument("--ee-quat-thresh-deg", type=float, default=15.0)

    parser.add_argument("--tactile-in-channels", type=int, default=3)
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)

    parser.add_argument("--fusion-type", type=str, default="concat", choices=["concat", "gate", "film", "attn"])
    parser.add_argument("--fusion-latent-dim", type=int, default=None)
    parser.add_argument("--fusion-hidden-dim", type=int, default=None)
    parser.add_argument("--attn-d-model", type=int, default=256)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--attn-layers", type=int, default=2)
    parser.add_argument("--attn-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)

    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])
    parser.add_argument("--use-proj", action="store_true")
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--std-coeff", type=float, default=1.0)
    parser.add_argument("--sigreg-coeff", type=float, default=0.1)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sim-coeff-t", type=float, default=0.1)
    parser.add_argument("--idm-coeff", type=float, default=0.1)
    parser.add_argument("--idm-after-proj", action="store_true")
    parser.add_argument("--sim-t-after-proj", action="store_true")
    parser.add_argument("--reg-on-vision-only", action="store_true")
    parser.add_argument("--stop-on-success", action="store_true")

    if cfg:
        parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    keys_to_load = [
        "action",
        args.vision_key,
        "dof_pos",
        "dof_vel",
        "plug_pos",
        "plug_quat",
        "socket_pos_gt",
        "socket_quat",
        "ee_pos",
        "ee_quat",
    ]


    if args.vision_type == "pc" and "front" not in keys_to_load:
        keys_to_load.append("front")

    if args.use_tactile:
        keys_to_load.append(args.tactile_key)

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=1,
        num_steps=1,
        keys_to_load=keys_to_load,
        keys_to_cache=[
            "action",
            "dof_pos",
            "dof_vel",
            "plug_pos",
            "plug_quat",
            "socket_pos_gt",
            "socket_quat",
            "ee_pos",
            "ee_quat",
        ],
    )
    action_dim = dataset.get_dim("action")

    all_actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = (all_actions.std(axis=0) + 1e-6).astype(np.float32)
    print("Dataset action mean:", action_mean, flush=True)
    print("Dataset action std:", action_std, flush=True)

    planner_cfg = PlannerConfig(
        history_size=args.history_size,
        horizon=args.horizon,
        candidates=args.candidates,
        topk=args.topk,
        iterations=args.iterations,
        action_dim=action_dim,
        use_action_prior=args.use_action_prior,
        action_prior_std_scale=args.action_prior_std_scale,
        action_prior_min_std=args.action_prior_min_std,
        warm_start_mode=args.warm_start_mode,
        warm_start_std=args.warm_start_std,
        warm_start_mix=args.warm_start_mix,
        cem_min_std=args.cem_min_std,
        action_smooth_weight=args.action_smooth_weight,
        action_magnitude_weight=args.action_magnitude_weight,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        pc_in_channels=args.pc_in_channels,
        sum_all_diffs=args.sum_all_diffs,
        discount=args.discount,
        vision_dim=args.vision_dim,
        tactile_dim=args.tactile_dim,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
        tactile_in_channels=args.tactile_in_channels,
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
        fusion_type=args.fusion_type,
        fusion_latent_dim=args.fusion_latent_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        attn_d_model=args.attn_d_model,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        attn_mlp_ratio=args.attn_mlp_ratio,
        attn_dropout=args.attn_dropout,
        reg_loss_type=args.reg_loss_type,
        use_proj=args.use_proj,
        cov_coeff=args.cov_coeff,
        std_coeff=args.std_coeff,
        sigreg_coeff=args.sigreg_coeff,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sim_coeff_t=args.sim_coeff_t,
        idm_coeff=args.idm_coeff,
        idm_after_proj=args.idm_after_proj,
        sim_t_after_proj=args.sim_t_after_proj,
        reg_on_vision_only=args.reg_on_vision_only,
    )

    model = build_model(planner_cfg)
    model = load_lightning_ckpt(model, args.ckpt_path)
    model = model.to(device).eval()
    model.requires_grad_(False)

    planner = BatchedCEMPlanner(
        model=model,
        cfg=planner_cfg,
        device=device,
        action_mean=action_mean if args.use_action_prior else None,
        action_std=action_std if args.use_action_prior else None,
    )

    batch = sample_init_goal_segments(
        dataset=dataset,
        num_goals=args.num_envs,
        goal_offset_steps=args.goal_offset_steps,
        seed=args.seed,
        vision_key=args.vision_key,
        tactile_key=args.tactile_key if args.use_tactile else None,
    )

    goal_info = build_goal_info_numpy(batch, args)
    goal_plug_pos = np.asarray(batch["goal_plug_pos"], dtype=np.float32)
    goal_plug_quat = np.asarray(batch["goal_plug_quat"], dtype=np.float32)
    goal_ee_pos = np.asarray(batch["goal_ee_pos"], dtype=np.float32)
    goal_ee_quat = np.asarray(batch["goal_ee_quat"], dtype=np.float32)

    env = make_env(args)
    if hasattr(env, "seed"):
        env.seed(safe_env_seed(args.seed))

    _ = env.reset()
    reset_env_to_dataset_state(env, batch)

    zero_action = np.zeros((args.num_envs, action_dim), dtype=np.float32)
    obs, _, _, _ = env.step(zero_action)

    modality_video_keys = [args.vision_key]

    if args.vision_type == "pc" and "front" not in modality_video_keys:
        modality_video_keys.append("front")

    if args.use_tactile:
        modality_video_keys.append(args.tactile_key)

    rollout_frames = {
        k: [np.asarray(obs[k]).copy()]
        for k in modality_video_keys
        if k in obs
    }

    history = init_history_buffers(
        obs=obs,
        history_size=args.history_size,
        vision_key=args.vision_key,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
    )

    first_success_step = np.full(args.num_envs, fill_value=-1, dtype=np.int32)
    done_mask = np.zeros(args.num_envs, dtype=bool)
    plug_done_mask = np.zeros(args.num_envs, dtype=bool)
    ee_done_mask = np.zeros(args.num_envs, dtype=bool)
    metrics_over_time = []
    last_cost = np.zeros(args.num_envs, dtype=np.float32)
    prev_action = np.zeros((args.num_envs, action_dim), dtype=np.float32)

    for step_idx in range(args.max_steps):
        plan_out = planner.plan(
            current_info=history,
            goal_info=goal_info,
            prev_action=prev_action,
        )

        if isinstance(plan_out, tuple):
            action_np = np.asarray(plan_out[0], dtype=np.float32)
            if len(plan_out) > 2 and plan_out[2] is not None:
                last_cost = np.asarray(plan_out[2], dtype=np.float32)
        else:
            action_np = np.asarray(plan_out, dtype=np.float32)

        if args.stop_on_success:
            action_np = action_np.copy()
            action_np[done_mask] = 0.0


        if action_dim > 6:
            ep_ids = batch["episode_idx"]
            local_steps = batch["start_step"] + step_idx

            max_local_steps = dataset.lengths[ep_ids] - 1
            local_steps = np.minimum(local_steps, max_local_steps)

            gt_action_rows = dataset.offsets[ep_ids] + local_steps
            gt_actions_step = np.asarray(dataset.get_col_data("action")[gt_action_rows], dtype=np.float32)

            active_mask = ~done_mask if args.stop_on_success else np.ones(args.num_envs, dtype=bool)
            action_np[active_mask, 6] = gt_actions_step[active_mask, 6]


        obs, _, _, _ = env.step(action_np)
        prev_action = action_np.copy()

        for k in modality_video_keys:
            if k in obs:
                rollout_frames[k].append(np.asarray(obs[k]).copy())

        history = update_history_buffers(
            history=history,
            obs=obs,
            vision_key=args.vision_key,
            use_tactile=args.use_tactile,
            tactile_key=args.tactile_key,
        )

        m = end_pose_metrics_joint(
            current_plug_pos=np.asarray(obs["plug_pos"], dtype=np.float32),
            current_plug_quat=np.asarray(obs["plug_quat"], dtype=np.float32),
            goal_plug_pos=goal_plug_pos,
            goal_plug_quat=goal_plug_quat,
            current_ee_pos=np.asarray(obs["ee_pos"], dtype=np.float32),
            current_ee_quat=np.asarray(obs["ee_quat"], dtype=np.float32),
            goal_ee_pos=goal_ee_pos,
            goal_ee_quat=goal_ee_quat,
            plug_pos_thresh=args.pos_thresh,
            plug_quat_thresh_deg=args.quat_thresh_deg,
            ee_pos_thresh=args.ee_pos_thresh,
            ee_quat_thresh_deg=args.ee_quat_thresh_deg,
        )
        metrics_over_time.append(m)

        success_now = m["success"].astype(bool)
        plug_success_now = m["plug_success"].astype(bool)
        ee_success_now = m["ee_success"].astype(bool)

        newly_success = (~done_mask) & success_now
        first_success_step[newly_success] = step_idx + 1

        done_mask |= success_now
        plug_done_mask |= plug_success_now
        ee_done_mask |= ee_success_now

        print(
            f"[step {step_idx + 1:03d}/{args.max_steps}] "
            f"joint_success_now={success_now.mean():.4f} "
            f"joint_ever_success={done_mask.mean():.4f} "
            f"plug_ever_success={plug_done_mask.mean():.4f} "
            f"ee_ever_success={ee_done_mask.mean():.4f} "
            f"mean_plug_pos_err={m['plug_pos_err'].mean():.6f} "
            f"mean_plug_quat_err_deg={m['plug_quat_err_deg'].mean():.6f} "
            f"mean_ee_pos_err={m['ee_pos_err'].mean():.6f} "
            f"mean_ee_quat_err_deg={m['ee_quat_err_deg'].mean():.6f}",
            flush=True,
        )

        if args.stop_on_success and done_mask.all():
            break

    final_metrics = metrics_over_time[-1] if metrics_over_time else {
        "success": np.zeros(args.num_envs, dtype=np.int32),
        "plug_success": np.zeros(args.num_envs, dtype=np.int32),
        "ee_success": np.zeros(args.num_envs, dtype=np.int32),
        "plug_pos_err": np.full(args.num_envs, np.nan, dtype=np.float32),
        "plug_quat_err_deg": np.full(args.num_envs, np.nan, dtype=np.float32),
        "ee_pos_err": np.full(args.num_envs, np.nan, dtype=np.float32),
        "ee_quat_err_deg": np.full(args.num_envs, np.nan, dtype=np.float32),
    }

    successful_steps = first_success_step[first_success_step > 0]

    summary = {
        "num_envs": int(args.num_envs),
        "max_steps": int(args.max_steps),
        "num_executed_steps": int(len(metrics_over_time)),
        "num_final_plug_success": int(final_metrics["plug_success"].sum()),
        "num_final_ee_success": int(final_metrics["ee_success"].sum()),
        "num_final_joint_success": int(final_metrics["success"].sum()),
        "mean_first_success_step": float(successful_steps.mean()) if len(successful_steps) > 0 else None,
        "mean_plug_pos_err": float(np.mean(final_metrics["plug_pos_err"])),
        "mean_plug_quat_err_deg": float(np.mean(final_metrics["plug_quat_err_deg"])),
        "mean_ee_pos_err": float(np.mean(final_metrics["ee_pos_err"])),
        "mean_ee_quat_err_deg": float(np.mean(final_metrics["ee_quat_err_deg"])),
        "last_cost_mean": float(np.mean(last_cost)) if last_cost is not None else None,
        "last_cost_std": float(np.std(last_cost)) if last_cost is not None else None,
    }

    print("\n===== Final Summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    with open(args.output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    save_matched_rollout_and_gt_videos(
        dataset=dataset,
        batch=batch,
        rollout_frames=rollout_frames,
        args=args,
    )

    print("Finished planning and saved all results.", flush=True)

    # Avoid IsaacGym/CUDA C++ destructor segfault on exit.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    os._exit(0)


if __name__ == "__main__":
    main()
