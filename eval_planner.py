from __future__ import annotations

# IsaacGym MUST be imported before torch.
import isaacgym

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch

from hydra import compose, initialize
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
        initialize(config_path="config", version_base="1.1")

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


def _depth_to_rgb(frame: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        if frame.shape[0] == 1:
            frame = frame[0]
        elif frame.shape[-1] == 1:
            frame = frame[..., 0]

    if vmin is None:
        vmin = float(np.nanmin(frame))
    if vmax is None:
        vmax = float(np.nanmax(frame))

    x = (frame - vmin) / (vmax - vmin + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    rgb = plt.get_cmap("viridis")(x)[..., :3]
    return (rgb * 255).astype(np.uint8)


def _tacff_to_rgb(frame: np.ndarray, mag_max=None) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.moveaxis(frame, 0, -1)

    assert frame.ndim == 3 and frame.shape[-1] == 3, (
        f"Expected TacFF frame [H,W,3] or [3,H,W], got {frame.shape}"
    )

    h, w, _ = frame.shape
    fx = frame[..., 0]
    fy = frame[..., 1]
    mag = np.linalg.norm(frame, axis=-1)

    if mag_max is None:
        mag_max = float(np.max(mag)) + 1e-8

    norm_mag = np.clip(mag / mag_max, 0.0, 1.0)
    x = np.arange(w)
    y = np.arange(h)
    x_grid, y_grid = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(w * 0.4, h * 0.4), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.quiver(
        x_grid,
        y_grid,
        fx,
        -fy,
        norm_mag,
        cmap="RdYlGn_r",
        clim=(0.0, 1.0),
        angles="xy",
        scale_units="xy",
        scale=0.001,
        width=0.01,
        pivot="middle",
    )

    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.35)
    ax.tick_params(colors="white", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout()
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
        mag_max = float(np.max(np.linalg.norm(data_hwc, axis=-1))) + 1e-8
        for t in range(seq.shape[0]):
            frames.append(_tacff_to_rgb(seq[t], mag_max=mag_max))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--isaacgym-cfg-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

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
