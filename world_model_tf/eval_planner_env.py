from __future__ import annotations

# IsaacGym MUST be imported before torch.
import isaacgym  # noqa: F401

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict

from config_utils import build_config_aware_parser
from dataset import DEFAULT_LOWDIM_KEYS, ManiFeelSequenceDataset, parse_key_list
from encoders import DINO_TOKEN_MODES, DINO_TOKEN_STRATEGIES, DinoSpatialEncoder
from model import TACTILE_POOL_MODES, PointCloudSpatialEncoder, PredictorConfig, SpatialActionConditionedModel
from pointcloud_utils import PC_TOKENIZERS, effective_pc_num_points
from planner_utils import (
    PLANNER_COST_MODES,
    BatchedCEMPlanner,
    PlannerConfig,
    init_history_buffers,
    load_lightning_ckpt,
    update_history_buffers,
)


WORLD_MODEL_TF_DIR = Path(__file__).resolve().parent
MANIFEEL_DIR = Path(os.environ.get("MANIFEEL_DIR", WORLD_MODEL_TF_DIR.parent)).expanduser().resolve()
if str(MANIFEEL_DIR) not in sys.path:
    sys.path.append(str(MANIFEEL_DIR))

from envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper  # noqa: E402


def is_rgb_like_key(key: str) -> bool:
    key = key.lower()
    return any(tok in key for tok in ["image", "camera", "rgb", "taxim", "vision"])


def ensure_pointcloud_depth_camera(cfg) -> None:
    camera_configs = cfg.task.env.get("camera_configs")
    if camera_configs is None:
        raise ValueError("Pointcloud eval requires task.env.camera_configs.")

    names = [str(cam.get("name")) for cam in camera_configs]
    if "front_depth" in names:
        return
    if "front" not in names:
        raise ValueError("Pointcloud eval requires a front RGB camera to clone into front_depth.")

    front_cfg = next(cam for cam in camera_configs if str(cam.get("name")) == "front")
    depth_cfg = OmegaConf.create(OmegaConf.to_container(front_cfg, resolve=True))
    with open_dict(depth_cfg):
        depth_cfg.name = "front_depth"
        depth_cfg.image_type = "depth"

    with open_dict(cfg.task.env):
        camera_configs.append(depth_cfg)


def make_env(args) -> MultipleIsaacEnvWrapper:
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=str(MANIFEEL_DIR / "config"), version_base="1.1")

    cfg = compose(config_name=args.isaacgym_cfg_name)
    cfg.num_envs = args.num_envs
    cfg.headless = True
    cfg.capture_video = False
    cfg.force_render = bool(args.vision_type == "pc")
    if args.vision_type == "pc":
        ensure_pointcloud_depth_camera(cfg)

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

    obs_meta["front"] = {"type": "rgb"}
    if args.vision_type == "image":
        obs_meta[args.vision_key] = {"type": "rgb"}
    else:
        obs_meta[args.vision_key] = {"type": "low_dim"}

    if args.use_tactile:
        obs_meta[args.tactile_key] = {"type": "rgb" if is_rgb_like_key(args.tactile_key) else "low_dim"}

    cfg["shape_meta"] = OmegaConf.create({"obs": obs_meta})
    # The IsaacGym task allocates obs_dict from task.env.obsDims before the wrapper
    # filters by shape_meta. Keep these in sync with the keys the USB task writes.
    required_obs_dims = {
        "ee_pos": [3],
        "ee_quat": [4],
        "plug_pos": [3],
        "plug_quat": [4],
        "socket_pos": [3],
        "socket_pos_gt": [3],
        "socket_quat": [4],
        "dof_pos": [9],
        "dof_vel": [9],
        "tactile_depth_left": [320, 240],
        "tactile_depth_right": [320, 240],
        "tactile_rgb_left": [320, 240, 3],
        "tactile_rgb_right": [320, 240, 3],
        "left_tactile_camera_taxim": [320, 240, 3],
        "right_tactile_camera_taxim": [320, 240, 3],
        "tactile_force_field_left": [10, 14, 3],
        "tactile_force_field_right": [10, 14, 3],
    }
    if args.vision_type == "pc":
        raw_num_points = getattr(args, "pc_raw_num_points", None) or 1024
        required_obs_dims[args.vision_key] = [int(raw_num_points), args.pc_in_channels]
    if args.use_tactile:
        tactile_key = args.tactile_key.lower()
        if "force_field" in tactile_key or "tacff" in tactile_key:
            required_obs_dims[args.tactile_key] = [args.tactile_height, args.tactile_width, 3]
        elif "depth" in tactile_key:
            required_obs_dims[args.tactile_key] = [320, 240]
        else:
            required_obs_dims[args.tactile_key] = [320, 240, 3]
    with open_dict(cfg.task.env.obsDims):
        for key, dims in required_obs_dims.items():
            cfg.task.env.obsDims[key] = dims
    return MultipleIsaacEnvWrapper(cfg)


def np_quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


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
) -> dict[str, np.ndarray]:
    plug_pos_err = np.linalg.norm(current_plug_pos - goal_plug_pos, axis=-1)
    plug_quat_err_deg = np_quat_angle_deg(current_plug_quat, goal_plug_quat)
    ee_pos_err = np.linalg.norm(current_ee_pos - goal_ee_pos, axis=-1)
    ee_quat_err_deg = np_quat_angle_deg(current_ee_quat, goal_ee_quat)
    plug_success = (plug_pos_err < plug_pos_thresh) & (plug_quat_err_deg < plug_quat_thresh_deg)
    ee_success = (ee_pos_err < ee_pos_thresh) & (ee_quat_err_deg < ee_quat_thresh_deg)
    success = plug_success & ee_success
    return {
        "success": success.astype(np.int32),
        "plug_success": plug_success.astype(np.int32),
        "ee_success": ee_success.astype(np.int32),
        "plug_pos_err": plug_pos_err,
        "plug_quat_err_deg": plug_quat_err_deg,
        "ee_pos_err": ee_pos_err,
        "ee_quat_err_deg": ee_quat_err_deg,
    }


def sample_init_goal_segments(dataset: ManiFeelSequenceDataset, args, lowdim_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    valid = []
    for ep_idx, ep_len in enumerate(dataset.lengths):
        max_start = int(ep_len - args.goal_offset_steps - 1)
        if max_start >= 0:
            for start in range(max_start + 1):
                valid.append((ep_idx, start, start + args.goal_offset_steps))
    if not valid:
        raise RuntimeError("No valid init/goal segments found.")

    choice_idx = rng.choice(len(valid), size=args.num_envs, replace=len(valid) < args.num_envs)
    chosen = [valid[int(i)] for i in choice_idx]
    ep_ids = np.asarray([x[0] for x in chosen], dtype=np.int64)
    start_steps = np.asarray([x[1] for x in chosen], dtype=np.int64)
    goal_steps = np.asarray([x[2] for x in chosen], dtype=np.int64)
    init_rows = dataset.offsets[ep_ids] + start_steps
    goal_rows = dataset.offsets[ep_ids] + goal_steps

    data = dataset.data
    batch = {
        "episode_idx": ep_ids,
        "start_step": start_steps,
        "goal_step": goal_steps,
        "init_row_idx": init_rows.astype(np.int64),
        "goal_row_idx": goal_rows.astype(np.int64),
        "init_dof_pos": np.asarray(data["dof_pos"][init_rows]),
        "init_dof_vel": np.asarray(data["dof_vel"][init_rows]),
        "init_plug_pos": np.asarray(data["plug_pos"][init_rows]),
        "init_plug_quat": np.asarray(data["plug_quat"][init_rows]),
        "init_socket_pos": np.asarray(data["socket_pos_gt"][init_rows]),
        "init_socket_quat": np.asarray(data["socket_quat"][init_rows]),
        "goal_plug_pos": np.asarray(data["plug_pos"][goal_rows]),
        "goal_plug_quat": np.asarray(data["plug_quat"][goal_rows]),
        "goal_ee_pos": np.asarray(data["ee_pos"][goal_rows]),
        "goal_ee_quat": np.asarray(data["ee_quat"][goal_rows]),
        args.vision_key: np.asarray(data[args.vision_key][goal_rows]),
    }

    for key in lowdim_keys:
        batch[f"goal_{key}"] = np.asarray(data[key][goal_rows])
    if args.use_tactile:
        batch[args.tactile_key] = np.asarray(data[args.tactile_key][goal_rows])
    return batch


def reset_env_to_dataset_state(env: MultipleIsaacEnvWrapper, batch: dict) -> None:
    env.reset_to_dataset_state(
        dof_pos=batch["init_dof_pos"],
        dof_vel=batch["init_dof_vel"],
        plug_pos=batch["init_plug_pos"],
        plug_quat=batch["init_plug_quat"],
        socket_pos=batch["init_socket_pos"],
        socket_quat=batch["init_socket_quat"],
    )


def build_goal_info_numpy(batch: dict, args, lowdim_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    goal_info = {args.vision_key: batch[args.vision_key][:, None]}
    for key in lowdim_keys:
        goal_info[key] = batch[f"goal_{key}"][:, None]
    if args.use_tactile:
        goal_info[args.tactile_key] = batch[args.tactile_key][:, None]
    return goal_info


def build_dataset(args, lowdim_keys):
    return ManiFeelSequenceDataset(
        root=args.data_root,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        action_key="action",
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        image_size=args.image_size,
        pc_in_channels=args.pc_in_channels,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
        tactile_force_scale=args.tactile_force_scale,
        pointcloud_scale=args.pointcloud_scale,
        pc_tokenizer=args.pc_tokenizer,
        pc_num_points=args.pc_num_points,
        pc_order_mode=args.pc_order_mode,
        pc_voxel_grid=args.pc_voxel_grid,
        pc_bounds=args.pc_bounds,
        lowdim_keys=lowdim_keys,
    )


def build_model(args, dataset):
    if args.vision_type == "pc":
        encoder = PointCloudSpatialEncoder(
            in_channels=args.pc_in_channels,
            num_points=effective_pc_num_points(
                int(dataset.data[args.vision_key].shape[1]),
                pc_tokenizer=args.pc_tokenizer,
                pc_num_points=args.pc_num_points,
                pc_voxel_grid=args.pc_voxel_grid,
            ),
            embed_dim=args.predictor_embed_dim,
        )
    else:
        encoder = DinoSpatialEncoder(
            model_name=args.dino_name,
            image_size=args.image_size,
            checkpoint_path=args.dino_checkpoint,
            token_mode=args.dino_token_mode,
            token_strategy=args.dino_token_strategy,
            last_layers=args.dino_last_layers,
        )

    cfg = PredictorConfig(
        predictor_embed_dim=args.predictor_embed_dim,
        depth=args.predictor_depth,
        num_heads=args.predictor_heads,
        mlp_ratio=args.predictor_mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        reg_loss_type=args.reg_loss_type,
    )
    model = SpatialActionConditionedModel(
        encoder=encoder,
        action_dim=dataset.action_dim(),
        lowdim_dim=dataset.lowdim_dim(),
        num_frames=args.num_steps,
        config=cfg,
        use_tactile=args.use_tactile,
        tactile_channels=dataset.tactile_channels(),
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
        tactile_pool_mode=args.tactile_pool_mode,
    )
    model = load_lightning_ckpt(model, args.checkpoint)
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    return model


def _build_parser(pre_parser):
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--isaacgym-cfg-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--encoder", type=str, default="dino", choices=["dino"])
    parser.add_argument("--vision-key", type=str, default="front")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pc-in-channels", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=6)
    parser.add_argument("--lowdim-keys", type=str, default=",".join(DEFAULT_LOWDIM_KEYS))
    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="tactile_force_field_right")
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)
    parser.add_argument("--tactile-force-scale", type=float, default=0.002)
    parser.add_argument("--tactile-pool-mode", type=str, default="mean", choices=TACTILE_POOL_MODES)
    parser.add_argument("--pointcloud-scale", type=float, default=0.4)
    parser.add_argument("--pc-tokenizer", type=str, default="none", choices=PC_TOKENIZERS)
    parser.add_argument("--pc-num-points", type=int, default=None)
    parser.add_argument("--pc-raw-num-points", type=int, default=None)
    parser.add_argument("--pc-order-mode", type=str, default="none", choices=["none", "xyz"])
    parser.add_argument("--pc-voxel-grid", default="16,16,1")
    parser.add_argument("--pc-bounds", default="-0.4,0.8,-0.6,0.6,-0.2,0.8")

    parser.add_argument("--dino-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--dino-token-mode", type=str, default="patch", choices=DINO_TOKEN_MODES)
    parser.add_argument("--dino-token-strategy", type=str, default="patch_only", choices=DINO_TOKEN_STRATEGIES)
    parser.add_argument("--dino-last-layers", type=int, default=4)
    parser.add_argument("--predictor-embed-dim", type=int, default=384)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--predictor-heads", type=int, default=None)
    parser.add_argument("--predictor-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])

    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--ctxt-window", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--goal-offset-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-mode", type=str, default="joint", choices=list(PLANNER_COST_MODES))
    parser.add_argument("--visual-cost-weight", type=float, default=1.0)
    parser.add_argument("--lowdim-cost-weight", type=float, default=1.0)
    parser.add_argument("--tactile-cost-weight", type=float, default=1.0)
    parser.add_argument("--sum-all-diffs", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)
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
    parser.add_argument("--stop-on-success", action="store_true")

    parser.add_argument("--pos-thresh", type=float, default=0.01)
    parser.add_argument("--quat-thresh-deg", type=float, default=15.0)
    parser.add_argument("--ee-pos-thresh", type=float, default=0.01)
    parser.add_argument("--ee-quat-thresh-deg", type=float, default=15.0)
    return parser


def parse_args():
    parser = build_config_aware_parser(_build_parser)
    args = parser.parse_args()
    missing = []
    for name in ("checkpoint", "data_root", "isaacgym_cfg_name", "output_dir"):
        if getattr(args, name) is None:
            missing.append("--" + name.replace("_", "-"))
    if missing:
        parser.error(f"{', '.join(missing)} is required (provide it in command line or config YAML).")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    lowdim_keys = tuple(parse_key_list(args.lowdim_keys))
    print("[INFO] loading dataset", flush=True)
    dataset = build_dataset(args, lowdim_keys)
    if args.pc_raw_num_points is None and args.vision_type == "pc":
        args.pc_raw_num_points = int(dataset.data[args.vision_key].shape[1])
    action_dim = dataset.action_dim()
    all_actions = np.asarray(dataset.data["action"], dtype=np.float32).reshape(-1, action_dim)
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = (all_actions.std(axis=0) + 1e-6).astype(np.float32)
    print("Dataset action mean:", action_mean, flush=True)
    print("Dataset action std:", action_std, flush=True)

    print("[INFO] loading model", flush=True)
    model = build_model(args, dataset)
    print("[INFO] initializing planner", flush=True)
    planner = BatchedCEMPlanner(
        model=model,
        cfg=PlannerConfig(
            horizon=args.horizon,
            candidates=args.candidates,
            candidate_chunk_size=args.candidate_chunk_size,
            topk=args.topk,
            iterations=args.iterations,
            ctxt_window=args.ctxt_window,
            action_dim=action_dim,
            cost_mode=args.cost_mode,
            visual_cost_weight=args.visual_cost_weight,
            lowdim_cost_weight=args.lowdim_cost_weight,
            tactile_cost_weight=args.tactile_cost_weight,
            sum_all_diffs=args.sum_all_diffs,
            discount=args.discount,
            cem_min_std=args.cem_min_std,
            use_action_prior=args.use_action_prior,
            action_prior_std_scale=args.action_prior_std_scale,
            action_prior_min_std=args.action_prior_min_std,
            warm_start_mode=args.warm_start_mode,
            warm_start_std=args.warm_start_std,
            warm_start_mix=args.warm_start_mix,
            action_smooth_weight=args.action_smooth_weight,
            action_magnitude_weight=args.action_magnitude_weight,
        ),
        device=args.device,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        lowdim_keys=lowdim_keys,
        lowdim_stats=dataset.lowdim_stats,
        pc_in_channels=args.pc_in_channels,
        pointcloud_scale=args.pointcloud_scale,
        pc_tokenizer=args.pc_tokenizer,
        pc_num_points=args.pc_num_points,
        pc_order_mode=args.pc_order_mode,
        pc_voxel_grid=args.pc_voxel_grid,
        pc_bounds=args.pc_bounds,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
        tactile_force_scale=args.tactile_force_scale,
        action_mean=action_mean if args.use_action_prior else None,
        action_std=action_std if args.use_action_prior else None,
    )

    print("[INFO] sampling init/goal segments", flush=True)
    batch = sample_init_goal_segments(dataset, args, lowdim_keys)
    goal_info = build_goal_info_numpy(batch, args, lowdim_keys)
    goal_plug_pos = np.asarray(batch["goal_plug_pos"], dtype=np.float32)
    goal_plug_quat = np.asarray(batch["goal_plug_quat"], dtype=np.float32)
    goal_ee_pos = np.asarray(batch["goal_ee_pos"], dtype=np.float32)
    goal_ee_quat = np.asarray(batch["goal_ee_quat"], dtype=np.float32)

    print("[INFO] creating IsaacGym env", flush=True)
    env = make_env(args)
    print("[INFO] resetting IsaacGym env", flush=True)
    _ = env.reset()
    reset_env_to_dataset_state(env, batch)
    zero_action = np.zeros((args.num_envs, action_dim), dtype=np.float32)
    obs, _, _, _ = env.step(zero_action)
    print("[INFO] starting planner rollout", flush=True)

    history = init_history_buffers(
        obs=obs,
        ctxt_window=args.ctxt_window,
        action_dim=action_dim,
        vision_key=args.vision_key,
        lowdim_keys=lowdim_keys,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
    )

    first_success_step = np.full(args.num_envs, -1, dtype=np.int32)
    done_mask = np.zeros(args.num_envs, dtype=bool)
    plug_done_mask = np.zeros(args.num_envs, dtype=bool)
    ee_done_mask = np.zeros(args.num_envs, dtype=bool)
    metrics_over_time = []
    last_cost = np.zeros(args.num_envs, dtype=np.float32)
    prev_action = zero_action.copy()

    for step_idx in range(args.max_steps):
        action_np, _, cost_np = planner.plan(history, goal_info, prev_action=prev_action)
        last_cost = np.asarray(cost_np, dtype=np.float32)
        if args.stop_on_success:
            action_np = action_np.copy()
            action_np[done_mask] = 0.0

        obs, _, _, _ = env.step(action_np)
        prev_action = action_np.copy()
        history = update_history_buffers(
            history=history,
            obs=obs,
            action=action_np,
            vision_key=args.vision_key,
            lowdim_keys=lowdim_keys,
            use_tactile=args.use_tactile,
            tactile_key=args.tactile_key,
        )

        metrics = end_pose_metrics_joint(
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
        metrics_over_time.append(metrics)
        success_now = metrics["success"].astype(bool)
        plug_success_now = metrics["plug_success"].astype(bool)
        ee_success_now = metrics["ee_success"].astype(bool)
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
            f"mean_plug_pos_err={metrics['plug_pos_err'].mean():.6f} "
            f"mean_plug_quat_err_deg={metrics['plug_quat_err_deg'].mean():.6f} "
            f"mean_ee_pos_err={metrics['ee_pos_err'].mean():.6f} "
            f"mean_ee_quat_err_deg={metrics['ee_quat_err_deg'].mean():.6f} "
            f"plan_cost={last_cost.mean():.6f}",
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
        "joint_success_rate": float(final_metrics["success"].mean()),
        "plug_success_rate": float(final_metrics["plug_success"].mean()),
        "ee_success_rate": float(final_metrics["ee_success"].mean()),
        "mean_first_success_step": float(successful_steps.mean()) if successful_steps.size > 0 else None,
        "mean_plug_pos_err": float(np.mean(final_metrics["plug_pos_err"])),
        "mean_plug_quat_err_deg": float(np.mean(final_metrics["plug_quat_err_deg"])),
        "mean_ee_pos_err": float(np.mean(final_metrics["ee_pos_err"])),
        "mean_ee_quat_err_deg": float(np.mean(final_metrics["ee_quat_err_deg"])),
        "last_cost_mean": float(np.mean(last_cost)),
        "last_cost_std": float(np.std(last_cost)),
    }
    print("\n===== Final Summary =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    with (args.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (args.output_dir / "sampled_goals.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "episode_idx": batch["episode_idx"].tolist(),
                "start_step": batch["start_step"].tolist(),
                "goal_step": batch["goal_step"].tolist(),
                "goal_row_idx": batch["goal_row_idx"].tolist(),
            },
            handle,
            indent=2,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    os._exit(0)


if __name__ == "__main__":
    main()
