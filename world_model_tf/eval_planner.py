from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config_utils import build_config_aware_parser
from dataset import DEFAULT_LOWDIM_KEYS, ManiFeelSequenceDataset, parse_key_list
from encoders import DINO_TOKEN_MODES, DINO_TOKEN_STRATEGIES, DinoSpatialEncoder
from model import TACTILE_POOL_MODES, PointCloudSpatialEncoder, PredictorConfig, SpatialActionConditionedModel
from planner_utils import PLANNER_COST_MODES, BatchedCEMPlanner, PlannerConfig, load_lightning_ckpt


def _build_parser(pre_parser):
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
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

    parser.add_argument("--ctxt-window", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--candidate-chunk-size", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--goal-offset-steps", type=int, default=20)
    parser.add_argument("--num-goals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-mode", type=str, default="joint", choices=list(PLANNER_COST_MODES))
    parser.add_argument("--visual-cost-weight", type=float, default=1.0)
    parser.add_argument("--lowdim-cost-weight", type=float, default=1.0)
    parser.add_argument("--tactile-cost-weight", type=float, default=1.0)
    parser.add_argument("--sum-all-diffs", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)
    return parser


def parse_args():
    parser = build_config_aware_parser(_build_parser)
    args = parser.parse_args()
    missing = []
    if args.checkpoint is None:
        missing.append("--checkpoint")
    if args.data_root is None:
        missing.append("--data-root")
    if missing:
        parser.error(f"{', '.join(missing)} is required (provide it in the command line or config YAML).")
    return args


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
        lowdim_keys=lowdim_keys,
    )


def build_model(args, dataset):
    if args.vision_type == "pc":
        encoder = PointCloudSpatialEncoder(
            in_channels=args.pc_in_channels,
            num_points=int(dataset.data[args.vision_key].shape[1]),
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


def sample_goal_pairs(dataset, args, lowdim_keys):
    rng = np.random.default_rng(args.seed)
    valid = []
    for ep_idx, ep_len in enumerate(dataset.lengths):
        max_start = int(ep_len - args.goal_offset_steps - 1)
        if max_start >= 0:
            for start in range(max_start + 1):
                valid.append((ep_idx, start, start + args.goal_offset_steps))
    if not valid:
        raise RuntimeError("No valid init/goal pairs found.")
    picks = rng.choice(len(valid), size=args.num_goals, replace=len(valid) < args.num_goals)
    pairs = [valid[int(i)] for i in picks]
    start_rows = np.asarray([dataset.offsets[ep] + start for ep, start, _ in pairs], dtype=np.int64)
    goal_rows = np.asarray([dataset.offsets[ep] + goal for ep, _, goal in pairs], dtype=np.int64)

    current = {args.vision_key: np.repeat(np.asarray(dataset.data[args.vision_key][start_rows])[:, None], args.ctxt_window, axis=1)}
    goal = {args.vision_key: np.asarray(dataset.data[args.vision_key][goal_rows])[:, None]}
    for key in lowdim_keys:
        current[key] = np.repeat(np.asarray(dataset.data[key][start_rows])[:, None], args.ctxt_window, axis=1)
        goal[key] = np.asarray(dataset.data[key][goal_rows])[:, None]
    if args.use_tactile:
        current[args.tactile_key] = np.repeat(
            np.asarray(dataset.data[args.tactile_key][start_rows])[:, None],
            args.ctxt_window,
            axis=1,
        )
        goal[args.tactile_key] = np.asarray(dataset.data[args.tactile_key][goal_rows])[:, None]
    current["action"] = np.zeros((args.num_goals, max(args.ctxt_window - 1, 0), dataset.action_dim()), dtype=np.float32)
    return current, goal


def main():
    args = parse_args()
    lowdim_keys = tuple(parse_key_list(args.lowdim_keys))
    dataset = build_dataset(args, lowdim_keys)
    model = build_model(args, dataset)
    current, goal = sample_goal_pairs(dataset, args, lowdim_keys)
    planner = BatchedCEMPlanner(
        model=model,
        cfg=PlannerConfig(
            horizon=args.horizon,
            candidates=args.candidates,
            candidate_chunk_size=args.candidate_chunk_size,
            topk=args.topk,
            iterations=args.iterations,
            ctxt_window=args.ctxt_window,
            action_dim=dataset.action_dim(),
            cost_mode=args.cost_mode,
            visual_cost_weight=args.visual_cost_weight,
            lowdim_cost_weight=args.lowdim_cost_weight,
            tactile_cost_weight=args.tactile_cost_weight,
            sum_all_diffs=args.sum_all_diffs,
            discount=args.discount,
        ),
        device=args.device,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        lowdim_keys=lowdim_keys,
        lowdim_stats=dataset.lowdim_stats,
        pc_in_channels=args.pc_in_channels,
        pointcloud_scale=args.pointcloud_scale,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
        tactile_force_scale=args.tactile_force_scale,
    )
    best_action, _, best_cost = planner.plan(current, goal)
    print(f"planned batch: {best_action.shape}")
    print(f"mean cost: {float(np.mean(best_cost)):.6f}")
    print(f"first action: {best_action[0].tolist()}")


if __name__ == "__main__":
    main()
