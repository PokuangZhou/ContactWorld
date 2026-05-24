from __future__ import annotations

import argparse
from pathlib import Path

import lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from config_utils import build_config_aware_parser
from dataset import DEFAULT_LOWDIM_KEYS, ManiFeelSequenceDataset, parse_key_list
from encoders import DINO_TOKEN_MODES, DINO_TOKEN_STRATEGIES, DinoSpatialEncoder, VJEPASpatialEncoder
from lightning_module import PredictorTrainingModule
from model import PointCloudSpatialEncoder, PredictorConfig, SpatialActionConditionedModel


def _build_parser(pre_parser):
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="dino", choices=["dino", "vjepa"])
    parser.add_argument("--vision-key", type=str, default="front")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pc-in-channels", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--eval-split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--log-every-n-batches", type=int, default=10)
    parser.add_argument(
        "--action-mode",
        type=str,
        default="gt",
        choices=["gt", "zero", "random_uniform", "all"],
    )
    parser.add_argument(
        "--proprio-mode",
        type=str,
        default="use_ground_truth",
        choices=["use_ground_truth", "predict_proprio", "gt", "predict"],
    )

    parser.add_argument("--dino-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--dino-checkpoint", type=str, default=None)
    parser.add_argument("--dino-token-mode", type=str, default="patch", choices=DINO_TOKEN_MODES)
    parser.add_argument("--dino-token-strategy", type=str, default="patch_only", choices=DINO_TOKEN_STRATEGIES)
    parser.add_argument("--dino-last-layers", type=int, default=4)
    parser.add_argument("--vjepa-arch", type=str, default="vit_large_rope")
    parser.add_argument("--vjepa-checkpoint", type=str, default=None)
    parser.add_argument("--vjepa-checkpoint-key", type=str, default="target_encoder")
    parser.add_argument("--no-vjepa-batchify-video", dest="vjepa_batchify_video", action="store_false")
    parser.add_argument("--no-vjepa-dup-image", dest="vjepa_dup_image", action="store_false")
    parser.add_argument("--vjepa-normalize-reps", action="store_true")
    parser.set_defaults(vjepa_batchify_video=True, vjepa_dup_image=True, vjepa_normalize_reps=False)

    parser.add_argument("--predictor-embed-dim", type=int, default=384)
    parser.add_argument("--predictor-depth", type=int, default=6)
    parser.add_argument("--predictor-heads", type=int, default=None)
    parser.add_argument("--predictor-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--attn-drop-rate", type=float, default=0.0)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="tactile_force_field_right")
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)
    parser.add_argument("--tactile-force-scale", type=float, default=0.002)
    parser.add_argument("--pointcloud-scale", type=float, default=0.4)
    parser.add_argument("--lowdim-keys", type=str, default=",".join(DEFAULT_LOWDIM_KEYS))

    parser.add_argument("--visual-l2-weight", type=float, default=1.0)
    parser.add_argument("--visual-cos-weight", type=float, default=0.0)
    parser.add_argument("--lowdim-l2-weight", type=float, default=1.0)
    parser.add_argument("--lowdim-cos-weight", type=float, default=0.0)
    parser.add_argument("--tactile-l2-weight", type=float, default=1.0)
    parser.add_argument("--tactile-cos-weight", type=float, default=0.0)
    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--std-coeff", type=float, default=1.0)
    parser.add_argument("--sigreg-coeff", type=float, default=0.1)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sim-coeff-t", type=float, default=0.1)
    parser.add_argument("--idm-coeff", type=float, default=0.1)
    parser.add_argument("--rollout-ctxt-window", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=None)
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


def build_encoder(args, dataset: ManiFeelSequenceDataset):
    if args.vision_type == "pc":
        num_points = int(dataset.data[args.vision_key].shape[1])
        return PointCloudSpatialEncoder(
            in_channels=args.pc_in_channels,
            num_points=num_points,
            embed_dim=args.predictor_embed_dim,
        )
    if args.encoder == "dino":
        return DinoSpatialEncoder(
            model_name=args.dino_name,
            image_size=args.image_size,
            checkpoint_path=args.dino_checkpoint,
            token_mode=args.dino_token_mode,
            token_strategy=args.dino_token_strategy,
            last_layers=args.dino_last_layers,
        )
    return VJEPASpatialEncoder(
        arch_name=args.vjepa_arch,
        checkpoint_path=args.vjepa_checkpoint,
        checkpoint_key=args.vjepa_checkpoint_key,
        image_size=args.image_size,
        num_steps=args.num_steps,
        batchify_video=args.vjepa_batchify_video,
        dup_image=args.vjepa_dup_image,
        normalize_reps=args.vjepa_normalize_reps,
    )


def validate_temporal_config(args, encoder):
    encoded_steps = encoder.output_num_steps(args.num_steps)
    if args.rollout_ctxt_window >= encoded_steps:
        raise ValueError(
            f"rollout_ctxt_window={args.rollout_ctxt_window} must be smaller than encoded_steps={encoded_steps}. "
            "When using V-JEPA with batchify_video=False and dup_image=False, encoded_steps can be smaller than num_steps."
        )
    if args.rollout_steps is not None and args.rollout_ctxt_window + args.rollout_steps > encoded_steps:
        raise ValueError(
            f"rollout_ctxt_window + rollout_steps = {args.rollout_ctxt_window + args.rollout_steps} exceeds "
            f"encoded_steps={encoded_steps}."
        )
    return encoded_steps


def build_module(args, dataset: ManiFeelSequenceDataset, encoder) -> PredictorTrainingModule:
    predictor_cfg = PredictorConfig(
        predictor_embed_dim=args.predictor_embed_dim,
        depth=args.predictor_depth,
        num_heads=args.predictor_heads,
        mlp_ratio=args.predictor_mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        visual_l2_weight=args.visual_l2_weight,
        visual_cos_weight=args.visual_cos_weight,
        lowdim_l2_weight=args.lowdim_l2_weight,
        lowdim_cos_weight=args.lowdim_cos_weight,
        tactile_l2_weight=args.tactile_l2_weight,
        tactile_cos_weight=args.tactile_cos_weight,
        reg_loss_type=args.reg_loss_type,
        reg_weight=args.reg_weight,
        cov_coeff=args.cov_coeff,
        std_coeff=args.std_coeff,
        sigreg_coeff=args.sigreg_coeff,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sim_coeff_t=args.sim_coeff_t,
        idm_coeff=args.idm_coeff,
    )
    model = SpatialActionConditionedModel(
        encoder=encoder,
        action_dim=dataset.action_dim(),
        lowdim_dim=dataset.lowdim_dim(),
        num_frames=args.num_steps,
        config=predictor_cfg,
        use_tactile=args.use_tactile,
        tactile_channels=dataset.tactile_channels(),
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
    )
    module = PredictorTrainingModule(
        model=model,
        rollout_ctxt_window=args.rollout_ctxt_window,
        rollout_steps=args.rollout_steps,
        log_rollout_metrics=False,
    )
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    module.load_state_dict(state_dict, strict=False)
    module.eval()
    return module


def main():
    args = parse_args()
    pl.seed_everything(args.split_seed, workers=True)

    lowdim_keys = parse_key_list(args.lowdim_keys)
    dataset = ManiFeelSequenceDataset(
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
    if args.eval_split == "all":
        eval_set = dataset
    else:
        train_len = int(len(dataset) * args.train_ratio)
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(args.split_seed),
        )
        eval_set = train_set if args.eval_split == "train" else val_set

    loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    encoder = build_encoder(args, dataset)
    encoded_steps = validate_temporal_config(args, encoder)
    module = build_module(args, dataset, encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)

    action_modes = ["gt", "zero", "random_uniform"] if args.action_mode == "all" else [args.action_mode]
    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)

    print(f"[INFO] evaluating checkpoint: {args.checkpoint}")
    print(f"[INFO] eval split: {args.eval_split}, dataset samples: {len(eval_set)}, batches: {total_batches}")
    print(
        "[INFO] rollout config: "
        f"ctxt_window={args.rollout_ctxt_window}, rollout_steps={args.rollout_steps}, proprio_mode={args.proprio_mode}"
    )
    print(
        "[INFO] temporal alignment: "
        f"raw_steps={args.num_steps}, encoded_steps={encoded_steps}, temporal_stride={encoder.temporal_stride}"
    )

    results = {}
    for action_mode in action_modes:
        print(f"[INFO] action mode: {action_mode}")
        running = {}
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                merged = module.model.compute_rollout_metrics(
                    batch=batch,
                    ctxt_window=args.rollout_ctxt_window,
                    rollout_steps=args.rollout_steps,
                    action_mode=action_mode,
                    proprio_mode=args.proprio_mode,
                )
                if action_mode == "gt":
                    merged = {**module.model.compute_losses(batch), **merged}
                for key, value in merged.items():
                    running[key] = running.get(key, 0.0) + float(value.detach().cpu())
                count += 1
                if (
                    batch_idx == 0
                    or (args.log_every_n_batches > 0 and (batch_idx + 1) % args.log_every_n_batches == 0)
                    or batch_idx + 1 == total_batches
                ):
                    print(f"[INFO] action mode {action_mode}: processed batch {batch_idx + 1}/{total_batches}")
                if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                    break

        if count == 0:
            raise RuntimeError("No evaluation batches processed.")
        results[action_mode] = {key: value / count for key, value in running.items()}
        print(f"[INFO] completed action mode {action_mode}")

    if len(action_modes) == 1:
        for key in sorted(results[action_modes[0]]):
            print(f"{key}: {results[action_modes[0]][key]:.6f}")
        return

    for action_mode in action_modes:
        print(f"[{action_mode}]")
        for key in sorted(results[action_mode]):
            print(f"{key}: {results[action_mode][key]:.6f}")


if __name__ == "__main__":
    main()
