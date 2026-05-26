from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, random_split

from config_utils import build_config_aware_parser
from dataset import DEFAULT_LOWDIM_KEYS, ManiFeelSequenceDataset, parse_key_list
from encoders import DINO_TOKEN_MODES, DINO_TOKEN_STRATEGIES, DinoSpatialEncoder, VJEPASpatialEncoder
from lightning_module import PredictorTrainingModule
from model import TACTILE_POOL_MODES, PointCloudSpatialEncoder, PredictorConfig, SpatialActionConditionedModel
from pointcloud_utils import PC_TOKENIZERS, effective_pc_num_points


def _build_parser(pre_parser):
    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--task", type=str, default=None)
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
    parser.add_argument("--train-num-workers", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.set_defaults(pin_memory=True, persistent_workers=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="outputs/world_model_tf")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--save-every-n-epochs", type=int, default=25)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-monitor", type=str, default="val/rollout_visual_l2")
    parser.add_argument("--early-stop-mode", type=str, default="min", choices=["min", "max"])
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="world_model_tf")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default="outputs/wandb")
    parser.add_argument("--wandb-log-model", type=str, default="false", choices=["false", "all"])

    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="tactile_force_field_right")
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)
    parser.add_argument("--tactile-force-scale", type=float, default=0.002)
    parser.add_argument("--tactile-pool-mode", type=str, default="mean", choices=TACTILE_POOL_MODES)
    parser.add_argument("--pointcloud-scale", type=float, default=0.4)
    parser.add_argument("--pc-tokenizer", type=str, default="none", choices=PC_TOKENIZERS)
    parser.add_argument("--pc-num-points", type=int, default=None)
    parser.add_argument("--pc-order-mode", type=str, default="none", choices=["none", "xyz"])
    parser.add_argument("--pc-voxel-grid", default="16,16,1")
    parser.add_argument("--pc-bounds", default="-0.4,0.8,-0.6,0.6,-0.2,0.8")
    parser.add_argument("--lowdim-keys", type=str, default=",".join(DEFAULT_LOWDIM_KEYS))

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
    parser.add_argument(
        "--val-proprio-rollout-mode",
        type=str,
        default="use_ground_truth",
        choices=["use_ground_truth", "predict_proprio"],
    )
    parser.add_argument("--precision", type=str, default="32")
    return parser


def parse_args():
    parser = build_config_aware_parser(_build_parser)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("--data-root is required (provide it in the command line or config YAML).")
    return args


def build_encoder(args, dataset: ManiFeelSequenceDataset):
    if args.vision_type == "pc":
        num_points = effective_pc_num_points(
            int(dataset.data[args.vision_key].shape[1]),
            pc_tokenizer=args.pc_tokenizer,
            pc_num_points=args.pc_num_points,
            pc_voxel_grid=args.pc_voxel_grid,
        )
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


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
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
        pc_tokenizer=args.pc_tokenizer,
        pc_num_points=args.pc_num_points,
        pc_order_mode=args.pc_order_mode,
        pc_voxel_grid=args.pc_voxel_grid,
        pc_bounds=args.pc_bounds,
        lowdim_keys=lowdim_keys,
    )
    train_len = int(len(dataset) * args.train_ratio)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    train_num_workers = args.num_workers if args.train_num_workers is None else args.train_num_workers
    val_num_workers = args.num_workers if args.val_num_workers is None else args.val_num_workers

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": train_num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": True,
    }
    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": args.pin_memory,
        "drop_last": False,
    }
    if train_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = args.persistent_workers
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = args.persistent_workers
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_set, **train_loader_kwargs)
    val_loader = DataLoader(val_set, **val_loader_kwargs)

    encoder = build_encoder(args, dataset)
    encoded_steps = validate_temporal_config(args, encoder)
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
        tactile_pool_mode=args.tactile_pool_mode,
    )
    module = PredictorTrainingModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rollout_ctxt_window=args.rollout_ctxt_window,
        rollout_steps=args.rollout_steps,
        log_rollout_metrics=True,
        val_proprio_rollout_mode=args.val_proprio_rollout_mode,
    )

    exp_prefix = f"{args.task}/" if args.task else ""
    exp_name = args.exp_name or f"{exp_prefix}{args.encoder}_{args.vision_key}_tf"
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints" / exp_name
    csv_logger = CSVLogger(save_dir=str(save_dir / "logs"), name=exp_name)
    logger = csv_logger
    if args.wandb:
        wandb_root = Path(args.wandb_save_dir).expanduser().resolve()
        wandb_root.mkdir(parents=True, exist_ok=True)
        wandb_cache_dir = wandb_root / "cache"
        wandb_data_dir = wandb_root / "data"
        wandb_artifact_dir = wandb_root / "artifacts"
        wandb_run_dir = wandb_root / "runs"
        for path in (wandb_cache_dir, wandb_data_dir, wandb_artifact_dir, wandb_run_dir):
            path.mkdir(parents=True, exist_ok=True)

        os.environ["WANDB_DIR"] = str(wandb_run_dir)
        os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)
        os.environ["WANDB_DATA_DIR"] = str(wandb_data_dir)
        os.environ["WANDB_ARTIFACT_DIR"] = str(wandb_artifact_dir)

        wandb_log_model = False if args.wandb_log_model == "false" else args.wandb_log_model
        logger = [
            csv_logger,
            WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or exp_name,
                save_dir=args.wandb_save_dir,
                log_model=wandb_log_model,
            ),
        ]
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-val-loss",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-rollout-visual-l2",
            monitor="val/rollout_visual_l2",
            mode="min",
            save_top_k=1,
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch-{epoch:03d}",
            every_n_epochs=args.save_every_n_epochs,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        ),
    ]
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stop_monitor,
                mode=args.early_stop_mode,
                patience=args.early_stop_patience,
            )
        )

    print(f"[INFO] experiment: {exp_name}")
    print(f"[INFO] checkpoint dir: {ckpt_dir}")
    print("[INFO] checkpoint policy: keep best val/loss, keep best rollout_visual_l2, keep last.ckpt, save periodic snapshots")
    print(f"[INFO] periodic snapshot frequency: every {args.save_every_n_epochs} epochs")
    print(
        "[INFO] dataloader: "
        f"batch_size={args.batch_size}, train_num_workers={train_num_workers}, "
        f"val_num_workers={val_num_workers}, pin_memory={args.pin_memory}, "
        f"persistent_workers={args.persistent_workers}, prefetch_factor={args.prefetch_factor}"
    )
    print(
        "[INFO] VJEPA encoding: "
        f"batchify_video={args.vjepa_batchify_video}, dup_image={args.vjepa_dup_image}, "
        f"normalize_reps={args.vjepa_normalize_reps}"
    )
    print(
        f"[INFO] DINO token mode: {args.dino_token_mode}, "
        f"strategy={args.dino_token_strategy}, last_layers={args.dino_last_layers}"
    )
    print(f"[INFO] temporal alignment: raw_steps={args.num_steps}, encoded_steps={encoded_steps}, temporal_stride={encoder.temporal_stride}")
    print(f"[INFO] validation proprio rollout mode: {args.val_proprio_rollout_mode}")
    if args.early_stop_patience > 0:
        print(
            "[INFO] early stopping: "
            f"monitor={args.early_stop_monitor}, mode={args.early_stop_mode}, patience={args.early_stop_patience}"
        )
    else:
        print("[INFO] early stopping: disabled")
    if args.wandb:
        print(
            "[INFO] wandb logging enabled: "
            f"project={args.wandb_project}, run={args.wandb_name or exp_name}, save_dir={args.wandb_save_dir}, "
            f"log_model={args.wandb_log_model}"
        )
        print(
            "[INFO] wandb local dirs: "
            f"runs={os.environ['WANDB_DIR']}, cache={os.environ['WANDB_CACHE_DIR']}, "
            f"artifacts={os.environ['WANDB_ARTIFACT_DIR']}"
        )
    if args.resume_from is not None:
        print(f"[INFO] resuming from checkpoint: {args.resume_from}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        precision=args.precision,
        default_root_dir=str(save_dir),
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from,
    )


if __name__ == "__main__":
    main()
