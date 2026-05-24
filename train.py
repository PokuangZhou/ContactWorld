import argparse

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import os

from architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    PointCloudTemporalEncoder,
    PointNetEncoderXYZRGB,
    TacFFEncoder,
    Projector,
    RNNPredictor,
    build_vision_tactile_encoder,
)
from dataset.zarr_dataset import ZarrDataset
from jepa import WorldModel
from losses import (
    SquareLossSeq,
    VC_IDM_Sim_Regularizer,
    SIGReg,
    SIGReg_IDM_Sim_Regularizer,
)


class JointTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: WorldModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        nsteps: int = 2,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.nsteps = nsteps

    def _forward_loss(self, batch):
        return self.model.training_losses(batch, nsteps=self.nsteps)

    def training_step(self, batch, batch_idx):
        out = self._forward_loss(batch)
        self.log("train/loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/pred_loss", out["pred_loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/reg_loss", out["reg_loss"], prog_bar=True, on_step=True, on_epoch=True)
        for k, v in out["reg_dict"].items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._forward_loss(batch)
        self.log("val/loss", out["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/pred_loss", out["pred_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/reg_loss", out["reg_loss"], prog_bar=True, on_step=False, on_epoch=True)
        for k, v in out["reg_dict"].items():
            self.log(f"val/{k}", v, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


def build_dataloaders(args):
    keys_to_load = ["action", args.vision_key]
    if args.use_tactile:
        keys_to_load.append(args.tactile_key)

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        keys_to_load=keys_to_load,
        keys_to_cache=["action"],
    )

    train_len = int(args.train_ratio * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return dataset, train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--vision-key", type=str, default="wrist")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])

    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=5)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=560)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)

    parser.add_argument("--pc-in-channels", type=int, default=3)

    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="left_tactile_camera_taxim")
    parser.add_argument("--tactile-in-channels", type=int, default=3)
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)
    parser.add_argument("--tactile-dim", type=int, default=512)

    parser.add_argument("--fusion-type", type=str, default="concat",
                        choices=["concat", "gate", "film", "attn"])
    parser.add_argument("--fusion-latent-dim", type=int, default=None)
    parser.add_argument("--fusion-hidden-dim", type=int, default=None)
    parser.add_argument("--attn-d-model", type=int, default=256)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--attn-layers", type=int, default=2)
    parser.add_argument("--attn-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)

    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Reg choice
    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])
    parser.add_argument("--use-proj", action="store_true")

    # VC loss
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--std-coeff", type=float, default=1.0)

    # SIGReg loss
    parser.add_argument("--sigreg-coeff", type=float, default=0.1)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)

    # Shared extras
    parser.add_argument("--sim-coeff-t", type=float, default=0.1)
    parser.add_argument("--idm-coeff", type=float, default=0.1)
    parser.add_argument("--idm-after-proj", action="store_true")
    parser.add_argument("--sim-t-after-proj", action="store_true")

    parser.add_argument("--reg-on-vision-only", action="store_true")
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vt-wm")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default="logs/wandb")
    parser.add_argument("--wandb-name", type=str, default=None)

    parser.add_argument("--save-dir", type=str, default="logs/ckpts")
    parser.add_argument("--ckpt-every-n-steps", type=int, default=10000)


    return parser.parse_args()


def build_vision_encoder(args):
    if args.vision_type == "image":
        return ImpalaEncoder(
            input_channels=3,
            input_shape=(3, args.image_size, args.image_size),
            mlp_output_dim=args.vision_dim,
            final_ln=True,
        )

    if args.vision_type == "pc":
        point_encoder = PointNetEncoderXYZRGB(
            in_channels=args.pc_in_channels,
            out_channels=args.vision_dim,
        )
        return PointCloudTemporalEncoder(
            point_encoder=point_encoder,
            out_dim=args.vision_dim,
            final_ln=True,
        )

    raise ValueError(f"Unknown vision_type: {args.vision_type}")

def build_tactile_encoder(args):
    tactile_key = args.tactile_key.lower()

    if "force_field" in tactile_key or "tacff" in tactile_key:
        return TacFFEncoder(
            input_channels=args.tactile_in_channels,
            height=args.tactile_height,
            width=args.tactile_width,
            out_dim=args.tactile_dim,
            hidden_dim=256,
            final_ln=True,
        )

    return ImpalaEncoder(
        input_channels=args.tactile_in_channels,
        input_shape=(args.tactile_in_channels, args.tactile_height, args.tactile_width),
        mlp_output_dim=args.tactile_dim,
        final_ln=True,
    )

def build_regularizer(reg_hidden_dim: int, action_dim: int, args):
    projector = None
    if args.use_proj:
        projector = Projector(f"{reg_hidden_dim}-{reg_hidden_dim*4}-{reg_hidden_dim*4}")

    idm_in_dim = projector.out_dim if projector is not None and args.idm_after_proj else reg_hidden_dim
    idm = InverseDynamicsModel(
        state_dim=idm_in_dim,
        hidden_dim=256,
        action_dim=action_dim,
    )

    if args.reg_loss_type == "vc":
        return VC_IDM_Sim_Regularizer(
            cov_coeff=args.cov_coeff,
            std_coeff=args.std_coeff,
            sim_coeff_t=args.sim_coeff_t,
            idm_coeff=args.idm_coeff,
            idm=idm,
            projector=projector,
            spatial_as_samples=False,
            idm_after_proj=args.idm_after_proj,
            sim_t_after_proj=args.sim_t_after_proj,
        )

    if args.reg_loss_type == "sigreg":
        sigreg = SIGReg(
            knots=args.sigreg_knots,
            num_proj=args.sigreg_num_proj,
        )
        return SIGReg_IDM_Sim_Regularizer(
            sigreg_coeff=args.sigreg_coeff,
            sim_coeff_t=args.sim_coeff_t,
            idm_coeff=args.idm_coeff,
            sigreg=sigreg,
            idm=idm,
            projector=projector,
            idm_after_proj=args.idm_after_proj,
            sim_t_after_proj=args.sim_t_after_proj,
        )

    raise ValueError(f"Unknown reg_loss_type: {args.reg_loss_type}")


def build_model(args, action_dim: int):
    vision_encoder = build_vision_encoder(args)

    if args.use_tactile:
        tactile_encoder = build_tactile_encoder(args)
        encoder, predictor_hidden = build_vision_tactile_encoder(
            fusion_type=args.fusion_type,
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
            vision_dim=args.vision_dim,
            tactile_dim=args.tactile_dim,
            fusion_latent_dim=args.fusion_latent_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            attn_d_model=args.attn_d_model,
            attn_heads=args.attn_heads,
            attn_layers=args.attn_layers,
            attn_mlp_ratio=args.attn_mlp_ratio,
            attn_dropout=args.attn_dropout,
        )
    else:
        encoder = vision_encoder
        predictor_hidden = args.vision_dim

    predictor = RNNPredictor(
        hidden_size=predictor_hidden,
        action_dim=action_dim,
        num_layers=1,
        final_ln=nn.LayerNorm(predictor_hidden),
    )

    # concat special case:
    # predictor works on fused latent, regularizer works on fused vision slice only.
    reg_hidden_dim = predictor_hidden
    if args.use_tactile and args.reg_on_vision_only:
        reg_hidden_dim = args.vision_dim

    regularizer = build_regularizer(reg_hidden_dim, action_dim, args)
    predcost = SquareLossSeq()

    model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        regularizer=regularizer,
        predcost=predcost,
        action_dim=action_dim,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
        tactile_size=(args.tactile_height, args.tactile_width),
        vision_dim=args.vision_dim,
        tactile_dim=args.tactile_dim,
        fusion_type=args.fusion_type,
        reg_on_vision_only=args.reg_on_vision_only,
    )
    return model


def main():
    args = parse_args()

    dataset, train_loader, val_loader = build_dataloaders(args)
    action_dim = dataset.get_dim("action")
    model = build_model(args, action_dim)

    lit_model = JointTrainingModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        nsteps=args.nsteps,
    )

    exp_name = f"{args.task}/{args.vision_key}/{args.reg_loss_type}/only"
    if args.use_tactile:
        exp_name = f"{args.task}/{args.vision_key}/{args.reg_loss_type}/{args.tactile_key}/{args.fusion_type}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_dir}/{exp_name}",
        filename="{step:06d}",
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=args.ckpt_every_n_steps,
        auto_insert_metric_name=False,
    )

    logger = None
    if args.wandb:
        run_name = args.wandb_name or exp_name
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            save_dir=args.wandb_save_dir,
            log_model=False,
        )
        logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    ckpt_dir = f"{args.save_dir}/{exp_name}"
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")

    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        ckpt_path = last_ckpt
    else:
        print("[INFO] No checkpoint found, training from scratch")
        ckpt_path = None

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()