from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

from dataset.zarr_dataset import ZarrDataset
from planner_utils import PlannerConfig, build_model, load_lightning_ckpt


@torch.no_grad()
def compute_rollout_errors(
    model,
    loader,
    history_size=1,
    max_rollout=6,
    device="cuda",
    action_mode="gt",
):
    model.eval()
    step_errors = {k: [] for k in range(1, max_rollout + 1)}

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        emb = model.encode(batch)                 # [B, T, D]
        act = batch["action"].float()             # [B, T, A]

        if action_mode == "gt":
            act_used = act
        elif action_mode == "random_uniform":
            act_used = 2.0 * torch.rand_like(act) - 1.0
        elif action_mode == "zero":
            act_used = torch.zeros_like(act)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        B, T, A = act_used.shape
        assert T >= history_size + max_rollout, (
            f"T={T} too short for history_size={history_size}, max_rollout={max_rollout}"
        )

        # history latent: use first history_size states
        z_hist = emb[:, :history_size].clone()

        preds = []
        for step in range(max_rollout):
            # IMPORTANT:
            # if history_size == 1, next action is act[:, 0]
            # if history_size == 2, next action should start from act[:, 1], etc.
            action_idx = history_size - 1 + step
            a_t = act_used[:, action_idx: action_idx + 1, :]    # [B,1,A]

            z_pred_seq = model.predict_sequence(z_hist[:, -1:], a_t)
            z_next = z_pred_seq[:, -1:]
            preds.append(z_next)
            z_hist = torch.cat([z_hist, z_next], dim=1)

        pred_rollout = torch.cat(preds, dim=1)   # [B, max_rollout, D]
        target_rollout = emb[:, history_size:history_size + max_rollout]

        for k in range(1, max_rollout + 1):
            mse_k = ((pred_rollout[:, :k] - target_rollout[:, :k]) ** 2).mean().item()
            step_errors[k].append(mse_k)

    return {f"{k}_step": sum(v) / len(v) for k, v in step_errors.items()}


@torch.no_grad()
def compute_rollout_errors_for_modes(
    model,
    loader,
    history_size=1,
    max_rollout=6,
    device="cuda",
    modes=None,
):
    if modes is None:
        modes = ["gt", "random_uniform", "zero"]

    out = {}
    for mode in modes:
        out[mode] = compute_rollout_errors(
            model=model,
            loader=loader,
            history_size=history_size,
            max_rollout=max_rollout,
            device=device,
            action_mode=mode,
        )
    return out


def save_metrics_json(metrics: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics json to: {save_path}")


def plot_rollout_metrics(metrics: dict, save_path: Path, title: str = "Multi-step Rollout Error"):
    """
    metrics example:
    {
        "gt": {"1_step": ..., "2_step": ..., ...},
        "random_uniform": {...},
        "zero": {...}
    }
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    preferred_order = ["gt", "random_uniform", "zero"]
    mode_names = [m for m in preferred_order if m in metrics] + [m for m in metrics if m not in preferred_order]

    for mode in mode_names:
        step_dict = metrics[mode]
        steps = sorted(int(k.split("_")[0]) for k in step_dict.keys())
        values = [step_dict[f"{s}_step"] for s in steps]
        plt.plot(steps, values, marker="o", linewidth=2, label=mode)

    plt.xlabel("Rollout step")
    plt.ylabel("MSE")
    plt.title(title)
    plt.xticks(sorted({int(k.split('_')[0]) for v in metrics.values() for k in v.keys()}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved rollout plot to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--vision-key", type=str, default="wrist")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])

    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="left_tactile_camera_taxim")

    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--max-rollout", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--num-workers", type=int, default=6)

    parser.add_argument("--action-mode", type=str, default="all",
                        choices=["gt", "random_uniform", "zero", "all"])

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)
    parser.add_argument("--pc-in-channels", type=int, default=3)

    parser.add_argument("--tactile-dim", type=int, default=512)
    parser.add_argument("--tactile-in-channels", type=int, default=3)
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)

    parser.add_argument("--fusion-type", type=str, default="concat",
                        choices=["concat", "gate", "film", "attn"])
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

    # new save/plot args
    parser.add_argument("--output-dir", type=str, default="eval_rollout_outputs")
    parser.add_argument("--plot-filename", type=str, default="rollout_plot.png")
    parser.add_argument("--json-filename", type=str, default="rollout_metrics.json")
    parser.add_argument("--plot-title", type=str, default="Multi-step Rollout Error")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys_to_load = ["action", args.vision_key]
    if args.use_tactile:
        keys_to_load.append(args.tactile_key)

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=1,
        num_steps=max(args.history_size + args.max_rollout + 1, 8),
        keys_to_load=keys_to_load,
        keys_to_cache=["action"],
    )

    val_len = min(2048, len(dataset))
    train_len = len(dataset) - val_len
    _, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    action_dim = dataset.get_dim("action")
    cfg = PlannerConfig(
        action_dim=action_dim,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        pc_in_channels=args.pc_in_channels,
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

    model = build_model(cfg)
    model = load_lightning_ckpt(model, args.ckpt_path)
    model = model.to(device).eval()
    model.requires_grad_(False)

    if args.action_mode == "all":
        metrics = compute_rollout_errors_for_modes(
            model=model,
            loader=loader,
            history_size=args.history_size,
            max_rollout=args.max_rollout,
            device=device,
            modes=["gt", "random_uniform", "zero"],
        )
    else:
        metrics = {
            args.action_mode: compute_rollout_errors(
                model=model,
                loader=loader,
                history_size=args.history_size,
                max_rollout=args.max_rollout,
                device=device,
                action_mode=args.action_mode,
            )
        }

    print(json.dumps(metrics, indent=2))

    json_path = output_dir / args.json_filename
    plot_path = output_dir / args.plot_filename

    save_metrics_json(metrics, json_path)
    plot_rollout_metrics(metrics, plot_path, title=args.plot_title)


if __name__ == "__main__":
    main()