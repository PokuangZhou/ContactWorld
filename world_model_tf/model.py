from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ac_predictor import VisionTransformerPredictorAC
from encoders import FrozenSpatialEncoder

TACTILE_POOL_MODES = ("mean", "attention")


@dataclass
class PredictorConfig:
    predictor_embed_dim: int = 384
    depth: int = 6
    num_heads: int | None = None
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    visual_l2_weight: float = 1.0
    visual_cos_weight: float = 0.0
    lowdim_l2_weight: float = 1.0
    lowdim_cos_weight: float = 0.0
    tactile_l2_weight: float = 1.0
    tactile_cos_weight: float = 0.0

    reg_loss_type: str = "vc"
    reg_weight: float = 1.0
    cov_coeff: float = 1.0
    std_coeff: float = 1.0
    sigreg_coeff: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sim_coeff_t: float = 0.1
    idm_coeff: float = 0.1


def init_module_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class PointCloudSpatialEncoder(FrozenSpatialEncoder):
    def __init__(
        self,
        in_channels: int,
        num_points: int,
        embed_dim: int = 512,
        image_size: int | None = None,
    ) -> None:
        super().__init__()
        side = int(round(math.sqrt(num_points)))
        if side * side != num_points:
            raise ValueError(f"Pointcloud token count must be square for AC predictor, got {num_points}")
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        from encoders import EncoderSpec

        self.spec = EncoderSpec(
            kind="pointcloud",
            image_size=image_size or side,
            patch_size=1,
            embed_dim=embed_dim,
            num_heads=8,
            num_patches=num_points,
            temporal_stride=1,
        )
        self.apply(init_module_weights)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points.float())


class TactileTokenEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, height: int, width: int) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.num_tokens = self.height * self.width
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, self.num_tokens, embed_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.apply(init_module_weights)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = tactile.shape
        if h != self.height or w != self.width:
            raise ValueError(f"Expected tactile H,W=({self.height},{self.width}), got ({h},{w})")
        x = tactile.permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
        return self.encoder(x) + self.pos_emb


class TactileAttentionPool(nn.Module):
    """Single-query dot-product attention pooling: [B,T,N,D] → [B,T,D]."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, N, D]
        scores = (tokens * self.query).sum(dim=-1, keepdim=True)  # [B, T, N, 1]
        weights = torch.softmax(scores, dim=2)
        return (tokens * weights).sum(dim=2)  # [B, T, D]


class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        random_proj = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        random_proj = random_proj.div_(random_proj.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12))
        x_t = (proj @ random_proj).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class LatentRegularizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        cfg: PredictorConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_num_proj)
        self.idm = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.apply(init_module_weights)

    @staticmethod
    def _std_loss(z):
        return torch.mean(F.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)))

    @staticmethod
    def _cov_loss(z):
        z = z - z.mean(dim=0, keepdim=True)
        n, d = z.shape
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return (off_diag ** 2).mean()

    @staticmethod
    def _sim_t_loss(z):
        if z.size(1) < 2:
            return z.new_tensor(0.0)
        return F.mse_loss(z[:, 1:], z[:, :-1])

    def _idm_loss(self, z, actions):
        if self.cfg.idm_coeff <= 0 or z.size(1) < 2:
            return z.new_tensor(0.0)
        pred = self.idm(torch.cat([z[:, :-1], z[:, 1:]], dim=-1).reshape(-1, z.size(-1) * 2))
        target = actions[:, :-1].reshape(-1, actions.size(-1))
        return F.mse_loss(pred, target)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = z.reshape(-1, z.size(-1))
        sim_t = self._sim_t_loss(z) if self.cfg.sim_coeff_t > 0 else z.new_tensor(0.0)
        idm = self._idm_loss(z, actions)

        if self.cfg.reg_loss_type == "vc":
            std = self._std_loss(flat)
            cov = self._cov_loss(flat)
            total = (
                self.cfg.std_coeff * std
                + self.cfg.cov_coeff * cov
                + self.cfg.sim_coeff_t * sim_t
                + self.cfg.idm_coeff * idm
            )
            return total, {
                "std_loss": std.detach(),
                "cov_loss": cov.detach(),
                "sim_t_loss": sim_t.detach(),
                "idm_loss": idm.detach(),
            }

        if self.cfg.reg_loss_type == "sigreg":
            sig = self.sigreg(z.transpose(0, 1).contiguous())
            total = (
                self.cfg.sigreg_coeff * sig
                + self.cfg.sim_coeff_t * sim_t
                + self.cfg.idm_coeff * idm
            )
            return total, {
                "sigreg_loss": sig.detach(),
                "sim_t_loss": sim_t.detach(),
                "idm_loss": idm.detach(),
            }

        raise ValueError(f"Unknown reg_loss_type: {self.cfg.reg_loss_type}")


class SpatialActionConditionedModel(nn.Module):
    def __init__(
        self,
        encoder: FrozenSpatialEncoder,
        action_dim: int,
        lowdim_dim: int,
        num_frames: int,
        config: PredictorConfig,
        use_tactile: bool = False,
        tactile_channels: int = 0,
        tactile_height: int = 10,
        tactile_width: int = 14,
        tactile_pool_mode: str = "mean",
        decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if tactile_pool_mode not in TACTILE_POOL_MODES:
            raise ValueError(f"tactile_pool_mode must be one of {TACTILE_POOL_MODES}, got {tactile_pool_mode!r}")
        self.encoder = encoder
        self.config = config
        self.decoder = decoder
        self.action_dim = int(action_dim)
        self.lowdim_dim = int(lowdim_dim)
        self.use_tactile = bool(use_tactile)
        self.tactile_pool_mode = tactile_pool_mode
        self.temporal_stride = int(self.encoder.temporal_stride)

        self.grid_size = int(round(math.sqrt(self.encoder.num_patches)))
        if self.grid_size * self.grid_size != self.encoder.num_patches:
            raise ValueError(f"Expected square spatial token grid, got {self.encoder.num_patches} patches.")

        self.lowdim_encoder = nn.Sequential(
            nn.LayerNorm(self.lowdim_dim),
            nn.Linear(self.lowdim_dim, config.predictor_embed_dim),
            nn.GELU(),
            nn.Linear(config.predictor_embed_dim, config.predictor_embed_dim),
            nn.LayerNorm(config.predictor_embed_dim),
        )

        self.tactile_encoder = None
        self.tactile_pool = None
        if self.use_tactile:
            self.tactile_encoder = TactileTokenEncoder(
                in_channels=tactile_channels,
                embed_dim=config.predictor_embed_dim,
                height=tactile_height,
                width=tactile_width,
            )
            if tactile_pool_mode == "attention":
                self.tactile_pool = TactileAttentionPool(config.predictor_embed_dim)

        cond_dim = config.predictor_embed_dim * (2 if self.use_tactile else 1)
        predictor_heads = config.num_heads if config.num_heads is not None else self.encoder.num_heads
        self.predictor = VisionTransformerPredictorAC(
            img_size=(self.encoder.spec.image_size, self.encoder.spec.image_size),
            patch_size=self.encoder.patch_size,
            num_frames=self.encoder.output_num_steps(num_frames),
            tubelet_size=1,
            embed_dim=self.encoder.embed_dim,
            predictor_embed_dim=config.predictor_embed_dim,
            depth=config.depth,
            num_heads=predictor_heads,
            mlp_ratio=config.mlp_ratio,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            action_embed_dim=self.action_dim,
            state_embed_dim=cond_dim,
            use_rope=True,
            is_frame_causal=True,
        )

        lowdim_head_in = self.encoder.embed_dim + self.lowdim_dim + self.action_dim
        self.lowdim_predictor = nn.Sequential(
            nn.LayerNorm(lowdim_head_in),
            nn.Linear(lowdim_head_in, config.predictor_embed_dim),
            nn.GELU(),
            nn.Linear(config.predictor_embed_dim, self.lowdim_dim),
        )

        if self.use_tactile:
            tactile_head_in = config.predictor_embed_dim + self.encoder.embed_dim + self.action_dim
            self.tactile_predictor = nn.Sequential(
                nn.LayerNorm(tactile_head_in),
                nn.Linear(tactile_head_in, config.predictor_embed_dim),
                nn.GELU(),
                nn.Linear(config.predictor_embed_dim, config.predictor_embed_dim),
            )
        else:
            self.tactile_predictor = None

        self.joint_projector = nn.Sequential(
            nn.Linear(self.encoder.embed_dim + cond_dim, config.predictor_embed_dim),
            nn.LayerNorm(config.predictor_embed_dim),
        )
        self.regularizer = LatentRegularizer(config.predictor_embed_dim, self.action_dim, config)
        self.lowdim_encoder.apply(init_module_weights)
        self.lowdim_predictor.apply(init_module_weights)
        self.joint_projector.apply(init_module_weights)
        if self.tactile_predictor is not None:
            self.tactile_predictor.apply(init_module_weights)

    def _align_temporal_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.temporal_stride == 1:
            return batch
        raise NotImplementedError("Only temporal_stride=1 is supported in the ManiFeel DINO path.")

    def encode_visual(self, vision: torch.Tensor) -> torch.Tensor:
        return self.encoder(vision)

    def encode_lowdim(self, lowdim: torch.Tensor) -> torch.Tensor:
        return self.lowdim_encoder(lowdim)

    def encode_tactile(self, tactile: torch.Tensor | None) -> torch.Tensor | None:
        if not self.use_tactile:
            return None
        if tactile is None:
            raise ValueError("Tactile input is required for this model.")
        return self.tactile_encoder(tactile)

    def _pool_tactile(self, tactile_tokens: torch.Tensor) -> torch.Tensor:
        if self.tactile_pool is not None:
            return self.tactile_pool(tactile_tokens)
        return tactile_tokens.mean(dim=2)

    def _condition_tokens(self, lowdim_tokens: torch.Tensor, tactile_tokens: torch.Tensor | None) -> torch.Tensor:
        parts = [lowdim_tokens]
        if self.use_tactile:
            parts.append(self._pool_tactile(tactile_tokens))
        return torch.cat(parts, dim=-1)

    def _joint_latent(
        self,
        visual_tokens: torch.Tensor,
        lowdim_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        visual_pool = visual_tokens.mean(dim=2)
        cond = self._condition_tokens(lowdim_tokens, tactile_tokens)
        return self.joint_projector(torch.cat([visual_pool, cond], dim=-1))

    def forward_predictor(
        self,
        visual_tokens: torch.Tensor,
        actions: torch.Tensor,
        lowdim: torch.Tensor,
        tactile_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        b, t, n, d = visual_tokens.shape
        lowdim_tokens = self.encode_lowdim(lowdim)
        cond = self._condition_tokens(lowdim_tokens, tactile_tokens)
        pred_visual_flat = self.predictor(
            visual_tokens.reshape(b, t * n, d),
            actions,
            cond,
        )
        pred_visual = pred_visual_flat.view(b, t, n, d)

        pooled_pred = pred_visual.mean(dim=2)
        pred_lowdim = self.lowdim_predictor(torch.cat([pooled_pred, lowdim, actions], dim=-1))

        pred_tactile = None
        if self.use_tactile:
            if tactile_tokens is None:
                raise ValueError("tactile_tokens required when use_tactile=True")
            pooled = pooled_pred[:, :, None, :].expand(-1, -1, tactile_tokens.size(2), -1)
            act = actions[:, :, None, :].expand(-1, -1, tactile_tokens.size(2), -1)
            pred_tactile = self.tactile_predictor(torch.cat([tactile_tokens, pooled, act], dim=-1))

        return pred_visual, pred_lowdim, pred_tactile

    @staticmethod
    def _cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        target_visual = self.encode_visual(batch["vision"])
        target_lowdim = batch["lowdim"]
        target_tactile = self.encode_tactile(batch.get("tactile"))
        actions = batch["action"]

        pred_visual, pred_lowdim, pred_tactile = self.forward_predictor(
            visual_tokens=target_visual,
            actions=actions,
            lowdim=target_lowdim,
            tactile_tokens=target_tactile,
        )

        visual_l2 = F.mse_loss(pred_visual[:, :-1], target_visual[:, 1:].detach())
        visual_cos = self._cosine_loss(pred_visual[:, :-1], target_visual[:, 1:].detach())
        lowdim_l2 = F.mse_loss(pred_lowdim[:, :-1], target_lowdim[:, 1:])
        lowdim_cos = self._cosine_loss(pred_lowdim[:, :-1], target_lowdim[:, 1:])

        tactile_l2 = target_visual.new_tensor(0.0)
        tactile_cos = target_visual.new_tensor(0.0)
        if self.use_tactile:
            tactile_l2 = F.mse_loss(pred_tactile[:, :-1], target_tactile[:, 1:].detach())
            tactile_cos = self._cosine_loss(pred_tactile[:, :-1], target_tactile[:, 1:].detach())

        lowdim_tokens = self.encode_lowdim(target_lowdim)
        joint_latent = self._joint_latent(target_visual, lowdim_tokens, target_tactile)
        reg_loss, reg_dict = self.regularizer(joint_latent, actions)

        loss = (
            self.config.visual_l2_weight * visual_l2
            + self.config.visual_cos_weight * visual_cos
            + self.config.lowdim_l2_weight * lowdim_l2
            + self.config.lowdim_cos_weight * lowdim_cos
            + self.config.tactile_l2_weight * tactile_l2
            + self.config.tactile_cos_weight * tactile_cos
            + self.config.reg_weight * reg_loss
        )
        out = {
            "loss": loss,
            "visual_l2_loss": visual_l2,
            "visual_cos_loss": visual_cos,
            "lowdim_l2_loss": lowdim_l2,
            "lowdim_cos_loss": lowdim_cos,
            "tactile_l2_loss": tactile_l2,
            "tactile_cos_loss": tactile_cos,
            "reg_loss": reg_loss,
        }
        out.update(reg_dict)
        return out

    def rollout(
        self,
        batch: dict[str, torch.Tensor],
        ctxt_window: int = 4,
        rollout_steps: int | None = None,
        action_mode: str = "gt",
        proprio_mode: str = "predict_proprio",
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            target_visual = self.encode_visual(batch["vision"])
            target_tactile = self.encode_tactile(batch.get("tactile"))
        target_lowdim = batch["lowdim"]
        actions = batch["action"]
        if action_mode == "gt":
            rollout_actions = actions
        elif action_mode == "zero":
            rollout_actions = torch.zeros_like(actions)
        elif action_mode == "random_uniform":
            rollout_actions = 2.0 * torch.rand_like(actions) - 1.0
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        b, t, _, _ = target_visual.shape
        if rollout_steps is None:
            rollout_steps = t - ctxt_window
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")

        visual_hist = target_visual[:, :ctxt_window].clone()
        lowdim_hist = target_lowdim[:, :ctxt_window].clone()
        tactile_hist = target_tactile[:, :ctxt_window].clone() if self.use_tactile else None
        pred_visual_steps = []
        pred_lowdim_steps = []
        pred_tactile_steps = []

        for step in range(rollout_steps):
            ctx_vis = visual_hist[:, -ctxt_window:]
            ctx_low = lowdim_hist[:, -ctxt_window:]
            ctx_tac = tactile_hist[:, -ctxt_window:] if self.use_tactile else None
            ctx_actions = rollout_actions[:, step : step + ctxt_window]
            pred_visual_ctx, pred_lowdim_ctx, pred_tactile_ctx = self.forward_predictor(
                visual_tokens=ctx_vis,
                actions=ctx_actions,
                lowdim=ctx_low,
                tactile_tokens=ctx_tac,
            )
            next_visual = pred_visual_ctx[:, -1:]
            next_lowdim = pred_lowdim_ctx[:, -1:]
            next_tactile = pred_tactile_ctx[:, -1:] if self.use_tactile else None

            pred_visual_steps.append(next_visual)
            pred_lowdim_steps.append(next_lowdim)
            if self.use_tactile:
                pred_tactile_steps.append(next_tactile)

            visual_hist = torch.cat([visual_hist, next_visual], dim=1)
            if proprio_mode in ("gt", "use_ground_truth"):
                lowdim_hist = torch.cat([lowdim_hist, target_lowdim[:, ctxt_window + step : ctxt_window + step + 1]], dim=1)
                if self.use_tactile:
                    tactile_hist = torch.cat(
                        [tactile_hist, target_tactile[:, ctxt_window + step : ctxt_window + step + 1]],
                        dim=1,
                    )
            elif proprio_mode in ("predict", "predict_proprio"):
                lowdim_hist = torch.cat([lowdim_hist, next_lowdim], dim=1)
                if self.use_tactile:
                    tactile_hist = torch.cat([tactile_hist, next_tactile], dim=1)
            else:
                raise ValueError(f"Unsupported proprio_mode: {proprio_mode}")

        result = {
            "pred_visual": torch.cat(pred_visual_steps, dim=1),
            "target_visual": target_visual[:, ctxt_window : ctxt_window + rollout_steps],
            "pred_lowdim": torch.cat(pred_lowdim_steps, dim=1),
            "target_lowdim": target_lowdim[:, ctxt_window : ctxt_window + rollout_steps],
        }
        if self.use_tactile:
            result["pred_tactile"] = torch.cat(pred_tactile_steps, dim=1)
            result["target_tactile"] = target_tactile[:, ctxt_window : ctxt_window + rollout_steps]
        return result

    def compute_rollout_metrics(
        self,
        batch: dict[str, torch.Tensor],
        ctxt_window: int = 4,
        rollout_steps: int | None = None,
        action_mode: str = "gt",
        proprio_mode: str = "predict_proprio",
    ) -> dict[str, torch.Tensor]:
        rollout = self.rollout(
            batch=batch,
            ctxt_window=ctxt_window,
            rollout_steps=rollout_steps,
            action_mode=action_mode,
            proprio_mode=proprio_mode,
        )
        metrics = {
            "rollout_visual_l2": F.mse_loss(rollout["pred_visual"], rollout["target_visual"]),
            "rollout_visual_cos": self._cosine_loss(rollout["pred_visual"], rollout["target_visual"]),
            "rollout_lowdim_l2": F.mse_loss(rollout["pred_lowdim"], rollout["target_lowdim"]),
            "rollout_lowdim_cos": self._cosine_loss(rollout["pred_lowdim"], rollout["target_lowdim"]),
        }
        if self.use_tactile:
            metrics["rollout_tactile_l2"] = F.mse_loss(rollout["pred_tactile"], rollout["target_tactile"])
            metrics["rollout_tactile_cos"] = self._cosine_loss(rollout["pred_tactile"], rollout["target_tactile"])
        return metrics

    def decode_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("No decoder attached.")
        b, t, n, d = visual_tokens.shape
        feats = visual_tokens.view(b, t, 1, self.grid_size, self.grid_size, d)
        return self.decoder.decode(feats)
