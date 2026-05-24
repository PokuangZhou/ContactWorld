from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from dataset import DEFAULT_LOWDIM_KEYS, IMAGENET_MEAN, IMAGENET_STD


# cost_mode choices: vision | vision_lowdim | vision_tactile | vision_tactile_lowdim
# "joint" is kept as a backward-compatible alias for "vision_lowdim"
PLANNER_COST_MODES = ("vision", "lowdim", "joint", "vision_lowdim", "vision_tactile", "vision_tactile_lowdim")


@dataclass
class PlannerConfig:
    horizon: int = 6
    candidates: int = 64
    candidate_chunk_size: int = 4
    topk: int = 8
    iterations: int = 4
    ctxt_window: int = 4
    action_dim: int = 7
    action_low: float = -1.0
    action_high: float = 1.0
    cost_mode: str = "joint"  # vision | vision_lowdim | vision_tactile | vision_tactile_lowdim | joint (alias for vision_lowdim)
    visual_cost_weight: float = 1.0
    lowdim_cost_weight: float = 1.0
    tactile_cost_weight: float = 1.0
    sum_all_diffs: bool = False
    discount: float = 1.0
    cem_min_std: float = 0.03
    use_action_prior: bool = True
    action_prior_std_scale: float = 1.0
    action_prior_min_std: float = 0.05
    warm_start_mode: str = "prev_action"  # none | prev_action
    warm_start_std: float = 0.15
    warm_start_mix: float = 0.5
    action_smooth_weight: float = 0.05
    action_magnitude_weight: float = 0.01


def load_lightning_ckpt(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = {(key[len("model.") :] if key.startswith("model.") else key): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] ckpt load - missing: {missing}")
    print(f"[INFO] ckpt load - unexpected: {unexpected}")
    return model


def _to_tensor(x: np.ndarray | torch.Tensor, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device).float()
    return torch.from_numpy(np.asarray(x)).to(device=device).float()


def _preprocess_image_sequence(images, *, image_size: int, device: str) -> torch.Tensor:
    x = _to_tensor(images, device)
    if x.ndim == 4:
        x = x.unsqueeze(1)
    if x.ndim != 5:
        raise ValueError(f"Expected image [B,T,H,W,C] or [B,T,C,H,W], got {tuple(x.shape)}")
    if x.shape[-1] in (1, 3, 4):
        if x.shape[-1] == 4:
            x = x[..., :3]
        if x.shape[-1] == 1:
            x = x.expand(*x.shape[:-1], 3)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
    if x.max() > 1.5:
        x = x / 255.0
    b, t, c, h, w = x.shape
    if (h, w) != (image_size, image_size):
        x = F.interpolate(
            x.view(b * t, c, h, w),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, c, image_size, image_size)
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - mean) / std


def _preprocess_pointcloud_sequence(points, *, device: str, pc_in_channels: int, pointcloud_scale: float) -> torch.Tensor:
    x = _to_tensor(points, device)
    if x.ndim != 4:
        raise ValueError(f"Expected pointcloud [B,T,N,C], got {tuple(x.shape)}")
    x = x[..., :pc_in_channels].clone()
    x[..., :3] = x[..., :3] / float(pointcloud_scale)
    if x.size(-1) >= 6:
        rgb = x[..., 3:6]
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        x[..., 3:6] = rgb
    return x


def _preprocess_tactile_sequence(
    tactile,
    *,
    device: str,
    tactile_key: str,
    tactile_height: int,
    tactile_width: int,
    tactile_force_scale: float,
) -> torch.Tensor:
    x = _to_tensor(tactile, device)
    if x.ndim == 4:
        if x.shape[-1] in (1, 3, 4):
            x = x.unsqueeze(1)
        else:
            x = x.unsqueeze(2)
    if x.ndim != 5:
        raise ValueError(f"Expected tactile [B,T,H,W,C] or [B,T,C,H,W], got {tuple(x.shape)}")
    if x.shape[-1] in (1, 3, 4):
        if x.shape[-1] == 4:
            x = x[..., :3]
        x = x.permute(0, 1, 4, 2, 3).contiguous()
    if x.max() > 1.5:
        x = x / 255.0
    b, t, c, h, w = x.shape
    if (h, w) != (tactile_height, tactile_width):
        x = F.interpolate(
            x.view(b * t, c, h, w),
            size=(tactile_height, tactile_width),
            mode="bilinear",
            align_corners=False,
        ).view(b, t, c, tactile_height, tactile_width)
    key = tactile_key.lower()
    if "force_field" in key or "tacff" in key:
        x = x / float(tactile_force_scale)
    elif "depth" in key or "rgb" in key or "image" in key or "taxim" in key:
        x = x.clamp(0.0, 1.0)
    return x


def _build_lowdim(
    raw_batch: dict,
    *,
    device: str,
    lowdim_keys: tuple[str, ...],
    lowdim_stats: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    parts = []
    for key in lowdim_keys:
        x = _to_tensor(raw_batch[key], device)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim == 1:
            x = x.view(1, 1, -1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        mean = lowdim_stats[key]["mean"].to(device=x.device, dtype=x.dtype)
        std = lowdim_stats[key]["std"].to(device=x.device, dtype=x.dtype)
        parts.append((x - mean.view(1, 1, -1)) / std.view(1, 1, -1))
    return torch.cat(parts, dim=-1)


def build_model_batch_from_raw(
    raw_batch: dict,
    *,
    device: str,
    vision_key: str,
    vision_type: str,
    image_size: int,
    lowdim_keys: tuple[str, ...],
    lowdim_stats: dict[str, dict[str, torch.Tensor]],
    pc_in_channels: int = 3,
    pointcloud_scale: float = 0.4,
    use_tactile: bool = False,
    tactile_key: str = "tactile_force_field_right",
    tactile_height: int = 10,
    tactile_width: int = 14,
    tactile_force_scale: float = 0.002,
) -> dict[str, torch.Tensor]:
    if vision_type == "image":
        vision = _preprocess_image_sequence(raw_batch[vision_key], image_size=image_size, device=device)
    elif vision_type == "pc":
        vision = _preprocess_pointcloud_sequence(
            raw_batch[vision_key],
            device=device,
            pc_in_channels=pc_in_channels,
            pointcloud_scale=pointcloud_scale,
        )
    else:
        raise ValueError(f"Unknown vision_type: {vision_type}")

    batch = {
        "vision": vision,
        "lowdim": _build_lowdim(
            raw_batch,
            device=device,
            lowdim_keys=lowdim_keys,
            lowdim_stats=lowdim_stats,
        ),
    }
    if "action" in raw_batch:
        batch["action"] = _to_tensor(raw_batch["action"], device)
    if use_tactile:
        batch["tactile"] = _preprocess_tactile_sequence(
            raw_batch[tactile_key],
            device=device,
            tactile_key=tactile_key,
            tactile_height=tactile_height,
            tactile_width=tactile_width,
            tactile_force_scale=tactile_force_scale,
        )
    return batch


def init_history_buffers(
    obs: Dict[str, np.ndarray],
    *,
    ctxt_window: int,
    action_dim: int,
    vision_key: str,
    lowdim_keys: tuple[str, ...] = DEFAULT_LOWDIM_KEYS,
    use_tactile: bool = False,
    tactile_key: Optional[str] = None,
    init_action: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    batch_size = int(obs[vision_key].shape[0])
    history = {vision_key: np.repeat(obs[vision_key][:, None], ctxt_window, axis=1)}
    for key in lowdim_keys:
        history[key] = np.repeat(obs[key][:, None], ctxt_window, axis=1)
    if use_tactile:
        history[tactile_key] = np.repeat(obs[tactile_key][:, None], ctxt_window, axis=1)

    action_history_len = max(ctxt_window - 1, 0)
    if init_action is None:
        init_action = np.zeros((batch_size, action_dim), dtype=np.float32)
    history["action"] = np.repeat(np.asarray(init_action, dtype=np.float32)[:, None], action_history_len, axis=1)
    return history


def update_history_buffers(
    history: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    *,
    vision_key: str,
    lowdim_keys: tuple[str, ...] = DEFAULT_LOWDIM_KEYS,
    use_tactile: bool = False,
    tactile_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    history[vision_key] = np.concatenate([history[vision_key][:, 1:], obs[vision_key][:, None]], axis=1)
    for key in lowdim_keys:
        history[key] = np.concatenate([history[key][:, 1:], obs[key][:, None]], axis=1)
    if use_tactile:
        history[tactile_key] = np.concatenate([history[tactile_key][:, 1:], obs[tactile_key][:, None]], axis=1)
    if history["action"].shape[1] > 0:
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 1:
            action_arr = action_arr[None]
        history["action"] = np.concatenate([history["action"][:, 1:], action_arr[:, None]], axis=1)
    return history


def rollout_token_sequences(
    model,
    visual_context: torch.Tensor,
    lowdim_context: torch.Tensor,
    tactile_context: torch.Tensor | None,
    action_history: torch.Tensor,
    action_candidates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch_size, num_samples, horizon, action_dim = action_candidates.shape
    ctxt_window = int(visual_context.size(1))
    visual_hist = visual_context[:, None].expand(-1, num_samples, -1, -1, -1).reshape(
        batch_size * num_samples,
        ctxt_window,
        visual_context.size(2),
        visual_context.size(3),
    )
    lowdim_hist = lowdim_context[:, None].expand(-1, num_samples, -1, -1).reshape(
        batch_size * num_samples,
        ctxt_window,
        lowdim_context.size(-1),
    )
    tactile_hist = None
    if tactile_context is not None:
        tactile_hist = tactile_context[:, None].expand(-1, num_samples, -1, -1, -1).reshape(
            batch_size * num_samples,
            ctxt_window,
            tactile_context.size(2),
            tactile_context.size(3),
        )
    action_hist = action_history[:, None].expand(-1, num_samples, -1, -1).reshape(
        batch_size * num_samples,
        max(ctxt_window - 1, 0),
        action_dim,
    )
    actions = action_candidates.reshape(batch_size * num_samples, horizon, action_dim)

    pred_visual_steps = []
    pred_lowdim_steps = []
    pred_tactile_steps = []
    for step_idx in range(horizon):
        a_next = actions[:, step_idx : step_idx + 1]
        ctx_actions = torch.cat([action_hist, a_next], dim=1) if ctxt_window > 1 else a_next
        pred_visual_ctx, pred_lowdim_ctx, pred_tactile_ctx = model.forward_predictor(
            visual_tokens=visual_hist[:, -ctxt_window:],
            actions=ctx_actions,
            lowdim=lowdim_hist[:, -ctxt_window:],
            tactile_tokens=tactile_hist[:, -ctxt_window:] if tactile_hist is not None else None,
        )
        next_visual = pred_visual_ctx[:, -1:]
        next_lowdim = pred_lowdim_ctx[:, -1:]
        pred_visual_steps.append(next_visual)
        pred_lowdim_steps.append(next_lowdim)
        visual_hist = torch.cat([visual_hist, next_visual], dim=1)
        lowdim_hist = torch.cat([lowdim_hist, next_lowdim], dim=1)
        if tactile_hist is not None:
            next_tactile = pred_tactile_ctx[:, -1:]
            pred_tactile_steps.append(next_tactile)
            tactile_hist = torch.cat([tactile_hist, next_tactile], dim=1)
        if ctxt_window > 1:
            action_hist = torch.cat([action_hist[:, 1:], a_next], dim=1)

    pred_visual = torch.cat(pred_visual_steps, dim=1).reshape(
        batch_size,
        num_samples,
        horizon,
        visual_context.size(2),
        visual_context.size(3),
    )
    pred_lowdim = torch.cat(pred_lowdim_steps, dim=1).reshape(batch_size, num_samples, horizon, lowdim_context.size(-1))
    pred_tactile = None
    if pred_tactile_steps:
        pred_tactile = torch.cat(pred_tactile_steps, dim=1).reshape(
            batch_size,
            num_samples,
            horizon,
            tactile_context.size(2),
            tactile_context.size(3),
        )
    return pred_visual, pred_lowdim, pred_tactile


class BatchedCEMPlanner:
    def __init__(
        self,
        *,
        model,
        cfg: PlannerConfig,
        device: str,
        vision_key: str,
        vision_type: str,
        image_size: int,
        lowdim_keys: tuple[str, ...],
        lowdim_stats: dict[str, dict[str, torch.Tensor]],
        pc_in_channels: int = 3,
        pointcloud_scale: float = 0.4,
        use_tactile: bool = False,
        tactile_key: str = "tactile_force_field_right",
        tactile_height: int = 10,
        tactile_width: int = 14,
        tactile_force_scale: float = 0.002,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.vision_key = vision_key
        self.vision_type = vision_type
        self.image_size = image_size
        self.lowdim_keys = tuple(lowdim_keys)
        self.lowdim_stats = lowdim_stats
        self.pc_in_channels = pc_in_channels
        self.pointcloud_scale = pointcloud_scale
        self.use_tactile = use_tactile
        self.tactile_key = tactile_key
        self.tactile_height = tactile_height
        self.tactile_width = tactile_width
        self.tactile_force_scale = tactile_force_scale
        self.action_mean = None
        self.action_std = None
        if action_mean is not None:
            self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
        if action_std is not None:
            self.action_std = torch.as_tensor(action_std, dtype=torch.float32, device=device)

    def _model_batch(self, raw_batch: dict) -> dict[str, torch.Tensor]:
        return build_model_batch_from_raw(
            raw_batch,
            device=self.device,
            vision_key=self.vision_key,
            vision_type=self.vision_type,
            image_size=self.image_size,
            lowdim_keys=self.lowdim_keys,
            lowdim_stats=self.lowdim_stats,
            pc_in_channels=self.pc_in_channels,
            pointcloud_scale=self.pointcloud_scale,
            use_tactile=self.use_tactile,
            tactile_key=self.tactile_key,
            tactile_height=self.tactile_height,
            tactile_width=self.tactile_width,
            tactile_force_scale=self.tactile_force_scale,
        )

    def _init_cem_distribution(
        self,
        batch_size: int,
        horizon: int,
        action_dim: int,
        prev_action: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.cfg.action_low
        high = self.cfg.action_high
        if self.cfg.use_action_prior and self.action_mean is not None and self.action_std is not None:
            mean = self.action_mean.view(1, 1, action_dim).expand(batch_size, horizon, action_dim).clone()
            std = self.action_std.view(1, 1, action_dim).expand(batch_size, horizon, action_dim).clone()
            std = (std * float(self.cfg.action_prior_std_scale)).clamp_min(float(self.cfg.action_prior_min_std))
        else:
            mean = torch.zeros(batch_size, horizon, action_dim, device=self.device)
            std = torch.ones(batch_size, horizon, action_dim, device=self.device)

        if self.cfg.warm_start_mode == "prev_action" and prev_action is not None:
            prev_action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
            if prev_action_t.ndim == 1:
                prev_action_t = prev_action_t[None]
            prev_action_t = torch.clamp(prev_action_t, low, high)
            prev_mean = prev_action_t[:, None, :].expand(batch_size, horizon, action_dim)
            mix = max(0.0, min(1.0, float(self.cfg.warm_start_mix)))
            mean = mix * prev_mean + (1.0 - mix) * mean
            std = torch.minimum(std, torch.full_like(std, float(self.cfg.warm_start_std)))

        return torch.clamp(mean, low, high), std.clamp_min(float(self.cfg.cem_min_std))

    @torch.no_grad()
    def plan(
        self,
        current_info: dict[str, np.ndarray],
        goal_info: dict[str, np.ndarray],
        prev_action: np.ndarray | None = None,
    ):
        cost_mode = self.cfg.cost_mode
        needs_tactile_cost = cost_mode in ("vision_tactile", "vision_tactile_lowdim")
        if needs_tactile_cost and not self.use_tactile:
            raise ValueError(f"cost_mode={cost_mode!r} requires use_tactile=True")

        current = self._model_batch(current_info)
        goal = self._model_batch(goal_info)
        current_visual = self.model.encode_visual(current["vision"])
        current_lowdim = current["lowdim"]
        current_tactile = self.model.encode_tactile(current.get("tactile")) if self.use_tactile else None
        goal_visual = self.model.encode_visual(goal["vision"])[:, -1]
        goal_lowdim = goal["lowdim"][:, -1]
        goal_tactile = None
        if needs_tactile_cost:
            goal_tactile = self.model.encode_tactile(goal["tactile"])[:, -1]  # [B, N, D]

        b = current_visual.size(0)
        h = self.cfg.horizon
        a = self.cfg.action_dim
        s = self.cfg.candidates
        raw_action_history = current_info.get("action")
        if raw_action_history is None:
            action_history = torch.zeros(b, max(self.cfg.ctxt_window - 1, 0), a, device=self.device)
        else:
            action_history = _to_tensor(raw_action_history, self.device)
            if action_history.ndim == 2:
                action_history = action_history.unsqueeze(1)
        low = self.cfg.action_low
        high = self.cfg.action_high
        mean, std = self._init_cem_distribution(
            batch_size=b,
            horizon=h,
            action_dim=a,
            prev_action=prev_action,
        )

        final_cost = None
        final_topk_idx = None
        final_topk_actions = None
        for _ in range(self.cfg.iterations):
            samples = torch.clamp(mean[:, None] + std[:, None] * torch.randn(b, s, h, a, device=self.device), low, high)
            cost_chunks = []
            chunk_size = max(1, int(self.cfg.candidate_chunk_size))
            for start in range(0, s, chunk_size):
                samples_chunk = samples[:, start : start + chunk_size]
                pred_visual, pred_lowdim, pred_tactile = rollout_token_sequences(
                    self.model,
                    visual_context=current_visual,
                    lowdim_context=current_lowdim,
                    tactile_context=current_tactile,
                    action_history=action_history,
                    action_candidates=samples_chunk,
                )
                visual_cost = self.cfg.visual_cost_weight * (
                    pred_visual - goal_visual[:, None, None]
                ).pow(2).mean(dim=(-1, -2))
                lowdim_cost = (pred_lowdim - goal_lowdim[:, None, None]).pow(2).mean(dim=-1)
                if cost_mode in ("vision",):
                    total_step_cost = visual_cost
                elif cost_mode == "lowdim":
                    total_step_cost = lowdim_cost
                elif cost_mode in ("joint", "vision_lowdim"):
                    total_step_cost = visual_cost + self.cfg.lowdim_cost_weight * lowdim_cost
                elif cost_mode == "vision_tactile":
                    tactile_cost = (pred_tactile - goal_tactile[:, None, None]).pow(2).mean(dim=(-1, -2))
                    total_step_cost = visual_cost + self.cfg.tactile_cost_weight * tactile_cost
                elif cost_mode == "vision_tactile_lowdim":
                    tactile_cost = (pred_tactile - goal_tactile[:, None, None]).pow(2).mean(dim=(-1, -2))
                    total_step_cost = (
                        visual_cost
                        + self.cfg.tactile_cost_weight * tactile_cost
                        + self.cfg.lowdim_cost_weight * lowdim_cost
                    )
                else:
                    raise ValueError(f"Unknown cost_mode: {cost_mode!r}. Valid: {PLANNER_COST_MODES}")
                if self.cfg.sum_all_diffs:
                    if self.cfg.discount != 1.0:
                        weights = total_step_cost.new_tensor([self.cfg.discount ** t for t in range(h)]).view(1, 1, h)
                        total_step_cost = total_step_cost * weights
                    cost_chunk = total_step_cost.sum(dim=-1)
                else:
                    cost_chunk = total_step_cost[:, :, -1]

                if self.cfg.action_magnitude_weight > 0:
                    cost_chunk = cost_chunk + float(self.cfg.action_magnitude_weight) * (samples_chunk ** 2).mean(dim=(2, 3))
                if self.cfg.action_smooth_weight > 0 and h > 1:
                    smooth_cost = ((samples_chunk[:, :, 1:] - samples_chunk[:, :, :-1]) ** 2).mean(dim=(2, 3))
                    cost_chunk = cost_chunk + float(self.cfg.action_smooth_weight) * smooth_cost
                cost_chunks.append(cost_chunk)

            cost = torch.cat(cost_chunks, dim=1)

            topk_idx = torch.topk(cost, k=self.cfg.topk, dim=1, largest=False).indices
            topk_actions = torch.gather(samples, 1, topk_idx[:, :, None, None].expand(-1, -1, h, a))
            mean = topk_actions.mean(dim=1)
            std = topk_actions.std(dim=1, unbiased=False).clamp_min(self.cfg.cem_min_std)
            final_cost = cost
            final_topk_idx = topk_idx
            final_topk_actions = topk_actions

        best_action_seq = final_topk_actions[:, 0]
        best_action = best_action_seq[:, 0]
        best_cost = torch.gather(final_cost, 1, final_topk_idx[:, :1]).squeeze(1)
        return (
            best_action.detach().cpu().numpy().astype(np.float32),
            best_action_seq.detach().cpu().numpy().astype(np.float32),
            best_cost.detach().cpu().numpy().astype(np.float32),
        )
