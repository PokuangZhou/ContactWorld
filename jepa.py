from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAbase(nn.Module):
    def __init__(self, encoder, aencoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = aencoder
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    @torch.no_grad()
    def encode_raw(self, observations):
        return self.encoder(observations)


class JEPA(JEPAbase):
    def __init__(
        self,
        encoder,
        aencoder,
        predictor,
        regularizer,
        predcost,
        vision_dim: int,
        tactile_dim: int,
        fusion_type: str = "concat",
        reg_on_vision_only: bool = False,
    ):
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost
        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.fusion_type = fusion_type.lower()
        self.reg_on_vision_only = reg_on_vision_only

    def _encode_state_and_modalities(self, observations):
        if (
            isinstance(observations, dict)
            and hasattr(self.encoder, "encode_modalities")
            and hasattr(self.encoder, "fuse_latents")
        ):
            encoded = self.encoder.encode_modalities(observations)

            # concat/gate/film: return z_v, z_t
            if len(encoded) == 2:
                z_v, z_t = encoded
                state = self.encoder.fuse_latents(z_v, z_t)
                return state, z_v, z_t

            # attn: return z_v, z_t, v_feats, t_feats
            if len(encoded) == 4:
                z_v, z_t, v_feats, t_feats = encoded
                state = self.encoder.fuse_latents(z_v, z_t, v_feats, t_feats)
                return state, z_v, z_t

        state = self.encoder(observations)
        return state, None, None

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    ):
        state, z_v, z_t = self._encode_state_and_modalities(observations)
        actions_encoded = self.action_encoder(actions) if actions is not None else None

        if compute_loss:
            reg_state = state

            if (
                self.reg_on_vision_only
                and z_v is not None
            ):
                reg_state = z_v

            rloss, rloss_unweight, rloss_dict = self.regularizer(reg_state, actions_encoded)

            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        all_steps = [] if return_all_steps else None

        if unroll_mode != "autoregressive":
            raise ValueError("This TVB EB wrapper only supports autoregressive mode.")

        effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time
        predicted_states = state[:, :, :effective_ctxt_window]

        for i in range(nsteps):
            context_states = predicted_states[:, :, -effective_ctxt_window:]
            context_actions = actions_encoded[
                :, :, max(0, i + 1 - effective_ctxt_window): i + 1
            ]

            pred_joint_step = self.predictor(context_states, context_actions)[:, :, -1:]
            predicted_states = torch.cat([predicted_states, pred_joint_step], dim=2)

            if return_all_steps:
                all_steps.append(predicted_states.clone())

            if compute_loss:
                target = state[:, :, i + 1: i + 2]
                ploss += self.predcost(pred_joint_step, target) / nsteps

        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        if return_all_steps:
            return all_steps, losses
        return predicted_states, losses


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        regularizer: nn.Module,
        predcost: nn.Module,
        action_dim: int,
        vision_key: str,
        vision_type: str = "image",
        image_size: Union[int, Tuple[int, int]] = 224,
        use_tactile: bool = False,
        tactile_key: str = None,
        tactile_size=None,
        vision_dim: int = 512,
        tactile_dim: int = 512,
        fusion_type: str = "concat",
        reg_on_vision_only: bool = False,
        pointcloud_scale: float = 0.4,
        tactile_force_scale: float = 0.002,
    ):
        super().__init__()
        self.vision_key = vision_key
        self.vision_type = vision_type.lower()
        self.image_size = image_size
        self.action_dim = action_dim

        self.use_tactile = use_tactile
        self.tactile_key = tactile_key
        self.tactile_size = tactile_size

        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.fusion_type = fusion_type.lower()
        self.reg_on_vision_only = reg_on_vision_only
        self.pointcloud_scale = float(pointcloud_scale)
        self.tactile_force_scale = float(tactile_force_scale)

        self.jepa = JEPA(
            encoder=encoder,
            aencoder=nn.Identity(),
            predictor=predictor,
            regularizer=regularizer,
            predcost=predcost,
            vision_dim=vision_dim,
            tactile_dim=tactile_dim,
            fusion_type=self.fusion_type,
            reg_on_vision_only=reg_on_vision_only,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("vision_mean", mean, persistent=False)
        self.register_buffer("vision_std", std, persistent=False)

    def _resolve_image_hw(self) -> Tuple[int, int]:
        if isinstance(self.image_size, int):
            return self.image_size, self.image_size
        if isinstance(self.image_size, (tuple, list)) and len(self.image_size) == 2:
            return int(self.image_size[0]), int(self.image_size[1])
        raise ValueError(f"Unsupported image_size: {self.image_size}")

    def _preprocess_image_btchw_to_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        """
        image-like input supports:
          [B, T, H, W]      -> treated as single-channel
          [B, T, C, H, W]

        output:
          [B, C, T, H, W]
        """
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B,T,1,H,W]

        if x.ndim != 5:
            raise ValueError(f"Image input must be [B,T,C,H,W] or [B,T,H,W], got {tuple(x.shape)}")

        out_h, out_w = self._resolve_image_hw()

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        if h != out_h or w != out_w:
            x = F.interpolate(
                x,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )

        if c == 3:
            x = (x - self.vision_mean) / self.vision_std

        x = x.reshape(b, t, c, out_h, out_w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def _preprocess_pc_btnc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Point cloud input:
          [B, T, N, C], usually C=3 (xyz) or C=6 (xyzrgb)

        For your current dataset:
          xyz roughly in [-0.4, 1.0], std ~ 0.15
        So normalize xyz by pointcloud_scale (default 0.4).

        If C >= 6, assume last 3 dims are rgb in [0,1] or [0,255].
        """
        x = x.float()

        if x.ndim != 4:
            raise ValueError(f"Point cloud input must be [B,T,N,C], got {tuple(x.shape)}")

        c = x.shape[-1]
        if c < 3:
            raise ValueError(f"Point cloud last dim must be >= 3, got {c}")

        x = x.clone()

        # normalize xyz
        x[..., :3] = x[..., :3] / self.pointcloud_scale

        # optional rgb handling for xyzrgb point clouds
        if c >= 6:
            rgb = x[..., 3:6]
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            x[..., 3:6] = rgb

        return x.contiguous()

    def _preprocess_tactile_btchw_to_bcthw(
        self,
        x: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        """
        tactile input supports:
          [B, T, H, W]      -> single-channel
          [B, T, C, H, W]

        output:
          [B, C, T, H, W]

        Normalization policy for your dataset:
          - tactile_rgb_right: already roughly in [0,1], keep as is
          - tactile_depth_right: already in [0,1], keep as is
          - tactile_force_field_right: roughly [-0.003, 0.003], divide by 0.002
        """
        x = x.float()

        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B,T,1,H,W]

        if x.ndim != 5:
            raise ValueError(f"Unsupported tactile shape: {tuple(x.shape)}")

        # generic uint8-like safeguard
        if x.max() > 1.5:
            x = x / 255.0

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        if h != out_h or w != out_w:
            x = F.interpolate(
                x,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )

        x = x.reshape(b, t, c, out_h, out_w)

        # modality-specific normalization
        tactile_key = self.tactile_key.lower() if self.tactile_key is not None else ""

        # tactile force field: scale up from ~1e-3 to O(1e-1)
        if "force_field" in tactile_key or "tacff" in tactile_key:
            x = x / self.tactile_force_scale

        # tactile depth: already in [0,1], optionally clamp for safety
        elif "depth" in tactile_key:
            x = x.clamp(0.0, 1.0)

        # tactile rgb: already float [0,1]-ish in your saved data, keep as is
        elif "rgb" in tactile_key or "image" in tactile_key:
            x = x.clamp(0.0, 1.0)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def preprocess_vision(self, x: torch.Tensor) -> torch.Tensor:
        if self.vision_type == "image":
            return self._preprocess_image_btchw_to_bcthw(x)
        if self.vision_type == "pc":
            return self._preprocess_pc_btnc(x)
        raise ValueError(f"Unknown vision_type: {self.vision_type}")

    def _prepare_batch_obs(self, batch: dict):
        vision_obs = self.preprocess_vision(batch[self.vision_key])

        if self.use_tactile:
            tactile_obs = self._preprocess_tactile_btchw_to_bcthw(
                batch[self.tactile_key],
                out_h=self.tactile_size[0],
                out_w=self.tactile_size[1],
            )
            return {
                "vision": vision_obs,
                "tactile": tactile_obs,
            }
        return vision_obs

    def training_losses(self, batch: dict, nsteps: int):
        obs = self._prepare_batch_obs(batch)
        act = batch["action"].float().transpose(1, 2)  # [B,A,T]

        _, losses = self.jepa.unroll(
            obs,
            act,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=True,
            return_all_steps=False,
        )
        total_loss, reg_loss, _, reg_dict, pred_loss = losses
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "reg_loss": reg_loss,
            "reg_dict": reg_dict,
        }

    def encode(self, batch: dict):
        obs = self._prepare_batch_obs(batch)
        enc_out = self.jepa.encoder(obs)
        emb = enc_out.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        return emb

    def predict_sequence(self, z_hist: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        z5 = z_hist.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()  # [B,D,T,1,1]
        a = action_seq.transpose(1, 2).contiguous()  # [B,A,T]
        pred5 = self.jepa.predictor(z5, a)
        pred = pred5.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        return pred

    def rollout_latent(
        self,
        z0: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        B, S, T, A = action_sequence.shape
        D = z0.size(-1)

        z = z0.unsqueeze(1).expand(B, S, 1, D).reshape(B * S, 1, D)
        act = action_sequence.reshape(B * S, T, A)

        preds = []
        for t in range(T):
            a_t = act[:, t:t + 1, :]
            pred = self.predict_sequence(z[:, -1:, :], a_t)
            z_next = pred[:, -1:, :]
            preds.append(z_next)
            z = torch.cat([z, z_next], dim=1)

        pred_rollout = torch.cat(preds, dim=1)
        pred_rollout = pred_rollout.reshape(B, S, T, D).contiguous()
        return pred_rollout

    def _expand_goal_to_rollout(self, goal_emb: torch.Tensor, pred_emb: torch.Tensor) -> torch.Tensor:
        if goal_emb.ndim == 2:
            goal_emb = goal_emb[:, None, None, :].expand(
                pred_emb.size(0), pred_emb.size(1), pred_emb.size(2), -1
            )
        elif goal_emb.ndim == 3:
            goal_emb = goal_emb[:, :, None, :].expand_as(pred_emb)
        else:
            raise ValueError(f"Unsupported goal_emb shape: {tuple(goal_emb.shape)}")
        return goal_emb

    def criterion(
        self,
        pred_rollout: torch.Tensor,
        goal_emb: torch.Tensor,
        sum_all_diffs: bool = False,
        discount: float = 1.0,
    ) -> torch.Tensor:
        goal_emb = self._expand_goal_to_rollout(goal_emb, pred_rollout)
        total_step_cost = F.mse_loss(pred_rollout, goal_emb.detach(), reduction="none").sum(dim=-1)

        if sum_all_diffs:
            if discount != 1.0:
                T = total_step_cost.size(-1)
                w = total_step_cost.new_tensor([discount ** t for t in range(T)]).view(1, 1, T)
                total_step_cost = total_step_cost * w
            return total_step_cost.sum(dim=-1)

        return total_step_cost[:, :, -1]

    def get_cost(
        self,
        current_info: dict,
        action_candidates: torch.Tensor,
        goal_info: dict,
        sum_all_diffs: bool = False,
        discount: float = 1.0,
    ) -> torch.Tensor:
        device = action_candidates.device

        current_batch = {
            self.vision_key: current_info[self.vision_key].to(device),
        }
        goal_batch = {
            self.vision_key: goal_info[self.vision_key].to(device),
        }

        if self.use_tactile:
            current_batch[self.tactile_key] = current_info[self.tactile_key].to(device)
            goal_batch[self.tactile_key] = goal_info[self.tactile_key].to(device)

        emb = self.encode(current_batch)
        goal_emb = self.encode(goal_batch)

        z0 = emb[:, -1:, :]  # [B,1,D]
        pred_rollout = self.rollout_latent(z0, action_candidates)  # [B,S,T,D]

        return self.criterion(
            pred_rollout=pred_rollout,
            goal_emb=goal_emb[:, -1, :],
            sum_all_diffs=sum_all_diffs,
            discount=discount,
        )