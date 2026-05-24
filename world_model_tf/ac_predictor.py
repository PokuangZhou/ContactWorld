# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from vjepa2/src/models/ac_predictor.py and made self-contained for
# ManiFeel world_model_tf. The predictor keeps the V-JEPA2 action/state-token
# causal layout, but does not depend on the external src package.

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def build_action_block_causal_attention_mask(T, H, W, add_tokens=2):
    tokens_per_step = add_tokens + (H * W)
    n_tokens = T * tokens_per_step
    mask = torch.zeros(n_tokens, n_tokens).bool()
    mask_block = torch.ones(tokens_per_step, tokens_per_step).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[
                t1 * tokens_per_step : (t1 + 1) * tokens_per_step,
                t2 * tokens_per_step : (t2 + 1) * tokens_per_step,
            ] = mask_block
    return mask


def rotate_queries_or_keys(x, pos):
    _, _, _, dim = x.size()
    if dim == 0:
        return x
    assert dim % 2 == 0, "Embedding dimension must be a multiple of 2 for RoPE"

    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freq.cos().squeeze(-1).repeat(1, 1, 1, 2)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        swiglu_hidden_features = hidden_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            swiglu_hidden_features = (swiglu_hidden_features + 7) // 8 * 8
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        return self.fc3(self.act(self.fc1(x)) * self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, attn_mask=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.proj_drop_prob if self.training else 0.0,
                is_causal=self.is_causal,
                attn_mask=attn_mask,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = self.attn_drop(attn.softmax(dim=-1))
            x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class ACRoPEAttention(Attention):
    def __init__(self, *args, grid_size=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size
        self.head_dim = self.proj.in_features // self.num_heads
        self.d_dim = int(2 * ((self.head_dim // 3) // 2))
        self.h_dim = int(2 * ((self.head_dim // 3) // 2))
        self.w_dim = int(2 * ((self.head_dim // 3) // 2))

    @staticmethod
    def _separate_positions(ids, h_patches, w_patches):
        tokens_per_frame = int(h_patches * w_patches)
        tokens_per_row = int(w_patches)
        frame_ids = ids // tokens_per_frame
        rem = ids - tokens_per_frame * frame_ids
        height_ids = rem // tokens_per_row
        width_ids = rem - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        b, n, c = x.size()
        if action_tokens > 0:
            x = x.view(b, -1, action_tokens + H * W, c)
            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                token = x[:, :, i : i + 1, :].flatten(1, 2)
                qkv = self.qkv(token).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                action_q.append(torch.cat([qd, q[..., self.d_dim :]], dim=-1).view(b, self.num_heads, T, 1, -1))
                action_k.append(torch.cat([kd, k[..., self.d_dim :]], dim=-1).view(b, self.num_heads, T, 1, -1))
                action_v.append(v.view(b, self.num_heads, T, 1, -1))
            action_q = torch.cat(action_q, dim=3).flatten(2, 3)
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)
            x = x[:, :, action_tokens:, :].flatten(1, 2)

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ids = torch.arange(int(T * H * W), device=x.device)
        d_pos, h_pos, w_pos = self._separate_positions(ids, H, W)
        h_pos *= self.grid_size / H
        w_pos *= self.grid_size / W

        s = 0
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_pos)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_pos)
        s += self.d_dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_pos)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_pos)
        s += self.h_dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_pos)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_pos)
        s += self.w_dim
        q = torch.cat([qd, qh, qw, q[..., s:]], dim=-1)
        k = torch.cat([kd, kh, kw, k[..., s:]], dim=-1)

        if action_tokens > 0:
            def merge(frame_tokens, action_tokens_tensor):
                frame_tokens = frame_tokens.view(b, self.num_heads, T, H * W, -1)
                action_tokens_tensor = action_tokens_tensor.view(b, self.num_heads, T, action_tokens, -1)
                return torch.cat([action_tokens_tensor, frame_tokens], dim=3).flatten(2, 3)

            q = merge(q, action_q)
            k = merge(k, action_k)
            v = merge(v, action_v)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.proj_drop_prob if self.training else 0.0,
            attn_mask=attn_mask,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_rope=False,
        grid_size=16,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        attn_cls = ACRoPEAttention if use_rope else Attention
        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            grid_size=grid_size,
        ) if use_rope else attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(dim, hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop)
        else:
            self.mlp = MLP(dim, hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerPredictorAC(nn.Module):
    """Action-conditioned causal ViT predictor for per-frame spatial tokens."""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=1,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,
        state_embed_dim: int | None = None,
    ):
        super().__init__()
        self.is_frame_causal = is_frame_causal
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        self.state_encoder = nn.Linear(state_embed_dim or action_embed_dim, predictor_embed_dim, bias=True)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_height = self.img_height // self.patch_size
        self.grid_width = self.img_width // self.patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                ACBlock(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            attn_mask = build_action_block_causal_attention_mask(
                self.num_frames // self.tubelet_size,
                self.grid_height,
                self.grid_width,
                add_tokens=2,
            )
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data if hasattr(layer.mlp, "fc2") else layer.mlp.fc3.weight.data, layer_id + 1)

    def forward(self, x, actions, states):
        x = self.predictor_embed(x)
        b, n_ctxt, d = x.size()
        tokens_per_frame = self.grid_height * self.grid_width
        t = n_ctxt // tokens_per_frame

        state_tokens = self.state_encoder(states).unsqueeze(2)
        action_tokens = self.action_encoder(actions).unsqueeze(2)
        x = x.view(b, t, tokens_per_frame, d)
        x = torch.cat([action_tokens, state_tokens, x], dim=2).flatten(1, 2)

        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        for block in self.predictor_blocks:
            x = block(
                x,
                attn_mask=attn_mask,
                T=t,
                H=self.grid_height,
                W=self.grid_width,
                action_tokens=2,
            )

        x = x.view(b, t, 2 + tokens_per_frame, d)[:, :, 2:, :].flatten(1, 2)
        x = self.predictor_norm(x)
        return self.predictor_proj(x)


def vit_ac_predictor(**kwargs):
    return VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
