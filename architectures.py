from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_module_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Projector(nn.Module):
    """
    spec like "512-2048-2048"
    """
    def __init__(self, mlp_spec: str):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]
        self.apply(init_module_weights)

    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if max_pooling else nn.Identity()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """
    Input:  [B, C, T, H, W]
    Output: [B, D, T, 1, 1]

    Also supports:
      - forward_features(x) -> [B, C_f, T, H_f, W_f]
      - project_features(feats) -> [B, D, T, 1, 1]
    """
    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=3,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(3, 224, 224),
    ):
        super().__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            out = dummy
            for block in self.stack_blocks:
                out = block(out)
            self.feature_channels = out.shape[1]
            self.feature_hw = (out.shape[2], out.shape[3])
            flattened_dim = out.reshape(out.size(0), -1).shape[1]

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)
        self.final_ln = nn.LayerNorm(self.mlp_output_dim) if final_ln else nn.Identity()

        self.apply(init_module_weights)

    def forward_features(self, x):
        """
        x: [B, C, T, H, W]
        return: [B, C_f, T, H_f, W_f]
        """
        b, c, t, h, w = x.shape
        x = x.permute(2, 0, 1, 3, 4)  # [T, B, C, H, W]

        feats = []
        for i in range(t):
            conv_out = x[i]
            for block in self.stack_blocks:
                conv_out = block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            feats.append(conv_out)

        feats = torch.stack(feats, dim=2)  # [B, C_f, T, H_f, W_f]
        return feats

    def project_features(self, feats):
        """
        feats: [B, C_f, T, H_f, W_f]
        return: [B, D, T, 1, 1]
        """
        b, c, t, h, w = feats.shape
        x = feats.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c * h * w)
        x = self.mlp(x)
        x = self.final_ln(x)
        x = x.view(b, t, self.mlp_output_dim).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x.contiguous()

    def forward(self, x):
        feats = self.forward_features(x)
        return self.project_features(feats)


class TacFFEncoder(nn.Module):
    """
    TacFF token-style encoder.

    Input:
        x: [B, C, T, H, W]
    Output:
        z: [B, D_out, T, 1, 1]

    Feature output for attn:
        forward_features(x) -> [B, D_token, T, HW, 1]

    Design:
      - treat each taxel as one token
      - shared MLP over per-taxel 3D force vector
      - optional learnable position embedding over HW tokens
      - pool over tokens, then final projection
    """
    def __init__(
        self,
        input_channels=3,
        height=10,
        width=14,
        hidden_dim=256,
        out_dim=128,
        use_pos_emb=True,
        pooling="max",
        final_ln=True,
    ):
        super().__init__()
        assert pooling in ["mean", "max"]

        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.num_tokens = height * width
        self.token_dim = hidden_dim
        self.out_dim = out_dim
        self.use_pos_emb = use_pos_emb
        self.pooling = pooling

        self.feature_channels = hidden_dim          # for attn compatibility
        self.feature_hw = (self.num_tokens, 1)     # tokenized layout
        self.mlp_output_dim = out_dim

        self.token_mlp = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
            nn.init.normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = None

        self.final_projection = nn.Linear(hidden_dim, out_dim)
        self.final_ln = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

        self.apply(init_module_weights)

    def forward_features(self, x):
        """
        x: [B, C, T, H, W]
        return: [B, D_token, T, HW, 1]
        """
        b, c, t, h, w = x.shape
        assert c == self.input_channels
        assert h == self.height and w == self.width, \
            f"Expected H,W=({self.height},{self.width}), got ({h},{w})"

        # [B, C, T, H, W] -> [B, T, H, W, C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # [B, T, H, W, C] -> [B*T, HW, C]
        x = x.view(b * t, h * w, c)

        # token-wise MLP: [B*T, HW, C] -> [B*T, HW, D_token]
        feats = self.token_mlp(x)

        if self.pos_emb is not None:
            feats = feats + self.pos_emb

        # [B*T, HW, D_token] -> [B, D_token, T, HW, 1]
        feats = feats.view(b, t, h * w, self.token_dim)
        feats = feats.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
        return feats

    def project_features(self, feats):
        """
        feats: [B, D_token, T, HW, 1]
        return: [B, D_out, T, 1, 1]
        """
        b, d_token, t, n_tok, one = feats.shape
        assert one == 1

        # [B, D_token, T, HW, 1] -> [B, T, HW, D_token]
        x = feats.squeeze(-1).permute(0, 2, 3, 1).contiguous()

        # pool over token dim HW
        if self.pooling == "mean":
            x = x.mean(dim=2)              # [B, T, D_token]
        else:
            x = x.max(dim=2)[0]            # [B, T, D_token]

        # final proj
        x = self.final_projection(x)       # [B, T, D_out]
        x = self.final_ln(x)

        # [B, T, D_out] -> [B, D_out, T, 1, 1]
        x = x.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        return self.project_features(feats)


class PointNetEncoderXYZRGB(nn.Module):
    """
    Input:  [B, N, C]
    Output: [B, D]

    Also supports:
      - forward_features(x) -> [B, N, 512]
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024
    ):
        super().__init__()

        self.in_channels = in_channels

        block_channel = [64, 128, 256, 512]
        self.feature_channels = block_channel[-1]

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        self.final_projection = nn.Sequential(
            nn.Linear(block_channel[-1], out_channels),
            nn.LayerNorm(out_channels),
        )

        self.out_channels = out_channels
        self.mlp_output_dim = out_channels
        self.apply(init_module_weights)

    def forward_features(self, x):
        """
        x: [B, N, C]
        return: [B, N, 512]
        """
        x = x[..., :self.in_channels]
        return self.mlp(x)

    def forward(self, x):
        x = self.forward_features(x)   # [B, N, 512]
        feat = torch.max(x, 1)[0]      # [B, 512]
        feat = self.final_projection(feat)
        return feat


class PointCloudTemporalEncoder(nn.Module):
    """
    Wrap point encoder to support both:
      - token features for attention
      - pooled latent for non-attention fusion

    Input:  [B, T, N, C]
    Output:
      - forward_features(x): [B, C_f, T, N, 1]
      - forward(x):          [B, D,   T, 1, 1]
    """
    def __init__(
        self,
        point_encoder: nn.Module,
        out_dim: int,
        final_ln: bool = True,
    ):
        super().__init__()
        self.point_encoder = point_encoder
        self.out_dim = out_dim
        self.mlp_output_dim = out_dim
        self.feature_channels = getattr(point_encoder, "feature_channels", 512)
        self.final_ln = nn.LayerNorm(out_dim) if final_ln else nn.Identity()

    def forward_features(self, x):
        """
        x: [B, T, N, C]
        return: [B, C_f, T, N, 1]
        """
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)

        feats = self.point_encoder.forward_features(x)   # [B*T, N, C_f]
        feats = feats.view(b, t, n, self.feature_channels)
        feats = feats.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
        return feats  # [B, C_f, T, N, 1]

    def forward(self, x):
        """
        x: [B, T, N, C]
        return: [B, D, T, 1, 1]
        """
        b, t, n, c = x.shape
        x = x.reshape(b * t, n, c)

        z = self.point_encoder(x)   # [B*T, D]
        z = self.final_ln(z)
        z = z.view(b, t, self.out_dim).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return z.contiguous()


class RNNPredictor(nn.Module):
    """
    Input:
      state:  [B, D, T, 1, 1]
      action: [B, A, T]
    Output:
      pred:   [B, D, T, 1, 1]
    """
    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: int = 7,
        num_layers: int = 1,
        final_ln: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.final_ln = final_ln if final_ln is not None else nn.Identity()
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        b, d, t, _, _ = state.shape
        _, a, ta = action.shape
        assert t == ta

        outs = []
        h = None
        for i in range(t):
            s_i = state[:, :, i].reshape(1, b, d).contiguous()
            a_i = action[:, :, i].reshape(1, b, a).contiguous()
            h0 = s_i if h is None else h
            out_i, h = self.rnn(a_i, h0)
            out_i = self.final_ln(out_i)
            outs.append(out_i[0])

        outs = torch.stack(outs, dim=2).unsqueeze(-1).unsqueeze(-1)
        return outs


class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_tp1):
        x = torch.cat([state_t, state_tp1], dim=-1)
        return self.model(x)


class VisionTactileConcatEncoder(nn.Module):
    """
    Standard concat fusion with unified interface.
    """
    def __init__(self, vision_encoder: nn.Module, tactile_encoder: nn.Module):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder

    def encode_modalities(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("VisionTactileConcatEncoder expects dict obs.")
        z_v = self.vision_encoder(obs["vision"])
        z_t = self.tactile_encoder(obs["tactile"])
        return z_v, z_t

    def fuse_latents(self, z_v, z_t):
        return torch.cat([z_v, z_t], dim=1)

    def forward(self, obs):
        z_v, z_t = self.encode_modalities(obs)
        return self.fuse_latents(z_v, z_t)


class VisionTactileGateEncoder(nn.Module):
    """
    Vision-anchored gated residual fusion with unified interface.
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        tactile_encoder: nn.Module,
        latent_dim: int,
        fusion_hidden_dim: Optional[int] = None,
        final_ln: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.latent_dim = latent_dim

        vdim = getattr(vision_encoder, "mlp_output_dim", latent_dim)
        tdim = getattr(tactile_encoder, "mlp_output_dim", latent_dim)
        hidden = fusion_hidden_dim if fusion_hidden_dim is not None else latent_dim

        self.v_proj = nn.Linear(vdim, latent_dim)
        self.t_proj = nn.Linear(tdim, latent_dim)

        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.Sigmoid(),
        )

        self.delta = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

        self.fusion_ln = nn.LayerNorm(latent_dim) if final_ln else nn.Identity()
        self.apply(init_module_weights)

    def encode_modalities(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("VisionTactileGateEncoder expects dict obs.")
        z_v = self.vision_encoder(obs["vision"])
        z_t = self.tactile_encoder(obs["tactile"])
        return z_v, z_t

    def fuse_latents(self, z_v, z_t):
        v = z_v.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()  # [B,T,Dv]
        t = z_t.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()  # [B,T,Dt]

        v = self.v_proj(v)
        t = self.t_proj(t)

        joint = torch.cat([v, t], dim=-1)
        g = self.gate(joint)
        d = self.delta(t)

        fused = v + g * d
        fused = self.fusion_ln(fused)
        fused = fused.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return fused.contiguous()

    def forward(self, obs):
        z_v, z_t = self.encode_modalities(obs)
        return self.fuse_latents(z_v, z_t)


class VisionTactileFiLMEncoder(nn.Module):
    """
    FiLM fusion with unified interface.
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        tactile_encoder: nn.Module,
        latent_dim: int,
        fusion_hidden_dim: Optional[int] = None,
        final_ln: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.latent_dim = latent_dim

        vdim = getattr(vision_encoder, "mlp_output_dim", latent_dim)
        tdim = getattr(tactile_encoder, "mlp_output_dim", latent_dim)
        hidden = fusion_hidden_dim if fusion_hidden_dim is not None else latent_dim

        self.v_proj = nn.Linear(vdim, latent_dim)
        self.t_proj = nn.Linear(tdim, latent_dim)

        self.gamma_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

        self.fusion_ln = nn.LayerNorm(latent_dim) if final_ln else nn.Identity()
        self.apply(init_module_weights)

    def encode_modalities(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("VisionTactileFiLMEncoder expects dict obs.")
        z_v = self.vision_encoder(obs["vision"])
        z_t = self.tactile_encoder(obs["tactile"])
        return z_v, z_t

    def fuse_latents(self, z_v, z_t):
        v = z_v.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        t = z_t.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()

        v = self.v_proj(v)
        t = self.t_proj(t)

        gamma = self.gamma_mlp(t)
        beta = self.beta_mlp(t)

        fused = (1.0 + gamma) * v + beta
        fused = self.fusion_ln(fused)
        fused = fused.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return fused.contiguous()

    def forward(self, obs):
        z_v, z_t = self.encode_modalities(obs)
        return self.fuse_latents(z_v, z_t)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q2 = self.norm_q(q)
        kv2 = self.norm_kv(kv)
        attn_out, _ = self.attn(q2, kv2, kv2, need_weights=False)
        x = q + attn_out
        x = x + self.ffn(x)
        return x


class VisionTactileAttnEncoder(nn.Module):
    """
    Token-level cross attention fusion with unified interface.

    Expect both encoders to support:
      - forward_features(...)
      - forward(...)
      - feature_channels
      - mlp_output_dim
    """
    def __init__(
        self,
        vision_encoder: nn.Module,
        tactile_encoder: nn.Module,
        latent_dim: int,
        final_ln: bool = True,
        attn_d_model: int = 256,
        attn_heads: int = 4,
        attn_layers: int = 2,
        attn_mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        use_modality_emb: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.latent_dim = latent_dim
        self.attn_d_model = attn_d_model
        self.use_modality_emb = use_modality_emb

        cv = vision_encoder.feature_channels
        ct = tactile_encoder.feature_channels

        self.v_proj = nn.Linear(cv, attn_d_model)
        self.t_proj = nn.Linear(ct, attn_d_model)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=attn_d_model,
                n_heads=attn_heads,
                mlp_ratio=attn_mlp_ratio,
                dropout=attn_dropout,
            )
            for _ in range(attn_layers)
        ])

        if use_modality_emb:
            self.v_type_emb = nn.Parameter(torch.zeros(1, 1, attn_d_model))
            self.t_type_emb = nn.Parameter(torch.zeros(1, 1, attn_d_model))
            nn.init.normal_(self.v_type_emb, std=0.02)
            nn.init.normal_(self.t_type_emb, std=0.02)

        self.head = nn.Linear(attn_d_model, latent_dim)
        self.fusion_ln = nn.LayerNorm(latent_dim) if final_ln else nn.Identity()
        self.mlp_output_dim = latent_dim

        self.apply(init_module_weights)

    def encode_modalities(self, obs):
        if not isinstance(obs, dict):
            raise TypeError("VisionTactileAttnEncoder expects dict obs.")

        z_v = self.vision_encoder(obs["vision"])                    # [B, Dv, T, 1, 1]
        z_t = self.tactile_encoder(obs["tactile"])                  # [B, Dt, T, 1, 1]
        v_feats = self.vision_encoder.forward_features(obs["vision"])
        t_feats = self.tactile_encoder.forward_features(obs["tactile"])
        return z_v, z_t, v_feats, t_feats

    def fuse_latents(self, z_v, z_t, v_feats, t_feats):
        """
        z_v, z_t are kept for unified interface / future ablation use.
        v_feats: [B, C_v, T, N_v, 1] or [B, C_v, T, H_v, W_v]
        t_feats: [B, C_t, T, N_t, 1] or [B, C_t, T, H_t, W_t]
        """
        b, cv, t = v_feats.shape[:3]
        _, ct, _ = t_feats.shape[:3]

        # flatten token dims
        v_tokens = v_feats.view(b, cv, t, -1).permute(0, 2, 3, 1).contiguous()  # [B,T,Nv,Cv]
        t_tokens = t_feats.view(b, ct, t, -1).permute(0, 2, 3, 1).contiguous()  # [B,T,Nt,Ct]

        nv = v_tokens.shape[2]
        nt = t_tokens.shape[2]

        v_tokens = v_tokens.view(b * t, nv, cv)   # [BT,Nv,Cv]
        t_tokens = t_tokens.view(b * t, nt, ct)   # [BT,Nt,Ct]

        v_tokens = self.v_proj(v_tokens)
        t_tokens = self.t_proj(t_tokens)

        if self.use_modality_emb:
            v_tokens = v_tokens + self.v_type_emb
            t_tokens = t_tokens + self.t_type_emb

        x = v_tokens
        for blk in self.blocks:
            x = blk(x, t_tokens)

        pooled = x.mean(dim=1)  # [BT, d_model]
        fused = self.head(pooled)
        fused = self.fusion_ln(fused)
        fused = fused.view(b, t, self.latent_dim).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return fused.contiguous()

    def forward(self, obs):
        z_v, z_t, v_feats, t_feats = self.encode_modalities(obs)
        return self.fuse_latents(z_v, z_t, v_feats, t_feats)


def build_vision_tactile_encoder(
    fusion_type,
    vision_encoder,
    tactile_encoder,
    vision_dim,
    tactile_dim,
    fusion_latent_dim=None,
    fusion_hidden_dim=None,
    attn_d_model=256,
    attn_heads=4,
    attn_layers=2,
    attn_mlp_ratio=4.0,
    attn_dropout=0.0,
):
    fusion_type = fusion_type.lower()

    if fusion_type == "concat":
        encoder = VisionTactileConcatEncoder(
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
        )
        return encoder, vision_dim + tactile_dim

    if fusion_type == "gate":
        latent_dim = fusion_latent_dim if fusion_latent_dim is not None else vision_dim
        encoder = VisionTactileGateEncoder(
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
            latent_dim=latent_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            final_ln=True,
        )
        return encoder, latent_dim

    if fusion_type == "film":
        latent_dim = fusion_latent_dim if fusion_latent_dim is not None else vision_dim
        encoder = VisionTactileFiLMEncoder(
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
            latent_dim=latent_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            final_ln=True,
        )
        return encoder, latent_dim

    if fusion_type == "attn":
        latent_dim = fusion_latent_dim if fusion_latent_dim is not None else vision_dim
        encoder = VisionTactileAttnEncoder(
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
            latent_dim=latent_dim,
            final_ln=True,
            attn_d_model=attn_d_model,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            attn_mlp_ratio=attn_mlp_ratio,
            attn_dropout=attn_dropout,
            use_modality_emb=True,
        )
        return encoder, latent_dim

    raise ValueError(f"Unsupported fusion_type: {fusion_type}")