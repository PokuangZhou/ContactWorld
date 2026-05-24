from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
DEFAULT_LOWDIM_KEYS = (
    "ee_pos",
    "ee_quat",
    "plug_pos",
    "plug_quat",
    "socket_pos_gt",
)


def parse_key_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


class ManiFeelSequenceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        vision_key: str = "front",
        vision_type: str = "image",
        action_key: str = "action",
        frameskip: int = 1,
        num_steps: int = 6,
        image_size: int = 224,
        pc_in_channels: int = 3,
        use_tactile: bool = False,
        tactile_key: str | None = None,
        tactile_height: int = 10,
        tactile_width: int = 14,
        tactile_force_scale: float = 0.002,
        pointcloud_scale: float = 0.4,
        lowdim_keys: list[str] | tuple[str, ...] = DEFAULT_LOWDIM_KEYS,
        normalize_images: bool = True,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.root = Path(root)
        self.zroot = zarr.open_group(str(self.root), mode="r")
        self.data = self.zroot["data"]
        self.vision_key = vision_key
        self.vision_type = vision_type.lower()
        self.action_key = action_key
        self.frameskip = int(frameskip)
        self.num_steps = int(num_steps)
        self.span = self.num_steps * self.frameskip
        self.image_size = int(image_size)
        self.pc_in_channels = int(pc_in_channels)
        self.use_tactile = bool(use_tactile)
        self.tactile_key = tactile_key
        self.tactile_height = int(tactile_height)
        self.tactile_width = int(tactile_width)
        self.tactile_force_scale = float(tactile_force_scale)
        self.pointcloud_scale = float(pointcloud_scale)
        self.lowdim_keys = tuple(lowdim_keys)
        self.normalize_images = normalize_images
        self.transform = transform

        episode_ends = np.asarray(self.zroot["meta"]["episode_ends"], dtype=np.int64)
        self.lengths = np.diff(np.concatenate([[0], episode_ends]))
        self.offsets = np.concatenate([[0], episode_ends[:-1]])
        self.clip_indices = [
            (ep_idx, start)
            for ep_idx, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

        keys = [self.vision_key, self.action_key, *self.lowdim_keys]
        if self.use_tactile:
            if self.tactile_key is None:
                raise ValueError("tactile_key is required when use_tactile=True")
            keys.append(self.tactile_key)
        missing = [key for key in keys if key not in self.data]
        if missing:
            raise KeyError(f"Requested keys not found in zarr data/: {missing}")
        self.keys = list(dict.fromkeys(keys))
        self._arrays = {key: self.data[key] for key in self.keys}
        self.lowdim_stats = self._compute_lowdim_stats()

    def __len__(self) -> int:
        return len(self.clip_indices)

    def _load_raw_slice(self, ep_idx: int, start: int, end: int) -> dict[str, np.ndarray]:
        global_start = int(self.offsets[ep_idx] + start)
        global_end = int(self.offsets[ep_idx] + end)
        out = {}
        for key, array in self._arrays.items():
            data = np.asarray(array[global_start:global_end])
            if key != self.action_key:
                data = data[:: self.frameskip]
            out[key] = data
        return out

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(image))
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.ndim != 4:
            raise ValueError(f"Expected image [T,H,W,C] or [T,H,W], got {tuple(x.shape)}")
        if x.shape[-1] == 4:
            x = x[..., :3]
        if x.shape[-1] == 1:
            x = x.expand(*x.shape[:-1], 3)
        x = x.permute(0, 3, 1, 2).contiguous().float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if self.normalize_images:
            x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        return x

    def _preprocess_pointcloud(self, pc: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(pc)).float()
        if x.ndim != 3:
            raise ValueError(f"Expected pointcloud [T,N,C], got {tuple(x.shape)}")
        if x.size(-1) < 3:
            raise ValueError(f"Pointcloud last dim must be >=3, got {x.size(-1)}")
        x = x[..., : self.pc_in_channels].clone()
        x[..., :3] = x[..., :3] / self.pointcloud_scale
        if x.size(-1) >= 6:
            rgb = x[..., 3:6]
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            x[..., 3:6] = rgb
        return x.contiguous()

    def _preprocess_tactile(self, tactile: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(tactile)).float()
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.ndim != 4:
            raise ValueError(f"Expected tactile [T,H,W,C] or [T,H,W], got {tuple(x.shape)}")
        if x.shape[-1] == 4:
            x = x[..., :3]
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-2:] != (self.tactile_height, self.tactile_width):
            x = F.interpolate(
                x,
                size=(self.tactile_height, self.tactile_width),
                mode="bilinear",
                align_corners=False,
            )

        tactile_key = (self.tactile_key or "").lower()
        if "force_field" in tactile_key or "tacff" in tactile_key:
            x = x / self.tactile_force_scale
        elif "depth" in tactile_key:
            x = x.clamp(0.0, 1.0)
        elif "rgb" in tactile_key or "image" in tactile_key or "taxim" in tactile_key:
            x = x.clamp(0.0, 1.0)
        return x.contiguous()

    @staticmethod
    def _dense_to_tensor(data: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(data)).float()
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        return x.reshape(x.shape[0], -1)

    def _build_lowdim(self, raw_steps: dict[str, np.ndarray]) -> torch.Tensor:
        parts = []
        for key in self.lowdim_keys:
            x = self._dense_to_tensor(raw_steps[key])
            mean = self.lowdim_stats[key]["mean"]
            std = self.lowdim_stats[key]["std"]
            parts.append((x - mean.view(1, -1)) / std.view(1, -1))
        return torch.cat(parts, dim=-1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.clip_indices[idx]
        raw_steps = self._load_raw_slice(ep_idx, start, start + self.span)

        if self.vision_type == "image":
            vision = self._preprocess_image(raw_steps[self.vision_key])
        elif self.vision_type == "pc":
            vision = self._preprocess_pointcloud(raw_steps[self.vision_key])
        else:
            raise ValueError(f"Unknown vision_type: {self.vision_type}")

        item = {
            "vision": vision,
            "action": self._dense_to_tensor(raw_steps[self.action_key]).reshape(self.num_steps, -1),
            "lowdim": self._build_lowdim(raw_steps),
        }
        if self.use_tactile:
            item["tactile"] = self._preprocess_tactile(raw_steps[self.tactile_key])
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _compute_lowdim_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        stats = {}
        for key in self.lowdim_keys:
            arr = np.asarray(self.data[key])
            if arr.ndim == 1:
                arr = arr[:, None]
            flat = torch.as_tensor(arr.reshape(-1, arr.shape[-1]), dtype=torch.float32)
            stats[key] = {
                "mean": flat.mean(dim=0),
                "std": flat.std(dim=0, unbiased=False).clamp_min(1e-6),
            }
        return stats

    def action_dim(self) -> int:
        data = np.asarray(self.data[self.action_key])
        base_dim = int(np.prod(data.shape[1:])) if data.ndim > 1 else 1
        return base_dim * self.frameskip

    def lowdim_dim(self) -> int:
        return sum(self.lowdim_stats[key]["mean"].numel() for key in self.lowdim_keys)

    def tactile_channels(self) -> int:
        if not self.use_tactile or self.tactile_key is None:
            return 0
        shape = self.data[self.tactile_key].shape
        if len(shape) >= 4:
            return int(shape[-1]) if shape[-1] in (1, 3, 4) else 1
        return 1

    def vision_channels(self) -> int:
        if self.vision_type == "pc":
            return self.pc_in_channels
        return 3
