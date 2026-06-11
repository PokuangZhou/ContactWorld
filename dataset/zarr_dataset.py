from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import zarr


class BaseEpisodeDataset:
    """
    Minimal episode-based dataset abstraction.

    This is a lightweight local replacement for stable_worldmodel.data.dataset.Dataset.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        offsets: np.ndarray,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.offsets = np.asarray(offsets)
        self.frameskip = int(frameskip)
        self.num_steps = int(num_steps)
        self.span = self.num_steps * self.frameskip
        self.transform = transform

        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)

        # Match Le-WM/stable_worldmodel convention:
        # action is reshaped to [num_steps, -1]
        if "action" in steps:
            steps["action"] = steps["action"].reshape(self.num_steps, -1)

        return steps

    def load_chunk(
        self,
        episodes_idx: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(int(ep), int(s), int(e))
            if "action" in steps:
                steps["action"] = steps["action"].reshape((e - s) // self.frameskip, -1)
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        return self._load_slice(int(episode_idx), 0, int(self.lengths[episode_idx]))

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        raise NotImplementedError

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        raise NotImplementedError


class ZarrDataset(BaseEpisodeDataset):
    """
    Episode dataset for TVB zarr data.

    Expected structure:
        root/
          data/
            action
            front
            left_tactile_camera_taxim
            right_tactile_camera_taxim
            state
            tactile_force_field_right
            tactile_depth_right
            ...
          meta/
            episode_ends
    """

    def __init__(
        self,
        root: str | Path,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.zroot = zarr.open_group(str(self.root), mode="r")

        if "meta" not in self.zroot or "episode_ends" not in self.zroot["meta"]:
            raise KeyError(f"Could not find meta/episode_ends in zarr root: {self.root}")

        episode_ends = np.asarray(self.zroot["meta"]["episode_ends"])
        if episode_ends.ndim != 1 or len(episode_ends) == 0:
            raise ValueError("meta/episode_ends must be a non-empty 1D array")

        lengths = np.diff(np.concatenate([[0], episode_ends]))
        offsets = np.concatenate([[0], episode_ends[:-1]])

        self.data_group = self.zroot["data"]
        available_keys = list(self.data_group.keys())
        self._keys = keys_to_load or available_keys

        missing = [k for k in self._keys if k not in available_keys]
        if missing:
            raise KeyError(f"Requested keys not found in zarr data/: {missing}")

        self._arrays: dict[str, Any] = {k: self.data_group[k] for k in self._keys}
        self._cache: dict[str, np.ndarray] = {}

        for k in keys_to_cache or []:
            if k not in self._arrays:
                raise KeyError(f"Cannot cache missing key: {k}")
            self._cache[k] = np.asarray(self._arrays[k])

        super().__init__(
            lengths=lengths,
            offsets=offsets,
            frameskip=frameskip,
            num_steps=num_steps,
            transform=transform,
        )

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        if col not in self._arrays:
            raise KeyError(col)
        return np.asarray(self._arrays[col])

    def _maybe_to_tensor(self, data: np.ndarray) -> Any:
        if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
            val = data[0] if len(data) > 0 else b""
            return val.decode() if isinstance(val, bytes) else val

        x = torch.from_numpy(np.asarray(data))

        # Convert [T, H, W, C] -> [T, C, H, W]
        if x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        return x

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start = int(self.offsets[ep_idx] + start)
        g_end = int(self.offsets[ep_idx] + end)

        steps: dict[str, Any] = {}
        for col in self._keys:
            src = self._cache[col] if col in self._cache else self._arrays[col]
            data = np.asarray(src[g_start:g_end])

            # Match Le-WM convention:
            # non-action obs are frame-skipped, action is kept dense
            if col != "action":
                data = data[:: self.frameskip]

            steps[col] = self._maybe_to_tensor(data)

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {col: self._get_col(col)[row_idx] for col in self._keys}

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        if isinstance(source, str):
            source = [source]

        for s in source:
            if s not in self._keys:
                raise KeyError(f"merge_col source key not found: {s}")

        merged = np.concatenate([self._get_col(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)