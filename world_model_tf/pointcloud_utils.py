from __future__ import annotations

import math
from collections.abc import Sequence

import torch


PC_TOKENIZERS = ("none", "voxel")


def parse_int_tuple(value: str | Sequence[int], *, expected_len: int, name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {value!r}")
    return tuple(int(part) for part in parts)


def parse_float_tuple(value: str | Sequence[float], *, expected_len: int, name: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {value!r}")
    return tuple(float(part) for part in parts)


def effective_pc_num_points(
    raw_num_points: int,
    *,
    pc_tokenizer: str = "none",
    pc_num_points: int | None = None,
    pc_voxel_grid: str | Sequence[int] = (16, 16, 1),
) -> int:
    pc_tokenizer = str(pc_tokenizer).lower()
    if pc_tokenizer == "none":
        return int(raw_num_points if pc_num_points is None else pc_num_points)
    if pc_tokenizer == "voxel":
        grid = parse_int_tuple(pc_voxel_grid, expected_len=3, name="pc_voxel_grid")
        grid_points = int(grid[0] * grid[1] * grid[2])
        if pc_num_points is not None and int(pc_num_points) != grid_points:
            raise ValueError(f"pc_num_points={pc_num_points} must equal product(pc_voxel_grid)={grid_points}")
        return grid_points
    raise ValueError(f"Unknown pc_tokenizer={pc_tokenizer!r}. Valid: {PC_TOKENIZERS}")


def _normalize_pc_features(x: torch.Tensor, *, pointcloud_scale: float) -> torch.Tensor:
    x = x.clone()
    x[..., :3] = x[..., :3] / float(pointcloud_scale)
    if x.size(-1) >= 6:
        rgb = x[..., 3:6]
        if rgb.numel() > 0 and rgb.max() > 1.5:
            rgb = rgb / 255.0
        x[..., 3:6] = rgb
    return x.contiguous()


def _lexsort_xyz_indices(xyz: torch.Tensor) -> torch.Tensor:
    ranges = xyz.max(dim=0).values - xyz.min(dim=0).values
    weights = torch.empty(3, dtype=xyz.dtype, device=xyz.device)
    weights[2] = 1.0
    weights[1] = ranges[2] + 1.0
    weights[0] = (ranges[1] + 1.0) * weights[1]
    return torch.argsort((xyz * weights).sum(dim=-1), stable=True)


def _sample_ordered_points(x: torch.Tensor, *, pc_num_points: int | None, pc_order_mode: str) -> torch.Tensor:
    if pc_order_mode == "xyz":
        x = x[_lexsort_xyz_indices(x[:, :3])]
    elif pc_order_mode != "none":
        raise ValueError("pc_order_mode must be one of: none, xyz")

    if pc_num_points is None or int(pc_num_points) == x.size(0):
        return x
    pc_num_points = int(pc_num_points)
    if pc_num_points <= 0:
        raise ValueError(f"pc_num_points must be positive, got {pc_num_points}")
    if x.size(0) >= pc_num_points:
        idx = torch.linspace(0, x.size(0) - 1, pc_num_points, device=x.device).round().long()
        return x[idx]
    repeat_idx = torch.arange(pc_num_points - x.size(0), device=x.device) % x.size(0)
    return torch.cat([x, x[repeat_idx]], dim=0)


def _voxelize_single(
    points: torch.Tensor,
    *,
    pc_in_channels: int,
    pc_voxel_grid: tuple[int, int, int],
    pc_bounds: tuple[float, float, float, float, float, float],
) -> torch.Tensor:
    device = points.device
    dtype = points.dtype
    gx, gy, gz = pc_voxel_grid
    xmin, xmax, ymin, ymax, zmin, zmax = pc_bounds
    num_cells = gx * gy * gz

    xs = torch.linspace(xmin, xmax, gx + 1, dtype=dtype, device=device)
    ys = torch.linspace(ymin, ymax, gy + 1, dtype=dtype, device=device)
    zs = torch.linspace(zmin, zmax, gz + 1, dtype=dtype, device=device)
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    cz = 0.5 * (zs[:-1] + zs[1:])
    mesh = torch.meshgrid(cx, cy, cz, indexing="ij")
    anchors = torch.stack(mesh, dim=-1).reshape(num_cells, 3)

    out = torch.zeros(num_cells, pc_in_channels, dtype=dtype, device=device)
    out[:, :3] = anchors

    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] <= ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )
    if not mask.any():
        return out

    pts = points[mask]
    xyz = pts[:, :3]
    ix = ((xyz[:, 0] - xmin) / max(xmax - xmin, 1e-6) * gx).floor().long().clamp(0, gx - 1)
    iy = ((xyz[:, 1] - ymin) / max(ymax - ymin, 1e-6) * gy).floor().long().clamp(0, gy - 1)
    iz = ((xyz[:, 2] - zmin) / max(zmax - zmin, 1e-6) * gz).floor().long().clamp(0, gz - 1)
    cell_idx = ix * (gy * gz) + iy * gz + iz

    sums = torch.zeros_like(out)
    counts = torch.zeros(num_cells, 1, dtype=dtype, device=device)
    sums.index_add_(0, cell_idx, pts)
    counts.index_add_(0, cell_idx, torch.ones(pts.size(0), 1, dtype=dtype, device=device))
    filled = counts.squeeze(-1) > 0
    out[filled] = sums[filled] / counts[filled]
    return out


def canonicalize_pointcloud(
    points,
    *,
    pc_in_channels: int = 3,
    pointcloud_scale: float = 0.4,
    pc_tokenizer: str = "none",
    pc_num_points: int | None = None,
    pc_order_mode: str = "none",
    pc_voxel_grid: str | Sequence[int] = (16, 16, 1),
    pc_bounds: str | Sequence[float] = (-0.4, 0.8, -0.6, 0.6, -0.2, 0.8),
) -> torch.Tensor:
    x = torch.as_tensor(points).float()
    if x.ndim < 2:
        raise ValueError(f"Expected pointcloud [...,N,C], got {tuple(x.shape)}")
    if x.size(-1) < 3:
        raise ValueError(f"Pointcloud last dim must be >=3, got {x.size(-1)}")
    x = x[..., : int(pc_in_channels)].clone()

    pc_tokenizer = str(pc_tokenizer).lower()
    if pc_tokenizer == "none":
        leading = x.shape[:-2]
        flat = x.reshape(-1, x.size(-2), x.size(-1))
        if pc_order_mode != "none" or pc_num_points is not None:
            flat = torch.stack(
                [
                    _sample_ordered_points(frame, pc_num_points=pc_num_points, pc_order_mode=pc_order_mode)
                    for frame in flat
                ],
                dim=0,
            )
            x = flat.reshape(*leading, flat.size(-2), flat.size(-1))
        return _normalize_pc_features(x, pointcloud_scale=pointcloud_scale)

    if pc_tokenizer == "voxel":
        grid = parse_int_tuple(pc_voxel_grid, expected_len=3, name="pc_voxel_grid")
        bounds = parse_float_tuple(pc_bounds, expected_len=6, name="pc_bounds")
        num_points = effective_pc_num_points(
            x.size(-2),
            pc_tokenizer=pc_tokenizer,
            pc_num_points=pc_num_points,
            pc_voxel_grid=grid,
        )
        side = int(round(math.sqrt(num_points)))
        if side * side != num_points:
            raise ValueError(f"Voxel token count must be square for AC predictor, got {num_points}")

        leading = x.shape[:-2]
        flat = x.reshape(-1, x.size(-2), x.size(-1))
        tokens = torch.stack(
            [
                _voxelize_single(
                    frame,
                    pc_in_channels=int(pc_in_channels),
                    pc_voxel_grid=grid,
                    pc_bounds=bounds,
                )
                for frame in flat
            ],
            dim=0,
        )
        tokens = tokens.reshape(*leading, num_points, int(pc_in_channels))
        return _normalize_pc_features(tokens, pointcloud_scale=pointcloud_scale)

    raise ValueError(f"Unknown pc_tokenizer={pc_tokenizer!r}. Valid: {PC_TOKENIZERS}")
