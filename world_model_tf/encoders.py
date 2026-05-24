from __future__ import annotations

from dataclasses import dataclass
import importlib.abc
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn


WORLD_MODEL_TF_DIR = Path(__file__).resolve().parent
MANIFEEL_PACKAGE_DIR = WORLD_MODEL_TF_DIR.parent
REPO_ROOT = WORLD_MODEL_TF_DIR.parents[2]
DINO3_ROOT = Path(os.environ.get("DINOV3_ROOT", REPO_ROOT / "dinov3")).expanduser().resolve()
DINO3_DEFAULT_CKPT_DIR = WORLD_MODEL_TF_DIR / "wm_tf_data" / "pretrianed_model" / "dino3"


class _FutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source_path = self.get_filename(fullname)
        source_bytes = self.get_data(source_path)
        return self.source_to_code(source_bytes, source_path)

    def source_to_code(self, data, path, *, _optimize=-1):
        text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        if "from __future__ import annotations" not in text[:300]:
            text = "from __future__ import annotations\n" + text
        return compile(text, path, "exec", dont_inherit=True, optimize=_optimize)


class _LocalDinov3Finder(importlib.abc.MetaPathFinder):
    def __init__(self, package_root: Path):
        self.package_root = package_root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "dinov3" and not fullname.startswith("dinov3."):
            return None

        rel_parts = fullname.split(".")[1:]
        base = self.package_root.joinpath(*rel_parts)

        if fullname == "dinov3":
            file_path = self.package_root / "__init__.py"
            search_locations = [str(self.package_root)]
        elif (base / "__init__.py").exists():
            file_path = base / "__init__.py"
            search_locations = [str(base)]
        elif base.with_suffix(".py").exists():
            file_path = base.with_suffix(".py")
            search_locations = None
        else:
            return None

        loader = _FutureAnnotationsLoader(fullname, str(file_path))
        return importlib.util.spec_from_file_location(
            fullname,
            str(file_path),
            loader=loader,
            submodule_search_locations=search_locations,
        )


def _install_local_dinov3_importer() -> None:
    package_root = DINO3_ROOT / "dinov3"
    if not package_root.exists():
        return
    for finder in sys.meta_path:
        if isinstance(finder, _LocalDinov3Finder) and finder.package_root == package_root:
            return
    sys.meta_path.insert(0, _LocalDinov3Finder(package_root))
    if str(DINO3_ROOT) not in sys.path:
        sys.path.insert(0, str(DINO3_ROOT))


_install_local_dinov3_importer()

try:
    from dinov3.hub import backbones as dinov3_backbones  # noqa: E402
    _DINOV3_IMPORT_ERROR = None
except Exception as exc:
    dinov3_backbones = None
    _DINOV3_IMPORT_ERROR = exc


DINO_TOKEN_MODES = ("patch", "cls")
DINO_TOKEN_STRATEGIES = ("patch_only", "last4_avg", "last4_concat_project")


@dataclass
class EncoderSpec:
    kind: str
    image_size: int
    patch_size: int
    embed_dim: int
    num_heads: int
    num_patches: int
    temporal_stride: int = 1


class FrozenSpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec: EncoderSpec | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        if self.spec is None:
            raise RuntimeError("Encoder spec not initialized.")
        return self.spec.embed_dim

    @property
    def num_heads(self) -> int:
        if self.spec is None:
            raise RuntimeError("Encoder spec not initialized.")
        return self.spec.num_heads

    @property
    def num_patches(self) -> int:
        if self.spec is None:
            raise RuntimeError("Encoder spec not initialized.")
        return self.spec.num_patches

    @property
    def patch_size(self) -> int:
        if self.spec is None:
            raise RuntimeError("Encoder spec not initialized.")
        return self.spec.patch_size

    @property
    def temporal_stride(self) -> int:
        if self.spec is None:
            raise RuntimeError("Encoder spec not initialized.")
        return self.spec.temporal_stride

    def output_num_steps(self, input_steps: int) -> int:
        return input_steps

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()


class DinoSpatialEncoder(FrozenSpatialEncoder):
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 224,
        checkpoint_path: str | None = None,
        token_mode: str = "patch",
        token_strategy: str = "patch_only",
        last_layers: int = 4,
    ) -> None:
        super().__init__()
        if token_mode not in DINO_TOKEN_MODES:
            raise ValueError(f"Unsupported DINO token mode: {token_mode}. Choices: {DINO_TOKEN_MODES}")
        if token_strategy not in DINO_TOKEN_STRATEGIES:
            raise ValueError(f"Unsupported DINO token strategy: {token_strategy}. Choices: {DINO_TOKEN_STRATEGIES}")
        if not model_name.startswith("dinov3"):
            raise ValueError("world_model_tf now supports local DINOv3 backbones only.")
        if dinov3_backbones is None:
            raise ImportError(
                f"Could not import local DINOv3 package from {DINO3_ROOT}. "
                "Make sure the repository-level dinov3 directory is present. "
                f"Original import error: {type(_DINOV3_IMPORT_ERROR).__name__}: {_DINOV3_IMPORT_ERROR}"
            )
        if not hasattr(dinov3_backbones, model_name):
            raise ValueError(f"Unknown DINOv3 backbone: {model_name}")
        if last_layers <= 0:
            raise ValueError(f"last_layers must be positive, got {last_layers}")

        self.token_mode = token_mode
        self.token_strategy = token_strategy
        self.last_layers = int(last_layers)
        self.encoder = self._load_dinov3_encoder(model_name, checkpoint_path)

        patch_size = int(getattr(self.encoder, "patch_size", 16))
        embed_dim = int(getattr(self.encoder, "num_features", getattr(self.encoder, "embed_dim")))
        num_heads = int(getattr(self.encoder, "num_heads", 16))
        self.token_projector = (
            nn.Linear(self.last_layers * embed_dim, embed_dim)
            if token_strategy == "last4_concat_project"
            else nn.Identity()
        )

        if token_mode == "cls":
            spec_patch_size = int(image_size)
            num_patches = 1
        else:
            spec_patch_size = patch_size
            num_patches_side = int(image_size) // patch_size
            num_patches = num_patches_side * num_patches_side
        self.spec = EncoderSpec(
            kind="dino",
            image_size=int(image_size),
            patch_size=spec_patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_patches=num_patches,
            temporal_stride=1,
        )
        self._freeze_backbone()

    @staticmethod
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for key in ("state_dict", "teacher", "model", "backbone"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        if not isinstance(ckpt, dict):
            raise TypeError(f"Expected checkpoint dict, got {type(ckpt)}")

        state_dict = {}
        for key, value in ckpt.items():
            if not torch.is_tensor(value):
                continue
            cleaned_key = key
            for prefix in ("module.", "teacher.", "student.", "model.", "backbone."):
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[len(prefix) :]
            state_dict[cleaned_key] = value
        if not state_dict:
            raise ValueError("No tensor state_dict entries found in checkpoint.")
        return state_dict

    def _load_dinov3_encoder(self, model_name: str, checkpoint_path: str | None):
        ctor = getattr(dinov3_backbones, model_name)
        model = ctor(pretrained=False)
        if checkpoint_path is None:
            checkpoint_path = self._resolve_default_checkpoint(model_name)
        if checkpoint_path:
            ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
            state_dict = self._extract_state_dict(ckpt)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] loaded local DINOv3 checkpoint: {checkpoint_path}")
            print(f"[INFO] DINOv3 load_state_dict msg: {msg}")
        else:
            print("[WARN] no DINO checkpoint provided; using randomly initialized DINOv3 backbone.")
        return model

    @staticmethod
    def _resolve_default_checkpoint(model_name: str) -> str | None:
        candidates = sorted(DINO3_DEFAULT_CKPT_DIR.glob(f"{model_name}_pretrain_*.pth*"))
        if candidates:
            return str(candidates[0])
        expected = DINO3_DEFAULT_CKPT_DIR / f"{model_name}_pretrain_lvd1689m-8aa4cbdd.pth"
        print(
            "[WARN] no DINO checkpoint provided and no default checkpoint found. "
            f"Looked under {DINO3_DEFAULT_CKPT_DIR}; expected something like {expected}."
        )
        return None

    def _freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _forward_final_tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder.forward_features(x)
        if isinstance(out, dict):
            if self.token_mode == "cls":
                return out["x_norm_clstoken"].unsqueeze(1)
            return out["x_norm_patchtokens"]
        if isinstance(out, (tuple, list)):
            out = out[0]
        if self.token_mode == "cls" and out.ndim == 2:
            return out.unsqueeze(1)
        return out

    def _forward_intermediate_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not hasattr(self.encoder, "get_intermediate_layers"):
            raise RuntimeError(
                f"DINO model {type(self.encoder).__name__} does not expose get_intermediate_layers, "
                f"required by token_strategy={self.token_strategy}."
            )
        layers = self.encoder.get_intermediate_layers(
            x,
            n=self.last_layers,
            return_class_token=self.token_mode == "cls",
            norm=True,
        )
        if len(layers) != self.last_layers:
            raise RuntimeError(f"Expected {self.last_layers} DINO layers, got {len(layers)}.")
        if self.token_mode == "cls":
            return tuple(cls_token.unsqueeze(1) for _, cls_token in layers)
        return tuple(layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w)
        with torch.no_grad():
            if self.token_strategy == "patch_only":
                tokens = self._forward_final_tokens(x)
            else:
                tokens = self._forward_intermediate_tokens(x)

        if self.token_strategy == "last4_avg":
            tokens = torch.stack(tokens, dim=0).mean(dim=0)
        elif self.token_strategy == "last4_concat_project":
            tokens = self.token_projector(torch.cat(tokens, dim=-1))

        return tokens.view(b, t, tokens.size(1), tokens.size(2)).contiguous()


class VJEPASpatialEncoder(FrozenSpatialEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "This ManiFeel-adapted world_model_tf pass focuses on DINOv3. "
            "Use --encoder dino."
        )
