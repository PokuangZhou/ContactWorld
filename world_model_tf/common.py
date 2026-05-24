from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
JEPA_WMS_ROOT = ROOT_DIR / "third_party" / "jepa-wms"


def ensure_jepa_wms_on_path() -> None:
    path = str(JEPA_WMS_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def clean_state_dict_keys(state_dict: dict[str, object]) -> dict[str, object]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_key = new_key.replace("backbone.", "")
        cleaned[new_key] = value
    return cleaned
