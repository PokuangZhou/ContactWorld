#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/world_model_tf}

CONDA_ENV=${CONDA_ENV:-/home/pokuang/miniforge3/envs/cw}
CONFIG_FILE=${CONFIG_FILE:-${REPO_ROOT}/config/tf/train/dino/train_dino_front_rgb.yaml}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/demo_data/insertion_usb}
DINO_CHECKPOINT=${DINO_CHECKPOINT:-${REPO_ROOT}/data/pretrained_model/dino3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
USE_TINY_DATA=${USE_TINY_DATA:-false}

if [ -z "${DINOV3_ROOT:-}" ]; then
    for candidate in \
        "${REPO_ROOT}/thirdparty/dinov3" \
        "${REPO_ROOT}/dinov3"; do
        if [ -d "${candidate}/dinov3" ]; then
            DINOV3_ROOT="${candidate}"
            break
        fi
    done
fi
DINOV3_ROOT=${DINOV3_ROOT:-${REPO_ROOT}/thirdparty/dinov3}

CACHE_DIR=${CACHE_DIR:-${REPO_ROOT}/.cache/world_model_tf/test}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}
SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/outputs/world_model_tf_test/front_rgb}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-${REPO_ROOT}/outputs/wandb_test}
WANDB_MODE=${WANDB_MODE:-disabled}
TINY_DATA_ROOT=${TINY_DATA_ROOT:-${CACHE_DIR}/tiny_insertion_usb.zarr}

CONFIG_NAME=${CONFIG_FILE##*/}
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}
RUN_NAME=${RUN_NAME:-${CONFIG_NAME}_smoke_$(date +%Y%m%d_%H%M%S)}
LOG_PREFIX=${LOG_DIR}/train_tf_test_${RUN_NAME}

require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

require_path "${CONDA_ENV}/bin/python" "conda env python"
require_path "${WORK_DIR}/train.py" "world_model_tf train.py"
require_path "${CONFIG_FILE}" "train config"
require_path "${DATA_ROOT}" "demo dataset"
require_path "${DINO_CHECKPOINT}" "DINO checkpoint"
require_path "${DINOV3_ROOT}/dinov3" "DINOv3 source package"

mkdir -p \
    "${CACHE_DIR}/matplotlib" \
    "${CACHE_DIR}/xdg" \
    "${CACHE_DIR}/torch" \
    "${LOG_DIR}" \
    "${SAVE_DIR}" \
    "${WANDB_SAVE_DIR}"

if [ "${USE_TINY_DATA}" = "true" ]; then
    echo "Preparing tiny training dataset: ${TINY_DATA_ROOT}"
    export DATA_ROOT
    export TINY_DATA_ROOT
    "${CONDA_ENV}/bin/python" - <<'PY'
import os
import shutil
from pathlib import Path

import numpy as np
import zarr

src_root = Path(os.environ["DATA_ROOT"])
dst_root = Path(os.environ["TINY_DATA_ROOT"])
if dst_root.exists():
    shutil.rmtree(dst_root)

src = zarr.open_group(str(src_root), mode="r")
dst = zarr.open_group(str(dst_root), mode="w")
src_data = src["data"]
dst_data = dst.create_group("data")
dst_meta = dst.create_group("meta")

episode_ends = np.asarray(src["meta"]["episode_ends"], dtype=np.int64)
lengths = np.diff(np.concatenate([[0], episode_ends]))
offsets = np.concatenate([[0], episode_ends[:-1]])
chosen = [idx for idx, length in enumerate(lengths) if length >= 8][:2]
if len(chosen) < 2:
    raise RuntimeError("Need at least two source episodes with length >= 8 for tiny training data.")

segments = [(int(offsets[idx]), int(offsets[idx]) + 8) for idx in chosen]
tiny_episode_ends = np.cumsum([end - start for start, end in segments]).astype(np.int64)
dst_meta.create_dataset("episode_ends", data=tiny_episode_ends, shape=tiny_episode_ends.shape, dtype=tiny_episode_ends.dtype)

keys = [
    "front",
    "tactile_rgb_right",
    "action",
    "ee_pos",
    "ee_quat",
    "plug_pos",
    "plug_quat",
    "socket_pos_gt",
]
for key in keys:
    parts = [np.asarray(src_data[key][start:end]) for start, end in segments]
    arr = np.concatenate(parts, axis=0)
    chunks = tuple(min(dim, 8) if axis == 0 else dim for axis, dim in enumerate(arr.shape))
    dst_data.create_dataset(key, data=arr, shape=arr.shape, chunks=chunks, dtype=arr.dtype)

print(f"created {dst_root} with {int(tiny_episode_ends[-1])} rows")
PY
    DATA_ROOT="${TINY_DATA_ROOT}"
fi

echo "=========================================="
echo "ContactWorld TF smoke training"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Work dir: ${WORK_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Config: ${CONFIG_FILE}"
echo "Data root: ${DATA_ROOT}"
echo "Use tiny data: ${USE_TINY_DATA}"
echo "DINOv3 root: ${DINOV3_ROOT}"
echo "DINO checkpoint: ${DINO_CHECKPOINT}"
echo "Save dir: ${SAVE_DIR}"
echo "W&B mode: ${WANDB_MODE}"
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_PREFIX}.log"
echo ""

export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${WORK_DIR}:${DINOV3_ROOT}:${PYTHONPATH:-}"
export DINOV3_ROOT
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export TORCH_HOME="${CACHE_DIR}/torch"
export WANDB_MODE

cd "${WORK_DIR}"

"${CONDA_ENV}/bin/python" train.py \
    --config "${CONFIG_FILE}" \
    --data-root "${DATA_ROOT}" \
    --dino-checkpoint "${DINO_CHECKPOINT}" \
    --save-dir "${SAVE_DIR}" \
    --wandb-save-dir "${WANDB_SAVE_DIR}" \
    --wandb-name "${RUN_NAME}" \
    --epochs "${EPOCHS:-0}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --train-num-workers "${TRAIN_NUM_WORKERS:-0}" \
    --val-num-workers "${VAL_NUM_WORKERS:-0}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE:-0}" \
    --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS:-99}" \
    "$@" 2>&1 | tee "${LOG_PREFIX}.log"

echo ""
echo "Smoke training finished at $(date)"
