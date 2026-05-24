#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/world_model_tf}

CONDA_ENV=${CONDA_ENV:-/home/pokuang/miniforge3/envs/cw}
CONFIG_FILE=${CONFIG_FILE:-${REPO_ROOT}/config/tf/train/dino/train_dino_front_rgb.yaml}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/demo_data/insertion_usb}
DINO_CHECKPOINT=${DINO_CHECKPOINT:-${REPO_ROOT}/data/pretrained_model/dino3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
DINOV3_ROOT=${DINOV3_ROOT:-${REPO_ROOT}/thirdparty/dinov3}

CACHE_DIR=${CACHE_DIR:-${REPO_ROOT}/.cache/world_model_tf/train}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}
SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/outputs/world_model_tf/usb/front_rgb}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-${REPO_ROOT}/outputs/wandb}
WANDB_MODE=${WANDB_MODE:-disabled}

CONFIG_NAME=${CONFIG_FILE##*/}
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}
CONFIG_TAG=${CONFIG_NAME#train_}
RUN_NAME=${RUN_NAME:-${CONFIG_TAG}_local_$(date +%Y%m%d_%H%M%S)}
LOG_PREFIX=${LOG_DIR}/train_tf_${RUN_NAME}

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
require_path "${DATA_ROOT}" "training dataset"
require_path "${DINO_CHECKPOINT}" "DINO checkpoint"
require_path "${DINOV3_ROOT}/dinov3" "DINOv3 source package"

mkdir -p \
    "${CACHE_DIR}/matplotlib" \
    "${CACHE_DIR}/xdg" \
    "${CACHE_DIR}/torch" \
    "${LOG_DIR}" \
    "${SAVE_DIR}" \
    "${WANDB_SAVE_DIR}"

echo "=========================================="
echo "ContactWorld TF local training"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Work dir: ${WORK_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Config: ${CONFIG_FILE}"
echo "Data root: ${DATA_ROOT}"
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
    "$@" 2>&1 | tee "${LOG_PREFIX}.log"

echo ""
echo "Local training finished at $(date)"
