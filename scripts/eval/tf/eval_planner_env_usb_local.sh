#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/world_model_tf}
THIRDPARTY_ROOT=${THIRDPARTY_ROOT:-${REPO_ROOT}/thirdparty}
MANIFEEL_DIR=${MANIFEEL_DIR:-${THIRDPARTY_ROOT}/manifeel/manifeel}
MANIFEEL_ISAACGYM_ROOT=${MANIFEEL_ISAACGYM_ROOT:-${THIRDPARTY_ROOT}/manifeel-isaacgymenvs}

CONDA_ENV=${CONDA_ENV:-/home/pokuang/miniforge3/envs/cw}
DINOV3_ROOT=${DINOV3_ROOT:-${THIRDPARTY_ROOT}/dinov3}
CONFIG_FILE=${CONFIG_FILE:-${REPO_ROOT}/config/tf/eval/dino/eval_planner_env_dino_front_rgb.yaml}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/demo_data/insertion_usb}
DINO_CHECKPOINT=${DINO_CHECKPOINT:-${REPO_ROOT}/data/pretrained_model/dino3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
CHECKPOINT=${CHECKPOINT:-${REPO_ROOT}/outputs/world_model_tf/usb/front_rgb/checkpoints/usb/dinob_front_rgb_patch_vc/best-rollout-visual-l2.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/world_model_tf/planning/usb/front_rgb_local}

CACHE_DIR=${CACHE_DIR:-${REPO_ROOT}/.cache/world_model_tf_eval/local}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}
DEVICE=${DEVICE:-cuda}
TACTILE_POOL_MODE=${TACTILE_POOL_MODE:-attention}

CONFIG_NAME=${CONFIG_FILE##*/}
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}
CONFIG_TAG=${CONFIG_NAME#eval_planner_env_}
RUN_TAG=${RUN_TAG:-${CONFIG_TAG}_local_$(date +%Y%m%d_%H%M%S)}
LOG_PREFIX=${LOG_DIR}/eval_planner_env_local_${RUN_TAG}

mkdir -p "${CACHE_DIR}/matplotlib" "${CACHE_DIR}/xdg" "${CACHE_DIR}/torch" "${CACHE_DIR}/torch_extensions" "${LOG_DIR}" "${OUTPUT_DIR}"

echo "Local planner eval started at $(date)"
echo "Repo root: ${REPO_ROOT}"
echo "Work dir: ${WORK_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "DINOv3 root: ${DINOV3_ROOT}"
echo "DINO checkpoint: ${DINO_CHECKPOINT}"
echo "ManiFeel dir: ${MANIFEEL_DIR}"
echo "ManiFeel IsaacGymEnvs dir: ${MANIFEEL_ISAACGYM_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Tactile pool mode: ${TACTILE_POOL_MODE}"
echo "Log file: ${LOG_PREFIX}.log"

export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${WORK_DIR}:${MANIFEEL_DIR}:${MANIFEEL_ISAACGYM_ROOT}:${DINOV3_ROOT}:${PYTHONPATH:-}"
export MANIFEEL_DIR
export DINOV3_ROOT
export MPLCONFIGDIR="${CACHE_DIR}/matplotlib"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export TORCH_HOME="${CACHE_DIR}/torch"
export TORCH_EXTENSIONS_DIR="${CACHE_DIR}/torch_extensions"

cd "${WORK_DIR}"

"${CONDA_ENV}/bin/python" eval_planner_env.py \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --dino-checkpoint "${DINO_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --tactile-pool-mode "${TACTILE_POOL_MODE}" \
    "$@" 2>&1 | tee "${LOG_PREFIX}.log"

echo "Local planner eval finished at $(date)"
