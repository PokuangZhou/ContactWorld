#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
WORK_DIR=${WORK_DIR:-${REPO_ROOT}/world_model_tf}
THIRDPARTY_ROOT=${THIRDPARTY_ROOT:-${REPO_ROOT}/thirdparty}
MANIFEEL_DIR=${MANIFEEL_DIR:-${THIRDPARTY_ROOT}/manifeel/manifeel}
MANIFEEL_ISAACGYM_ROOT=${MANIFEEL_ISAACGYM_ROOT:-${THIRDPARTY_ROOT}/manifeel-isaacgymenvs}

CONDA_ENV=${CONDA_ENV:-/home/pokuang/miniforge3/envs/cw}
CONFIG_FILE=${CONFIG_FILE:-${REPO_ROOT}/config/tf/eval/dino/eval_planner_env_dino_front_rgb_smoke.yaml}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/demo_data/insertion_usb}
DINO_CHECKPOINT=${DINO_CHECKPOINT:-${REPO_ROOT}/data/pretrained_model/dino3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
DINOV3_ROOT=${DINOV3_ROOT:-${THIRDPARTY_ROOT}/dinov3}

DEFAULT_CKPT_DIR=${REPO_ROOT}/outputs/world_model_tf_test/front_rgb/checkpoints/usb/dinob_front_rgb_patch_vc
if [ -z "${CHECKPOINT:-}" ]; then
    for candidate in \
        "${DEFAULT_CKPT_DIR}/best-rollout-visual-l2-v1.ckpt" \
        "${DEFAULT_CKPT_DIR}/best-rollout-visual-l2.ckpt" \
        "${DEFAULT_CKPT_DIR}/last-v1.ckpt" \
        "${DEFAULT_CKPT_DIR}/last.ckpt"; do
        if [ -f "${candidate}" ]; then
            CHECKPOINT="${candidate}"
            break
        fi
    done
fi
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CKPT_DIR}/best-rollout-visual-l2.ckpt}

CACHE_DIR=${CACHE_DIR:-${REPO_ROOT}/.cache/world_model_tf_eval/test}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/world_model_tf_eval_test/front_rgb_smoke}
DEVICE=${DEVICE:-cuda}
TACTILE_POOL_MODE=${TACTILE_POOL_MODE:-attention}

CONFIG_NAME=${CONFIG_FILE##*/}
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}
RUN_NAME=${RUN_NAME:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}
LOG_PREFIX=${LOG_DIR}/eval_planner_tf_test_${RUN_NAME}

require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

require_path "${CONDA_ENV}/bin/python" "conda env python"
require_path "${WORK_DIR}/eval_planner_env.py" "world_model_tf eval_planner_env.py"
require_path "${CONFIG_FILE}" "eval config"
require_path "${DATA_ROOT}" "demo dataset"
require_path "${DINO_CHECKPOINT}" "DINO checkpoint"
require_path "${DINOV3_ROOT}/dinov3" "DINOv3 source package"
require_path "${MANIFEEL_DIR}/config/isaacgym_config_usb.yaml" "ManiFeel IsaacGym Hydra config"
require_path "${MANIFEEL_ISAACGYM_ROOT}/isaacgymenvs" "manifeel-isaacgymenvs package"
require_path "${CHECKPOINT}" "world model checkpoint"

mkdir -p \
    "${CACHE_DIR}/matplotlib" \
    "${CACHE_DIR}/xdg" \
    "${CACHE_DIR}/torch" \
    "${CACHE_DIR}/torch_extensions" \
    "${LOG_DIR}" \
    "${OUTPUT_DIR}"

echo "=========================================="
echo "ContactWorld TF planner eval smoke test"
echo "=========================================="
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
echo ""

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
    --num-envs "${NUM_ENVS:-2}" \
    --max-steps "${MAX_STEPS:-1}" \
    --candidates "${CANDIDATES:-4}" \
    --candidate-chunk-size "${CANDIDATE_CHUNK_SIZE:-4}" \
    --topk "${TOPK:-2}" \
    --iterations "${ITERATIONS:-1}" \
    --goal-offset-steps "${GOAL_OFFSET_STEPS:-1}" \
    "$@" 2>&1 | tee "${LOG_PREFIX}.log"

echo ""
echo "Planner eval smoke test finished at $(date)"
