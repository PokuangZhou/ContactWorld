#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${SCRIPT_DIR}}

PYTHON=${PYTHON:-python}
PYTHON_PREFIX=${PYTHON_PREFIX:-$("${PYTHON}" -c 'import sys; print(sys.prefix)')}
MANIFEEL_DIR=${MANIFEEL_DIR:-${REPO_ROOT}/thirdparty/manifeel/manifeel}
MANIFEEL_ISAACGYM_ROOT=${MANIFEEL_ISAACGYM_ROOT:-${REPO_ROOT}/thirdparty/manifeel-isaacgymenvs}
MANIFEEL_CONFIG_DIR=${MANIFEEL_CONFIG_DIR:-${MANIFEEL_DIR}/config}
MANIFEEL_ASSETS_DIR=${MANIFEEL_ASSETS_DIR:-${REPO_ROOT}/thirdparty/manifeel/assets}
CONFIG_FILE=${CONFIG_FILE:-${REPO_ROOT}/config/plan.yaml}
ARCHIVE_DIR=${ARCHIVE_DIR:-${REPO_ROOT}/releases}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/demo_data/insertion_usb}
CKPT_ROOT=${CKPT_ROOT:-${REPO_ROOT}/logs/ckpts}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/logs/planning_smoke/insertion_usb/front_ff}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}

DATASET_ARCHIVE=${DATASET_ARCHIVE:-${ARCHIVE_DIR}/insertion_usb_dataset.tar.gz}
DATASET_URL=${DATASET_URL:-https://huggingface.co/datasets/Pokuang/ContactWorld/resolve/main/releases/insertion_usb_dataset.tar.gz}
CKPT_ARCHIVE=${CKPT_ARCHIVE:-${ARCHIVE_DIR}/insertion_usb_ckpt.tar.gz}
CKPT_URL=${CKPT_URL:-https://huggingface.co/datasets/Pokuang/ContactWorld/resolve/main/releases/insertion_usb_ckpt.tar.gz}

NUM_ENVS=${NUM_ENVS:-2}
NUM_RECORD=${NUM_RECORD:-1}
MAX_STEPS=${MAX_STEPS:-1}
CANDIDATES=${CANDIDATES:-4}
TOPK=${TOPK:-2}
ITERATIONS=${ITERATIONS:-1}
GOAL_OFFSET_STEPS=${GOAL_OFFSET_STEPS:-12}
DEVICE=${DEVICE:-cuda}
RUN_NAME=${RUN_NAME:-insertion_usb_front_ff_plan_smoke}
CREATED_HYDRA_CONFIG_LINK=false
CREATED_HYDRA_TASK_LINK=false
CREATED_HYDRA_ASSETS_LINK=false

download_file() {
    local url="$1"
    local output="$2"

    if [ -s "${output}" ]; then
        echo "Found archive: ${output}"
        return
    fi

    mkdir -p "$(dirname "${output}")"
    echo "Downloading ${url}"
    if command -v wget >/dev/null 2>&1; then
        wget -O "${output}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${output}"
    else
        echo "ERROR: install wget or curl to download ${url}" >&2
        exit 1
    fi
}

extract_dataset() {
    if [ -d "${DATA_ROOT}" ]; then
        echo "Found dataset: ${DATA_ROOT}"
        return
    fi

    local data_parent
    data_parent=$(dirname "${DATA_ROOT}")
    mkdir -p "${data_parent}"
    echo "Extracting dataset to ${data_parent}"
    tar -xzf "${DATASET_ARCHIVE}" -C "${data_parent}"

    if [ ! -d "${DATA_ROOT}" ]; then
        echo "ERROR: expected dataset directory not found after extraction: ${DATA_ROOT}" >&2
        exit 1
    fi
}

extract_checkpoint() {
    local expected="${CKPT_ROOT}/insertion_usb/front/vc/tactile_force_field_right/concat/100000.ckpt"
    if [ -f "${expected}" ]; then
        echo "Found checkpoint: ${expected}"
        return
    fi

    mkdir -p "${CKPT_ROOT}"
    echo "Extracting checkpoint to ${CKPT_ROOT}"
    tar -xzf "${CKPT_ARCHIVE}" -C "${CKPT_ROOT}"
}

find_checkpoint() {
    if [ -n "${CHECKPOINT:-}" ]; then
        echo "${CHECKPOINT}"
        return
    fi

    local expected="${CKPT_ROOT}/insertion_usb/front/vc/tactile_force_field_right/concat/100000.ckpt"
    if [ -f "${expected}" ]; then
        echo "${expected}"
        return
    fi

    local found
    found=$(find "${CKPT_ROOT}/insertion_usb" -type f -name '*.ckpt' | sort | tail -n 1 || true)
    if [ -n "${found}" ]; then
        echo "${found}"
        return
    fi

    echo "ERROR: no checkpoint found under ${CKPT_ROOT}/insertion_usb" >&2
    exit 1
}

require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

cleanup_hydra_links() {
    if [ "${CREATED_HYDRA_CONFIG_LINK}" = "true" ]; then
        rm -f "${REPO_ROOT}/config/isaacgym_config_usb.yaml"
    fi
    if [ "${CREATED_HYDRA_TASK_LINK}" = "true" ]; then
        rm -f "${REPO_ROOT}/config/task"
    fi
    if [ "${CREATED_HYDRA_ASSETS_LINK}" = "true" ]; then
        rm -f "${REPO_ROOT}/config/assets"
    fi
}

prepare_hydra_links() {
    local cfg_src="${MANIFEEL_CONFIG_DIR}/isaacgym_config_usb.yaml"
    local cfg_dst="${REPO_ROOT}/config/isaacgym_config_usb.yaml"
    local task_src="${MANIFEEL_CONFIG_DIR}/task"
    local task_dst="${REPO_ROOT}/config/task"
    local assets_src="${MANIFEEL_ASSETS_DIR}"
    local assets_dst="${REPO_ROOT}/config/assets"

    require_path "${cfg_src}" "ManiFeel IsaacGym config"
    require_path "${task_src}/TacSLTaskUSB.yaml" "ManiFeel USB task config"
    require_path "${assets_src}/tacsl/yaml/tacsl_asset_info_power.yaml" "ManiFeel TacSL asset info"

    if [ ! -e "${cfg_dst}" ]; then
        ln -s "${cfg_src}" "${cfg_dst}"
        CREATED_HYDRA_CONFIG_LINK=true
    fi

    if [ ! -e "${task_dst}" ]; then
        ln -s "${task_src}" "${task_dst}"
        CREATED_HYDRA_TASK_LINK=true
    fi

    if [ ! -e "${assets_dst}" ]; then
        ln -s "${assets_src}" "${assets_dst}"
        CREATED_HYDRA_ASSETS_LINK=true
    fi
}

download_file "${DATASET_URL}" "${DATASET_ARCHIVE}"
extract_dataset
download_file "${CKPT_URL}" "${CKPT_ARCHIVE}"
extract_checkpoint
CHECKPOINT=$(find_checkpoint)

require_path "${CONFIG_FILE}" "planner config"
require_path "${REPO_ROOT}/eval_planner.py" "eval_planner.py"
require_path "${CHECKPOINT}" "checkpoint"
require_path "${MANIFEEL_DIR}/dataset/zarr_dataset.py" "ManiFeel dataset package"
require_path "${MANIFEEL_ISAACGYM_ROOT}/isaacgymenvs" "ManiFeel IsaacGymEnvs package"
require_path "${PYTHON_PREFIX}/lib" "Python environment lib directory"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
trap cleanup_hydra_links EXIT
prepare_hydra_links

export PYTHONPATH="${REPO_ROOT}:${MANIFEEL_DIR}:${MANIFEEL_ISAACGYM_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export MANIFEEL_DIR

echo "=========================================="
echo "ContactWorld insertion USB planner smoke test"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON}"
echo "Python prefix: ${PYTHON_PREFIX}"
echo "ManiFeel dir: ${MANIFEEL_DIR}"
echo "ManiFeel IsaacGymEnvs dir: ${MANIFEEL_ISAACGYM_ROOT}"
echo "ManiFeel config dir: ${MANIFEEL_CONFIG_DIR}"
echo "ManiFeel assets dir: ${MANIFEEL_ASSETS_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Data root: ${DATA_ROOT}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Device: ${DEVICE}"
echo "Num envs: ${NUM_ENVS}"
echo "Max steps: ${MAX_STEPS}"
echo "Candidates/topk/iterations: ${CANDIDATES}/${TOPK}/${ITERATIONS}"
echo "Run name: ${RUN_NAME}"
echo ""

cd "${REPO_ROOT}"

"${PYTHON}" eval_planner.py \
    --config "${CONFIG_FILE}" \
    --data-root "${DATA_ROOT}" \
    --ckpt-path "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --num-envs "${NUM_ENVS}" \
    --num-record "${NUM_RECORD}" \
    --max-steps "${MAX_STEPS}" \
    --candidates "${CANDIDATES}" \
    --topk "${TOPK}" \
    --iterations "${ITERATIONS}" \
    --goal-offset-steps "${GOAL_OFFSET_STEPS}" \
    "$@" 2>&1 | tee "${LOG_DIR}/smoke_test_plan_${RUN_NAME}.log"

echo ""
echo "Planner smoke test finished at $(date)"
