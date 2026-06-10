#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${SCRIPT_DIR}}

PYTHON=${PYTHON:-python}
MANIFEEL_DIR=${MANIFEEL_DIR:-${REPO_ROOT}/thirdparty/manifeel/manifeel}
CONFIG_FILE=${CONFIG_FILE:-${REPO_ROOT}/config/train.yaml}
ARCHIVE_DIR=${ARCHIVE_DIR:-${REPO_ROOT}/releases}
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/demo_data/insertion_usb}
SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/logs/ckpts_smoke}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-${REPO_ROOT}/logs/wandb_smoke}

DATASET_ARCHIVE=${DATASET_ARCHIVE:-${ARCHIVE_DIR}/insertion_usb_dataset.tar.gz}
DATASET_URL=${DATASET_URL:-https://huggingface.co/datasets/Pokuang/ContactWorld/resolve/main/releases/insertion_usb_dataset.tar.gz}

EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-2}
CKPT_EVERY_N_STEPS=${CKPT_EVERY_N_STEPS:-1}
RUN_NAME=${RUN_NAME:-insertion_usb_front_ff_smoke}

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

require_path() {
    local path="$1"
    local label="$2"
    if [ ! -e "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

download_file "${DATASET_URL}" "${DATASET_ARCHIVE}"
extract_dataset

require_path "${CONFIG_FILE}" "training config"
require_path "${REPO_ROOT}/train.py" "train.py"
require_path "${MANIFEEL_DIR}/dataset/zarr_dataset.py" "ManiFeel dataset package"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${WANDB_SAVE_DIR}"

export PYTHONPATH="${REPO_ROOT}:${MANIFEEL_DIR}:${PYTHONPATH:-}"

echo "=========================================="
echo "ContactWorld insertion USB smoke training"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "ManiFeel dir: ${MANIFEEL_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Data root: ${DATA_ROOT}"
echo "Save dir: ${SAVE_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Run name: ${RUN_NAME}"
echo ""

cd "${REPO_ROOT}"

"${PYTHON}" train.py \
    --config "${CONFIG_FILE}" \
    --data-root "${DATA_ROOT}" \
    --save-dir "${SAVE_DIR}" \
    --wandb-save-dir "${WANDB_SAVE_DIR}" \
    --wandb-name "${RUN_NAME}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --ckpt-every-n-steps "${CKPT_EVERY_N_STEPS}" \
    "$@" 2>&1 | tee "${LOG_DIR}/smoke_test_train_${RUN_NAME}.log"

echo ""
echo "Smoke training finished at $(date)"
echo "Checkpoints are under:"
echo "${SAVE_DIR}/insertion_usb/front/vc/tactile_force_field_right/concat"
