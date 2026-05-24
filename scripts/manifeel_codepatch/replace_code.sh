#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH_ROOT="${REPO_ROOT}/data/replace_part"
THIRDPARTY_ROOT="${REPO_ROOT}/thirdparty"
MANIFEEL_ROOT="${THIRDPARTY_ROOT}/manifeel"
MANIFEEL_ISAACGYM_ROOT="${THIRDPARTY_ROOT}/manifeel-isaacgymenvs"

DRY_RUN="${DRY_RUN:-false}"
BACKUP="${BACKUP:-true}"
BACKUP_ROOT="${BACKUP_ROOT:-${REPO_ROOT}/.cache/manifeel_codepatch_backup/$(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "ContactWorld ManiFeel Code Patch"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Patch root: ${PATCH_ROOT}"
echo "Thirdparty root: ${THIRDPARTY_ROOT}"
echo "Dry run: ${DRY_RUN}"
echo "Backup existing files: ${BACKUP}"
if [ "${BACKUP}" = "true" ]; then
    echo "Backup root: ${BACKUP_ROOT}"
fi
echo ""

require_dir() {
    local path="$1"
    local label="$2"
    if [ ! -d "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    if [ ! -f "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

copy_patch() {
    local src_rel="$1"
    local dst_rel="$2"
    local src="${PATCH_ROOT}/${src_rel}"
    local dst="${THIRDPARTY_ROOT}/${dst_rel}"

    require_file "${src}" "patch source"

    echo "Patch:"
    echo "  ${src_rel}"
    echo "  -> thirdparty/${dst_rel}"

    if [ "${DRY_RUN}" = "true" ]; then
        if [ -f "${dst}" ]; then
            echo "  status: would replace existing file"
        else
            echo "  status: would create new file"
        fi
        echo ""
        return
    fi

    mkdir -p "$(dirname "${dst}")"

    if [ -f "${dst}" ] && [ "${BACKUP}" = "true" ]; then
        local backup_path="${BACKUP_ROOT}/${dst_rel}"
        mkdir -p "$(dirname "${backup_path}")"
        cp -p "${dst}" "${backup_path}"
        echo "  backup: ${backup_path}"
    fi

    cp -p "${src}" "${dst}"
    echo "  done"
    echo ""
}

require_dir "${PATCH_ROOT}/manifeel" "ManiFeel patch directory"
require_dir "${PATCH_ROOT}/manifeel_issacgym" "ManiFeel IsaacGym patch directory"
require_dir "${MANIFEEL_ROOT}" "thirdparty ManiFeel repo"
require_dir "${MANIFEEL_ISAACGYM_ROOT}" "thirdparty manifeel-isaacgymenvs repo"

echo "Applying patches..."
echo ""

copy_patch "manifeel/TacSLTaskBoltNut.yaml" \
    "manifeel/manifeel/config/task/TacSLTaskBoltNut.yaml"
copy_patch "manifeel/TacSLTaskBulb.yaml" \
    "manifeel/manifeel/config/task/TacSLTaskBulb.yaml"
copy_patch "manifeel/TacSLTaskUSB.yaml" \
    "manifeel/manifeel/config/task/TacSLTaskUSB.yaml"
copy_patch "manifeel/env_wrapper.py" \
    "manifeel/manifeel/envs/env_wrapper.py"
copy_patch "manifeel/isaacgym_config_gui.yaml" \
    "manifeel/manifeel/config/isaacgym_config_gui.yaml"
copy_patch "manifeel/tacsl_asset_info_power.yaml" \
    "manifeel/assets/tacsl/yaml/tacsl_asset_info_power.yaml"
copy_patch "manifeel/vistac_isaacgym_multiple_env_wrapper.py" \
    "manifeel/manifeel/envs/vistac_isaacgym_multiple_env_wrapper.py"
copy_patch "manifeel/zarr_dataset.py" \
    "manifeel/manifeel/dataset/zarr_dataset.py"

copy_patch "manifeel_issacgym/tacsl_env_insertion.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_env_insertion.py"
copy_patch "manifeel_issacgym/tacsl_sensors.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tacsl_sensors/tacsl_sensors.py"
copy_patch "manifeel_issacgym/tacsl_task_bolt_nut.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_bolt_nut.py"
copy_patch "manifeel_issacgym/tacsl_task_bulb.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_bulb.py"
copy_patch "manifeel_issacgym/tacsl_task_image_augmentation.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_image_augmentation.py"
copy_patch "manifeel_issacgym/tacsl_task_USB.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl/tacsl_task_USB.py"

echo "=========================================="
echo "Patch complete"
echo "=========================================="
if [ "${DRY_RUN}" != "true" ] && [ "${BACKUP}" = "true" ]; then
    echo "Original files backed up under:"
    echo "  ${BACKUP_ROOT}"
fi
echo ""
echo "Usage:"
echo "  DRY_RUN=true ${0}"
echo "  BACKUP=false ${0}"
