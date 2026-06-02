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

copy_patch_dir() {
    local src_rel="$1"
    local dst_rel="$2"
    local src="${PATCH_ROOT}/${src_rel}"
    local dst="${THIRDPARTY_ROOT}/${dst_rel}"

    require_dir "${src}" "patch source directory"

    echo "Patch directory:"
    echo "  ${src_rel}/"
    echo "  -> thirdparty/${dst_rel}/"

    if [ "${DRY_RUN}" = "true" ]; then
        local file_count
        file_count="$(find "${src}" -type f | wc -l)"
        if [ -d "${dst}" ]; then
            echo "  status: would replace/add ${file_count} files in existing directory"
        else
            echo "  status: would create directory and copy ${file_count} files"
        fi
        echo ""
        return
    fi

    mkdir -p "${dst}"

    if [ -d "${dst}" ] && [ "${BACKUP}" = "true" ]; then
        local backup_path="${BACKUP_ROOT}/${dst_rel}"
        mkdir -p "$(dirname "${backup_path}")"
        cp -a "${dst}" "${backup_path}"
        echo "  backup: ${backup_path}"
    fi

    cp -a "${src}/." "${dst}/"
    echo "  done"
    echo ""
}

copy_patch_if_missing() {
    local src_rel="$1"
    local dst_rel="$2"
    local src="${PATCH_ROOT}/${src_rel}"
    local dst="${THIRDPARTY_ROOT}/${dst_rel}"

    require_file "${src}" "patch source"

    echo "Patch if missing:"
    echo "  ${src_rel}"
    echo "  -> thirdparty/${dst_rel}"

    if [ -f "${dst}" ]; then
        echo "  status: already exists, skip"
        echo ""
        return
    fi

    if [ "${DRY_RUN}" = "true" ]; then
        echo "  status: would create missing file"
        echo ""
        return
    fi

    mkdir -p "$(dirname "${dst}")"
    cp -p "${src}" "${dst}"
    echo "  done"
    echo ""
}

require_dir "${PATCH_ROOT}/manifeel" "ManiFeel patch directory"
require_dir "${PATCH_ROOT}/manifeel/assets" "ManiFeel assets patch directory"
require_dir "${PATCH_ROOT}/manifeel/config" "ManiFeel config patch directory"
require_dir "${PATCH_ROOT}/manifeel_issacgym" "ManiFeel IsaacGym patch directory"
require_dir "${PATCH_ROOT}/manifeel_issacgym/tacsl" "ManiFeel IsaacGym TacSL patch directory"
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
copy_patch_dir "manifeel/assets" \
    "manifeel/assets"
copy_patch_dir "manifeel/config" \
    "manifeel/manifeel/config"

copy_patch_dir "manifeel_issacgym/tacsl" \
    "manifeel-isaacgymenvs/isaacgymenvs/tasks/tacsl"
copy_patch "manifeel_issacgym/tacsl_sensors.py" \
    "manifeel-isaacgymenvs/isaacgymenvs/tacsl_sensors/tacsl_sensors.py"
copy_patch_if_missing "manifeel_issacgym/my_vt_lookup_table.pkl" \
    "manifeel-isaacgymenvs/isaacgymenvs/tacsl_sensors/my_vt_lookup_table_impedance.pkl"

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
