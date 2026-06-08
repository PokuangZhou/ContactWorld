#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
THIRDPARTY_ROOT="${REPO_ROOT}/thirdparty"
PATCH_SCRIPT_DIR="${REPO_ROOT}/scripts/manifeel_codepatch"
PATCHED_INSTALL="${PATCH_SCRIPT_DIR}/install.sh"
REPLACE_SCRIPT="${PATCH_SCRIPT_DIR}/replace_code.sh"
MANIFEEL_ROOT="${THIRDPARTY_ROOT}/manifeel"
ISAACGYM_TAR="${THIRDPARTY_ROOT}/IsaacGym_Preview_TacSL_Package.tar.gz"
ISAACGYM_DIR="${THIRDPARTY_ROOT}/IsaacGym_Preview_TacSL_Package"
DINO3_DIR="${REPO_ROOT}/data/pretrained_model/dino3"
DINO3_CKPT="${DINO3_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

MANIFEEL_REPO_URL="${MANIFEEL_REPO_URL:-https://github.com/purdue-mars/manifeel.git}"
ISAACGYM_GDRIVE_URL="${ISAACGYM_GDRIVE_URL:-https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view?usp=sharing}"
DINO3_GDRIVE_URL="${DINO3_GDRIVE_URL:-https://drive.google.com/file/d/1m_WYeLRM50KT6M2MfTUtJro2px5e0Rmt/view?usp=sharing}"
SKIP_ISAACGYM_DOWNLOAD="${SKIP_ISAACGYM_DOWNLOAD:-false}"
SKIP_DINO3_DOWNLOAD="${SKIP_DINO3_DOWNLOAD:-false}"
SKIP_MANIFEEL_INSTALL="${SKIP_MANIFEEL_INSTALL:-false}"
SKIP_REPLACE_CODE="${SKIP_REPLACE_CODE:-false}"

echo "=========================================="
echo "ContactWorld Setup"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Thirdparty root: ${THIRDPARTY_ROOT}"
echo ""

require_file() {
    local path="$1"
    local label="$2"

    if [ ! -f "${path}" ]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        exit 1
    fi
}

ensure_gdown() {
    if command -v gdown >/dev/null 2>&1; then
        GDOWN_BIN="$(command -v gdown)"
        return
    fi

    echo "gdown not found. Installing gdown with pip..."
    if command -v python3 >/dev/null 2>&1; then
        python3 -m pip install --user gdown
    elif command -v python >/dev/null 2>&1; then
        python -m pip install --user gdown
    else
        echo "ERROR: python/python3 not found; cannot install gdown." >&2
        exit 1
    fi

    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v gdown >/dev/null 2>&1; then
        echo "ERROR: gdown was installed, but it is still not on PATH." >&2
        echo "Try: export PATH=\"${HOME}/.local/bin:\${PATH}\"" >&2
        exit 1
    fi

    GDOWN_BIN="$(command -v gdown)"
}

download_with_gdown() {
    local url="$1"
    local output_path="$2"

    ensure_gdown
    if "${GDOWN_BIN}" --help 2>/dev/null | grep -q -- "--fuzzy"; then
        "${GDOWN_BIN}" --fuzzy "${url}" -O "${output_path}"
    else
        "${GDOWN_BIN}" "${url}" -O "${output_path}"
    fi
}

require_file "${PATCHED_INSTALL}" "patched ManiFeel install script"
require_file "${REPLACE_SCRIPT}" "ContactWorld ManiFeel replace script"
mkdir -p "${THIRDPARTY_ROOT}"

echo "=========================================="
echo "1. Clone ManiFeel"
echo "=========================================="
if [ -d "${MANIFEEL_ROOT}/.git" ]; then
    echo "ManiFeel already exists: ${MANIFEEL_ROOT}"
elif [ -d "${MANIFEEL_ROOT}" ]; then
    echo "ERROR: ${MANIFEEL_ROOT} exists but is not a git repository." >&2
    exit 1
else
    git clone "${MANIFEEL_REPO_URL}" "${MANIFEEL_ROOT}"
fi
echo ""

echo "=========================================="
echo "2. Replace ManiFeel install.sh"
echo "=========================================="
cp "${PATCHED_INSTALL}" "${MANIFEEL_ROOT}/install.sh"
chmod +x "${MANIFEEL_ROOT}/install.sh"
echo "Copied ${PATCHED_INSTALL} -> ${MANIFEEL_ROOT}/install.sh"
echo ""

echo "=========================================="
echo "3. Download IsaacGym TacSL Package"
echo "=========================================="
if [ "${SKIP_ISAACGYM_DOWNLOAD}" = "true" ]; then
    echo "Skipping IsaacGym download because SKIP_ISAACGYM_DOWNLOAD=true"
elif [ -d "${ISAACGYM_DIR}" ]; then
    echo "IsaacGym already extracted: ${ISAACGYM_DIR}"
else
    if [ ! -f "${ISAACGYM_TAR}" ]; then
        download_with_gdown "${ISAACGYM_GDRIVE_URL}" "${ISAACGYM_TAR}"
    else
        echo "IsaacGym archive already exists: ${ISAACGYM_TAR}"
    fi

    tar -xzf "${ISAACGYM_TAR}" -C "${THIRDPARTY_ROOT}"
fi
echo ""

echo "=========================================="
echo "4. Download DINOv3 Checkpoint"
echo "=========================================="
if [ "${SKIP_DINO3_DOWNLOAD}" = "true" ]; then
    echo "Skipping DINOv3 download because SKIP_DINO3_DOWNLOAD=true"
elif [ -f "${DINO3_CKPT}" ]; then
    echo "DINOv3 checkpoint already exists: ${DINO3_CKPT}"
else
    mkdir -p "${DINO3_DIR}"
    download_with_gdown "${DINO3_GDRIVE_URL}" "${DINO3_CKPT}"
fi
echo ""

echo "=========================================="
echo "5. Install ManiFeel"
echo "=========================================="
if [ "${SKIP_MANIFEEL_INSTALL}" = "true" ]; then
    echo "Skipping ManiFeel install because SKIP_MANIFEEL_INSTALL=true"
else
    bash "${MANIFEEL_ROOT}/install.sh"
fi
echo ""

echo "=========================================="
echo "6. Apply ContactWorld Code Patch"
echo "=========================================="
if [ "${SKIP_REPLACE_CODE}" = "true" ]; then
    echo "Skipping replace_code.sh because SKIP_REPLACE_CODE=true"
else
    bash "${REPLACE_SCRIPT}"
fi
echo ""

echo "=========================================="
echo "ContactWorld setup complete"
echo "=========================================="
