#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
THIRDPARTY_ROOT="${REPO_ROOT}/thirdparty"
REPLACE_PART_ROOT="${REPO_ROOT}/data/replace_part"
PATCH_SCRIPT_DIR="${REPO_ROOT}/scripts/manifeel_codepatch"
PATCHED_INSTALL="${PATCH_SCRIPT_DIR}/install.sh"
REPLACE_SCRIPT="${PATCH_SCRIPT_DIR}/replace_code.sh"
MANIFEEL_ROOT="${THIRDPARTY_ROOT}/manifeel"
MANIFEEL_ISAACGYM_ROOT="${THIRDPARTY_ROOT}/manifeel-isaacgymenvs"
ISAACGYM_TAR="${THIRDPARTY_ROOT}/IsaacGym_Preview_TacSL_Package.tar.gz"
ISAACGYM_DIR="${THIRDPARTY_ROOT}/IsaacGym_Preview_TacSL_Package"
INDUSTREAL_TAR="${REPLACE_PART_ROOT}/industreal.tar"
INDUSTREAL_DST="${MANIFEEL_ISAACGYM_ROOT}/assets/industreal"

MANIFEEL_REPO_URL="${MANIFEEL_REPO_URL:-https://github.com/purdue-mars/manifeel.git}"
ISAACGYM_GDRIVE_URL="${ISAACGYM_GDRIVE_URL:-https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view?usp=sharing}"
INDUSTREAL_URL="${INDUSTREAL_URL:-https://huggingface.co/datasets/Pokuang/ContactWorld/resolve/main/assets/industreal.tar}"
SKIP_ISAACGYM_DOWNLOAD="${SKIP_ISAACGYM_DOWNLOAD:-false}"
SKIP_INDUSTREAL_ASSETS="${SKIP_INDUSTREAL_ASSETS:-false}"
SKIP_MANIFEEL_INSTALL="${SKIP_MANIFEEL_INSTALL:-false}"
SKIP_REPLACE_CODE="${SKIP_REPLACE_CODE:-false}"
CONTACTWORLD_NONINTERACTIVE="${CONTACTWORLD_NONINTERACTIVE:-true}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-contactworld}"

echo "=========================================="
echo "ContactWorld Setup"
echo "=========================================="
echo "Repo root: ${REPO_ROOT}"
echo "Thirdparty root: ${THIRDPARTY_ROOT}"
echo "Non-interactive ManiFeel install: ${CONTACTWORLD_NONINTERACTIVE}"
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

download_file() {
    local url="$1"
    local output_path="$2"

    if command -v wget >/dev/null 2>&1; then
        wget -O "${output_path}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${output_path}"
    else
        echo "ERROR: neither wget nor curl was found; cannot download ${url}" >&2
        exit 1
    fi
}

find_industreal_source() {
    if [ -d "${REPLACE_PART_ROOT}/assets/industreal" ]; then
        INDUSTREAL_SRC="${REPLACE_PART_ROOT}/assets/industreal"
    elif [ -d "${REPLACE_PART_ROOT}/industreal" ]; then
        INDUSTREAL_SRC="${REPLACE_PART_ROOT}/industreal"
    elif [ -d "${REPLACE_PART_ROOT}/manifeel/assets/industreal" ]; then
        INDUSTREAL_SRC="${REPLACE_PART_ROOT}/manifeel/assets/industreal"
    else
        echo "ERROR: extracted IndustReal assets directory not found under ${REPLACE_PART_ROOT}" >&2
        exit 1
    fi
}

replace_industreal_assets() {
    find_industreal_source

    if [ ! -d "${MANIFEEL_ISAACGYM_ROOT}" ]; then
        echo "ERROR: manifeel-isaacgymenvs repo not found: ${MANIFEEL_ISAACGYM_ROOT}" >&2
        echo "Run ManiFeel install first, or set SKIP_INDUSTREAL_ASSETS=true." >&2
        exit 1
    fi

    mkdir -p "$(dirname "${INDUSTREAL_DST}")"

    rm -rf "${INDUSTREAL_DST}"
    mkdir -p "${INDUSTREAL_DST}"
    cp -a "${INDUSTREAL_SRC}/." "${INDUSTREAL_DST}/"
    echo "Copied ${INDUSTREAL_SRC} -> ${INDUSTREAL_DST}"
}

detect_conda_base() {
    local candidate=""

    if command -v conda >/dev/null 2>&1; then
        candidate="$(conda info --base 2>/dev/null || true)"
    fi

    if [ -z "${candidate}" ] && command -v mamba >/dev/null 2>&1; then
        candidate="$(dirname "$(dirname "$(command -v mamba)")")"
    fi

    echo "${candidate}"
}

get_contactworld_env_path() {
    if [ -n "${CONDA_ENV_PATH:-}" ]; then
        echo "${CONDA_ENV_PATH}"
        return
    fi

    if [ -n "${MINIFORGE_HOME:-}" ]; then
        echo "${MINIFORGE_HOME}/envs/${CONDA_ENV_NAME}"
        return
    fi

    local conda_base
    conda_base="$(detect_conda_base)"
    if [ -n "${conda_base}" ]; then
        echo "${conda_base}/envs/${CONDA_ENV_NAME}"
        return
    fi

    echo "${HOME}/miniforge3/envs/${CONDA_ENV_NAME}"
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
echo "4. Install ManiFeel"
echo "=========================================="
if [ "${SKIP_MANIFEEL_INSTALL}" = "true" ]; then
    echo "Skipping ManiFeel install because SKIP_MANIFEEL_INSTALL=true"
else
    CONTACTWORLD_NONINTERACTIVE="${CONTACTWORLD_NONINTERACTIVE}" \
        bash "${MANIFEEL_ROOT}/install.sh"
fi
echo ""

echo "=========================================="
echo "5. Download and Replace IndustReal Assets"
echo "=========================================="
if [ "${SKIP_INDUSTREAL_ASSETS}" = "true" ]; then
    echo "Skipping IndustReal assets because SKIP_INDUSTREAL_ASSETS=true"
else
    mkdir -p "${REPLACE_PART_ROOT}"
    if [ ! -f "${INDUSTREAL_TAR}" ]; then
        download_file "${INDUSTREAL_URL}" "${INDUSTREAL_TAR}"
    else
        echo "IndustReal archive already exists: ${INDUSTREAL_TAR}"
    fi

    tar -xf "${INDUSTREAL_TAR}" -C "${REPLACE_PART_ROOT}"
    replace_industreal_assets
fi
echo ""

echo "=========================================="
echo "6. Apply ContactWorld Code Patch"
echo "=========================================="
if [ "${SKIP_REPLACE_CODE}" = "true" ]; then
    echo "Skipping replace_code.sh because SKIP_REPLACE_CODE=true"
else
    BACKUP=false bash "${REPLACE_SCRIPT}"
fi
echo ""

echo "=========================================="
echo "ContactWorld setup complete"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  conda activate $(get_contactworld_env_path)"
echo "  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}"
