#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

CONTACTWORLD_NONINTERACTIVE="${CONTACTWORLD_NONINTERACTIVE:-true}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-contactworld}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
MANIFEEL_ISAACGYM_REPO_URL="${MANIFEEL_ISAACGYM_REPO_URL:-https://github.com/purdue-mars/manifeel-isaacgymenvs.git}"

echo "=========================================="
echo "ManiFeel Installation Script"
echo "=========================================="
echo "Script dir: ${SCRIPT_DIR}"
echo "Thirdparty dir: ${PARENT_DIR}"
echo ""

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

detect_conda_base() {
    local candidate=""

    if command -v conda >/dev/null 2>&1; then
        candidate="$(conda info --base 2>/dev/null || true)"
    fi

    if [ -z "${candidate}" ] && [ -n "${CONDA_EXE:-}" ]; then
        candidate="$(dirname "$(dirname "${CONDA_EXE}")")"
    fi

    echo "${candidate}"
}

ensure_conda() {
    if command -v mamba >/dev/null 2>&1; then
        CONDA_CREATE_CMD="mamba"
        CONDA_EXE="$(command -v mamba)"
        echo "Found mamba: ${CONDA_EXE}"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_CREATE_CMD="conda"
        CONDA_EXE="$(command -v conda)"
        echo "Found conda: ${CONDA_EXE}"
    else
        local miniforge_home="${MINIFORGE_HOME:-${HOME}/miniforge3}"
        local installer="${PARENT_DIR}/Miniforge3-Linux-x86_64.sh"

        echo "conda/mamba not found. Installing Miniforge3 to ${miniforge_home}..."
        if [ ! -x "${miniforge_home}/bin/conda" ]; then
            download_file \
                "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
                "${installer}"
            bash "${installer}" -b -p "${miniforge_home}"
        else
            echo "Miniforge already exists: ${miniforge_home}"
        fi

        CONDA_CREATE_CMD="${miniforge_home}/bin/conda"
        CONDA_EXE="${miniforge_home}/bin/conda"
    fi

    CONDA_BASE="$(detect_conda_base)"
    if [ -z "${CONDA_BASE}" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        echo "ERROR: could not find conda initialization script under ${CONDA_BASE}" >&2
        exit 1
    fi
}

get_conda_env_path() {
    local default_home="${MINIFORGE_HOME:-${CONDA_BASE}}"
    local selected_home

    echo ""
    echo "=========================================="
    echo "Conda Environment Path"
    echo "=========================================="

    if [ -n "${CONDA_ENV_PATH:-}" ]; then
        echo "Using CONDA_ENV_PATH: ${CONDA_ENV_PATH}"
        return
    fi

    echo "Detected conda home: ${default_home}"
    if [ "${CONTACTWORLD_NONINTERACTIVE}" = "true" ] || [ "${CI:-false}" = "true" ]; then
        selected_home="${default_home}"
        echo "Using default conda home in non-interactive mode"
    else
        read -r -p "Enter conda home directory (press Enter to use default): " selected_home
        selected_home="${selected_home:-${default_home}}"
    fi

    CONDA_ENV_PATH="${selected_home}/envs/${CONDA_ENV_NAME}"
    echo "Environment path: ${CONDA_ENV_PATH}"
}

create_or_reuse_env() {
    echo ""
    echo "=========================================="
    echo "Setting up ManiFeel environment"
    echo "=========================================="

    get_conda_env_path

    if [ -d "${CONDA_ENV_PATH}" ]; then
        echo "Conda environment already exists: ${CONDA_ENV_PATH}"
    else
        echo "Creating Python ${PYTHON_VERSION} environment at ${CONDA_ENV_PATH}..."
        "${CONDA_CREATE_CMD}" create --prefix "${CONDA_ENV_PATH}" "python=${PYTHON_VERSION}" -y
    fi

    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}"

    python -m pip install --upgrade pip
}

clone_or_reuse_repo() {
    local repo_url="$1"
    local repo_dir="$2"
    local repo_name
    repo_name="$(basename "${repo_dir}")"

    if [ -d "${repo_dir}/.git" ]; then
        echo "${repo_name} already exists: ${repo_dir}"
    elif [ -d "${repo_dir}" ]; then
        echo "ERROR: ${repo_dir} exists but is not a git repository." >&2
        echo "Remove it or move it aside, then rerun this script." >&2
        exit 1
    else
        echo "Cloning ${repo_name}..."
        git clone "${repo_url}" "${repo_dir}"
    fi
}

pip_install_editable() {
    local package_dir="$1"
    local label="$2"

    echo "Installing ${label} from ${package_dir}"
    python -m pip install -e "${package_dir}"
}

ensure_conda
create_or_reuse_env

echo ""
echo "=========================================="
echo "Installing IsaacGym TacSL"
echo "=========================================="
ISAACGYM_PYTHON_DIR="${PARENT_DIR}/IsaacGym_Preview_TacSL_Package/isaacgym/python"
if [ "${CI:-false}" = "true" ]; then
    echo "Skipping IsaacGym install in CI mode"
elif [ -d "${ISAACGYM_PYTHON_DIR}" ]; then
    pip_install_editable "${ISAACGYM_PYTHON_DIR}" "IsaacGym TacSL"
else
    echo "ERROR: IsaacGym TacSL package not found at ${ISAACGYM_PYTHON_DIR}" >&2
    echo "Run the top-level install_cw.sh so it can download/extract IsaacGym first." >&2
    exit 1
fi

echo ""
echo "=========================================="
echo "Cloning and Installing Repositories"
echo "=========================================="
if [ "${CI:-false}" = "true" ]; then
    echo "Skipping manifeel-isaacgymenvs clone/install in CI mode"
else
    clone_or_reuse_repo "${MANIFEEL_ISAACGYM_REPO_URL}" "${PARENT_DIR}/manifeel-isaacgymenvs"
    pip_install_editable "${PARENT_DIR}/manifeel-isaacgymenvs" "manifeel-isaacgymenvs"
fi

echo ""
echo "=========================================="
echo "Installing ManiFeel"
echo "=========================================="
pip_install_editable "${SCRIPT_DIR}" "ManiFeel"

echo "Installing additional dependencies..."
python -m pip install \
    wandb==0.12.21 dill==0.3.9 tqdm==4.67.1 av==12.3.0 numpy==1.23.3 \
    opencv-python==4.10.0.84 zarr==2.16.1 einops==0.4.1 huggingface-hub==0.25.0 \
    diffusers==0.11.1 pandas==2.0.3 numba==0.56.4 rtree==1.3.0 \
    lightning matplotlib

echo ""
echo "=========================================="
echo "ManiFeel installation complete"
echo "=========================================="
echo "To activate the environment:"
echo "  conda activate ${CONDA_ENV_PATH}"
echo "  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}"
echo ""
