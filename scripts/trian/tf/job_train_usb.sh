#!/bin/bash
#SBATCH --job-name=train_wm-usb
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=shey
#SBATCH --partition=a30
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=60:00:00
#SBATCH --chdir=/scratch/gilbreth/zhou1458/manifeel_isaacgym/manifeel/manifeel

set -euo pipefail

SCRATCH_ROOT=/scratch/gilbreth/zhou1458
MANIFEEL_DIR=${SCRATCH_ROOT}/manifeel_isaacgym/manifeel/manifeel
WORK_DIR=${MANIFEEL_DIR}/world_model_tf
CONTAINER_FILE=${SCRATCH_ROOT}/manifeel_isaacgym/manifeel/manifeel.sif
CONDA_ENV=${SCRATCH_ROOT}/miniforge3/envs/manifeel
DINOV3_ROOT=${SCRATCH_ROOT}/Projects/dinov3
CONFIG_FILE=${CONFIG_FILE:-${WORK_DIR}/configs/train_dino_front_rgb_last4conc.yaml}
CACHE_DIR=${SCRATCH_ROOT}/.cache/world_model_tf/${SLURM_JOB_ID:-local}
WANDB_HOME=${WANDB_HOME:-${SCRATCH_ROOT}/.wandb_home}
WANDB_CONFIG_DIR=${WANDB_HOME}/.config/wandb
LOG_DIR=${MANIFEEL_DIR}/logs
CONFIG_NAME=${CONFIG_FILE##*/}
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}
CONFIG_TAG=${CONFIG_NAME#train_}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${CONFIG_TAG}_${SLURM_JOB_ID:-local}}
LOG_PREFIX=${LOG_DIR}/${SLURM_JOB_NAME}_${CONFIG_TAG}_${SLURM_JOB_ID:-local}

mkdir -p "${CACHE_DIR}/matplotlib" "${CACHE_DIR}/xdg" "${CACHE_DIR}/torch" "${WANDB_CONFIG_DIR}" "${LOG_DIR}"
exec > "${LOG_PREFIX}.out" 2> "${LOG_PREFIX}.err"

echo "Job started on $(hostname) at $(date)"
echo "Work dir: ${WORK_DIR}"
echo "Container: ${CONTAINER_FILE}"
echo "Config: ${CONFIG_FILE}"
echo "Log prefix: ${LOG_PREFIX}"
echo "DINOv3 root: ${DINOV3_ROOT}"
echo "W&B home: ${WANDB_HOME}"
echo "W&B config dir: ${WANDB_CONFIG_DIR}"
echo "W&B run name: ${WANDB_RUN_NAME}"
if [[ -f "${WANDB_HOME}/.netrc" ]]; then
    echo "W&B login: found ${WANDB_HOME}/.netrc"
else
    echo "W&B login: missing ${WANDB_HOME}/.netrc; run wandb login with this HOME before submitting if wandb is enabled"
fi
echo "Config YAML name: ${CONFIG_NAME}"
echo "===== BEGIN CONFIG YAML: ${CONFIG_FILE} ====="
cat "${CONFIG_FILE}"
echo "===== END CONFIG YAML: ${CONFIG_FILE} ====="
echo "===== DISK USAGE BEFORE TRAINING ====="
df -h "${SCRATCH_ROOT}" || true
du -sh "${WORK_DIR}/outputs/world_model_tf" 2>/dev/null || true
du -sh "${WORK_DIR}/outputs/wandb" 2>/dev/null || true

apptainer exec --nv --cleanenv \
    --bind ${SCRATCH_ROOT}:${SCRATCH_ROOT} \
    --home ${WANDB_HOME} \
    --env LD_PRELOAD= \
    --env WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} \
    --env PYTHONPATH=${WORK_DIR}:${MANIFEEL_DIR}:${DINOV3_ROOT} \
    --env DINOV3_ROOT=${DINOV3_ROOT} \
    --env MPLCONFIGDIR=${CACHE_DIR}/matplotlib \
    --env XDG_CACHE_HOME=${CACHE_DIR}/xdg \
    --env TORCH_HOME=${CACHE_DIR}/torch \
    ${CONTAINER_FILE} \
    bash -lc "
        set -eo pipefail
        export PATH=${CONDA_ENV}/bin:\${PATH}
        export CONDA_PREFIX=${CONDA_ENV}
        export LD_LIBRARY_PATH=${CONDA_ENV}/lib:\${LD_LIBRARY_PATH:-}
        cd ${WORK_DIR}
        python train.py --config ${CONFIG_FILE} --wandb-name ${WANDB_RUN_NAME}
    "

echo "===== DISK USAGE AFTER TRAINING ====="
df -h "${SCRATCH_ROOT}" || true
du -sh "${WORK_DIR}/outputs/world_model_tf" 2>/dev/null || true
du -sh "${WORK_DIR}/outputs/wandb" 2>/dev/null || true
echo "Job finished at $(date)"
