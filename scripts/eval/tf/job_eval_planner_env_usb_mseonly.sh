#!/bin/bash
#SBATCH --job-name=eval_wm-usb-mse
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --account=shey
#SBATCH --partition=a30
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=15:00:00
#SBATCH --chdir=/home/pokuang/project/ContactWorld

set -euo pipefail

CONTACTWORLD_ROOT=${CONTACTWORLD_ROOT:-/home/pokuang/project/ContactWorld}
THIRDPARTY_ROOT=${CONTACTWORLD_ROOT}/thirdparty
MANIFEEL_DIR=${THIRDPARTY_ROOT}/manifeel/manifeel
MANIFEEL_ISAACGYM_ROOT=${THIRDPARTY_ROOT}/manifeel-isaacgymenvs
WORK_DIR=${CONTACTWORLD_ROOT}/world_model_tf
CONTAINER_FILE=${CONTAINER_FILE:-${CONTACTWORLD_ROOT}/thirdparty/manifeel/manifeel.sif}
CONDA_ENV=${CONDA_ENV:-/home/pokuang/miniforge3/envs/cw}
DINOV3_ROOT=${DINOV3_ROOT:-${THIRDPARTY_ROOT}/dinov3}
EVAL_VARIANT=${EVAL_VARIANT:-front_no}

case "${EVAL_VARIANT}" in
    front|front_no|no)
        DEFAULT_CONFIG=${CONTACTWORLD_ROOT}/config/tf/eval/dino/eval_planner_env_dino_front_no_mseonly.yaml
        ;;
    front_rgb|rgb)
        DEFAULT_CONFIG=${CONTACTWORLD_ROOT}/config/tf/eval/dino/eval_planner_env_dino_front_rgb_mseonly.yaml
        ;;
    front_depth|depth)
        DEFAULT_CONFIG=${CONTACTWORLD_ROOT}/config/tf/eval/dino/eval_planner_env_dino_front_depth_mseonly.yaml
        ;;
    front_ff|ff)
        DEFAULT_CONFIG=${CONTACTWORLD_ROOT}/config/tf/eval/dino/eval_planner_env_dino_front_ff_mseonly.yaml
        ;;
    *)
        echo "Unknown EVAL_VARIANT=${EVAL_VARIANT}; use front_no, front_rgb, front_depth, front_ff, or set CONFIG_FILE directly." >&2
        exit 2
        ;;
esac

CONFIG_FILE=${CONFIG_FILE:-${DEFAULT_CONFIG}}
CACHE_DIR=${CONTACTWORLD_ROOT}/.cache/world_model_tf_eval/${SLURM_JOB_ID:-local}
LOG_DIR=${CONTACTWORLD_ROOT}/logs
CONFIG_NAME=${CONFIG_FILE##*/}
CONFIG_NAME=${CONFIG_NAME%.yaml}
CONFIG_NAME=${CONFIG_NAME%.yml}
CONFIG_TAG=${CONFIG_NAME#eval_planner_env_}
LOG_PREFIX=${LOG_DIR}/${SLURM_JOB_NAME}_${CONFIG_TAG}_${SLURM_JOB_ID:-local}
CONFIG_SNAPSHOT=${CACHE_DIR}/${CONFIG_NAME}_${SLURM_JOB_ID:-local}.yaml

mkdir -p "${CACHE_DIR}/matplotlib" "${CACHE_DIR}/xdg" "${CACHE_DIR}/torch" "${LOG_DIR}"
cp "${CONFIG_FILE}" "${CONFIG_SNAPSHOT}"
exec > "${LOG_PREFIX}.out" 2> "${LOG_PREFIX}.err"

echo "Eval mseonly job started on $(hostname) at $(date)"
echo "Work dir: ${WORK_DIR}"
echo "Container: ${CONTAINER_FILE}"
echo "Eval variant: ${EVAL_VARIANT}"
echo "Config: ${CONFIG_FILE}"
echo "Config snapshot: ${CONFIG_SNAPSHOT}"
echo "Log prefix: ${LOG_PREFIX}"
echo "DINOv3 root: ${DINOV3_ROOT}"
echo
echo "===== YAML config snapshot: ${CONFIG_SNAPSHOT} ====="
sed -n '1,240p' "${CONFIG_SNAPSHOT}"
echo "===== End YAML config ====="
echo

apptainer exec --nv --cleanenv \
    --bind ${CONTACTWORLD_ROOT}:${CONTACTWORLD_ROOT} \
    --env LD_PRELOAD= \
    --env PYTHONPATH=${WORK_DIR}:${MANIFEEL_DIR}:${MANIFEEL_ISAACGYM_ROOT}:${DINOV3_ROOT} \
    --env MANIFEEL_DIR=${MANIFEEL_DIR} \
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
        python eval_planner_mseonly.py --config ${CONFIG_SNAPSHOT}
    "

echo "Eval mseonly job finished at $(date)"
