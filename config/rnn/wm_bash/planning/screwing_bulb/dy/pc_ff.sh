#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the manifeel conda environment before running this script."
  exit 1
fi

MANIFEEL_DIR="/home/pokuang/project/manifeel_isaacgym/manifeel/manifeel"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MANIFEEL_DIR}:${PYTHONPATH:-}"

cd "${MANIFEEL_DIR}"

python eval_planner_dy.py \
  --data-root /home/pokuang/project/manifeel_isaacgym/manifeel/data/bulb_0423_100 \
  --ckpt-path logs/ckpts/screwing_bulb/pointcloud/vc/tactile_force_field_right/concat/100000.ckpt \
  --isaacgym-cfg-name isaacgym_config_bulb \
  --output-dir logs/planning/insertion_usb/100000/step_24/front_depth \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --use-tactile \
  --tactile-key tactile_force_field_right \
  --tactile-in-channels 3 \
  --tactile-height 10 \
  --tactile-width 14 \
  --tactile-dim 64 \
  --fusion-type concat \
  --reg-loss-type vc \
  --reg-on-vision-only \
  --history-size 1 \
  --horizon 6 \
  --candidates 100 \
  --topk 8 \
  --iterations 4 \
  --num-envs 100 \
  --num-record 1 \
  --goal-offset-steps 48 \
  --max-steps 60 \
  --stop-on-success
