<div align="center">
<h2> ContactWorld: <br>
What Matters in Vision-Tactile World Models for Contact-Rich Manipulation</h2>

<a href="https://github.com/PokuangZhou/ContactWorld"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="https://github.com/PokuangZhou/ContactWorld"><img src="https://img.shields.io/badge/Paper-Link-red" alt="Paper Link"></a>
<a href="https://huggingface.co/datasets/Pokuang/ContactWorld/tree/main"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset"></a>

</div>

<p style="width: 90%; margin: 0 auto; text-align: justify;">
<strong>ContactWorld</strong> systematically studies vision-tactile world models across 12 contact-rich manipulation tasks and shows that spatially structured, temporally continuous representations are crucial for stable long-horizon planning. Experiments further demonstrate that point-cloud observations and compatible tactile force-field representations substantially improve planning success, highlighting the importance of representation structure, multimodal compatibility, and tactile sensing under long-horizon contact uncertainty.
</p>

<p align="center">
<img src="media/teaser.png" alt="ContactWorld teaser" width="90%" />
</p>

## Highlights

- Vision-tactile world-model training on ManiFeel-style zarr datasets.
- DINOv3 visual encoders with tactile and low-dimensional state prediction.
- CEM-style planner evaluation inside IsaacGym contact-rich environments.
- Local smoke-test scripts for checking installation, training, checkpoint
  loading, and planner rollout.

## Repository Layout

```text
ContactWorld/
  config/                 # Training and evaluation YAML configs
  data/
    demo_data/            # Demo zarr datasets
    pretrained_model/     # Local pretrained checkpoints, e.g. DINOv3
    replace_part/         # Patch files applied to thirdparty ManiFeel repos
  media/                  # Figures used in this README
  scripts/
    manifeel_codepatch/   # Install, data download, and thirdparty patch scripts
    train_tf_test.sh      # Minimal training/sanity test
    eval_planner_tf_test.sh
  thirdparty/             # ManiFeel, IsaacGym, DINOv3, diffusion_policy, etc.
  world_model_tf/         # ContactWorld Transformer world-model code
```

## Easy Setup

details can refer to [patch helper](scripts/manifeel_codepatch/readme.md).

Please read the patch helper's readme, it will git clone the thirdparty repo and download data, if you agree to precess, directly run
```bash
bash install_cw.sh
```

## demo data and checkpoint download link
💾 [huggingface link](https://huggingface.co/datasets/Pokuang/ContactWorld)

## Required Local Assets

The smoke-test scripts expect these paths by default:

```text
data/demo_data/insertion_usb
data/pretrained_model/dino3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
thirdparty/dinov3
thirdparty/manifeel
thirdparty/manifeel-isaacgymenvs
thirdparty/IsaacGym_Preview_TacSL_Package
```

## Quick Tests

Run a training smoke test. By default this uses the full demo dataset but only
runs Lightning's sanity-check path (`EPOCHS=0`):

```bash
cd /home/pokuang/project/ContactWorld
scripts/train_tf_test.sh
```

Run one actual minimal training epoch:

```bash
EPOCHS=1 BATCH_SIZE=32 scripts/train_tf_test.sh
```

Run a minimal planner evaluation in IsaacGym:

```bash
scripts/eval_planner_tf_test.sh
```

The planner smoke test defaults to a very small setting:

```text
NUM_ENVS=2
MAX_STEPS=1
CANDIDATES=4
TOPK=2
ITERATIONS=1
```

## Training

Main DINO front-RGB training config:

```text
config/tf/train/dino/train_dino_front_rgb.yaml
```

The test script forwards any extra arguments to `world_model_tf/train.py`, so
you can override config values from the command line:

```bash
scripts/train_tf_test.sh \
  --epochs 2 \
  --batch-size 32 \
  --train-num-workers 8 \
  --val-num-workers 4
```

## Planner Evaluation

Planner evaluation config:

```text
config/tf/eval/dino/eval_planner_env_dino_front_rgb_smoke.yaml
```

The test script forwards extra arguments to `world_model_tf/eval_planner_env.py`:

```bash
scripts/eval_planner_tf_test.sh \
  --num-envs 4 \
  --max-steps 3 \
  --candidates 8 \
  --candidate-chunk-size 8 \
  --topk 2 \
  --iterations 1
```

## Addtion Information
ContactWorld/eval_planner_dy.py is for dynamic-horizon, and current default is fixed
ContactWorld/eval_planner_sorting.py is for sorting task (for addiiton case id needed)

after download the assets/industreal.tar, need to replace 👉
ContactWorld/thirdparty/manifeel-isaacgymenvs/assets/industreal


## Citation

```bibtex
@article{zhang2026contactworld,
  title={ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation},
  author={},
  journal={},
  year={2026}
}
```

Related:

```bibtex
@article{luu2025manifeel,
  title={ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning},
  author={Luu, Quan Khanh and Zhou, Pokuang and Xu, Zhengtong and Zhang, Zhiyuan and Qiu, Qiang and She, Yu},
  journal={arXiv preprint arXiv:2505.18472},
  year={2025}
}
```
