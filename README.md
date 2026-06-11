<div align="center">
<img src="media/cw_logo.gif" alt="ContactWorld logo" width="180" />

<h2> ContactWorld: <br>
What Matters in Vision-Tactile World Models for Contact-Rich Manipulation</h2>

<a href="https://contact-world.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="https://drive.google.com/file/d/15ToTrPhCByiSU59WxpiK-_ueZB6LVd8c/view"><img src="https://img.shields.io/badge/Paper-Link-red" alt="Paper Link"></a>
<a href="https://huggingface.co/datasets/Pokuang/ContactWorld/tree/main"><img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow" alt="Dataset"></a>

</div>

<p align="center">
<a href="https://zhangzhiyuanzhang.github.io/personal_website/">Zhiyuan Zhang</a>*,
<a href="https://pokuangzhou.github.io/website/">Pokuang Zhou</a>*,
<a href="https://aaronzkd.github.io/KaidiZhang.web/">Kaidi Zhang</a>,
<a href="https://scholar.google.com/citations?user=A880yg0AAAAJ&hl=en&oi=ao">Adeesh Desai</a>,
<a href="https://scholar.google.com/citations?user=Lloa6s4AAAAJ&hl=en&oi=ao">Temitope Amosa</a>,
<br>
<a href="https://scholar.google.com/citations?user=Rmadq64AAAAJ&hl=en">Davood Soleymanzadeh</a>,
<a href="https://scholar.google.com/citations?user=X52xke0AAAAJ&hl=en">Jiuzhou Lei</a>,
<a href="https://zh.engr.tamu.edu/">Minghui Zheng</a>,
<a href="https://www.purduemars.com/">Yu She<sup>†</sup></a>
<br>
* Equal Contribution &nbsp; † Corresponding Author
</p>

<p style="width: 80%; margin: 0 auto; text-align: justify;">
<strong>ContactWorld</strong> is a benchmark for studying visual-tactile world models in contact-rich manipulation.
Across 12 tasks and 6 sensing modalities, ContactWorld reveals that representation structure, cross-modal compatibility, and long-horizon robustness are critical for reliable planning.
</p>

<p align="center">
<img src="media/cw_teaser.gif" alt="ContactWorld teaser" width="80%" />
</p>

## Highlights
- A JEPA-structured vision-tactile world model with a CEM-style planner.
- We reveal three key findings for contact-rich world modeling:
  - Representation structure matters.
  - Tactile benefits depend on cross-modal compatibility.
  - Tactile sensing becomes increasingly important at longer planning horizons.
- Easy-to-install and lightweight Isaac Gym environments covering 12 contact-rich manipulation tasks and 6 sensing modalities.
- Open-source datasets, pretrained checkpoints, and evaluation tools for reproducible research.


## Easy Setup

Please read the patch helper's README first. The script will clone the required third-party repositories and download the necessary data after you confirm to proceed. Then, you can directly run the installation command.
Details can refer to the note in the end.

This installation requires [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). Please make sure mamba is installed before running the script.


```bash
bash install_cw.sh
```

## Demo Data and Checkpoint Download Link
💾 [huggingface link](https://huggingface.co/datasets/Pokuang/ContactWorld/tree/main)

how to use those data please refer to [data readme](extract_dataset_ckpts_addto_README.md) 

## Repository Layout

```text
ContactWorld/
  config/                 # Training and evaluation YAML configs
  data/
    demo_data/            # Demo zarr datasets
    pretrained_model/     # Local pretrained checkpoints, e.g. DINOv3
    replace_part/         # Patch files applied to thirdparty ManiFeel repos
  media/                  # GIF used in README
  scripts/
    manifeel_codepatch/   # Install, data download, and thirdparty patch scripts
    train_tf_test.sh      # Minimal training/sanity test
    eval_planner_tf_test.sh # planner
```

## Quick Tests

Run a training smoke test. This downloads the USB insertion demo data if needed
and trains the front-camera + tactile-force-field model for 3 epochs:

```bash
bash smoke_test_train.sh
```

You can override the smoke-test defaults from the shell:

```bash
EPOCHS=1 BATCH_SIZE=32 NUM_WORKERS=0 bash smoke_test_train.sh
```

Run a minimal planner evaluation in IsaacGym. This downloads the USB insertion
demo data and pretrained checkpoint if needed, then uses
`config/plan.yaml` with a small planner setting:

```bash
bash smoke_test_plan.sh
```

## Training

Edit `config/train.yaml` to set your task, data path, and modality. Before
running from the repository root, expose the local ManiFeel dataset package:

```bash
export PYTHONPATH=$PWD/thirdparty/manifeel/manifeel:$PYTHONPATH
python train.py --config config/train.yaml
```


## Planner Evaluation

Edit `config/plan.yaml` to set your checkpoint path, data path, and environment.
`eval_planner.py` automatically adds the local ManiFeel/IsaacGym source paths
and prepares the ManiFeel Hydra configs/assets at runtime, so from the repository
root you can run:

```bash
python eval_planner.py --config config/plan.yaml
```

For the sorting task:

```bash
python eval_planner_sorting.py --config config/plan.yaml
```

For a first check with downloaded USB data and a pretrained checkpoint, use
`bash smoke_test_plan.sh`.

## Addtion Information
ContactWorld/eval_planner_dy.py is for dynamic-horizon, and current default is fixed
ContactWorld/eval_planner_sorting.py is for sorting task (for addiiton case id needed)

Install details please refer to [patch readme](scripts/manifeel_codepatch/readme.md).

## Citation
If you find our work useful, please consider citing our paper. <img src="media/cat_thumbup.gif" alt="Cat thumbs up" width="42" />

```bibtex
@article{zhang2026contactworld,
  title={ContactWorld: What Matters in Vision-Tactile World Models for Contact-Rich Manipulation},
  author={Zhiyuan Zhang, Pokuang Zhou, Kaidi Zhang, Adeesh Mahesh Desai, Temitope Ibrahim Amosa, Davood Soleymanzadeh, Jiuzhou Lei, Minghui Zheng, and Yu She},
  journal={},
  year={2026}
}
```

Related work ([ManiFeel](https://zhengtongxu.github.io/manifeel-website/)):

```bibtex
@article{luu2025manifeel,
  title={ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning},
  author={Luu, Quan Khanh and Zhou, Pokuang and Xu, Zhengtong and Zhang, Zhiyuan and Qiu, Qiang and She, Yu},
  journal={arXiv preprint arXiv:2505.18472},
  year={2025}
}
```
