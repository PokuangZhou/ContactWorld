# world_model_tf

Transformer world-model path adapted for the current ManiFeel IsaacGym zarr
datasets.

## What this path supports

- one visual key per run: `front`, `wrist`, or `pointcloud`
- DINOv3 frozen image encoder for RGB keys
- local action-conditioned transformer predictor copied into this directory
- optional tactile key:
  - `tactile_force_field_right`
  - `tactile_depth_right`
  - `tactile_rgb_right` / taxim-style RGB keys
- low-dimensional prediction for:
  - `ee_pos`
  - `ee_quat`
  - `plug_pos`
  - `plug_quat`
  - `socket_pos_gt`
- VC or SIGReg latent regularization with temporal similarity and IDM terms

## Train

```bash
python manifeel/manifeel/world_model_tf/train.py \
  --data-root /path/to/dataset.zarr \
  --encoder dino \
  --dino-name dinov3_vitl16 \
  --vision-key front \
  --vision-type image \
  --num-steps 6 \
  --image-size 224
```

With tactile:

```bash
python manifeel/manifeel/world_model_tf/train.py \
  --data-root /path/to/dataset.zarr \
  --encoder dino \
  --vision-key wrist \
  --use-tactile \
  --tactile-key tactile_force_field_right
```

Config-driven launch is still supported:

```bash
python manifeel/manifeel/world_model_tf/train.py \
  --config manifeel/manifeel/world_model_tf/configs/train_dino.yaml
```

## DINO token modes and strategies

- `dino_token_mode=patch`: use DINO patch tokens, e.g. `[B,T,196,D]` for 224px with patch16
- `dino_token_mode=cls`: use the DINO CLS token as one visual token, `[B,T,1,D]`
- `patch_only`: final-layer tokens for the selected mode
- `last4_avg`: average the last `dino_last_layers` token outputs
- `last4_concat_project`: concatenate the last `dino_last_layers` outputs and train a
  small projector back to the DINO feature dimension

The repository-level `dinov3` directory is used directly. No files inside
`dinov3` are modified.

If `--dino-checkpoint` is omitted, `dinov3_vitl16` checkpoints are searched
under:

```text
manifeel/manifeel/world_model_tf/wm_tf_data/pretrianed_model/dino3/
```

For example:

```text
manifeel/manifeel/world_model_tf/wm_tf_data/pretrianed_model/dino3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```
