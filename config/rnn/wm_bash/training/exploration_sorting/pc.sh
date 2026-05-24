python train.py \
  --data-root data/exploration_sorting \
  --task exploration_sorting \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 415
