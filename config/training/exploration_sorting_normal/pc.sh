python train.py \
  --data-root data/demo_data/exploration_sorting_normal \
  --task exploration_sorting_normal \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 650