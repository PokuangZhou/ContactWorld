python train.py \
  --data-root data/demo_data/exploration_sorting_dim \
  --task exploration_sorting_dim \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 650