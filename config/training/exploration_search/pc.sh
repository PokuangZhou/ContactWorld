python train.py \
  --data-root data/demo_data/exploration_search \
  --task exploration_search \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 1000