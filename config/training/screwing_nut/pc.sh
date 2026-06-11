python train.py \
  --data-root data/demo_data/screwing_nut \
  --task screwing_nut \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 160