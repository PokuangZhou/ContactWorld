python train.py \
  --data-root data/demo_data/insertion_power \
  --task insertion_power \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 310