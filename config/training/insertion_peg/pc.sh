python train.py \
  --data-root data/demo_data/insertion_peg \
  --task insertion_peg \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 560