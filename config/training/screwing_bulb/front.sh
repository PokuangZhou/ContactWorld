python train.py \
  --data-root data/demo_data/screwing_bulb \
  --task screwing_bulb \
  --vision-key front \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 145