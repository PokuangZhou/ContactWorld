python train.py \
  --data-root data/demo_data/screwing_nut \
  --task screwing_nut \
  --vision-key wrist \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 160