python train.py \
  --data-root data/demo_data/insertion_power \
  --task insertion_power \
  --vision-key wrist \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 310