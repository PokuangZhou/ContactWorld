python train.py \
  --data-root data/demo_data/insertion_peg \
  --task insertion_peg \
  --vision-key wrist \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 560