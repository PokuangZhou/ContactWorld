python train.py \
  --data-root data/demo_data/exploration_search \
  --task exploration_search \
  --vision-key wrist \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 1000