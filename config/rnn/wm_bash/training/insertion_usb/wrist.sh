python train.py \
  --data-root data/insertion_usb \
  --task insertion_usb \
  --vision-key wrist \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 560