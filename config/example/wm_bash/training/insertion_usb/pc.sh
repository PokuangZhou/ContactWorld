python train.py \
  --data-root data/insertion_usb \
  --task insertion_usb \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 560
