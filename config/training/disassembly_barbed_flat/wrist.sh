python train.py \
  --data-root data/demo_data/disassembly_barbed_flat \
  --task disassembly_barbed_flat \
  --vision-key wrist \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 410