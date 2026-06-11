python train.py \
  --data-root data/demo_data/disassembly_barbed_flat \
  --task disassembly_barbed_flat \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 410