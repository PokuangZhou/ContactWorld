python train.py \
  --data-root data/demo_data/disassembly_barbed_spike \
  --task disassembly_barbed_spike \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 550