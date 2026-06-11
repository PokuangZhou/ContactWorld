python train.py \
  --data-root data/demo_data/disassembly_lidded_loose \
  --task disassembly_lidded_loose \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 415
