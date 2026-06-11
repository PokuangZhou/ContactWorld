python train.py \
  --data-root data/demo_data/disassembly_lidded_loose \
  --task disassembly_lidded_loose \
  --vision-key front \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 415