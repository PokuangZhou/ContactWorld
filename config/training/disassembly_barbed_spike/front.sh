python train.py \
  --data-root data/demo_data/disassembly_barbed_spike \
  --task disassembly_barbed_spike \
  --vision-key front \
  --vision-type image \
  --image-size 224 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 550