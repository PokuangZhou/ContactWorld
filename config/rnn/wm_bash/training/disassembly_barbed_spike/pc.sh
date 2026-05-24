python train.py \
  --data-root /home/zhiyuan/Project/TVB/data/disassembly_barbed_spike \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 560
