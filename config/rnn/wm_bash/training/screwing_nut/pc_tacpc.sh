
python train.py \
  --task screwing_nut \
  --data-root /home/zhiyuan/Project/TVB/data/screwing_nut \
  --vision-key pointcloud \
  --vision-type pc \
  --pc-in-channels 6 \
  --use-tactile \
  --tactile-key tactile_depth_right \
  --tactile-type pc \
  --tactile-height 80 \
  --tactile-width 60 \
  --tactile-num-points 1024 \
  --fusion-type concat \
  --reg-on-vision-only-for-concat \
  --reg-loss-type vc \
  --batch-size 64 \
  --epochs 160