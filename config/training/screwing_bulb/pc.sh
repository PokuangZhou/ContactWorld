  python train.py \
    --data-root data/demo_data/screwing_bulb \
    --task screwing_bulb \
    --vision-key pointcloud \
    --vision-type pc \
    --pc-in-channels 6 \
    --reg-loss-type vc \
    --batch-size 64 \
    --epochs 145