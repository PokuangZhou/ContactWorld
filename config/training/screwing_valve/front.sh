  python train.py \
    --data-root data/demo_data/screwing_valve \
    --task screwing_valve \
    --vision-key front \
    --vision-type image \
    --image-size 224 \
    --reg-loss-type vc \
    --batch-size 64 \
    --epochs 450