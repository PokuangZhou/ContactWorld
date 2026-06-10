## Download and Extract

USB insertion example:

- Collected data: [insertion_usb_dataset.tar.gz](https://huggingface.co/datasets/Pokuang/ContactWorld/blob/main/releases/insertion_usb_dataset.tar.gz)
- Pretrained checkpoint: [insertion_usb_ckpt.tar.gz](https://huggingface.co/datasets/Pokuang/ContactWorld/blob/main/releases/insertion_usb_ckpt.tar.gz)

Download the archives:

```bash
mkdir -p releases
wget -O releases/insertion_usb_dataset.tar.gz \
  https://huggingface.co/datasets/Pokuang/ContactWorld/resolve/main/releases/insertion_usb_dataset.tar.gz
wget -O releases/insertion_usb_ckpt.tar.gz \
  https://huggingface.co/datasets/Pokuang/ContactWorld/resolve/main/releases/insertion_usb_ckpt.tar.gz
```

### Dataset

Extract datasets into the `data/` directory:

```bash
mkdir -p data/demo_data
tar -xzf releases/insertion_usb_dataset.tar.gz -C data/demo_data
```

This will create:

```text
data/demo_data/
└── insertion_usb/
```

### Checkpoint

Extract checkpoints into the `logs/ckpts/` directory:

```bash
mkdir -p logs/ckpts
tar -xzf releases/insertion_usb_ckpt.tar.gz -C logs/ckpts
```

This will create:

```text
logs/
└── ckpts/
    └── insertion_usb/
```

The dataset and checkpoint archives contain directories with the same name and therefore should be extracted into their corresponding target locations (`data/` and `logs/ckpts/`).
