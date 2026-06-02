## Download and Extract

### Dataset

Extract datasets into the `data/` directory:

```bash
tar -xzf releases/disassembly_barbed_flat_dataset.tar.gz -C data/demo_data
```

This will create:

```text
data/demo_data/
└── disassembly_barbed_flat/
```

### Checkpoint

Extract checkpoints into the `logs/ckpts/` directory:

```bash
mkdir -p logs/ckpts
tar -xzf releases/disassembly_barbed_flat_ckpt.tar.gz -C logs/ckpts
```

This will create:

```text
logs/
└── ckpts/
    └── disassembly_barbed_flat/
```

The dataset and checkpoint archives contain directories with the same name and therefore should be extracted into their corresponding target locations (`data/` and `logs/ckpts/`).
