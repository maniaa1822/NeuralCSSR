"""Convenience entry point for supervised training.

Example commands:

```bash
uv run --with numpy --with torch run_training.py \
  --epochs 3 --batch_size 6 --lr 3e-4 \
  --checkpoint results/eps_decoder.pt \
  --log_dir results/tb_logs

uv run --with numpy --with torch run_training.py \
  --epochs 1 --batch_size 4 --device cpu

uv run --with numpy --with torch run_training.py \
  --epochs 5 --batch_size 16 --lr 2e-4 \
  --checkpoint results/eps_decoder_large.pt \
  --log_dir results/tb_large \
  --val_fraction 0.1

uv run --with numpy --with torch run_training.py \
  --dataset results/ood_supervised_dataset_200machines_10M.npz \
  --train_split train --val_split val \
  --epochs 50 --batch_size 32 --lr 2e-4 \
  --checkpoint results/eps_decoder_10M.pt \
  --log_dir results/tb_10M
```
"""

from ood_supervised.run_training import main


if __name__ == "__main__":
    main()
