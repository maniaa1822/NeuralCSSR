"""Convenience wrapper for epsilon-machine evaluation.

Example usage:

```bash
uv run --with numpy --with torch evaluate.py \
  --dataset results/ood_supervised_dataset_4machines_100k.npz \
  --checkpoint results/demo_checkpoint.pt \
  --batch_size 16 \
  --device auto

uv run --with numpy --with torch evaluate.py \
  --dataset results/ood_supervised_dataset_200machines_10M.npz \
  --checkpoint results/demo_checkpoint.pt \
  --batch_size 32 \
  --device cuda
```
"""

from ood_supervised.evaluate import main


if __name__ == "__main__":
    main()
