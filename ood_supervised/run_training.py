from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

"""Trainer CLI for supervised epsilon-machine decoder.

Example usage:

```bash
uv run --with numpy --with torch run_training.py \
  --epochs 3 --batch_size 6 --lr 3e-4 \
  --checkpoint results/eps_decoder.pt \
  --log_dir results/tb_logs
```
"""

from ood_supervised import (
    EpsMachineDecoder,
    MachineDatasetConfig,
    TrainingConfig,
    collate_sequences,
    generate_dataset,
    load_npz_dataset,
    split_dataset,
    subset_by_split,
    train_model,
)
from ood_supervised.model import DecoderConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train supervised epsilon-machine decoder")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto", help="Training device")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for dataset generation")
    parser.add_argument("--checkpoint", type=Path, help="Optional path to save model weights")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Fraction of data for validation (0 disables)")
    parser.add_argument("--log_dir", type=Path, help="TensorBoard log directory")
    parser.add_argument("--dataset", type=Path, help="Pre-generated NPZ dataset path")
    parser.add_argument("--train_split", type=str, default="train", help="Split name to use for training")
    parser.add_argument("--val_split", type=str, default="val", help="Split name to use for validation (optional)")
    parser.add_argument("--max_length", type=int, default=0, help="Model max sequence length (0=auto)")
    return parser


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return choice


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    if args.dataset:
        full_dataset = load_npz_dataset(args.dataset)
        if args.train_split:
            train_dataset = subset_by_split(full_dataset, args.train_split)
            if len(train_dataset) == 0:
                raise ValueError(f"No samples found for train split '{args.train_split}'")
        else:
            train_dataset = full_dataset

        if args.val_split:
            val_dataset = subset_by_split(full_dataset, args.val_split)
            if len(val_dataset) == 0:
                val_dataset = None
        else:
            val_dataset = None

        if val_dataset is None and args.val_fraction and 0.0 < args.val_fraction < 1.0:
            train_dataset, val_dataset = split_dataset(train_dataset, 1.0 - args.val_fraction)

        dataset_tuple = (train_dataset, val_dataset) if val_dataset else train_dataset
    else:
        dataset = generate_dataset(
            MachineDatasetConfig(
                n_machines=6,
                seqs_per_machine=3,
                lengths=[32, 64, 96, 128],
                pad_K=12,
                rng=args.seed,
            )
        )

        if args.val_fraction and 0.0 < args.val_fraction < 1.0:
            train_dataset, val_dataset = split_dataset(dataset, 1.0 - args.val_fraction)
            dataset_tuple = (train_dataset, val_dataset)
        else:
            dataset_tuple = dataset

    max_length = args.max_length
    if max_length <= 0:
        source = dataset_tuple[0] if isinstance(dataset_tuple, tuple) else dataset_tuple
        max_length = max(sample["length"] for sample in source.samples)

    model = EpsMachineDecoder(
        DecoderConfig(
            max_states=12,
            hidden_size=64,
            num_layers=3,
            num_heads=4,
            mlp_dim=128,
            max_length=max_length,
        )
    )

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        row_entropy_coeff=0.1,
        label_smoothing=0.01,
        device=resolve_device(args.device),
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
    )

    losses = train_model(model, dataset_tuple, config, collate_sequences)
    for name, value in losses.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
