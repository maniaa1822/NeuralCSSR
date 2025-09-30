"""Supervised epsilon-machine decoder components."""

from .data import (
    MachineDataset,
    MachineDatasetConfig,
    collate_sequences,
    generate_dataset,
    load_preview_dataset,
    load_npz_dataset,
    split_dataset,
    subset_by_split,
)
from .evaluate import EvaluationConfig, evaluate_model
from .metrics import compute_metrics
from .model import EpsMachineDecoder
from .train import TrainingConfig, train_model

__all__ = [
    "EpsMachineDecoder",
    "MachineDataset",
    "MachineDatasetConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "generate_dataset",
    "split_dataset",
    "collate_sequences",
    "compute_metrics",
    "load_preview_dataset",
    "load_npz_dataset",
    "evaluate_model",
    "subset_by_split",
    "train_model",
]
