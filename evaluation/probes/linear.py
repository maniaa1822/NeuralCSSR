"""Linear probe training and evaluation.

Adapted from entanglement_analysis/probes/ and nanoGPT/probes/state_probing/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProbeConfig:
    """Configuration for probe training."""
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_fraction: float = 0.1
    seed: int = 42


@dataclass
class ProbeMetrics:
    """Evaluation metrics for a probe."""
    loss: float
    accuracy: float
    samples: int


class LinearProbeSuite:
    """Suite of linear probes for multiple label types and layers."""

    def __init__(self, config: ProbeConfig = None):
        self.config = config or ProbeConfig()

    def fit_probes(
        self,
        activations: OrderedDict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, ProbeMetrics]]:
        """Fit linear probes for all layers and label types.

        Args:
            activations: {layer_name: features [N, D]}
            labels: {label_name: targets [N]}

        Returns:
            {layer_name: {label_name: ProbeMetrics}}
        """
        results = {}

        for layer_name, features in activations.items():
            results[layer_name] = {}
            for label_name, targets in labels.items():
                if features.size(0) != targets.size(0):
                    raise ValueError(
                        f"Mismatch: {layer_name} has {features.size(0)} samples, "
                        f"{label_name} has {targets.size(0)}"
                    )

                probe_metrics = train_linear_probe(
                    features, targets, self.config
                )
                results[layer_name][label_name] = probe_metrics

        return results


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: ProbeConfig,
) -> ProbeMetrics:
    """Train a single linear probe.

    Args:
        features: Input features [N, D]
        labels: Target labels [N]
        config: Probe configuration

    Returns:
        ProbeMetrics on validation set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create train/val split
    n_samples = features.size(0)
    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(n_samples, generator=generator)
    val_size = max(1, int(round(n_samples * config.val_fraction)))

    val_idx = permutation[:val_size]
    train_idx = permutation[val_size:]

    if train_idx.numel() == 0:
        train_idx = val_idx

    train_features = features[train_idx].contiguous()
    train_labels = labels[train_idx].contiguous()
    val_features = features[val_idx].contiguous()
    val_labels = labels[val_idx].contiguous()

    # Create dataloaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )

    # Create probe
    input_dim = features.size(1)
    num_classes = int(labels.max().item()) + 1
    probe = nn.Linear(input_dim, num_classes).to(device)

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)

    probe.train()
    for _ in range(config.epochs):
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            total_loss += float(loss.item()) * batch_labels.size(0)

            preds = logits.argmax(dim=-1)
            total_correct += int((preds == batch_labels).sum().item())
            total_samples += batch_labels.size(0)

    if total_samples == 0:
        return ProbeMetrics(loss=float("nan"), accuracy=float("nan"), samples=0)

    return ProbeMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
    )
