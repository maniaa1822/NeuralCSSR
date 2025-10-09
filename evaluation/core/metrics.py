"""Shared metric computation functions."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_lm_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute language modeling metrics.

    Args:
        logits: Model logits [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]

    Returns:
        Dictionary with 'loss', 'bits_per_token', 'perplexity'
    """
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean",
    )
    loss_val = float(loss.item())
    bits_per_token = loss_val / math.log(2)
    perplexity = math.exp(loss_val)

    return {
        "loss": loss_val,
        "bits_per_token": bits_per_token,
        "perplexity": perplexity,
    }


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        logits: Model logits [batch, seq_len, num_classes] or [batch, num_classes]
        targets: Target class indices [batch, seq_len] or [batch]

    Returns:
        Dictionary with 'loss', 'accuracy'
    """
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    preds = logits_flat.argmax(dim=-1)
    accuracy = float((preds == targets_flat).float().mean().item())

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy,
    }


def compute_distance_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute distance prediction metrics for geometry head.

    Args:
        logits: Distance logits [batch, seq_len, num_states, num_distance_classes]
        targets: Distance class targets [batch, seq_len, num_states]

    Returns:
        Dictionary with 'loss', 'accuracy'
    """
    B, T, S, C = logits.shape
    logits_flat = logits.reshape(B * T * S, C)
    targets_flat = targets.reshape(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    preds = logits_flat.argmax(dim=-1)
    accuracy = float((preds == targets_flat).float().mean().item())

    return {
        "loss": float(loss.item()),
        "accuracy": accuracy,
    }
