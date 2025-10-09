"""OOD evaluation logic.

Extracted and adapted from nanoGPT/mtl/eval_ood.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from evaluation.core.models import ModelHandle


@dataclass
class OODMetrics:
    """Metrics from OOD evaluation."""
    lm_loss: float
    bits_per_token: float
    token_count: int
    epsilon_loss: Optional[float] = None
    epsilon_accuracy: Optional[float] = None
    distance_loss: Optional[float] = None
    distance_accuracy: Optional[float] = None


def evaluate_model_ood(
    handle: ModelHandle,
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> OODMetrics:
    """Evaluate model on OOD data.

    Args:
        handle: Model handle
        data: (x, y, eps, dist) tensors
        batch_size: Batch size for evaluation
        device: Device to evaluate on

    Returns:
        OODMetrics with evaluation results
    """
    x, y, eps, dist = data
    total_sequences = x.size(0)

    lm_loss_sum = 0.0
    lm_token_count = 0
    epsilon_loss_sum = 0.0
    epsilon_correct = 0
    epsilon_count = 0
    distance_loss_sum = 0.0
    distance_correct = 0
    distance_count = 0

    with torch.no_grad():
        for start in range(0, total_sequences, batch_size):
            end = min(start + batch_size, total_sequences)
            xb = x[start:end].to(device)
            yb = y[start:end].to(device)
            epsb = eps[start:end].to(device)
            distb = dist[start:end].to(device)

            if handle.kind == "mtl":
                # Multitask model
                out = handle.model(xb, yb, epsb, distb)
                lm_logits = out.get("lm_logits")
                if lm_logits is None:
                    raise RuntimeError("Multitask model did not return lm_logits")

                # LM loss
                loss = F.cross_entropy(
                    lm_logits.reshape(-1, lm_logits.size(-1)),
                    yb.reshape(-1),
                    reduction="mean",
                )
                token_count = yb.numel()
                lm_loss_sum += float(loss.item()) * token_count
                lm_token_count += token_count

                # Epsilon loss
                epsilon_logits = out.get("epsilon_logits", {})
                if epsilon_logits:
                    # Use tap_preference or first available
                    tap_name = handle.tap_preference
                    if tap_name not in epsilon_logits:
                        tap_name = list(epsilon_logits.keys())[0]

                    logits = epsilon_logits[tap_name]
                    eps_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        epsb.reshape(-1),
                        reduction="mean",
                    )
                    preds = logits.argmax(dim=-1)
                    eps_count = epsb.numel()
                    epsilon_loss_sum += float(eps_loss.item()) * eps_count
                    epsilon_correct += int((preds == epsb).sum().item())
                    epsilon_count += eps_count

                # Distance loss
                distance_logits = out.get("distance_logits", {})
                if distance_logits and handle.has_distance_head:
                    tap_name = handle.tap_preference
                    if tap_name not in distance_logits:
                        tap_name = list(distance_logits.keys())[0]

                    logits = distance_logits[tap_name]
                    B, T, S, C = logits.shape
                    logits_flat = logits.reshape(B * T * S, C)
                    targets_flat = distb.reshape(-1)
                    dist_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
                    preds = logits.argmax(dim=-1)
                    dist_count = distb.numel()
                    distance_loss_sum += float(dist_loss.item()) * dist_count
                    distance_correct += int((preds == distb).sum().item())
                    distance_count += dist_count

            else:
                # Baseline model
                logits, _ = handle.model(xb, yb)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    yb.reshape(-1),
                    reduction="mean",
                )
                token_count = yb.numel()
                lm_loss_sum += float(loss.item()) * token_count
                lm_token_count += token_count

    # Compute final metrics
    lm_loss = lm_loss_sum / lm_token_count if lm_token_count > 0 else float("nan")
    bits_per_token = lm_loss / math.log(2) if lm_token_count > 0 else float("nan")

    epsilon_loss = epsilon_loss_sum / epsilon_count if epsilon_count > 0 else None
    epsilon_accuracy = epsilon_correct / epsilon_count if epsilon_count > 0 else None

    distance_loss = distance_loss_sum / distance_count if distance_count > 0 else None
    distance_accuracy = distance_correct / distance_count if distance_count > 0 else None

    return OODMetrics(
        lm_loss=lm_loss,
        bits_per_token=bits_per_token,
        token_count=lm_token_count,
        epsilon_loss=epsilon_loss,
        epsilon_accuracy=epsilon_accuracy,
        distance_loss=distance_loss,
        distance_accuracy=distance_accuracy,
    )
