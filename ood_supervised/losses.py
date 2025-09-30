from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


EPS = 1e-6


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numer = (torch.abs(pred - target) * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return numer / denom


def _masked_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    batch, num_states, num_symbols, vocab = logits.shape
    logits_flat = logits.view(batch * num_states * num_symbols, vocab)
    target_flat = target.view(-1)
    mask_expand = mask.unsqueeze(-1).expand(batch, num_states, num_symbols)
    mask_flat = mask_expand.reshape(-1)

    loss = F.cross_entropy(
        logits_flat,
        target_flat,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    numer = (loss * mask_flat).sum()
    denom = mask_flat.sum().clamp_min(1.0)
    return numer / denom


def _dice_loss(mask_probs: torch.Tensor, mask_target: torch.Tensor) -> torch.Tensor:
    intersection = (mask_probs * mask_target).sum()
    total = mask_probs.sum() + mask_target.sum()
    dice = (2 * intersection + EPS) / (total + EPS)
    return 1.0 - dice


def _row_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1).clamp_min(EPS)
    entropy = -(probs * probs.log()).sum(dim=-1)
    mask_expand = mask.unsqueeze(-1).expand_as(entropy)
    numer = (entropy * mask_expand).sum()
    denom = mask_expand.sum().clamp_min(1.0)
    max_states = logits.size(-1)
    return numer / (denom * math.log(max_states))


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    row_entropy_coeff: float = 0.1,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    emissions = outputs["emissions"]
    transition_logits = outputs["transition_logits"]
    mask_logits = outputs["mask_logits"]

    emission_target = targets["emissions"]
    transition_target = targets["transitions"]
    mask_target = targets["mask"]

    mask_probs = torch.sigmoid(mask_logits)

    loss_emit = _masked_mae(emissions, emission_target, mask_target)
    loss_trans = _masked_ce(transition_logits, transition_target, mask_target, label_smoothing)
    loss_mask = _dice_loss(mask_probs, mask_target)
    loss_uni = _row_entropy(transition_logits, mask_target)

    total = loss_emit + loss_trans + loss_mask + row_entropy_coeff * loss_uni

    return {
        "emit": loss_emit,
        "trans": loss_trans,
        "mask": loss_mask,
        "row_entropy": loss_uni,
        "total": total,
    }
