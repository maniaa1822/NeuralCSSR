from __future__ import annotations

from typing import Dict

import torch


def compute_metrics(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    mask = targets["mask"]
    mask_expand = mask.unsqueeze(-1)

    emit_pred = predictions["emissions"]
    emit_target = targets["emissions"]
    diff = (emit_pred - emit_target) * mask
    mae = diff.abs().sum() / mask.sum().clamp_min(1.0)
    rmse = torch.sqrt((diff.pow(2).sum() / mask.sum().clamp_min(1.0)))

    trans_pred = predictions["transitions"].argmax(dim=-1)
    trans_target = targets["transitions"]
    mask_symbols = mask_expand.expand_as(trans_pred)
    trans_correct = (trans_pred == trans_target) & mask_symbols.bool()
    trans_acc = trans_correct.float().sum() / mask_symbols.sum().clamp_min(1.0)

    mask_probs = torch.sigmoid(predictions["mask_logits"])
    intersection = (mask_probs * mask).sum()
    total = mask_probs.sum() + mask.sum()
    dice = (2 * intersection + 1e-6) / (total + 1e-6)

    probs = torch.softmax(predictions["transitions"], dim=-1).clamp_min(1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1)
    entropy = (entropy * mask_symbols).sum() / mask_symbols.sum().clamp_min(1.0)

    return {
        "emission_mae": mae.item(),
        "emission_rmse": rmse.item(),
        "transition_accuracy": trans_acc.item(),
        "mask_dice": dice.item(),
        "mean_row_entropy": entropy.item(),
    }
