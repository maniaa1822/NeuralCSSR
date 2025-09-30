from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data import MachineDataset, collate_sequences, load_npz_dataset
from .metrics import compute_metrics
from .model import DecoderConfig, EpsMachineDecoder

EPS = 1e-6


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _init_stats() -> Dict[str, float]:
    return defaultdict(float)


def _update_stats(stats: Dict[str, float], preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> None:
    emissions = preds["emissions"].detach()
    transition_logits = preds["transition_logits"].detach()
    mask_logits = preds["mask_logits"].detach()

    emission_target = targets["emissions"].detach()
    transition_target = targets["transitions"].detach()
    mask_target = targets["mask"].detach()

    mask_sum = mask_target.sum()
    diff = (emissions - emission_target) * mask_target
    stats["emit_abs"] += diff.abs().sum().item()
    stats["emit_sq"] += diff.pow(2).sum().item()
    stats["mask_sum"] += mask_sum.item()

    mask_probs = torch.sigmoid(mask_logits)
    stats["mask_intersection"] += (mask_probs * mask_target).sum().item()
    stats["mask_pred_sum"] += mask_probs.sum().item()
    stats["mask_true_sum"] += mask_target.sum().item()
    counts_pred = (mask_probs > 0.5).float().sum(dim=-1)
    counts_true = mask_target.sum(dim=-1)
    stats["state_count_correct"] += (counts_pred.round() == counts_true).float().sum().item()
    stats["num_sequences"] += mask_target.size(0)

    trans_probs = torch.softmax(transition_logits, dim=-1)
    trans_pred = trans_probs.argmax(dim=-1)
    mask_expand = mask_target.unsqueeze(-1).expand_as(transition_target)
    stats["trans_correct"] += ((trans_pred == transition_target) & mask_expand.bool()).sum().item()
    stats["trans_total"] += mask_expand.sum().item()

    entropy = -(trans_probs.clamp_min(EPS) * trans_probs.clamp_min(EPS).log()).sum(dim=-1)
    stats["row_entropy_sum"] += (entropy * mask_expand).sum().item()
    stats["row_count"] += mask_expand.sum().item()


def _finalize(stats: Dict[str, float], max_states: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    mask_sum = max(stats["mask_sum"], EPS)
    metrics["emission_mae"] = stats["emit_abs"] / mask_sum
    metrics["emission_rmse"] = math.sqrt(stats["emit_sq"] / mask_sum)

    trans_total = max(stats["trans_total"], EPS)
    metrics["transition_accuracy"] = stats["trans_correct"] / trans_total

    dice_den = stats["mask_pred_sum"] + stats["mask_true_sum"] + EPS
    metrics["mask_dice"] = (2 * stats["mask_intersection"] + EPS) / dice_den

    metrics["state_count_accuracy"] = stats["state_count_correct"] / max(stats["num_sequences"], EPS)

    row_count = max(stats["row_count"], EPS)
    metrics["mean_row_entropy"] = (stats["row_entropy_sum"] / row_count) / math.log(max_states)

    return metrics


def evaluate_model(
    model: EpsMachineDecoder,
    dataset: MachineDataset,
    config: EvaluationConfig,
) -> Dict[str, Dict[str, float]]:
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_sequences)
    device = config.device
    model.to(device)
    model.eval()

    overall_stats = _init_stats()
    split_stats: Dict[str, Dict[str, float]] = defaultdict(_init_stats)
    max_states = None

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            emissions = batch["emissions"].to(device)
            transitions = batch["transitions"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            splits = batch.get("split") or ["all"] * tokens.size(0)

            pred_emissions, pred_transitions, pred_mask_logits = model(tokens, attention_mask=attention_mask)

            batch_preds = {
                "emissions": pred_emissions,
                "transition_logits": pred_transitions,
                "mask_logits": pred_mask_logits,
            }
            batch_targets = {
                "emissions": emissions,
                "transitions": transitions,
                "mask": mask,
            }

            # overall statistics
            _update_stats(overall_stats, batch_preds, batch_targets)

            # per-split statistics
            for idx, split in enumerate(splits):
                split_tensor_slice = slice(idx, idx + 1)
                per_sample_preds = {
                    key: value[split_tensor_slice]
                    for key, value in batch_preds.items()
                }
                per_sample_targets = {
                    key: value[split_tensor_slice]
                    for key, value in batch_targets.items()
                }
                _update_stats(split_stats[split], per_sample_preds, per_sample_targets)

            if max_states is None:
                max_states = pred_transitions.size(-1)

    if max_states is None:
        raise RuntimeError("Evaluation dataset produced no samples")

    results = {
        "overall": _finalize(overall_stats, max_states),
        "splits": {
            split: _finalize(stats, max_states)
            for split, stats in split_stats.items()
        },
    }
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate epsilon-machine reconstruction")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to NPZ dataset")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint to load")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--max_states", type=int, default=12, help="Model max states")
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=0, help="Model max length (0=use dataset max)")
    parser.add_argument("--output", type=Path, help="Optional JSON path to write metrics")
    return parser


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return choice


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    logger = logging.getLogger(__name__)

    dataset = load_npz_dataset(args.dataset)
    logger.info("Loaded dataset %s with %d samples", args.dataset, len(dataset))

    state_dict = None
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        # Drop positional buffer to avoid size mismatch (deterministic, recomputed at init).
        state_dict.pop("positional.pe", None)
        logger.info("Loaded checkpoint %s", args.checkpoint)

    max_length = args.max_length
    if max_length <= 0:
        max_length = max(sample["length"] for sample in dataset.samples)

    model = EpsMachineDecoder(
        DecoderConfig(
            max_states=args.max_states,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            max_length=max_length,
        )
    )

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    eval_config = EvaluationConfig(batch_size=args.batch_size, device=resolve_device(args.device))
    results = evaluate_model(model, dataset, eval_config)

    logger.info("Overall metrics: %s", json.dumps(results["overall"], indent=2))
    for split, metrics in results["splits"].items():
        logger.info("Split %s: %s", split, json.dumps(metrics, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
