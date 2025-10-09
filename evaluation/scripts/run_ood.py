#!/usr/bin/env python3
"""Run OOD evaluation experiments on baseline and MTL models.

Replaces nanoGPT/mtl/eval_ood.py with unified infrastructure.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machines import get_machine
from evaluation.core import (
    load_model,
    parse_model_spec,
    generate_ood_data,
    compute_state_distances,
)
from evaluation.ood import evaluate_model_ood


def load_meta(dataset: str) -> tuple[dict[str, int], int]:
    """Load dataset metadata."""
    meta_path = REPO_ROOT / "nanoGPT" / "data" / dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], int(meta["vocab_size"])


def fmt_optional(value: Optional[float], csv_mode: bool = False) -> str:
    """Format optional float values."""
    if value is None:
        return "" if csv_mode else "NA"
    if math.isnan(value):
        return "" if csv_mode else "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model spec: name=path/to/ckpt.pt (repeatable)",
    )
    parser.add_argument(
        "--regime",
        choices=[
            "emission_mix",
            "start_state_uniform",
            "transition_noise",
            "alphabet_swap",
            "emission_bias",
            "state_dependent_swap",
            "temporal_swap",
            "mixed_regime",
        ],
        required=True,
        help="OOD perturbation regime",
    )
    parser.add_argument("--severity", type=float, default=0.0, help="Regime severity parameter")
    parser.add_argument("--tokens", type=int, default=65536, help="Number of evaluation tokens per model")
    parser.add_argument("--block-size", type=int, default=64, help="Sequence length per window")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--dataset", type=str, default="seven_state_human_mtl100k", help="Dataset for vocab metadata")
    parser.add_argument("--machine", type=str, default="seven_state_human", help="Machine name for ground truth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or cuda)")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--mixed-regimes", type=str, default=None, help="Comma-separated regime names for mixed_regime")
    parser.add_argument("--mixed-severities", type=str, default=None, help="Comma-separated severities for mixed_regime")

    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    stoi, vocab_size = load_meta(args.dataset)
    machine = get_machine(args.machine)
    distance_info = compute_state_distances(machine)

    # Parse mixed regime parameters
    mixed_regimes = None
    mixed_severities = None
    if args.regime == "mixed_regime":
        if not args.mixed_regimes or not args.mixed_severities:
            raise ValueError("mixed_regime requires both --mixed-regimes and --mixed-severities")
        mixed_regimes = [r.strip() for r in args.mixed_regimes.split(",")]
        mixed_severities = [float(s.strip()) for s in args.mixed_severities.split(",")]

    # Load models
    model_specs = [parse_model_spec(spec) for spec in args.models]
    handles = []
    for name, path in model_specs:
        ckpt_path = path if path.is_absolute() else (REPO_ROOT / path)
        print(f"Loading model {name}: {ckpt_path}")
        handle = load_model(name, ckpt_path, device, vocab_size, distance_info)
        print(f"  Type: {handle.kind}, Block size: {handle.block_size}")
        handles.append(handle)

    # Use block size from first model if not specified
    block_size = handles[0].block_size or args.block_size

    # Generate OOD data
    print(f"\nGenerating {args.tokens} tokens under regime '{args.regime}' (severity={args.severity})...")
    x, y, eps, dist = generate_ood_data(
        machine,
        stoi,
        block_size,
        args.tokens,
        distance_info,
        args.regime,
        args.severity,
        args.seed,
        mixed_regimes,
        mixed_severities,
    )
    print(f"Data shape: x={x.shape}, y={y.shape}")

    # Evaluate each model
    results = []
    print("\nEvaluating models...")
    for handle in handles:
        print(f"\n{handle.name} ({handle.kind}):")
        metrics = evaluate_model_ood(handle, (x, y, eps, dist), args.batch_size, device)

        print(f"  LM loss: {metrics.lm_loss:.6f}")
        print(f"  Bits/token: {metrics.bits_per_token:.6f}")
        if metrics.epsilon_accuracy is not None:
            print(f"  Epsilon accuracy: {metrics.epsilon_accuracy:.6f}")
        if metrics.distance_accuracy is not None:
            print(f"  Distance accuracy: {metrics.distance_accuracy:.6f}")

        results.append({
            "model": handle.name,
            "kind": handle.kind,
            "regime": args.regime,
            "severity": args.severity,
            "token_count": metrics.token_count,
            "lm_loss": metrics.lm_loss,
            "bits_per_token": metrics.bits_per_token,
            "epsilon_loss": metrics.epsilon_loss,
            "epsilon_accuracy": metrics.epsilon_accuracy,
            "distance_loss": metrics.distance_loss,
            "distance_accuracy": metrics.distance_accuracy,
        })

    # Print summary table
    print("\n" + "="*80)
    print("Summary:")
    headers = ["model", "kind", "regime", "severity", "token_count", "lm_loss", "bits_per_token",
               "epsilon_loss", "epsilon_accuracy", "distance_loss", "distance_accuracy"]
    print("\t".join(headers))
    for record in results:
        line = "\t".join([
            record["model"],
            record["kind"],
            args.regime,
            f"{args.severity:.3f}",
            str(record["token_count"]),
            fmt_optional(record["lm_loss"]),
            fmt_optional(record["bits_per_token"]),
            fmt_optional(record["epsilon_loss"]),
            fmt_optional(record["epsilon_accuracy"]),
            fmt_optional(record["distance_loss"]),
            fmt_optional(record["distance_accuracy"]),
        ])
        print(line)

    # Write CSV if requested
    if args.output:
        output_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for record in results:
                # Format floats for CSV
                csv_record = {
                    "model": record["model"],
                    "kind": record["kind"],
                    "regime": record["regime"],
                    "severity": f"{record['severity']:.6f}",
                    "token_count": record["token_count"],
                    "lm_loss": fmt_optional(record["lm_loss"], csv_mode=True),
                    "bits_per_token": fmt_optional(record["bits_per_token"], csv_mode=True),
                    "epsilon_loss": fmt_optional(record["epsilon_loss"], csv_mode=True),
                    "epsilon_accuracy": fmt_optional(record["epsilon_accuracy"], csv_mode=True),
                    "distance_loss": fmt_optional(record["distance_loss"], csv_mode=True),
                    "distance_accuracy": fmt_optional(record["distance_accuracy"], csv_mode=True),
                }
                writer.writerow(csv_record)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
