#!/usr/bin/env python3
"""Empirical optimal loss baseline given fixed context length L.

This script computes the best possible cross-entropy achievable by any predictor
that only conditions on the last L symbols, using the dataset itself to
estimate P(x_{t+1} | x_{t-L+1:t}).

It provides a faithful baseline that:
- Includes regime switching and pre-sync contexts.
- Does not rely on model internals or metadata.

Example:
  uv run python cssr_discovery/compute_markov_baseline.py \
    --data experiments/datasets/gm_seven_switch/combined.dat \
    --L 8 \
    --output_json results/gm_seven_switch_markov_L8.json
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from two_stage_oracle_cssr import load_binary_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute empirical Markov-L optimal loss baseline.")
    parser.add_argument("--data", required=True, help="Path to binary .dat dataset.")
    parser.add_argument("--L", type=int, required=True, help="Context length L for the Markov baseline.")
    parser.add_argument(
        "--max_positions",
        type=int,
        default=None,
        help="Optional cap on number of positions to use (default: all).",
    )
    parser.add_argument("--output_json", help="Optional path to write baseline summary JSON.")
    return parser.parse_args()


def build_history_counts(data: np.ndarray, L: int, max_positions: int | None) -> Dict[Tuple[int, ...], np.ndarray]:
    counts: Dict[Tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    n = len(data)
    limit = n if max_positions is None else min(n, max_positions + L)
    for t in range(L, limit):
        hist = tuple(int(x) for x in data[t - L : t])
        nxt = int(data[t])
        if nxt not in (0, 1):
            continue
        counts[hist][nxt] += 1
    return counts


def compute_markov_loss(data: np.ndarray, L: int, counts: Dict[Tuple[int, ...], np.ndarray]) -> Dict[str, float]:
    # Precompute probability tables
    probs: Dict[Tuple[int, ...], np.ndarray] = {}
    for hist, c in counts.items():
        total = int(c.sum())
        if total == 0:
            continue
        p = c.astype(np.float64) / float(total)
        probs[hist] = p

    total_loss = 0.0
    num_predictions = 0
    n = len(data)
    for t in range(L, n):
        hist = tuple(int(x) for x in data[t - L : t])
        nxt = int(data[t])
        dist = probs.get(hist)
        if dist is None:
            continue
        prob = float(dist[nxt])
        if prob <= 1e-12:
            continue
        total_loss -= math.log(prob)
        num_predictions += 1

    if num_predictions == 0:
        return {
            "total_loss": float("inf"),
            "avg_loss_per_symbol": float("inf"),
            "avg_loss_per_symbol_bits": float("inf"),
            "num_predictions": 0,
        }

    avg_loss = total_loss / num_predictions
    return {
        "total_loss": total_loss,
        "avg_loss_per_symbol": avg_loss,
        "avg_loss_per_symbol_bits": avg_loss / math.log(2.0),
        "num_predictions": num_predictions,
        "num_unique_histories": len(probs),
    }


def main() -> None:
    args = parse_args()
    data_str = load_binary_string(Path(args.data))
    data = np.array([int(c) for c in data_str], dtype=np.int64)

    if args.L <= 0:
        raise ValueError("L must be positive.")
    if args.L >= len(data):
        raise ValueError(f"L={args.L} is too large for dataset of length {len(data)}.")

    print(f"Computing Markov-{args.L} baseline on {len(data)} symbols...")
    counts = build_history_counts(data, args.L, args.max_positions)
    stats = compute_markov_loss(data, args.L, counts)
    print(
        f"Markov-{args.L} loss: {stats['avg_loss_per_symbol']:.4f} nats "
        f"({stats['avg_loss_per_symbol_bits']:.4f} bits) over {stats['num_predictions']} predictions."
    )
    print(f"Unique histories with counts: {stats['num_unique_histories']}")

    if args.output_json:
        payload = {
            "data": args.data,
            "L": args.L,
            "max_positions": args.max_positions,
            "baseline": stats,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"Wrote baseline summary to {args.output_json}")


if __name__ == "__main__":
    main()

