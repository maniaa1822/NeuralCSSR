#!/usr/bin/env python3
"""Merge CSSR states post hoc based on JS divergence of predictive distributions."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from js_metrics import get_next_token_distribution, js_divergence
from two_stage_oracle_cssr import _load_nano_gpt_model, load_binary_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CSSR states using JS divergence threshold.")
    parser.add_argument("--result_json", required=True, help="Path to CSSR result JSON.")
    parser.add_argument("--model_ckpt", required=True, help="Path to nanoGPT checkpoint used for CSSR.")
    parser.add_argument(
        "--tolerance_bits",
        type=float,
        default=1e-3,
        help="JS divergence threshold (in bits) for merging state emission distributions.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to binary .dat dataset used for CSSR (needed to recompute loss after merging).",
    )
    parser.add_argument("--output_json", help="Optional path to write merged-state summary JSON.")
    return parser.parse_args()


def suffix_to_array(suffix: str) -> np.ndarray:
    return np.array([int(ch) for ch in suffix], dtype=np.int64)


def union_find_merge(distributions: List[np.ndarray], tolerance_bits: float) -> Tuple[List[int], Dict[int, List[int]]]:
    n = len(distributions)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[rb] = ra

    tolerance = tolerance_bits * math.log(2.0)
    for i in range(n):
        for j in range(i + 1, n):
            js = js_divergence(distributions[i], distributions[j])
            if js <= tolerance:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)
    return parent, clusters


def build_suffix_lookup(minimal_suffixes: List[List[Dict[str, object]]], state_to_cluster: Dict[int, int]):
    suffix_lookup: Dict[int, Dict[Tuple[int, ...], int]] = {}
    lengths = set()
    for state_idx, suffixes in enumerate(minimal_suffixes):
        cluster_id = state_to_cluster[state_idx]
        for entry in suffixes:
            length = int(entry["length"])
            suffix_tuple = tuple(int(ch) for ch in entry["suffix"])
            suffix_lookup.setdefault(length, {})[suffix_tuple] = cluster_id
            lengths.add(length)
    sorted_lengths = sorted(lengths, reverse=True)
    return sorted_lengths, suffix_lookup


def resolve_state(history: np.ndarray, lengths: List[int], suffix_lookup: Dict[int, Dict[Tuple[int, ...], int]]):
    for length in lengths:
        if len(history) < length:
            continue
        key = tuple(int(x) for x in history[-length:].tolist())
        state = suffix_lookup[length].get(key)
        if state is not None:
            return state
    return None


def evaluate_loss(
    data: np.ndarray,
    L: int,
    lengths: List[int],
    suffix_lookup: Dict[int, Dict[Tuple[int, ...], int]],
    cluster_emissions: Dict[int, np.ndarray],
) -> Dict[str, float]:
    total_loss = 0.0
    num_predictions = 0
    for i in range(L, len(data)):
        context = data[i - L : i]
        state = resolve_state(context, lengths, suffix_lookup)
        if state is None:
            continue
        probs = cluster_emissions.get(state)
        if probs is None:
            continue
        next_symbol = int(data[i])
        prob = float(probs[next_symbol])
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
    }


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_json)
    data = json.loads(result_path.read_text())

    minimal_suffixes = data.get("minimal_suffixes", [])
    state_sizes = data.get("state_sizes", [1] * len(minimal_suffixes))

    # Build representative contexts from suffixes (pick the shortest available suffix for each state).
    contexts = []
    for suffix_list in minimal_suffixes:
        if not suffix_list:
            contexts.append(np.zeros(1, dtype=np.int64))
            continue
        shortest = min(suffix_list, key=lambda x: x["length"])
        contexts.append(suffix_to_array(shortest["suffix"]))

    # Load model once and compute predictive distributions for each context.
    model, _ = _load_nano_gpt_model(Path(args.model_ckpt), torch.device("cpu"))
    distributions = []
    for ctx in contexts:
        probs = get_next_token_distribution(model, ctx, platt_params=None, batch_size=None)
        distributions.append(probs)

    _, cluster_members = union_find_merge(distributions, args.tolerance_bits)
    state_to_cluster: Dict[int, int] = {}
    for cluster_idx, members in cluster_members.items():
        for m in members:
            state_to_cluster[m] = cluster_idx

    merged_entries = []
    cluster_emissions: Dict[int, np.ndarray] = {}
    for cluster_idx, members in cluster_members.items():
        total_size = int(sum(state_sizes[m] for m in members))
        representative = int(members[0])
        weights = np.array([state_sizes[m] for m in members], dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum == 0:
            avg = distributions[representative]
        else:
            weighted = np.zeros_like(distributions[representative])
            for m, w in zip(members, weights):
                weighted += w * distributions[m]
            avg = weighted / weight_sum
        cluster_emissions[cluster_idx] = avg
        merged_entries.append(
            {
                "merged_state_id": cluster_idx,
                "members": members,
                "total_size": total_size,
                "emission": distributions[representative].tolist(),
            }
        )

    merged_entries.sort(key=lambda entry: entry["total_size"], reverse=True)

    print(f"Merged {len(minimal_suffixes)} states into {len(merged_entries)} clusters at tolerance {args.tolerance_bits} bits.")
    for entry in merged_entries:
        print(f"  merged_id={entry['merged_state_id']}: size={entry['total_size']} members={entry['members']}")

    dataset_str = load_binary_string(Path(args.data))
    data_array = np.array([int(ch) for ch in dataset_str], dtype=np.int64)
    L = int(data.get("L_max", contexts[0].shape[0] if contexts else 1))
    lengths, suffix_lookup = build_suffix_lookup(minimal_suffixes, state_to_cluster)
    loss_stats = evaluate_loss(data_array, L, lengths, suffix_lookup, cluster_emissions)
    print(
        f"Merged-machine loss: {loss_stats['avg_loss_per_symbol']:.4f} nats ({loss_stats['avg_loss_per_symbol_bits']:.4f} bits) over {loss_stats['num_predictions']} predictions."
    )

    if args.output_json:
        payload = {
            "source": str(result_path),
            "tolerance_bits": args.tolerance_bits,
            "num_states_original": len(minimal_suffixes),
            "num_states_merged": len(merged_entries),
            "merged_states": merged_entries,
            "loss": loss_stats,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"Wrote merged state summary to {args.output_json}")


if __name__ == "__main__":
    main()
