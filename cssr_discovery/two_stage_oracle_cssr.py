"""
Oracle-driven two-stage CSSR implementation.

Stage 1 clusters fixed-length histories using oracle rollouts at increasing
prediction horizons. Stage 2 discovers minimal synchronizing suffixes (state
labels) directly from the clustered histories. All states therefore correspond
to histories of length L_max and short pre-sync contexts never become states.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration import fit_platt_params
from js_metrics import (
    extract_subsequences,
    get_kstep_distribution,
    get_next_token_distribution,
    js_divergence,
)
from transcssr_baseline.transcssr_neural_runner import (
    _load_nano_gpt_model,
    load_binary_string,
)


History = np.ndarray
HistoryKey = Tuple[int, ...]


@dataclass
class StageOneResult:
    partition: List[List[History]]
    representatives: List[History]
    k_values_used: List[int]
    state_of_history: Dict[HistoryKey, int]


@dataclass
class StageTwoResult:
    minimal_suffixes: List[List[Tuple[int, Tuple[int, ...]]]]
    suffix_to_states: Dict[Tuple[int, HistoryKey], set]


@dataclass
class TwoStageResult:
    stage_one: StageOneResult
    stage_two: StageTwoResult
    history_counts: Dict[HistoryKey, int]


def history_to_key(hist: History) -> HistoryKey:
    return tuple(int(x) for x in hist.tolist())


def unique_histories(histories: Sequence[History]) -> Tuple[List[History], Dict[HistoryKey, int]]:
    """Deduplicate histories while tracking their multiplicities."""
    unique: Dict[HistoryKey, History] = {}
    counts: Dict[HistoryKey, int] = {}
    ordered_keys: List[HistoryKey] = []

    for hist in histories:
        key = history_to_key(hist)
        counts[key] = counts.get(key, 0) + 1
        if key not in unique:
            unique[key] = hist.copy()
            ordered_keys.append(key)

    uniques = [unique[k] for k in ordered_keys]
    return uniques, counts


def partition_equal(a: Sequence[Sequence[History]], b: Sequence[Sequence[History]]) -> bool:
    """Compare partitions ignoring ordering inside clusters."""
    if len(a) != len(b):
        return False

    def normalize(partition: Sequence[Sequence[History]]) -> List[List[HistoryKey]]:
        normalized: List[List[HistoryKey]] = []
        for cluster in partition:
            cluster_keys = sorted(history_to_key(h) for h in cluster)
            normalized.append(cluster_keys)
        normalized.sort()
        return normalized

    return normalize(a) == normalize(b)


def histories_equivalent(
    h1: History,
    h2: History,
    k_values: Sequence[int],
    tolerance: float,
    preds: Dict[int, Dict[HistoryKey, np.ndarray]],
) -> bool:
    key1 = history_to_key(h1)
    key2 = history_to_key(h2)
    for k in k_values:
        p = preds[k][key1]
        q = preds[k][key2]
        if js_divergence(p, q) >= tolerance:
            return False
    return True


def cluster_histories(
    histories: Sequence[History],
    metrics_k: Sequence[int],
    tolerance: float,
    preds: Dict[int, Dict[HistoryKey, np.ndarray]],
) -> StageOneResult:
    if not metrics_k:
        raise ValueError("metrics_k must contain at least one horizon.")

    k_values_used = [metrics_k[0]]
    partition: List[List[History]] = []
    representatives: List[History] = []

    for hist in histories:
        assigned = None
        for idx, rep in enumerate(representatives):
            if histories_equivalent(hist, rep, k_values_used, tolerance, preds):
                assigned = idx
                break
        if assigned is None:
            assigned = len(representatives)
            representatives.append(hist.copy())
            partition.append([])
        partition[assigned].append(hist.copy())

    for k_new in metrics_k[1:]:
        trial_k_values = k_values_used + [k_new]
        new_partition: List[List[History]] = []
        new_reps: List[History] = []
        for cluster in partition:
            sub_partition: List[List[History]] = []
            sub_reps: List[History] = []
            for hist in cluster:
                assigned = None
                for sub_idx, rep in enumerate(sub_reps):
                    if histories_equivalent(hist, rep, trial_k_values, tolerance, preds):
                        assigned = sub_idx
                        break
                if assigned is None:
                    assigned = len(sub_reps)
                    sub_reps.append(hist.copy())
                    sub_partition.append([])
                sub_partition[assigned].append(hist.copy())
            new_partition.extend(sub_partition)
            new_reps.extend(sub_reps)

        if partition_equal(partition, new_partition):
            break
        partition = new_partition
        representatives = new_reps
        k_values_used.append(k_new)

    state_map: Dict[HistoryKey, int] = {}
    for idx, cluster in enumerate(partition):
        for hist in cluster:
            state_map[history_to_key(hist)] = idx

    return StageOneResult(
        partition=partition,
        representatives=representatives,
        k_values_used=k_values_used,
        state_of_history=state_map,
    )


def collect_suffix_maps(
    histories: Sequence[History],
    state_map: Dict[HistoryKey, int],
) -> Dict[Tuple[int, HistoryKey], set]:
    suffix_to_states: Dict[Tuple[int, HistoryKey], set] = defaultdict(set)
    for hist in histories:
        key = history_to_key(hist)
        state = state_map[key]
        L = len(hist)
        for length in range(1, L + 1):
            suffix = tuple(int(x) for x in hist[-length:].tolist())
            suffix_to_states[(length, suffix)].add(state)
    return suffix_to_states


def minimal_suffixes_per_state(
    num_states: int,
    suffix_to_states: Dict[Tuple[int, HistoryKey], set],
) -> List[List[Tuple[int, Tuple[int, ...]]]]:
    per_state: List[List[Tuple[int, Tuple[int, ...]]]] = [[] for _ in range(num_states)]
    for (length, suffix), states in suffix_to_states.items():
        if len(states) == 1:
            state = next(iter(states))
            per_state[state].append((length, suffix))

    minimal: List[List[Tuple[int, Tuple[int, ...]]]] = [[] for _ in range(num_states)]
    for state, suffixes in enumerate(per_state):
        suffixes.sort(key=lambda item: (item[0], item[1]))
        kept: List[Tuple[int, Tuple[int, ...]]] = []
        for length, suffix in suffixes:
            is_subsumed = False
            suffix_list = list(suffix)
            for kept_len, kept_suffix in kept:
                if kept_len <= length and suffix_list[-kept_len:] == list(kept_suffix):
                    is_subsumed = True
                    break
            if not is_subsumed:
                kept.append((length, suffix))
        minimal[state] = kept
    return minimal


def stage_two(
    histories: Sequence[History],
    stage_one: StageOneResult,
) -> StageTwoResult:
    suffix_map = collect_suffix_maps(histories, stage_one.state_of_history)
    minimal = minimal_suffixes_per_state(len(stage_one.partition), suffix_map)
    return StageTwoResult(minimal_suffixes=minimal, suffix_to_states=suffix_map)


def precompute_predictions(
    histories: Sequence[History],
    metrics_k: Sequence[int],
    model,
    platt_params: Optional[dict],
) -> Dict[int, Dict[HistoryKey, np.ndarray]]:
    preds: Dict[int, Dict[HistoryKey, np.ndarray]] = {k: {} for k in metrics_k}
    for idx, hist in enumerate(histories):
        key = history_to_key(hist)
        for k in metrics_k:
            preds[k][key] = get_kstep_distribution(model, hist, k, platt_params=platt_params)
        if (idx + 1) % 100 == 0 or idx == len(histories) - 1:
            print(f"  computed k-step distributions for history {idx + 1}/{len(histories)}", flush=True)
    return preds


def run_two_stage(
    data: np.ndarray,
    model,
    L_max: int,
    metrics_k: Sequence[int],
    tolerance_bits: float,
    platt_params: Optional[dict] = None,
) -> TwoStageResult:
    histories = extract_subsequences(data, L_max)
    print(f"Collected {len(histories)} length-{L_max} histories.")

    unique_hist, hist_counts = unique_histories(histories)
    print(f"Unique histories: {len(unique_hist)}")

    tolerance = tolerance_bits * math.log(2.0)
    preds = precompute_predictions(unique_hist, metrics_k, model, platt_params)

    print("\n=== Stage 1: oracle clustering ===")
    stage_one = cluster_histories(unique_hist, metrics_k, tolerance, preds)
    print(f"States discovered: {len(stage_one.partition)}")
    for idx, cluster in enumerate(stage_one.partition):
        example = "".join(str(int(b)) for b in cluster[0].tolist())
        print(f"  State {idx}: {len(cluster)} histories (example suffix {example})")

    print("\n=== Stage 2: synchronizing suffixes ===")
    stage_two_res = stage_two(unique_hist, stage_one)
    for idx, suffixes in enumerate(stage_two_res.minimal_suffixes):
        rendered = [f"{length}:{''.join(str(x) for x in suffix)}" for length, suffix in suffixes]
        label = rendered if rendered else ["<none>"]
        print(f"  State {idx}: minimal suffixes {label}")

    return TwoStageResult(stage_one=stage_one, stage_two=stage_two_res, history_counts=hist_counts)


def compute_epsilon_machine_loss(
    result: TwoStageResult,
    model,
    data: np.ndarray,
    L: int,
    platt_params: Optional[dict] = None,
) -> Dict[str, float]:
    """Compute negative log-likelihood of data under the discovered epsilon machine."""
    clusters = result.stage_one.partition
    if not clusters:
        return {
            "total_loss": float("inf"),
            "avg_loss_per_symbol": float("inf"),
            "avg_loss_per_symbol_bits": float("inf"),
            "num_predictions": 0,
            "num_states": 0,
        }

    print("\n=== Computing Epsilon Machine Loss ===")

    state_emissions: Dict[int, np.ndarray] = {}
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        repr_hist = cluster[0]
        probs = get_next_token_distribution(model, repr_hist, platt_params)
        state_emissions[i] = probs
        print(f"State {i}: P(0)={probs[0]:.4f}, P(1)={probs[1]:.4f} (from {len(cluster)} histories)")

    minimal_suffixes = result.stage_two.minimal_suffixes

    def get_state_from_context(history: np.ndarray) -> Optional[int]:
        """Map a context to discovered state using minimal synchronizing suffixes."""
        best_state: Optional[int] = None
        best_len = -1
        for state_idx, suffixes in enumerate(minimal_suffixes):
            for length, suffix in suffixes:
                if len(history) < length:
                    continue
                if np.array_equal(history[-length:], np.array(suffix, dtype=np.int64)):
                    if length > best_len:
                        best_len = length
                        best_state = state_idx
        return best_state

    total_loss = 0.0
    num_predictions = 0
    print(f"Evaluating epsilon machine on {len(data)} symbols...")

    for i in range(L, len(data)):
        context = data[i - L : i]
        next_symbol = int(data[i])
        state = get_state_from_context(context)
        if state is None or state not in state_emissions:
            continue
        emission_probs = state_emissions[state]
        prob_next = float(emission_probs[next_symbol])
        if prob_next > 1e-12:
            total_loss -= math.log(prob_next)
            num_predictions += 1

    if num_predictions == 0:
        avg_loss = float("inf")
        avg_bits = float("inf")
    else:
        avg_loss = total_loss / num_predictions
        avg_bits = avg_loss / math.log(2.0)

    print(f"Total predictions: {num_predictions}")
    print(f"Average loss: {avg_loss:.4f} nats ({avg_bits:.4f} bits)")

    return {
        "total_loss": total_loss,
        "avg_loss_per_symbol": avg_loss,
        "avg_loss_per_symbol_bits": avg_bits,
        "num_predictions": num_predictions,
        "num_states": len([c for c in clusters if c]),
    }


def save_result_json(path: Path, result: TwoStageResult, args, loss_stats: Optional[Dict[str, float]] = None) -> None:
    payload = {
        "preset": args.preset,
        "L_max": args.L_max,
        "metrics_k": result.stage_one.k_values_used,
        "tolerance_bits": args.tolerance_bits,
        "num_states": len(result.stage_one.partition),
        "state_sizes": [len(cluster) for cluster in result.stage_one.partition],
        "minimal_suffixes": [
            [{"length": length, "suffix": ''.join(str(x) for x in suffix)} for length, suffix in suffixes]
            for suffixes in result.stage_two.minimal_suffixes
        ],
        "histories_considered": len(result.history_counts),
        "total_histories": sum(result.history_counts.values()),
    }
    if loss_stats is not None:
        payload["loss"] = loss_stats
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved results to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-driven two-stage CSSR.")
    parser.add_argument("--preset", type=str, default="seven_state_human_char_large")
    parser.add_argument("--model_ckpt", type=str, help="Path to nanoGPT checkpoint.")
    parser.add_argument("--data", type=str, help="Path to binary .dat file.")
    parser.add_argument("--L_max", type=int, default=5, help="History length used to define states.")
    parser.add_argument(
        "--metrics_k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Sequence of k horizons for refining the partition.",
    )
    parser.add_argument(
        "--fixed_k",
        type=int,
        help="Use a single k horizon (overrides --metrics_k).",
    )
    parser.add_argument("--tolerance_bits", type=float, default=1e-3, help="JS tolerance in bits.")
    parser.add_argument(
        "--disable_platt",
        action="store_true",
        help="Disable Platt calibration before computing predictions.",
    )
    parser.add_argument(
        "--compute_loss",
        action="store_true",
        help="Compute epsilon-machine negative log-likelihood on the dataset.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Optional path to save JSON summary.",
    )
    return parser.parse_args()


def resolve_paths(preset: str, model_override: Optional[str], data_override: Optional[str]) -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "nanoGPT" / "out-seven-state-human-char-large" / "ckpt.pt"
    default_data = repo_root / "experiments" / "datasets" / "seven_state_human" / "seven_state_human.dat"

    if preset == "seven_state_human_100k":
        default_model = repo_root / "nanoGPT" / "out-seven-state-human-char_100k" / "ckpt.pt"
    elif preset == "seven_state_human_large":
        default_model = repo_root / "nanoGPT" / "out-seven-state-char_large" / "ckpt.pt"
    elif preset == "sevestateold":
        default_model = repo_root / "nanoGPT" / "out-sevestateold-char" / "ckpt.pt"
        default_data = repo_root / "experiments" / "datasets" / "sevestateold" / "sevestateold.dat"
    elif preset == "even_process":
        default_model = repo_root / "nanoGPT" / "out-even-process-char" / "ckpt.pt"
        default_data = repo_root / "experiments" / "datasets" / "even_process" / "even_process.dat"

    model_path = Path(model_override) if model_override else default_model
    data_path = Path(data_override) if data_override else default_data
    return model_path, data_path


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    metrics_k = args.metrics_k
    if args.fixed_k is not None:
        if args.fixed_k <= 0:
            raise ValueError("--fixed_k must be positive.")
        metrics_k = [args.fixed_k]

    model_path, data_path = resolve_paths(args.preset, args.model_ckpt, args.data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, block_size = _load_nano_gpt_model(model_path, device)

    dataset_str = load_binary_string(data_path)
    data = np.array([int(c) for c in dataset_str], dtype=np.int64)
    if args.L_max > block_size:
        print(f"Warning: L_max={args.L_max} exceeds model block size {block_size}. Contexts will be truncated.")

    platt_params = None
    if not args.disable_platt:
        print("Fitting Platt calibration parameters...")
        platt_params = fit_platt_params(data, model, L_max=min(args.L_max, 6))
        if platt_params:
            print(f"  Platt parameters: a={platt_params['a']:.3f}, b={platt_params['b']:.3f}")
        else:
            print("  Could not fit Platt parameters; proceeding without calibration.")

    print(
        f"Running 2-stage CSSR with L_max={args.L_max}, k horizons={metrics_k}, tolerance={args.tolerance_bits} bits"
    )
    result = run_two_stage(
        data=data,
        model=model,
        L_max=args.L_max,
        metrics_k=metrics_k,
        tolerance_bits=args.tolerance_bits,
        platt_params=platt_params,
    )

    loss_stats: Optional[Dict[str, float]] = None
    if args.compute_loss:
        loss_stats = compute_epsilon_machine_loss(result, model, data, L=args.L_max, platt_params=platt_params)

    if args.output_json:
        save_result_json(Path(args.output_json), result, args, loss_stats)


if __name__ == "__main__":
    main()
