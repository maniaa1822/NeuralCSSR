import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "nanoGPT"))
from model import GPT, GPTConfig  # noqa: E402


def load_sequence(sequence_path: Path) -> List[int]:
    text = sequence_path.read_text()
    digits = [c for c in text if c in ("0", "1")]
    if not digits:
        raise ValueError(f"No binary tokens found in {sequence_path}")
    return [int(c) for c in digits]


def load_machine_ids(machine_ids_path: Path) -> List[str]:
    text = machine_ids_path.read_text().strip()
    if not text:
        raise ValueError(f"No machine ids found in {machine_ids_path}")
    return text.split()


def load_segments(meta_path: Path) -> List[Dict]:
    with meta_path.open() as fh:
        meta = json.load(fh)
    if "segments" not in meta:
        raise ValueError(f"{meta_path} missing 'segments' entry")
    return meta["segments"]


def compute_unique_witnesses(
    sequence_bits: str,
    segments: Sequence[Dict],
    max_length: int,
    per_machine_limit: int,
) -> Dict[str, List[str]]:
    machines = sorted({segment["machine"] for segment in segments})
    per_machine_sets: Dict[str, List[set]] = {
        machine: [set() for _ in range(max_length + 1)] for machine in machines
    }

    for segment in segments:
        machine = segment["machine"]
        start = segment["start"]
        end = segment["end"]
        subseq = sequence_bits[start:end]
        seg_len = len(subseq)
        for length in range(1, min(max_length, seg_len) + 1):
            holder = per_machine_sets[machine][length]
            for i in range(seg_len - length + 1):
                holder.add(subseq[i : i + length])

    witnesses: Dict[str, List[str]] = {machine: [] for machine in machines}
    for length in range(1, max_length + 1):
        for machine in machines:
            if len(witnesses[machine]) >= per_machine_limit:
                continue
            unique_patterns = per_machine_sets[machine][length].copy()
            for other in machines:
                if other == machine:
                    continue
                unique_patterns -= per_machine_sets[other][length]
            if not unique_patterns:
                continue
            for pattern in sorted(unique_patterns):
                witnesses[machine].append(pattern)
                if len(witnesses[machine]) >= per_machine_limit:
                    break
    return witnesses


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**checkpoint["model_args"])
    model = GPT(config)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def sample_contexts(
    sequence: Sequence[int],
    machine_ids: Sequence[str],
    history_len: int,
    num_samples: int,
    rng: random.Random,
) -> Tuple[List[List[int]], List[str]]:
    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    if len(sequence) != len(machine_ids):
        raise ValueError("Sequence and machine ID streams must be the same length")

    max_index = len(sequence) - 1
    start_index = history_len - 1
    if start_index > max_index:
        return [], []

    available = max_index - start_index + 1
    sample_size = min(num_samples, available)
    positions = rng.sample(range(start_index, max_index + 1), sample_size)
    contexts: List[List[int]] = []
    labels: List[str] = []
    for pos in positions:
        start = pos - history_len + 1
        contexts.append(sequence[start : pos + 1])
        labels.append(machine_ids[pos])
    return contexts, labels


@torch.no_grad()
def sample_rollout_until_k(
    model: GPT, prefix: List[int], max_k: int, device: torch.device
) -> List[int]:
    generated: List[int] = []
    current = prefix.copy()
    for _ in range(max_k):
        if len(current) > model.config.block_size:
            raise ValueError(
                f"Sequence length {len(current)} exceeds block size {model.config.block_size}"
            )
        idx = torch.tensor(current, dtype=torch.long, device=device).unsqueeze(0)
        logits, _, _ = model(idx)
        probs = F.softmax(logits[0, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        current.append(next_token)
    return generated


def contains_any_pattern(sequence: Sequence[int], patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    text = "".join(str(x) for x in sequence)
    return any(pattern in text for pattern in patterns)


def evaluate_context_multi_k(
    model: GPT,
    context: List[int],
    horizons: Sequence[int],
    patterns: Sequence[str],
    num_rollouts: int,
    device: torch.device,
) -> Dict[int, float]:
    horizon_hits: Dict[int, int] = {k: 0 for k in horizons}
    max_horizon = max(horizons)
    for _ in range(num_rollouts):
        extension = sample_rollout_until_k(model, context, max_horizon, device)
        for horizon in horizons:
            combined = context + extension[:horizon]
            if contains_any_pattern(combined, patterns):
                horizon_hits[horizon] += 1
    return {k: horizon_hits[k] / num_rollouts if num_rollouts else 0.0 for k in horizons}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate how often model rollouts emit machine-specific forbidden patterns."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained nanoGPT checkpoint.",
    )
    parser.add_argument(
        "--sequence-path",
        type=Path,
        required=True,
        help="Path to combined .dat sequence file.",
    )
    parser.add_argument(
        "--machine-ids-path",
        type=Path,
        required=True,
        help="Path to combined .machine_ids.dat file.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        help="Path to combined .meta.json file (required for auto witnesses).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum history length to evaluate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
        help="Maximum history length to evaluate.",
    )
    parser.add_argument(
        "--length-step",
        type=int,
        default=4,
        help="History length increment.",
    )
    parser.add_argument(
        "--rollout-horizons",
        type=str,
        default="2,4,6",
        help="Comma-separated list of rollout horizons k to evaluate.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=32,
        help="Monte Carlo rollouts per context.",
    )
    parser.add_argument(
        "--samples-per-length",
        type=int,
        default=100,
        help="Contexts sampled per history length.",
    )
    parser.add_argument(
        "--forbidden-pattern",
        type=str,
        default=None,
        help="Optional fallback forbidden substring to detect (e.g., '00').",
    )
    parser.add_argument(
        "--machine-patterns",
        type=str,
        default="",
        help="Comma-separated machine=pattern overrides (e.g., 'golden_mean=00,seven=11100').",
    )
    parser.add_argument(
        "--auto-witnesses",
        action="store_true",
        help="Automatically discover machine-unique witness substrings.",
    )
    parser.add_argument(
        "--auto-witness-max-length",
        type=int,
        default=12,
        help="Maximum substring length to consider when discovering witnesses.",
    )
    parser.add_argument(
        "--auto-witness-limit",
        type=int,
        default=256,
        help="Maximum number of witness substrings to keep per machine.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default autodetect).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional TSV to store aggregated frequencies.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path to save a heatmap of collapse frequencies.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sequence = load_sequence(args.sequence_path)
    machine_ids = load_machine_ids(args.machine_ids_path)
    if len(sequence) != len(machine_ids):
        raise ValueError("Sequence and machine ID files must have equal length")

    sequence_bits = "".join(str(bit) for bit in sequence)
    witness_map: Dict[str, List[str]] = {}
    machine_pattern_map: Dict[str, List[str]] = {}
    fallback_patterns: Optional[List[str]] = None

    if args.auto_witnesses:
        if args.meta_path is None:
            raise ValueError(
                "--meta-path is required when --auto-witnesses is enabled."
            )
        segments = load_segments(args.meta_path)
        witness_map = compute_unique_witnesses(
            sequence_bits,
            segments,
            args.auto_witness_max_length,
            args.auto_witness_limit,
        )
        missing = [m for m, patterns in witness_map.items() if not patterns]
        if missing:
            raise ValueError(
                f"No witness substrings found for machines: {', '.join(missing)}. "
                "Increase --auto-witness-max-length or inspect the dataset."
            )
    else:
        if args.machine_patterns:
            for entry in args.machine_patterns.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if "=" not in entry:
                    raise ValueError(
                        f"Invalid machine pattern '{entry}'. Expected machine=pattern."
                    )
                machine_name, pattern_str = entry.split("=", 1)
                machine_name = machine_name.strip()
                pattern_str = pattern_str.strip()
                if not machine_name or not pattern_str:
                    raise ValueError(
                        f"Invalid machine pattern '{entry}'. Expected machine=pattern."
                    )
                machine_pattern_map[machine_name] = [pattern_str]

        fallback_patterns = (
            [args.forbidden_pattern] if args.forbidden_pattern else None
        )
        if fallback_patterns is None and not machine_pattern_map:
            raise ValueError(
                "Provide --forbidden-pattern, --machine-patterns, "
                "or enable --auto-witnesses."
            )
    model = load_model(args.checkpoint, device)
    rng = random.Random(args.seed)

    horizons = sorted({int(k) for k in args.rollout_horizons.split(",") if k.strip()})
    lengths = list(range(args.min_length, args.max_length + 1, args.length_step))
    machines = sorted(set(machine_ids))
    rows: List[Tuple[int, str, int, float]] = []

    print("L\tmachine\tk\twitness_freq\tsampled_contexts", flush=True)
    for history_len in lengths:
        contexts, labels = sample_contexts(
            sequence, machine_ids, history_len, args.samples_per_length, rng
        )
        if not contexts:
            for machine in machines:
                for horizon in horizons:
                    print(
                        f"{history_len}\t{machine}\t{horizon}\t0.0\t0", flush=True
                    )
                    rows.append((history_len, machine, horizon, 0.0))
            continue

        per_machine: Dict[str, Dict[int, List[float]]] = {
            machine: {k: [] for k in horizons} for machine in machines
        }
        for context, label in zip(contexts, labels):
            if args.auto_witnesses:
                pattern_list = witness_map.get(label)
            else:
                pattern_list = machine_pattern_map.get(label, fallback_patterns)
            if not pattern_list:
                raise ValueError(
                    f"No witness patterns available for machine '{label}'."
                )
            freqs = evaluate_context_multi_k(
                model,
                context,
                horizons,
                pattern_list,
                args.num_rollouts,
                device,
            )
            for horizon, freq in freqs.items():
                per_machine[label][horizon].append(freq)

        for machine in machines:
            for horizon in horizons:
                values = per_machine[machine][horizon]
                mean_freq = float(np.mean(values)) if values else 0.0
                print(
                    f"{history_len}\t{machine}\t{horizon}\t{mean_freq:.4f}\t{len(values)}",
                    flush=True,
                )
                rows.append((history_len, machine, horizon, mean_freq))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            fh.write("L\tmachine\tk\twitness_freq\n")
            for history_len, machine, horizon, freq in rows:
                fh.write(f"{history_len}\t{machine}\t{horizon}\t{freq:.6f}\n")

    if args.plot_output:
        fig, axes = plt.subplots(
            len(machines), 1, figsize=(8, 4 * len(machines)), sharex=True
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, machine in zip(axes, machines):
            heatmap = np.zeros((len(horizons), len(lengths)))
            for i, horizon in enumerate(horizons):
                for j, history_len in enumerate(lengths):
                    matches = [
                        freq
                        for L, m, k, freq in rows
                        if L == history_len and m == machine and k == horizon
                    ]
                    heatmap[i, j] = matches[0] if matches else 0.0
            im = ax.imshow(
                heatmap,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                extent=[min(lengths), max(lengths), min(horizons), max(horizons)],
            )
            ax.set_title(f"{machine} witness frequency")
            ax.set_ylabel("Rollout horizon k")
            fig.colorbar(im, ax=ax)
        axes[-1].set_xlabel("History length L")
        fig.tight_layout()
        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_output)
        plt.close(fig)


if __name__ == "__main__":
    main()
