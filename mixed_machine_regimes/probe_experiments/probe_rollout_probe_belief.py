import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from probe_machine_sync import (  # type: ignore
    encode_contexts,
    extract_last_hidden_batch,
    load_machine_ids,
    load_model,
    load_sequence,
    sample_contexts,
)


def train_probe(
    model,
    sequence: Sequence[int],
    machine_ids: Sequence[str],
    train_min_length: int,
    train_max_length: int,
    samples_per_length: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> Tuple[LogisticRegression, Dict[int, str]]:
    rng = random.Random(seed)
    machines = sorted(set(machine_ids))
    machine_to_idx: Dict[str, int] = {m: i for i, m in enumerate(machines)}

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    for history_len in range(train_min_length, train_max_length + 1):
        contexts, labels = sample_contexts(
            sequence, machine_ids, history_len, samples_per_length, rng
        )
        if not contexts:
            continue
        features_len = encode_contexts(model, contexts, batch_size, device)
        all_features.append(features_len)
        all_labels.extend([machine_to_idx[label] for label in labels])

    if not all_features:
        raise ValueError("No contexts sampled for probe training.")

    features = np.concatenate(all_features, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="auto")
    clf.fit(features, y)
    idx_to_machine: Dict[int, str] = {i: m for m, i in machine_to_idx.items()}
    return clf, idx_to_machine


def rollout_sequence(length: int) -> List[int]:
    # Ambiguous alternating pattern 1,0,1,0,...
    return [1 if i % 2 == 0 else 0 for i in range(length)]


def probe_belief_along_rollout(
    model,
    clf: LogisticRegression,
    idx_to_machine: Dict[int, str],
    rollout_len: int,
    device: torch.device,
) -> Tuple[List[int], np.ndarray]:
    seq = rollout_sequence(rollout_len)
    probs_over_time: List[np.ndarray] = []

    for t in range(1, rollout_len + 1):
        prefix = [seq[i] for i in range(t)]
        batch_tokens = np.array(prefix, dtype=np.int64)[None, :]
        hidden = extract_last_hidden_batch(model, batch_tokens, device)
        proba = clf.predict_proba(hidden)[0]
        probs_over_time.append(proba)

    return seq, np.stack(probs_over_time, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track probe-estimated machine belief along a rollout."
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
        "--train-min-length",
        type=int,
        default=1,
        help="Minimum history length for probe training.",
    )
    parser.add_argument(
        "--train-max-length",
        type=int,
        default=17,
        help="Maximum history length for probe training.",
    )
    parser.add_argument(
        "--train-samples-per-length",
        type=int,
        default=500,
        help="Number of contexts per length for probe training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size when encoding contexts.",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=64,
        help="Length of ambiguous rollout sequence to probe.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default uses cuda if available).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional TSV to save belief over time.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path to save belief vs. rollout step plot.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sequence = load_sequence(args.sequence_path)
    machine_ids = load_machine_ids(args.machine_ids_path)
    if len(sequence) != len(machine_ids):
        raise ValueError("Sequence and machine ID files have mismatched lengths")

    model = load_model(args.checkpoint, device)
    clf, idx_to_machine = train_probe(
        model,
        sequence,
        machine_ids,
        args.train_min_length,
        args.train_max_length,
        args.train_samples_per_length,
        args.batch_size,
        args.seed,
        device,
    )

    rollout_seq, belief_probs = probe_belief_along_rollout(
        model, clf, idx_to_machine, args.rollout_len, device
    )

    machines = [idx_to_machine[i] for i in sorted(idx_to_machine.keys())]
    print("t\tcontext\t" + "\t".join(f"P({m})" for m in machines), flush=True)
    for t in range(1, args.rollout_len + 1):
        context = "".join(str(x) for x in rollout_seq[:t])
        if len(context) > 20:
            context_str = "..." + context[-17:]
        else:
            context_str = context
        probs_t = belief_probs[t - 1]
        probs_str = "\t".join(f"{p:.4f}" for p in probs_t)
        print(f"{t}\t{context_str}\t{probs_str}", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            fh.write("t\tcontext\t" + "\t".join(machines) + "\n")
            for t in range(1, args.rollout_len + 1):
                context = "".join(str(x) for x in rollout_seq[:t])
                probs_t = belief_probs[t - 1]
                probs_str = "\t".join(f"{p:.6f}" for p in probs_t)
                fh.write(f"{t}\t{context}\t{probs_str}\n")

    if args.plot_output:
        steps = np.arange(1, args.rollout_len + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, m in enumerate(machines):
            ax.plot(steps, belief_probs[:, i], label=m)
        ax.set_xlabel("Rollout step t")
        ax.set_ylabel("Probe-estimated P(machine | context)")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Machine belief along ambiguous rollout")
        ax.legend()
        fig.tight_layout()
        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_output)
        plt.close(fig)


if __name__ == "__main__":
    main()
