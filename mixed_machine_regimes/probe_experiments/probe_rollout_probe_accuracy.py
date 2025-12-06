import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
) -> Tuple[LogisticRegression, Dict[str, int], Dict[int, str]]:
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
    return clf, machine_to_idx, idx_to_machine


@torch.no_grad()
def sample_next_token(model, seq: List[int], device: torch.device) -> int:
    idx = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    logits, _, _ = model(idx)
    probs = F.softmax(logits[0, -1], dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each history length L, roll out k steps and track "
            "linear-probe accuracy at each step."
        )
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
        default=400,
        help="Number of contexts per length for probe training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size when encoding contexts.",
    )
    parser.add_argument(
        "--max-history-length",
        type=int,
        default=10,
        help="Maximum history length L to evaluate.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=20,
        help="Number of rollout steps k to evaluate.",
    )
    parser.add_argument(
        "--samples-per-length",
        type=int,
        default=200,
        help="Number of starting contexts per history length L.",
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
        help="Optional TSV to save accuracy grid.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path to save accuracy heatmap.",
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
    clf, machine_to_idx, _ = train_probe(
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

    rng = random.Random(args.seed)
    max_L = args.max_history_length
    k = args.rollout_steps
    lengths = list(range(1, max_L + 1))

    correct = {L: [0] * (k + 1) for L in lengths}
    total = {L: [0] * (k + 1) for L in lengths}

    print("L\tstep\taccuracy\ttotal_samples", flush=True)
    for L in lengths:
        contexts, labels = sample_contexts(
            sequence, machine_ids, L, args.samples_per_length, rng
        )
        for ctx, label in zip(contexts, labels):
            label_idx = machine_to_idx[label]
            seq_ctx = ctx.copy()
            for step in range(k + 1):
                batch_tokens = np.array(seq_ctx, dtype=np.int64)[None, :]
                hidden = extract_last_hidden_batch(model, batch_tokens, device)
                proba = clf.predict_proba(hidden)[0]
                pred_idx = int(np.argmax(proba))
                total[L][step] += 1
                if pred_idx == label_idx:
                    correct[L][step] += 1
                if step < k:
                    next_token = sample_next_token(model, seq_ctx, device)
                    seq_ctx.append(next_token)

        for step in range(k + 1):
            if total[L][step] > 0:
                acc = correct[L][step] / total[L][step]
            else:
                acc = 0.0
            print(f"{L}\t{step}\t{acc:.4f}\t{total[L][step]}", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            fh.write("L\tstep\taccuracy\ttotal_samples\n")
            for L in lengths:
                for step in range(k + 1):
                    if total[L][step] > 0:
                        acc = correct[L][step] / total[L][step]
                    else:
                        acc = 0.0
                    fh.write(f"{L}\t{step}\t{acc:.6f}\t{total[L][step]}\n")

    if args.plot_output:
        heat = np.zeros((len(lengths), k + 1))
        for i, L in enumerate(lengths):
            for step in range(k + 1):
                heat[i, step] = (
                    correct[L][step] / total[L][step] if total[L][step] > 0 else 0.0
                )
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            heat,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            extent=[0, k, lengths[0], lengths[-1]],
        )
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("History length L")
        ax.set_title("Probe accuracy vs. history length and rollout step")
        fig.colorbar(im, ax=ax, label="Accuracy")
        fig.tight_layout()
        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_output)
        plt.close(fig)


if __name__ == "__main__":
    main()

