import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add nanoGPT to path for model loading
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


@torch.no_grad()
def extract_last_hidden_batch(
    model: GPT, batch_tokens: np.ndarray, device: torch.device
) -> np.ndarray:
    idx = torch.tensor(batch_tokens, dtype=torch.long, device=device)
    batch_size, seq_len = idx.shape
    if seq_len > model.config.block_size:
        raise ValueError(
            f"Sequence length {seq_len} exceeds block size {model.config.block_size}"
        )
    pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
    tok_emb = model.transformer.wte(idx)
    pos_emb = model.transformer.wpe(pos)
    x = model.transformer.drop(tok_emb + pos_emb)
    for block in model.transformer.h:
        x, _ = block(x)
    x = model.transformer.ln_f(x)
    hidden = x[:, -1, :].detach().cpu().numpy()
    return hidden


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


def encode_contexts(
    model: GPT,
    contexts: List[List[int]],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        batch_tokens = np.array(batch, dtype=np.int64)
        hidden = extract_last_hidden_batch(model, batch_tokens, device)
        features.append(hidden)
    return np.concatenate(features, axis=0) if features else np.zeros((0, 0))


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[float, float, int, int]:
    if features.size == 0:
        return 0.0, 0.0, 0, 0

    stratify = labels if len(np.unique(labels)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    proba = clf.predict_proba(X_test)
    true_prob = float(proba[np.arange(len(y_test)), y_test].mean()) if len(y_test) else 0.0
    return accuracy, true_prob, len(y_train), len(y_test)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit linear probes to predict machine IDs from model representations."
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
        "--min-length",
        type=int,
        default=1,
        help="Minimum history length to evaluate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum history length to evaluate.",
    )
    parser.add_argument(
        "--length-step",
        type=int,
        default=1,
        help="History length increment.",
    )
    parser.add_argument(
        "--samples-per-length",
        type=int,
        default=1000,
        help="Number of contexts sampled per history length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size when encoding contexts.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling and probe splits.",
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
        help="Optional TSV path to save probe accuracies.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path to save accuracy/belief plot.",
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
    rng = random.Random(args.seed)
    machines = sorted(set(machine_ids))
    machine_to_idx: Dict[str, int] = {m: i for i, m in enumerate(machines)}

    lengths = list(range(args.min_length, args.max_length + 1, args.length_step))
    rows = []

    print("L\taccuracy\ttrue_prob\ttrain_samples\ttest_samples\tnum_total_samples", flush=True)
    for history_len in lengths:
        contexts, labels = sample_contexts(
            sequence, machine_ids, history_len, args.samples_per_length, rng
        )
        if not contexts:
            print(f"{history_len}\t0.0\t0.0\t0\t0\t0", flush=True)
            rows.append((history_len, 0.0, 0.0, 0, 0, 0))
            continue
        features = encode_contexts(model, contexts, args.batch_size, device)
        y = np.array([machine_to_idx[label] for label in labels], dtype=np.int64)
        accuracy, true_prob, train_size, test_size = fit_probe(
            features, y, args.test_fraction, seed=args.seed
        )
        total_samples = train_size + test_size
        print(
            f"{history_len}\t{accuracy:.4f}\t{true_prob:.4f}\t{train_size}\t{test_size}\t{total_samples}",
            flush=True,
        )
        rows.append((history_len, accuracy, true_prob, train_size, test_size, total_samples))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            f.write("L\taccuracy\ttrue_prob\ttrain_samples\ttest_samples\ttotal_samples\n")
            for history_len, acc, true_prob, train_size, test_size, total in rows:
                f.write(
                    f"{history_len}\t{acc:.6f}\t{true_prob:.6f}\t{train_size}\t{test_size}\t{total}\n"
                )

    if args.plot_output:
        lengths_arr = np.array([row[0] for row in rows], dtype=float)
        acc_arr = np.array([row[1] for row in rows], dtype=float)
        prob_arr = np.array([row[2] for row in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lengths_arr, acc_arr, label="Probe accuracy", marker="o")
        ax.plot(lengths_arr, prob_arr, label="Mean true-class prob", marker="s")
        ax.set_xlabel("History length L")
        ax.set_ylabel("Probe performance")
        ax.set_title("Machine belief from linear probe")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        fig.tight_layout()
        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_output)
        plt.close(fig)


if __name__ == "__main__":
    main()
