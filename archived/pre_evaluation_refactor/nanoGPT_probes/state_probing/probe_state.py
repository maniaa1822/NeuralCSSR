#!/usr/bin/env python3
"""Probe GPT layer representations for machine state prediction.

This script trains linear and MLP probes on hidden representations extracted
from every transformer layer (including the token embedding layer) of a
baseline nanoGPT checkpoint. The probes are trained to predict the current
state of the underlying machine on IID sequences sampled from the same
distribution used during training.

The workflow is:
1. Load the specified baseline GPT checkpoint.
2. Generate IID sequences and epsilon-state labels from the requested machine.
3. Collect per-layer activations for every token in the dataset.
4. Train both linear and non-linear probes per layer.
5. Report evaluation metrics and optionally save them to CSV.

Run with ``uv`` from the project root, for example::

    # Baseline checkpoint (IID data)
    uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_state.py \
        --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
        --dataset seven_state_human_mtl100k \
        --machine seven_state_human \
        --tokens 65536 \
        --block-size 64 \
        --batch-size 64 \
        --epochs 5 \
        --mlp-hidden 256 \
        --output runs/probes/state_probe_baseline.csv

    # Multitask eps+dist checkpoint (IID data)
    uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_state.py \
        --model mtl=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
        --dataset seven_state_human_mtl100k \
        --machine seven_state_human \
        --tokens 65536 \
        --block-size 64 \
        --batch-size 64 \
        --epochs 5 \
        --mlp-hidden 256 \
        --output runs/probes/state_probe_epsdist.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machines import get_machine
from machines.base import Machine

from nanoGPT.mtl.eval_ood import (
    OODGenerator,
    build_eval_tensors,
    compute_state_distances,
    load_model,
    parse_model_spec,
)


def load_meta(dataset: str, repo_root: Path) -> Tuple[Dict[str, int], int]:
    """Load dataset metadata from meta.pkl."""

    meta_path = repo_root / "nanoGPT" / "data" / dataset / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {meta_path}")
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    stoi = meta["stoi"]
    vocab_size = int(meta["vocab_size"])
    return stoi, vocab_size


@dataclass
class ProbeConfig:
    """Configuration parameters for probe training."""

    epochs: int
    batch_size: int
    learning_rate: float
    mlp_hidden: int
    val_fraction: float
    seed: int


@dataclass
class ProbeMetrics:
    """Evaluation metrics for a single probe."""

    loss: float
    accuracy: float
    samples: int


def set_random_seed(seed: int) -> None:
    """Seed torch and NumPy RNGs for reproducibility."""

    torch.manual_seed(seed)
    np.random.seed(seed)


def load_iid_data(
    machine: Machine,
    stoi: Dict[str, int],
    block_size: int,
    num_tokens: int,
    distance_info,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate IID sequences and epsilon-state labels matching training data."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    num_sequences = max(1, num_tokens // block_size)
    generator = OODGenerator(machine, regime="emission_mix", severity=0.0, seed=seed)

    x, _, eps, _ = build_eval_tensors(
        generator,
        num_sequences,
        block_size,
        stoi,
        distance_info,
    )
    return x, eps


def create_split_indices(num_samples: int, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return train and validation index tensors."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    val_fraction = max(0.0, min(val_fraction, 0.5))
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    val_size = int(round(num_samples * val_fraction))
    val_size = max(1, val_size) if val_fraction > 0.0 else 0

    if val_size == 0:
        train_idx = permutation
        val_idx = permutation
    else:
        val_idx = permutation[:val_size]
        train_idx = permutation[val_size:]
        if train_idx.numel() == 0:
            train_idx = val_idx
    return train_idx, val_idx


def make_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a DataLoader for probe training or evaluation."""

    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_probe(probe: nn.Module, loader: DataLoader, device: torch.device) -> ProbeMetrics:
    """Compute loss and accuracy for a probe."""

    criterion = nn.CrossEntropyLoss()
    probe.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = probe(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == targets).sum().item())
            total_samples += targets.size(0)

    if total_samples == 0:
        return ProbeMetrics(loss=float("nan"), accuracy=float("nan"), samples=0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return ProbeMetrics(loss=avg_loss, accuracy=accuracy, samples=total_samples)


def train_probe(
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float,
) -> ProbeMetrics:
    """Train a probe and return validation metrics."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    probe.to(device)

    for _ in range(max(1, epochs)):
        probe.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = probe(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    return evaluate_probe(probe, val_loader, device)


def collect_layer_activations(
    model_core: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """Collect per-layer activations for all tokens in ``loader``."""

    layer_names = ["embedding"] + [f"layer_{idx}" for idx, _ in enumerate(model_core.transformer.h)] + ["ln_f"]
    storage: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    captured: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[name] = output.detach()

        return hook

    handles = []
    handles.append(model_core.transformer.drop.register_forward_hook(make_hook("embedding")))
    for idx, block in enumerate(model_core.transformer.h):
        handles.append(block.register_forward_hook(make_hook(f"layer_{idx}")))
    handles.append(model_core.transformer.ln_f.register_forward_hook(make_hook("ln_f")))

    model_core.eval()
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            captured.clear()
            model_core(inputs)
            for name in layer_names:
                if name not in captured:
                    raise RuntimeError(f"Activation for {name} not captured")
                storage[name].append(captured[name].cpu())

    for handle in handles:
        handle.remove()

    activations: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in layer_names:
        tensors = storage[name]
        if not tensors:
            continue
        stacked = torch.cat([
            tensor.reshape(tensor.size(0) * tensor.size(1), tensor.size(2))
            for tensor in tensors
        ], dim=0)
        activations[name] = stacked.float()

    return activations


def run_probes(
    activations: OrderedDict[str, torch.Tensor],
    labels: torch.Tensor,
    config: ProbeConfig,
    num_states: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Train probes for each layer and return metric records."""

    total_tokens = labels.numel()
    train_idx, val_idx = create_split_indices(total_tokens, config.val_fraction, config.seed)

    records: List[Dict[str, object]] = []

    for layer_idx, (layer_name, features) in enumerate(activations.items()):
        layer_features = features
        if layer_features.size(0) != total_tokens:
            raise ValueError(
                f"Mismatch between features ({layer_features.size(0)}) and labels ({total_tokens}) for {layer_name}"
            )

        train_features = layer_features.index_select(0, train_idx).contiguous()
        train_labels = labels.index_select(0, train_idx).contiguous()
        val_features = layer_features.index_select(0, val_idx).contiguous()
        val_labels = labels.index_select(0, val_idx).contiguous()

        train_loader = make_loader(train_features, train_labels, config.batch_size, shuffle=True)
        val_loader = make_loader(val_features, val_labels, config.batch_size, shuffle=False)

        input_dim = layer_features.size(1)

        linear_probe = nn.Linear(input_dim, num_states)
        linear_metrics = train_probe(
            linear_probe,
            train_loader,
            val_loader,
            config.epochs,
            device,
            config.learning_rate,
        )
        records.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "probe_type": "linear",
                "hidden_dim": "",
                "loss": linear_metrics.loss,
                "accuracy": linear_metrics.accuracy,
                "num_tokens": linear_metrics.samples,
            }
        )

        mlp_probe = nn.Sequential(
            nn.Linear(input_dim, config.mlp_hidden),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden, num_states),
        )
        mlp_metrics = train_probe(
            mlp_probe,
            train_loader,
            val_loader,
            config.epochs,
            device,
            config.learning_rate,
        )
        records.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "probe_type": "mlp",
                "hidden_dim": config.mlp_hidden,
                "loss": mlp_metrics.loss,
                "accuracy": mlp_metrics.accuracy,
                "num_tokens": mlp_metrics.samples,
            }
        )

    return records


def fmt_float(value: float) -> str:
    """Format floating-point numbers for console output."""

    if not isinstance(value, float) or np.isnan(value):
        return "NA"
    return f"{value:.6f}"


def print_records(records: Sequence[Dict[str, object]]) -> None:
    """Print probe metrics in a tab-separated table."""

    header = ["layer_idx", "layer_name", "probe_type", "hidden_dim", "loss", "accuracy", "num_tokens"]
    print("\t".join(header))
    for record in records:
        values = [
            str(record["layer_idx"]),
            str(record["layer_name"]),
            str(record["probe_type"]),
            str(record.get("hidden_dim", "")),
            fmt_float(record.get("loss", float("nan"))),
            fmt_float(record.get("accuracy", float("nan"))),
            str(record.get("num_tokens", "")),
        ]
        print("\t".join(values))


def write_csv(records: Sequence[Dict[str, object]], output_path: Path) -> None:
    """Write probe metrics to ``output_path`` as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer_idx", "layer_name", "probe_type", "hidden_dim", "loss", "accuracy", "num_tokens"]
    with output_path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model spec in the form name=path/to/ckpt.pt")
    parser.add_argument("--dataset", default="seven_state_human_mtl100k", help="Dataset name for vocab metadata")
    parser.add_argument("--machine", default="seven_state_human", help="Machine name for state labels")
    parser.add_argument("--tokens", type=int, default=65536, help="Total number of tokens to sample for probing")
    parser.add_argument("--block-size", type=int, default=64, help="Sequence length for probing windows")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for activation collection and probes")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for each probe")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for probe optimization")
    parser.add_argument("--mlp-hidden", type=int, default=256, help="Hidden dimension for the MLP probe")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of tokens reserved for validation")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (cpu or cuda:0, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits and sampling")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_random_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    stoi, vocab_size = load_meta(args.dataset, REPO_ROOT)
    machine = get_machine(args.machine)
    distance_info = compute_state_distances(machine)

    model_name, model_path = parse_model_spec(args.model)
    ckpt_path = model_path if model_path.is_absolute() else (REPO_ROOT / model_path)

    handle = load_model(model_name, ckpt_path, device, vocab_size, distance_info)

    # Support both baseline GPT and MultitaskGPT by extracting the GPT core
    model_core = handle.model.gpt if hasattr(handle.model, "gpt") else handle.model
    model_core.eval()

    block_size = handle.block_size or args.block_size
    if block_size != args.block_size:
        print(f"Using block size {block_size} inferred from checkpoint (overriding --block-size {args.block_size})")

    x, eps = load_iid_data(machine, stoi, block_size, args.tokens, distance_info, args.seed)

    token_dataset = TensorDataset(x)
    token_loader = DataLoader(token_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    activations = collect_layer_activations(model_core, token_loader, device)
    labels = eps.reshape(-1).to(torch.long)

    if labels.numel() == 0:
        raise ValueError("No labels generated for probing")

    config = ProbeConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mlp_hidden=args.mlp_hidden,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    records = run_probes(activations, labels, config, num_states=len(machine.states), device=device)

    print_records(records)
    if args.output is not None:
        write_csv(records, args.output if args.output.is_absolute() else REPO_ROOT / args.output)


if __name__ == "__main__":
    main()

