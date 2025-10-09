#!/usr/bin/env python3
"""Probe GPT layer representations for per-state distance prediction.

This script trains linear and MLP probes on hidden representations extracted
from every transformer layer (including the token embedding layer) of a
baseline nanoGPT checkpoint. The probes are trained to predict, for each
position and each target state, the shortest-hop distance class from the
current state to the target state, mirroring the distance head used in
`nanoGPT/mtl/eval_ood.py`.

Loss is averaged over all (token, target_state) pairs; accuracy is the
fraction of correctly predicted distance classes across all pairs.

Run with ``uv`` from the project root, for example::

    # Baseline checkpoint (IID data)
    uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_distance.py \
        --model baseline=nanoGPT/out-seven-state-human-mtl100k-baseline/ckpt.pt \
        --dataset seven_state_human_mtl100k \
        --machine seven_state_human \
        --tokens 65536 \
        --block-size 64 \
        --batch-size 64 \
        --epochs 5 \
        --mlp-hidden 256 \
        --output runs/probes/state_probe_distance_baseline.csv

    # Multitask eps+dist checkpoint (IID data)
    uv run --with torch --with numpy python nanoGPT/probes/state_probing/probe_distance.py \
        --model mtl=out-seven-state-human-mtl100k-mtl-eps-dist/ckpt_final.pt \
        --dataset seven_state_human_mtl100k \
        --machine seven_state_human \
        --tokens 65536 \
        --block-size 64 \
        --batch-size 64 \
        --epochs 5 \
        --mlp-hidden 256 \
        --output runs/probes/state_probe_distance_epsdist.csv
"""

from __future__ import annotations

import argparse
import csv
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
    load_meta,
    load_model,
    parse_model_spec,
)


@dataclass
class ProbeConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    mlp_hidden: int
    val_fraction: float
    seed: int


@dataclass
class ProbeMetrics:
    loss: float
    accuracy: float
    samples: int


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_iid_data(
    machine: Machine,
    stoi: Dict[str, int],
    block_size: int,
    num_tokens: int,
    distance_info,
    seed: int,
):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    num_sequences = max(1, num_tokens // block_size)
    generator = OODGenerator(machine, regime="emission_mix", severity=0.0, seed=seed)
    x, _, eps, dist = build_eval_tensors(
        generator,
        num_sequences,
        block_size,
        stoi,
        distance_info,
    )
    return x, eps, dist


def create_split_indices(num_samples: int, val_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    val_fraction = max(0.0, min(val_fraction, 0.5))
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    val_size = int(round(num_samples * val_fraction))
    val_size = max(1, val_size) if val_fraction > 0.0 else 0
    if val_size == 0:
        return permutation, permutation
    val_idx = permutation[:val_size]
    train_idx = permutation[val_size:]
    if train_idx.numel() == 0:
        train_idx = val_idx
    return train_idx, val_idx


def make_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class LinearDistanceProbe(nn.Module):
    def __init__(self, input_dim: int, num_states: int, num_classes: int) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_classes = num_classes
        self.out = nn.Linear(input_dim, num_states * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.out(x)
        return logits.view(x.size(0), self.num_states, self.num_classes)


class MLPDistanceProbe(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_states: int, num_classes: int) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_states * num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(x.size(0), self.num_states, self.num_classes)


def evaluate_probe(probe: nn.Module, loader: DataLoader, device: torch.device) -> ProbeMetrics:
    criterion = nn.CrossEntropyLoss()
    probe.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)  # [B, S]
            logits = probe(features)      # [B, S, C]
            B, S, C = logits.shape
            loss = criterion(logits.view(B * S, C), targets.view(B * S))
            total_loss += float(loss.item()) * (B * S)
            preds = logits.argmax(dim=-1)  # [B, S]
            total_correct += int((preds == targets).sum().item())
            total_samples += (B * S)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    probe.to(device)
    for _ in range(max(1, epochs)):
        probe.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = probe(features)
            B, S, C = logits.shape
            loss = criterion(logits.view(B * S, C), targets.view(B * S))
            loss.backward()
            optimizer.step()
    return evaluate_probe(probe, val_loader, device)


def collect_layer_activations(
    model_core: nn.Module,
    token_loader: DataLoader,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    layer_names = ["embedding"] + [f"layer_{idx}" for idx, _ in enumerate(model_core.transformer.h)] + ["ln_f"]
    storage: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    captured: Dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
            captured[name] = output.detach()
        return hook

    handles = []
    handles.append(model_core.transformer.drop.register_forward_hook(make_hook("embedding")))
    for idx, block in enumerate(model_core.transformer.h):
        handles.append(block.register_forward_hook(make_hook(f"layer_{idx}")))
    handles.append(model_core.transformer.ln_f.register_forward_hook(make_hook("ln_f")))

    model_core.eval()
    with torch.no_grad():
        for (inputs,) in token_loader:
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


def print_records(records: Sequence[Dict[str, object]]) -> None:
    header = [
        "layer_idx",
        "layer_name",
        "probe_type",
        "hidden_dim",
        "distance_loss",
        "distance_accuracy",
        "num_pairs",
    ]
    print("\t".join(header))
    for record in records:
        def fmt(v):
            if isinstance(v, float) and not np.isnan(v):
                return f"{v:.6f}"
            return str(v)
        row = [
            record["layer_idx"],
            record["layer_name"],
            record["probe_type"],
            record.get("hidden_dim", ""),
            fmt(record.get("distance_loss", float("nan"))),
            fmt(record.get("distance_accuracy", float("nan"))),
            record.get("num_pairs", 0),
        ]
        print("\t".join(map(str, row)))


def write_csv(records: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_idx",
        "layer_name",
        "probe_type",
        "hidden_dim",
        "distance_loss",
        "distance_accuracy",
        "num_pairs",
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args() -> argparse.Namespace:
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
    model_core = handle.model.gpt if hasattr(handle.model, "gpt") else handle.model
    model_core.eval()

    block_size = handle.block_size or args.block_size
    if block_size != args.block_size:
        print(f"Using block size {block_size} inferred from checkpoint (overriding --block-size {args.block_size})")

    # Data: x [B, T], eps [B, T], dist [B, T, S]
    x, _eps, dist = load_iid_data(machine, stoi, block_size, args.tokens, distance_info, args.seed)

    # Activation capture uses token sequences
    token_dataset = TensorDataset(x)
    token_loader = DataLoader(token_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    activations = collect_layer_activations(model_core, token_loader, device)

    # Labels for distance: flatten to [N, S]
    B, T, S = dist.size(0), dist.size(1), dist.size(2)
    labels = dist.reshape(B * T, S).to(torch.long)

    total_tokens = labels.size(0)
    train_idx, val_idx = create_split_indices(total_tokens, args.val_fraction, args.seed)

    records: List[Dict[str, object]] = []
    for layer_idx, (layer_name, features) in enumerate(activations.items()):
        if features.size(0) != total_tokens:
            raise ValueError(f"Feature/label size mismatch at {layer_name}: {features.size(0)} vs {total_tokens}")
        train_features = features.index_select(0, train_idx).contiguous()
        val_features = features.index_select(0, val_idx).contiguous()
        train_labels = labels.index_select(0, train_idx).contiguous()
        val_labels = labels.index_select(0, val_idx).contiguous()

        train_loader = make_loader(train_features, train_labels, args.batch_size, shuffle=True)
        val_loader = make_loader(val_features, val_labels, args.batch_size, shuffle=False)

        input_dim = features.size(1)
        num_states = S
        num_classes = int(distance_info.num_classes)

        # Linear probe
        linear_probe = LinearDistanceProbe(input_dim, num_states, num_classes)
        linear_metrics = train_probe(linear_probe, train_loader, val_loader, args.epochs, device, args.learning_rate)
        records.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "probe_type": "linear",
                "hidden_dim": "",
                "distance_loss": linear_metrics.loss,
                "distance_accuracy": linear_metrics.accuracy,
                "num_pairs": linear_metrics.samples,
            }
        )

        # MLP probe
        mlp_probe = MLPDistanceProbe(input_dim, args.mlp_hidden, num_states, num_classes)
        mlp_metrics = train_probe(mlp_probe, train_loader, val_loader, args.epochs, device, args.learning_rate)
        records.append(
            {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "probe_type": "mlp",
                "hidden_dim": args.mlp_hidden,
                "distance_loss": mlp_metrics.loss,
                "distance_accuracy": mlp_metrics.accuracy,
                "num_pairs": mlp_metrics.samples,
            }
        )

    print_records(records)
    if args.output is not None:
        out_path = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
        write_csv(records, out_path)


if __name__ == "__main__":
    main()


