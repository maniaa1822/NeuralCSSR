#!/usr/bin/env python3
"""Linear probes on nanoGPT hidden states for mixed-machine belief analysis.

This script:
- Loads a trained nanoGPT checkpoint.
- Extracts last-token hidden representations for contexts from a binary .dat sequence.
- Uses ground-truth `machine:state` labels (from *.state_ids.dat) to train simple
  linear classifiers for:
    - generator id (machine)
    - generator-state (machine:state)

Usage (example for gm_seven_switch):

  uv run python cssr_discovery/probe_belief_states.py \\
    --model_ckpt nanoGPT/out-gm-seven-switch-char/ckpt.pt \\
    --data experiments/datasets/gm_seven_switch/combined.dat \\
    --state_ids_dat experiments/datasets/gm_seven_switch/combined.state_ids.dat \\
    --context_len 64 \\
    --max_samples 20000 \\
    --output_json results/gm_seven_switch_probes.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from two_stage_oracle_cssr import _load_nano_gpt_model, load_binary_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probes on nanoGPT hidden states.")
    parser.add_argument("--model_ckpt", required=True, help="Path to nanoGPT checkpoint.")
    parser.add_argument("--data", required=True, help="Path to binary .dat sequence.")
    parser.add_argument("--state_ids_dat", required=True, help="Path to parallel machine:state labels.")
    parser.add_argument("--context_len", type=int, default=64, help="Context length for probe inputs.")
    parser.add_argument("--max_samples", type=int, default=20000, help="Max number of probe samples.")
    parser.add_argument("--batch_size", type=int, default=256, help="Probe training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Probe training epochs.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for probes (auto, cpu, cuda). Model is always loaded on same device.",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        help="Optional metadata JSON (e.g., combined.meta.json) for constructing sync-status labels.",
    )
    parser.add_argument(
        "--sync_window",
        type=int,
        default=None,
        help="Number of steps after a segment start treated as pre-sync (default: context_len).",
    )
    parser.add_argument("--output_json", help="Optional path to write probe metrics JSON.")
    return parser.parse_args()


@torch.no_grad()
def extract_hidden_representations(
    model,
    data_tokens: np.ndarray,
    context_len: int,
    positions: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Extract last-token hidden representations for selected positions."""

    hidden_list: List[torch.Tensor] = []

    # Forward hook on ln_f output
    captured: Dict[str, torch.Tensor] = {}

    def hook_ln_f(module, inp, out):
        captured["ln_f"] = out

    handle = model.transformer["ln_f"].register_forward_hook(hook_ln_f)

    model.eval()
    for start in range(0, len(positions), 1024):
        batch_pos = positions[start : start + 1024]
        if len(batch_pos) == 0:
            continue
        ctx_arr = []
        for p in batch_pos:
            left = int(p) - context_len + 1
            if left < 0:
                left = 0
            ctx = data_tokens[left : int(p) + 1]
            if len(ctx) < context_len:
                pad = np.zeros(context_len - len(ctx), dtype=np.int64)
                ctx = np.concatenate([pad, ctx])
            ctx_arr.append(ctx)
        ctx_np = np.stack(ctx_arr, axis=0)
        idx = torch.from_numpy(ctx_np).to(device=device, dtype=torch.long)
        captured.clear()
        _ = model(idx, targets=None)
        x = captured["ln_f"]  # (B, T, C)
        last_hidden = x[:, -1, :].detach().cpu()
        hidden_list.append(last_hidden)

    handle.remove()
    return torch.cat(hidden_list, dim=0)


def build_probe_dataset(
    tokens: np.ndarray,
    labels: List[str],
    segments: List[dict],
    context_len: int,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    """Prepare positions and label indices for probing (machine, state, sync)."""
    n = len(tokens)
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != data length {n}")

    # positions where we have full context_len (we will pad at left if needed)
    valid_positions = np.arange(context_len - 1, n)
    if len(valid_positions) > max_samples:
        rng = np.random.default_rng(0)
        valid_positions = rng.choice(valid_positions, size=max_samples, replace=False)
        valid_positions.sort()

    machine_labels = []
    state_labels = []
    sync_labels = []

    def segment_for_pos(pos: int) -> dict:
        for seg in segments:
            if seg["start"] <= pos < seg["end"]:
                return seg
        return {}
    for pos in valid_positions:
        pos_int = int(pos)
        lab = labels[pos_int]
        if ":" in lab:
            machine, state = lab.split(":", 1)
        else:
            machine, state = lab, lab
        machine_labels.append(machine)
        state_labels.append(f"{machine}:{state}")
        seg = segment_for_pos(pos_int)
        offset = pos_int - seg.get("start", 0) if seg else 0
        sync_labels.append(offset)

    # build vocabularies
    machines = sorted(set(machine_labels))
    machine_to_id = {m: i for i, m in enumerate(machines)}
    states = sorted(set(state_labels))
    state_to_id = {s: i for i, s in enumerate(states)}

    y_machine = np.array([machine_to_id[m] for m in machine_labels], dtype=np.int64)
    y_state = np.array([state_to_id[s] for s in state_labels], dtype=np.int64)
    y_sync_raw = np.array(sync_labels, dtype=np.int64)

    return valid_positions, y_machine, y_state, y_sync_raw, machine_to_id, state_to_id


def train_linear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> Dict[str, float]:
    """Train a simple linear classifier and report accuracy."""
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_dim = x.size(1)
    model = nn.Linear(in_dim, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=-1).cpu()
    acc = (preds == y).float().mean().item()
    return {"accuracy": acc, "num_classes": num_classes}


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    # Load data and labels
    data_tokens = np.array([int(c) for c in load_binary_string(Path(args.data))], dtype=np.int64)
    label_strs = Path(args.state_ids_dat).read_text().split()

    segments: List[dict] = []
    if args.metadata_json:
        meta_path = Path(args.metadata_json)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            segments = meta.get("segments", [])

    # Load model
    model, block_size = _load_nano_gpt_model(Path(args.model_ckpt), device)
    context_len = min(args.context_len, int(block_size))

    positions, y_machine_np, y_state_np, y_sync_raw_np, machine_to_id, state_to_id = build_probe_dataset(
        data_tokens,
        label_strs,
        segments=segments,
        context_len=context_len,
        max_samples=args.max_samples,
    )
    print(f"Probe dataset: {len(positions)} samples, context_len={context_len}")

    # Extract hidden representations
    h = extract_hidden_representations(model, data_tokens, context_len, positions, device)
    print(f"Hidden reps shape: {h.shape}")

    # Train probes
    y_machine = torch.from_numpy(y_machine_np)
    y_state = torch.from_numpy(y_state_np)
    sync_window = args.sync_window or context_len
    y_sync_np = (y_sync_raw_np >= sync_window).astype(np.int64)
    y_sync = torch.from_numpy(y_sync_np)

    machine_probe_stats = train_linear_probe(
        h, y_machine, num_classes=len(machine_to_id), batch_size=args.batch_size, epochs=args.epochs, device=device
    )
    state_probe_stats = train_linear_probe(
        h, y_state, num_classes=len(state_to_id), batch_size=args.batch_size, epochs=args.epochs, device=device
    )

    print(f"Machine-id probe accuracy: {machine_probe_stats['accuracy']:.4f}")
    print(f"State probe accuracy:      {state_probe_stats['accuracy']:.4f}")
    sync_probe_stats = None
    if len(segments) > 0:
        sync_probe_stats = train_linear_probe(
            h, y_sync, num_classes=2, batch_size=args.batch_size, epochs=args.epochs, device=device
        )
        print(f"Sync-status probe accuracy: {sync_probe_stats['accuracy']:.4f}")

    if args.output_json:
        payload = {
            "model_ckpt": args.model_ckpt,
            "data": args.data,
            "state_ids_dat": args.state_ids_dat,
            "context_len": context_len,
            "max_samples": int(len(positions)),
            "machine_vocab": machine_to_id,
            "state_vocab": state_to_id,
            "machine_probe": machine_probe_stats,
            "state_probe": state_probe_stats,
            "sync_window": sync_window,
            "sync_probe": sync_probe_stats,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2))
        print(f"Wrote probe metrics to {args.output_json}")


if __name__ == "__main__":
    main()
