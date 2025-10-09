"""Unified model loading for baseline and MTL checkpoints.

Extracted and adapted from nanoGPT/mtl/eval_ood.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from machines.base import Machine


@dataclass
class DistanceInfo:
    """State distance matrix information for geometry head."""
    matrix: torch.Tensor  # [num_states, num_states] distance matrix
    num_classes: int      # Number of distance classes (including unreachable)
    unreachable_class: int  # Class index for unreachable states


@dataclass
class ModelHandle:
    """Container for a loaded model with metadata."""
    name: str
    model: nn.Module
    kind: str  # "baseline" or "mtl"
    tap_preference: Optional[str]  # Preferred tap layer for MTL models
    has_distance_head: bool
    block_size: int
    vocab_size: int


def compute_state_distances(machine: Machine) -> DistanceInfo:
    """Compute shortest-hop distances between states (used for geometry head)."""
    import numpy as np

    states = machine.states
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    alphabet = machine.alphabet
    transitions = machine.transitions

    n_states = len(states)
    distances = np.full((n_states, n_states), fill_value=-1, dtype=np.int16)

    for state_idx, state in enumerate(states):
        distances[state_idx, state_idx] = 0
        frontier = [state]
        visited = {state}
        depth = 0

        while frontier:
            depth += 1
            next_frontier = []
            for current in frontier:
                for symbol in alphabet:
                    nxt = transitions.get((current, symbol))
                    if nxt is None or nxt in visited:
                        continue
                    visited.add(nxt)
                    next_frontier.append(nxt)
                    tgt_idx = state_to_idx[nxt]
                    if distances[state_idx, tgt_idx] == -1 or depth < distances[state_idx, tgt_idx]:
                        distances[state_idx, tgt_idx] = depth
            frontier = next_frontier

    finite_mask = distances >= 0
    max_distance = int(distances[finite_mask].max()) if finite_mask.any() else 0
    unreachable_class = max_distance + 1
    distances[~finite_mask] = unreachable_class
    num_classes = unreachable_class + 1

    return DistanceInfo(
        matrix=torch.tensor(distances, dtype=torch.long),
        num_classes=num_classes,
        unreachable_class=unreachable_class
    )


def parse_model_spec(spec: str) -> Tuple[str, Path]:
    """Parse model spec in format 'name=path'."""
    if "=" not in spec:
        raise ValueError(f"Model spec must be name=path format (got '{spec}')")
    name, path = spec.split("=", 1)
    return name, Path(path)


def strip_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove 'module.' prefix from DDP checkpoints."""
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model(
    name: str,
    ckpt_path: Path,
    device: torch.device,
    vocab_size: int,
    distance_info: DistanceInfo,
) -> ModelHandle:
    """Load baseline or MTL checkpoint.

    Args:
        name: Model name for identification
        ckpt_path: Path to checkpoint file
        device: Device to load model on
        vocab_size: Expected vocabulary size
        distance_info: State distance information for MTL models

    Returns:
        ModelHandle with loaded model and metadata
    """
    # Import here to avoid circular dependencies
    from nanoGPT.model import GPT, GPTConfig
    from nanoGPT.mtl.model_wrapper import MultitaskConfig, MultitaskGPT

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    block_size = int(config.get("block_size", 0))
    n_layer = int(config.get("n_layer", 0))
    n_head = int(config.get("n_head", 0))
    n_embd = int(config.get("n_embd", 0))
    dropout = float(config.get("dropout", 0.0))
    bias = bool(config.get("bias", False))

    state_dict = strip_ddp_prefix(checkpoint["model"])

    # Infer vocab size from checkpoint if provided size doesn't match
    weight_key = None
    for candidate in ("gpt.transformer.wte.weight", "transformer.wte.weight"):
        if candidate in state_dict:
            weight_key = candidate
            break
    if weight_key is None:
        raise KeyError("Could not locate embedding weights in checkpoint")
    vocab_size_ckpt = state_dict[weight_key].shape[0]
    if vocab_size != vocab_size_ckpt:
        vocab_size = vocab_size_ckpt

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )

    # Check if this is a multitask checkpoint
    multitask_layers = config.get("multitask_tap_layers")
    if multitask_layers:
        tap_layers = list(multitask_layers)
        multitask_config = MultitaskConfig(
            tap_layers=tap_layers,
            bottleneck_dim=config.get("multitask_bottleneck_dim"),
            epsilon_classes=int(config.get("multitask_epsilon_classes", 0)),
            epsilon_loss_weight=float(config.get("multitask_epsilon_loss_weight", 1.0)),
            num_states=len(distance_info.matrix),
            distance_num_classes=distance_info.num_classes,
            distance_loss_weight=float(config.get("multitask_distance_loss_weight", 0.0)),
        )
        model = MultitaskGPT(gpt_config, multitask_config)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        distance_weight = float(config.get("multitask_distance_loss_weight", 0.0) or 0.0)
        has_distance = distance_weight > 0 and any(key.startswith("distance_heads") for key in state_dict)
        tap_preference = tap_layers[-1] if tap_layers else None

        return ModelHandle(
            name=name,
            model=model,
            kind="mtl",
            tap_preference=tap_preference,
            has_distance_head=has_distance,
            block_size=block_size,
            vocab_size=vocab_size,
        )

    # Baseline GPT model
    model = GPT(gpt_config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return ModelHandle(
        name=name,
        model=model,
        kind="baseline",
        tap_preference=None,
        has_distance_head=False,
        block_size=block_size,
        vocab_size=vocab_size,
    )
