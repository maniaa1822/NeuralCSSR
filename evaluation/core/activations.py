"""Layer activation extraction for baseline and MTL models.

Extracted and adapted from nanoGPT/probes/state_probing/probe_state.py.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collect_layer_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """Collect per-layer activations for all tokens in loader.

    Works with both baseline GPT and MultitaskGPT models.
    For MultitaskGPT, extracts the core GPT model automatically.

    Args:
        model: GPT or MultitaskGPT model
        loader: DataLoader yielding (inputs,) tuples
        device: Device to run model on

    Returns:
        OrderedDict mapping layer names to activations [total_tokens, hidden_dim]
        Layer names: "embedding", "layer_0", "layer_1", ..., "ln_f"
    """
    # Extract core GPT model if this is a MultitaskGPT wrapper
    model_core = model.gpt if hasattr(model, "gpt") else model

    layer_names = [
        "embedding"
    ] + [f"layer_{idx}" for idx, _ in enumerate(model_core.transformer.h)] + ["ln_f"]
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

    # Stack activations: [batch, seq_len, hidden] -> [total_tokens, hidden]
    activations: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in layer_names:
        tensors = storage[name]
        if not tensors:
            continue
        stacked = torch.cat(
            [tensor.reshape(tensor.size(0) * tensor.size(1), tensor.size(2)) for tensor in tensors],
            dim=0,
        )
        activations[name] = stacked.float()

    return activations
