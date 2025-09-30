from __future__ import annotations

from typing import Optional

import numpy as np

from .meta_distribution import MachineSpec


def _pad_emissions(emissions: np.ndarray, target_size: int) -> np.ndarray:
    padded = np.full(target_size, 0.5, dtype=np.float64)
    padded[: emissions.shape[0]] = emissions
    return padded


def _transition_tensor(spec: MachineSpec, target_states: int) -> np.ndarray:
    tensor = np.zeros((target_states, spec.alphabet_size, target_states), dtype=np.float64)
    for state in range(spec.num_states):
        for symbol in range(spec.alphabet_size):
            tensor[state, symbol, spec.transitions[state, symbol]] = 1.0
    return tensor


def machine_distance(spec_a: MachineSpec, spec_b: MachineSpec, *, alpha: float = 0.5) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if spec_a.alphabet_size != spec_b.alphabet_size:
        raise ValueError("alphabet sizes must match before comparing machines")

    target_states = max(spec_a.num_states, spec_b.num_states)

    emis_a = _pad_emissions(spec_a.emissions, target_states)
    emis_b = _pad_emissions(spec_b.emissions, target_states)
    d_e = float(np.sum(np.abs(emis_a - emis_b)))

    tensor_a = _transition_tensor(spec_a, target_states)
    tensor_b = _transition_tensor(spec_b, target_states)
    diff = np.abs(tensor_a - tensor_b).sum(axis=-1) * 0.5
    d_t = float(diff.sum() / (target_states * spec_a.alphabet_size))

    return alpha * d_e + (1.0 - alpha) * d_t
