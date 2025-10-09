"""Unified data generation for IID and OOD sequences.

Extracted and adapted from nanoGPT/mtl/eval_ood.py.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from machines.base import Machine
from .models import DistanceInfo


class OODGenerator:
    """Generate token/state traces under an OOD regime."""

    def __init__(
        self,
        machine: Machine,
        regime: str,
        severity: float,
        seed: int,
        mixed_regimes: Optional[List[str]] = None,
        mixed_severities: Optional[List[float]] = None,
    ):
        self.machine = machine
        self.regime = regime
        self.severity = severity
        self.rng = np.random.default_rng(seed)
        self.states = machine.states
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.alphabet = machine.alphabet
        self.mixed_regimes = mixed_regimes or []
        self.mixed_severities = mixed_severities or []
        self._state_swap_set: Optional[set] = None

        # Validate severity ranges
        if regime == "emission_mix" and not (0.0 <= severity <= 1.0):
            raise ValueError("emission_mix severity must be between 0 and 1")
        if regime == "transition_noise" and not (0.0 <= severity <= 1.0):
            raise ValueError("transition_noise severity must be between 0 and 1")
        if regime == "emission_bias" and not (0.0 <= severity <= 1.0):
            raise ValueError("emission_bias severity must be between 0 and 1")
        if regime == "temporal_swap" and not (0.0 <= severity <= 1.0):
            raise ValueError("temporal_swap severity must be between 0 and 1")
        if regime == "state_dependent_swap":
            if not (0.0 <= severity <= 1.0):
                raise ValueError("state_dependent_swap severity must be between 0 and 1")
            n_states = len(self.states)
            n_swap = max(1, int(round(severity * n_states))) if n_states > 0 else 0
            chosen = self.rng.choice(self.states, size=n_swap, replace=False) if n_swap > 0 else []
            self._state_swap_set = set(chosen)
        if regime == "mixed_regime":
            if not mixed_regimes:
                raise ValueError("mixed_regime requires mixed_regimes parameter")
            if not mixed_severities or len(mixed_severities) != len(mixed_regimes):
                raise ValueError("mixed_regime requires mixed_severities with same length")

    def generate_sequence(self, length: int) -> Tuple[List[str], List[int]]:
        """Generate symbols and epsilon-state indices of given length."""
        tokens: List[str] = []
        eps_states: List[int] = []
        state = self._initial_state()

        for _ in range(length):
            symbol, next_state = self._sample_next(state)
            tokens.append(symbol)
            eps_states.append(self.state_to_idx[next_state])
            state = next_state

        return tokens, eps_states

    def _initial_state(self) -> str:
        if self.regime == "start_state_uniform":
            return self.rng.choice(self.states)
        return self.machine.start_state

    def _sample_next(self, state: str) -> Tuple[str, str]:
        emission_probs = self.machine.emissions[state]
        base_probs = np.array([emission_probs[sym] for sym in self.alphabet], dtype=np.float64)

        # Apply emission perturbations
        if self.regime == "mixed_regime":
            probs = base_probs.copy()
            for reg, sev in zip(self.mixed_regimes, self.mixed_severities):
                probs = self._apply_emission_perturbation(probs, reg, sev)
        else:
            probs = self._apply_emission_perturbation(base_probs, self.regime, self.severity)

        # State-dependent swap
        if self.regime == "state_dependent_swap" and self._state_swap_set is not None:
            if state in self._state_swap_set:
                probs = probs[::-1]
        # Temporal swap
        if self.regime == "temporal_swap" and self.rng.random() < self.severity:
            probs = probs[::-1]

        probs = probs / probs.sum()
        idx = self.rng.choice(len(self.alphabet), p=probs)
        symbol = self.alphabet[idx]

        next_state = self.machine.transitions.get((state, symbol))
        if next_state is None:
            raise RuntimeError(f"Transition undefined for state={state}, symbol={symbol}")

        # Apply transition perturbations
        if self.regime == "transition_noise" and self.rng.random() < self.severity:
            next_state = self.rng.choice(self.states)
        elif self.regime == "mixed_regime":
            for reg, sev in zip(self.mixed_regimes, self.mixed_severities):
                if reg == "transition_noise" and self.rng.random() < sev:
                    next_state = self.rng.choice(self.states)

        return symbol, next_state

    def _apply_emission_perturbation(self, probs: np.ndarray, regime: str, severity: float) -> np.ndarray:
        """Apply a single emission-based perturbation to probability distribution."""
        if regime == "emission_mix":
            uniform = np.full_like(probs, 1.0 / len(self.alphabet))
            return (1.0 - severity) * probs + severity * uniform
        elif regime == "alphabet_swap":
            return probs[::-1]
        elif regime == "emission_bias":
            # Push probabilities toward extremes (0.9 or 0.1)
            biased = np.zeros_like(probs)
            for i, p in enumerate(probs):
                target = 0.9 if p > 0.5 else 0.1
                biased[i] = (1.0 - severity) * p + severity * target
            return biased
        else:
            return probs


def build_eval_tensors(
    generator: OODGenerator,
    num_sequences: int,
    block_size: int,
    stoi: Dict[str, int],
    distance_info: DistanceInfo,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate non-overlapping evaluation blocks.

    Returns:
        x: Input token IDs [num_sequences, block_size]
        y: Target token IDs [num_sequences, block_size]
        eps: Epsilon state indices [num_sequences, block_size]
        dist: Distance matrices [num_sequences, block_size, num_states]
    """
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    eps_list: List[np.ndarray] = []
    dist_list: List[np.ndarray] = []

    for _ in range(num_sequences):
        tokens, eps_states = generator.generate_sequence(block_size + 1)
        token_ids = np.array([stoi[sym] for sym in tokens], dtype=np.int64)
        eps_arr = np.array(eps_states[:-1], dtype=np.int64)
        x_list.append(token_ids[:-1])
        y_list.append(token_ids[1:])
        eps_list.append(eps_arr)
        dist_list.append(distance_info.matrix[eps_arr].numpy())

    x = torch.tensor(np.stack(x_list), dtype=torch.long)
    y = torch.tensor(np.stack(y_list), dtype=torch.long)
    eps = torch.tensor(np.stack(eps_list), dtype=torch.long)
    dist = torch.tensor(np.stack(dist_list), dtype=torch.long)

    return x, y, eps, dist


def generate_iid_data(
    machine: Machine,
    stoi: Dict[str, int],
    block_size: int,
    num_tokens: int,
    distance_info: DistanceInfo,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate IID sequences matching the training distribution.

    Args:
        machine: Epsilon machine specification
        stoi: String-to-index vocabulary mapping
        block_size: Sequence length per block
        num_tokens: Total number of tokens to generate
        distance_info: State distance information
        seed: Random seed

    Returns:
        x, y, eps, dist tensors (same as build_eval_tensors)
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    num_sequences = max(1, num_tokens // block_size)
    generator = OODGenerator(machine, regime="emission_mix", severity=0.0, seed=seed)

    return build_eval_tensors(generator, num_sequences, block_size, stoi, distance_info)


def generate_ood_data(
    machine: Machine,
    stoi: Dict[str, int],
    block_size: int,
    num_tokens: int,
    distance_info: DistanceInfo,
    regime: str,
    severity: float,
    seed: int,
    mixed_regimes: Optional[List[str]] = None,
    mixed_severities: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate OOD sequences under a specific regime.

    Args:
        machine: Epsilon machine specification
        stoi: String-to-index vocabulary mapping
        block_size: Sequence length per block
        num_tokens: Total number of tokens to generate
        distance_info: State distance information
        regime: OOD regime name (e.g., "emission_mix", "alphabet_swap")
        severity: Regime severity parameter
        seed: Random seed
        mixed_regimes: List of regime names for mixed_regime
        mixed_severities: List of severities for mixed_regime

    Returns:
        x, y, eps, dist tensors (same as build_eval_tensors)
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    num_sequences = max(1, num_tokens // block_size)
    generator = OODGenerator(
        machine, regime, severity, seed, mixed_regimes, mixed_severities
    )

    return build_eval_tensors(generator, num_sequences, block_size, stoi, distance_info)
