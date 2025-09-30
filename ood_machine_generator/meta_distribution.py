from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .splits import categorize_emission, categorize_state_count, categorize_topology


_GRID_DENOMINATOR = 16
_HOLDOUT_GRID = {5, 11}


@dataclass(frozen=True)
class MachineSpec:
    emissions: np.ndarray
    transitions: np.ndarray
    stationary: np.ndarray
    canonical_perm: np.ndarray
    canonical_signature: str
    alphabet_size: int
    metadata: Dict[str, Any]

    @property
    def num_states(self) -> int:
        return int(self.emissions.shape[0])


def _sample_state_count(rng: np.random.Generator, k_min: int, k_max: int) -> int:
    support = np.arange(k_min, k_max + 1, dtype=np.int64)
    weights = 1.0 / support
    weights /= weights.sum()
    return int(rng.choice(support, p=weights))


def _initial_transitions(num_states: int, alphabet_size: int) -> np.ndarray:
    base = np.empty((num_states, alphabet_size), dtype=np.int64)
    for symbol in range(alphabet_size):
        base[:, symbol] = (np.arange(num_states, dtype=np.int64) + symbol + 1) % num_states
    return base


def _branching_entropy(transitions: np.ndarray) -> float:
    k, alphabet = transitions.shape
    entropy = 0.0
    for state in range(k):
        counts = np.bincount(transitions[state], minlength=k)
        probs = counts[counts > 0] / alphabet
        entropy -= float(np.sum(probs * np.log(probs)))
    return entropy / k


def _is_strongly_connected(transitions: np.ndarray) -> bool:
    k = transitions.shape[0]
    adjacency = [set(row.tolist()) for row in transitions]
    def dfs(graph: Sequence[set[int]]) -> bool:
        seen = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(graph[node] - seen)
        return len(seen) == k

    if not dfs(adjacency):
        return False

    reverse = [set() for _ in range(k)]
    for src, targets in enumerate(adjacency):
        for tgt in targets:
            reverse[tgt].add(src)
    return dfs(reverse)


def _is_aperiodic(transitions: np.ndarray) -> bool:
    k, alphabet = transitions.shape
    adjacency = [transitions[state].tolist() for state in range(k)]
    dist = [-1] * k
    dist[0] = 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)

    gcd_val = 0
    for state in range(k):
        for nxt in adjacency[state]:
            if dist[nxt] == -1:
                continue
            cycle_len = dist[state] + 1 - dist[nxt]
            if cycle_len <= 0:
                continue
            gcd_val = cycle_len if gcd_val == 0 else math.gcd(gcd_val, cycle_len)
    return gcd_val == 1


def _rewire_transitions(
    transitions: np.ndarray,
    rng: np.random.Generator,
    p_rewire: float,
) -> np.ndarray:
    k, alphabet = transitions.shape
    result = transitions.copy()
    for state in range(k):
        seen = {result[state, 0]}
        for symbol in range(1, alphabet):
            if rng.random() >= p_rewire:
                seen.add(result[state, symbol])
                continue
            choices = np.arange(k)
            available = np.setdiff1d(choices, np.fromiter(seen, dtype=np.int64), assume_unique=True)
            target_pool = available if available.size else choices
            result[state, symbol] = int(rng.choice(target_pool))
            seen.add(result[state, symbol])
    return result


def _stationary_distribution(transitions: np.ndarray, tol: float = 1e-9, max_steps: int = 10_000) -> np.ndarray:
    k, alphabet = transitions.shape
    kernel = np.zeros((k, k), dtype=np.float64)
    inv_alpha = 1.0 / alphabet
    for state in range(k):
        kernel[state, transitions[state]] += inv_alpha
    vec = np.full(k, 1.0 / k)
    for _ in range(max_steps):
        next_vec = vec @ kernel
        if np.linalg.norm(next_vec - vec, ord=1) <= tol:
            vec = next_vec
            break
        vec = next_vec
    vec /= vec.sum()
    return vec


def _canonicalize(
    transitions: np.ndarray,
    emissions: np.ndarray,
    stationary: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k, alphabet = transitions.shape
    order = sorted(
        range(k),
        key=lambda idx: (
            -stationary[idx],
            emissions[idx],
            tuple(transitions[idx]),
        ),
    )
    permutation = np.array(order, dtype=np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(k, dtype=np.int64)

    canon_transitions = np.empty_like(transitions)
    for new_idx, old_idx in enumerate(permutation):
        mapped = inverse[transitions[old_idx]]
        canon_transitions[new_idx] = mapped

    canon_emissions = emissions[permutation]
    canon_stationary = stationary[permutation]
    return canon_transitions, canon_emissions, canon_stationary, permutation


def _sample_emissions(
    rng: np.random.Generator,
    num_states: int,
) -> Tuple[np.ndarray, Sequence[Dict[str, Any]]]:
    emissions = np.empty(num_states, dtype=np.float64)
    info = []
    beta_grid: Sequence[Tuple[float, float]] = ((0.7, 0.7), (1.0, 1.0), (2.0, 2.0), (4.0, 4.0))
    for state in range(num_states):
        if rng.random() < 0.5:
            alpha, beta = beta_grid[rng.integers(0, len(beta_grid))]
            value = float(rng.beta(alpha, beta))
            emissions[state] = np.clip(value, 1e-4, 1 - 1e-4)
            info.append({"mode": "beta", "alpha": alpha, "beta": beta, "value": emissions[state]})
            continue

        grid_index = int(rng.integers(0, _GRID_DENOMINATOR + 1))
        sigma = float(rng.uniform(0.01, 0.05))
        center = grid_index / _GRID_DENOMINATOR
        jitter = float(rng.normal(0.0, sigma))
        value = np.clip(center + jitter, 1e-4, 1 - 1e-4)
        emissions[state] = value
        info.append(
            {
                "mode": "grid+jitter",
                "grid_index": grid_index,
                "sigma": sigma,
                "center": center,
                "value": value,
                "holdout": grid_index in _HOLDOUT_GRID,
            }
        )
    return emissions, info


def sample_machine(
    *,
    K_min: int = 2,
    K_max: int = 12,
    alphabet_size: int = 2,
    rng: Optional[np.random.Generator | int] = None,
    p_rewire_range: Tuple[float, float] = (0.2, 0.5),
) -> MachineSpec:
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    num_states = _sample_state_count(generator, K_min, K_max)
    p_rewire = float(generator.uniform(*p_rewire_range))

    base = _initial_transitions(num_states, alphabet_size)
    transitions = _rewire_transitions(base, generator, p_rewire)

    emissions, emission_info = _sample_emissions(generator, num_states)
    stationary = _stationary_distribution(transitions)

    canon_transitions, canon_emissions, canon_stationary, permutation = _canonicalize(
        transitions, emissions, stationary
    )

    if not _is_strongly_connected(canon_transitions):
        raise RuntimeError("canonical machine is not strongly connected")

    signature = hashlib.sha1(
        canon_transitions.astype(np.int64).tobytes() + canon_emissions.astype(np.float64).tobytes()
    ).hexdigest()

    row_entropy = np.zeros((num_states, alphabet_size), dtype=np.float64)
    aperiodic = _is_aperiodic(canon_transitions)

    branching_entropy = _branching_entropy(canon_transitions)
    state_regime = categorize_state_count(num_states)
    emission_regime = categorize_emission(emission_info)
    topology_regime = categorize_topology({"family": "random_unifilar_v2"})

    metadata: Dict[str, Any] = {
        "K": num_states,
        "alphabet_size": alphabet_size,
        "p_rewire": p_rewire,
        "branching_entropy": branching_entropy,
        "emission_info": emission_info,
        "splits": {
            "K_regime": state_regime,
            "emission_regime": emission_regime,
            "topology_regime": topology_regime,
        },
        "family": "random_unifilar_v2",
        "ergodicity": {"scc": True, "aperiodic": aperiodic},
        "row_entropy": row_entropy.tolist(),
        "canonical_signature": signature,
    }

    return MachineSpec(
        emissions=canon_emissions,
        transitions=canon_transitions,
        stationary=canon_stationary,
        canonical_perm=permutation,
        canonical_signature=signature,
        alphabet_size=alphabet_size,
        metadata=metadata,
    )
