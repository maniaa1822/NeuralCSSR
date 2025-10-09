from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .meta_distribution import MachineSpec


def sample_sequence(
    spec: MachineSpec,
    length: int,
    *,
    rng: Optional[np.random.Generator | int] = None,
    return_states: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if length <= 0:
        raise ValueError("length must be positive")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    if spec.alphabet_size != 2:
        raise NotImplementedError("non-binary alphabets are not yet supported")

    current_state = int(generator.choice(spec.num_states, p=spec.stationary))
    sequence = np.empty(length, dtype=np.uint8)
    states = np.empty(length, dtype=np.int64) if return_states else None

    for idx in range(length):
        prob_one = spec.emissions[current_state]
        symbol = int(generator.random() < prob_one)
        sequence[idx] = symbol
        if states is not None:
            states[idx] = current_state
        current_state = spec.transitions[current_state, symbol]

    if states is not None:
        return sequence, states
    return sequence, None
