"""
Golden Mean Process specification.

The Golden Mean process is a classic example with 2 causal states and
finite memory (L=1). It generates binary sequences where no two consecutive
0s can appear.

Causal states:
- State A: Last symbol was 1 → can emit 0 or 1 (probability 0.5 each)
- State B: Last symbol was 0 → must emit 1 (no consecutive 0s allowed)

This is a simple, well-understood machine often used for testing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .base import Machine
from . import register_machine


@register_machine("golden_mean")
class GoldenMeanMachine(Machine):
    """Golden Mean Process (no consecutive 0s)."""

    # ===== Identity =====

    @property
    def name(self) -> str:
        return "golden_mean"

    @property
    def display_name(self) -> str:
        return "Golden Mean"

    # ===== Structure =====

    @property
    def states(self) -> List[str]:
        return ["A", "B"]

    @property
    def alphabet(self) -> List[str]:
        return ["0", "1"]

    @property
    def start_state(self) -> str:
        return "A"

    @property
    def transitions(self) -> Dict[Tuple[str, str], str]:
        """Transitions enforce the 'no consecutive 0s' constraint."""
        return {
            ("A", "0"): "B",  # After emitting 0, must go to B
            ("A", "1"): "A",  # After emitting 1, stay in A
            ("B", "1"): "A",  # From B, can only emit 1, returning to A
        }

    @property
    def emissions(self) -> Dict[str, Dict[str, float]]:
        """State A has equal probabilities; State B must emit 1."""
        return {
            "A": {"0": 0.5, "1": 0.5},
            "B": {"0": 0.0, "1": 1.0},  # No consecutive 0s
        }

    # ===== Memory Properties =====

    @property
    def memory_type(self) -> str:
        return "finite"

    @property
    def memory_length(self) -> Optional[int]:
        return 1  # Only need to know last symbol

    # ===== Ground Truth Methods =====

    def get_gt_state(self, history: np.ndarray) -> Optional[str]:
        """Determine state based on last symbol.

        - Empty history → State A (start state)
        - Last symbol is 1 → State A
        - Last symbol is 0 → State B

        Args:
            history: Array of symbols

        Returns:
            'A' or 'B'
        """
        if len(history) == 0:
            return 'A'

        last_symbol = int(history[-1])
        return 'A' if last_symbol == 1 else 'B'

    def get_state_suffixes(self) -> Dict[str, List[str]]:
        """States defined by last symbol (not suffix patterns)."""
        # Returning empty lists indicates states are not suffix-based
        return {
            'A': [],  # All histories ending with 1 (or empty)
            'B': [],  # All histories ending with 0
        }

    # ===== Training Configuration =====

    def get_training_config(self, variant: str = "char") -> Dict:
        """Recommended training config for golden mean."""
        config = super().get_training_config(variant)

        # Golden mean is simple, doesn't need much capacity
        config.update({
            'block_size': 64,
            'n_embd': 64,
            'dropout': 0.1,
            'max_iters': 3000,
        })

        return config