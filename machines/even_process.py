"""
Even Process specification.

The Even Process is a sofic process with 2 causal states and infinite Markov
order. It generates binary sequences where runs of 1s have even length.

Causal states:
- State E (Even): Current run of trailing 1s has even parity (including 0)
- State O (Odd): Current run of trailing 1s has odd parity

From state E: can emit 0 (stay in E) or 1 (go to O)
From state O: must emit 1 (go back to E) - enforces even-length runs

This machine requires infinite history to reconstruct states from observations,
making it a challenging test case.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .base import Machine
from . import register_machine


@register_machine("even_process")
class EvenProcessMachine(Machine):
    """Even Process (runs of 1s have even length)."""

    # ===== Identity =====

    @property
    def name(self) -> str:
        return "even_process"

    @property
    def display_name(self) -> str:
        return "Even Process"

    # ===== Structure =====

    @property
    def states(self) -> List[str]:
        return ["E", "O"]

    @property
    def alphabet(self) -> List[str]:
        return ["0", "1"]

    @property
    def start_state(self) -> str:
        return "E"  # Start with even parity (0 ones)

    @property
    def transitions(self) -> Dict[Tuple[str, str], str]:
        """Transitions enforce even-length runs of 1s."""
        return {
            ("E", "0"): "E",  # Emitting 0 from even parity stays even
            ("E", "1"): "O",  # Starting a run → odd parity
            ("O", "1"): "E",  # Continuing run → even parity
            # Note: ("O", "0") is forbidden - cannot emit 0 from odd parity
        }

    @property
    def emissions(self) -> Dict[str, Dict[str, float]]:
        """E can emit 0 or 1; O must emit 1 to maintain even runs."""
        return {
            "E": {"0": 0.5, "1": 0.5},
            "O": {"0": 0.0, "1": 1.0},  # Must complete even run
        }

    # ===== Memory Properties =====

    @property
    def memory_type(self) -> str:
        return "infinite"

    @property
    def memory_length(self) -> Optional[int]:
        return None  # Requires arbitrarily long history in worst case

    # ===== Ground Truth Methods =====

    def get_gt_state(self, history: np.ndarray) -> Optional[str]:
        """Determine state by counting trailing 1s.

        State is determined by parity of the current run of 1s:
        - Even number of trailing 1s (including 0) → State E
        - Odd number of trailing 1s → State O

        Args:
            history: Array of symbols

        Returns:
            'E' or 'O'
        """
        ones_run = 0
        for symbol in reversed(history):
            if int(symbol) == 1:
                ones_run += 1
            else:
                break  # Stop at first 0

        return 'E' if ones_run % 2 == 0 else 'O'

    def get_state_suffixes(self) -> Dict[str, List[str]]:
        """States defined by trailing 1s parity (not fixed suffix patterns)."""
        return {
            'E': [],  # Even parity of trailing 1s
            'O': [],  # Odd parity of trailing 1s
        }

    # ===== Training Configuration =====

    def get_training_config(self, variant: str = "char") -> Dict:
        """Recommended training config for even process.

        Even process requires longer context window due to infinite memory.
        """
        config = super().get_training_config(variant)

        # Need longer context for infinite memory
        config.update({
            'block_size': 128,  # Longer context essential
            'n_embd': 128,
            'dropout': 0.0,  # Deterministic process
            'max_iters': 5000,
        })

        return config