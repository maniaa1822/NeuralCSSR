"""
Seven-State Human Machine specification.

This machine has 7 causal states with distinct emission probabilities
as described in Figure 3 of the original paper. States are defined by
suffix patterns of varying lengths (2-4 symbols).

Transition structure is deterministic and unifilar.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .base import Machine
from . import register_machine


@register_machine("seven_state_human")
class SevenStateHumanMachine(Machine):
    """Seven-State Human Machine (corrected emissions from Figure 3)."""

    # ===== Identity =====

    @property
    def name(self) -> str:
        return "seven_state_human"

    @property
    def display_name(self) -> str:
        return "Seven-State Human"

    # ===== Structure =====

    @property
    def states(self) -> List[str]:
        return ["bb", "aaa", "aaab", "ba", "bab", "baab", "baa"]

    @property
    def alphabet(self) -> List[str]:
        return ["0", "1"]

    @property
    def start_state(self) -> str:
        return "bb"

    @property
    def transitions(self) -> Dict[Tuple[str, str], str]:
        """Deterministic, unifilar transitions."""
        return {
            ("bb", "0"): "ba",
            ("bb", "1"): "bb",
            ("aaa", "0"): "aaa",
            ("aaa", "1"): "aaab",
            ("aaab", "0"): "ba",
            ("aaab", "1"): "bb",
            ("ba", "0"): "baa",
            ("ba", "1"): "bab",
            ("bab", "0"): "ba",
            ("bab", "1"): "bb",
            ("baab", "0"): "ba",
            ("baab", "1"): "bb",
            ("baa", "0"): "aaa",
            ("baa", "1"): "baab",
        }

    @property
    def emissions(self) -> Dict[str, Dict[str, float]]:
        """Emission probabilities (corrected from Figure 3)."""
        return {
            "bb": {"0": 15.0/16.0, "1": 1.0/16.0},
            "aaa": {"0": 3.0/16.0, "1": 13.0/16.0},
            "aaab": {"0": 7.0/16.0, "1": 9.0/16.0},
            "ba": {"0": 7.0/16.0, "1": 9.0/16.0},
            "bab": {"0": 8.0/16.0, "1": 8.0/16.0},
            "baab": {"0": 7.0/16.0, "1": 9.0/16.0},
            "baa": {"0": 3.0/16.0, "1": 13.0/16.0},
        }

    # ===== Memory Properties =====

    @property
    def memory_type(self) -> str:
        return "finite"

    @property
    def memory_length(self) -> Optional[int]:
        return 4  # Longest state suffix is 4 symbols (AAAB, BAAB)

    # ===== Ground Truth Methods =====

    def get_gt_state(self, history: np.ndarray) -> Optional[str]:
        """Determine ground truth state from history using suffix matching.

        States are defined by suffix patterns (longest match first):
        - AAAB: '0001' (4 symbols)
        - BAAB: '1001' (4 symbols)
        - BAB: '101' (3 symbols)
        - BAA: '100' (3 symbols)
        - AAA: '000' (3 symbols, but not '0001')
        - BB: '11' (2 symbols)
        - BA: '10' (2 symbols, but not longer patterns)

        Args:
            history: Array of symbols (0s and 1s)

        Returns:
            State name, or None if history too short
        """
        history_str = ''.join(map(str, history))

        # Check for longest patterns first (4 symbols)
        if len(history) >= 4:
            if history_str.endswith('0001'):  # AAAB
                return 'aaab'
            elif history_str.endswith('1001'):  # BAAB
                return 'baab'

        # Check for 3-symbol patterns
        if len(history) >= 3:
            if history_str.endswith('101'):  # BAB
                return 'bab'
            elif history_str.endswith('100'):  # BAA
                return 'baa'
            elif history_str.endswith('000') and not history_str.endswith('0001'):  # AAA (not AAAB)
                return 'aaa'

        # Check for 2-symbol patterns
        if len(history) >= 2:
            if history_str.endswith('11'):  # BB
                return 'bb'
            # BA: ends with '10' but not part of longer patterns
            elif history_str.endswith('10') and not (
                len(history) >= 3 and (
                    history_str.endswith('101') or history_str.endswith('100')
                )
            ):
                return 'ba'

        # Cannot determine state for shorter histories
        return None

    def get_state_suffixes(self) -> Dict[str, List[str]]:
        """Get minimal suffix patterns defining each state."""
        return {
            'bb': ['11'],
            'aaa': ['000'],
            'aaab': ['0001'],
            'ba': ['10'],
            'bab': ['101'],
            'baab': ['1001'],
            'baa': ['100'],
        }

    # ===== Training Configuration =====

    def get_training_config(self, variant: str = "char") -> Dict:
        """Recommended training config for seven-state machine."""
        config = super().get_training_config(variant)

        # Specialized configs for different variants
        if variant == "char":
            config.update({
                'block_size': 64,
                'n_embd': 16,
                'dropout': 0.2,
                'max_iters': 5000,
            })
        elif variant == "char_large":
            config.update({
                'block_size': 64,
                'n_embd': 128,
                'dropout': 0.0,
                'max_iters': 10000,
            })

        return config