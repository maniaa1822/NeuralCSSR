"""
Butterfly Machine specification.

A 2-cryptic process as described in Figure 3.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .base import Machine
from . import register_machine


@register_machine("butterfly")
class ButterflyMachine(Machine):
    """Butterfly Machine (2-cryptic process)."""

    # ===== Identity =====

    @property
    def name(self) -> str:
        return "butterfly"

    @property
    def display_name(self) -> str:
        return "Butterfly Process"

    # ===== Structure =====

    @property
    def states(self) -> List[str]:
        return ["A", "B", "C", "D", "E"]

    @property
    def alphabet(self) -> List[str]:
        return [str(i) for i in range(8)]

    @property
    def start_state(self) -> str:
        return "A"

    @property
    def transitions(self) -> Dict[Tuple[str, str], str]:
        return {
            ("A", "2"): "B",
            ("A", "3"): "D",
            ("B", "4"): "B",
            ("B", "0"): "C",
            ("C", "6"): "C",
            ("C", "1"): "A",
            ("D", "5"): "D",
            ("D", "0"): "E",
            ("E", "7"): "E",
            ("E", "1"): "A",
        }

    @property
    def emissions(self) -> Dict[str, Dict[str, float]]:
        # All transitions shown are 1/2 probability
        return {
            "A": {"2": 0.5, "3": 0.5},
            "B": {"4": 0.5, "0": 0.5},
            "C": {"6": 0.5, "1": 0.5},
            "D": {"5": 0.5, "0": 0.5},
            "E": {"7": 0.5, "1": 0.5},
        }

    # ===== Memory Properties =====

    @property
    def memory_type(self) -> str:
        return "finite"

    @property
    def memory_length(self) -> Optional[int]:
        return 2

    # ===== Ground Truth Methods =====

    def get_gt_state(self, history: np.ndarray) -> Optional[str]:
        """Determine ground truth state from history.
        
        Mappings:
        - Ends with 1 -> A
        - Ends with 2, 4 -> B
        - Ends with 3, 5 -> D
        - Ends with 6 -> C
        - Ends with 7 -> E
        - Ends with 0:
            - Prev is 2, 4 -> C
            - Prev is 3, 5 -> E
        
        Args:
            history: Array of symbols
            
        Returns:
            State name or None
        """
        if len(history) == 0:
            return None
            
        last = str(history[-1])
        
        if last == '1':
            return 'A'
        elif last in ['2', '4']:
            return 'B'
        elif last in ['3', '5']:
            return 'D'
        elif last == '6':
            return 'C'
        elif last == '7':
            return 'E'
        elif last == '0':
            if len(history) < 2:
                return None
            prev = str(history[-2])
            if prev in ['2', '4']: # Coming from B
                return 'C'
            elif prev in ['3', '5']: # Coming from D
                return 'E'
                
        return None
        
    def get_state_suffixes(self) -> Dict[str, List[str]]:
        return {
            'A': ['1'],
            'B': ['2', '4'],
            'C': ['6', '20', '40'],
            'D': ['3', '5'],
            'E': ['7', '30', '50'],
        }
