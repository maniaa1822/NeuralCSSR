"""
Primitive factor computation for ε-states.

Computes the 7 primitive factors:
1. entropy tertile - emission entropy tertiles
2. emission-bias bucket - bias towards 0 vs 1
3. community id - natural state communities
4. parity mod-2 - parity of 1s in suffix
5. mod-3 - length-based mod 3
6. stationary tertile - stationary distribution tertiles
7. out-degree class - number of outgoing transitions
8. self-loop flag - presence of self-loops
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import math


class PrimitiveFactors:
    """Computes primitive factors for ε-states."""

    @staticmethod
    def compute_all(state: str, machine) -> Dict[str, Any]:
        """Compute all primitive factors for a state."""
        return {
            'entropy_tertile': PrimitiveFactors._entropy_tertile(state, machine),
            'emission_bias_bucket': PrimitiveFactors._emission_bias_bucket(state, machine),
            'community_id': PrimitiveFactors._community_id(state, machine),
            'parity_mod2': PrimitiveFactors._parity_mod2(state, machine),
            'mod3': PrimitiveFactors._mod3(state, machine),
            'stationary_tertile': PrimitiveFactors._stationary_tertile(state, machine),
            'out_degree_class': PrimitiveFactors._out_degree_class(state, machine),
            'self_loop_flag': PrimitiveFactors._self_loop_flag(state, machine),
        }

    @staticmethod
    def _entropy_tertile(state: str, machine) -> int:
        """Emission entropy tertile (0=low, 1=medium, 2=high)."""
        emissions = machine.emissions[state]

        # Compute entropy: H = -∑ p_i log p_i
        entropy = 0.0
        for prob in emissions.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)

        # Tertiles across all states
        all_entropies = []
        for s in machine.states:
            em = machine.emissions[s]
            h = 0.0
            for p in em.values():
                if p > 0:
                    h -= p * math.log2(p)
            all_entropies.append(h)

        # Find tertile
        tertile_bounds = np.percentile(all_entropies, [33.33, 66.67])
        if entropy <= tertile_bounds[0]:
            return 0  # low
        elif entropy <= tertile_bounds[1]:
            return 1  # medium
        else:
            return 2  # high

    @staticmethod
    def _emission_bias_bucket(state: str, machine) -> str:
        """Emission bias bucket based on P(1) vs P(0)."""
        emissions = machine.emissions[state]
        p1 = emissions.get('1', 0.0)

        if p1 < 0.2:
            return 'strong_0'
        elif p1 < 0.4:
            return 'weak_0'
        elif p1 < 0.6:
            return 'balanced'
        elif p1 < 0.8:
            return 'weak_1'
        else:
            return 'strong_1'

    @staticmethod
    def _community_id(state: str, machine) -> str:
        """Natural communities in seven-state machine."""
        # Based on the paper's state groupings and emission patterns
        community_map = {
            'bb': 'recurrent',    # High P(0), recurrent on 1
            'aaa': 'recurrent',   # High P(1), recurrent on 0
            'aaab': 'transient',  # High P(1), transient
            'ba': 'balanced',     # Balanced emissions
            'bab': 'transient',   # High P(1), transient
            'baab': 'transient',  # Low P(1), transient
            'baa': 'balanced',    # Balanced emissions
        }
        return community_map.get(state, 'unknown')

    @staticmethod
    def _parity_mod2(state: str, machine) -> int:
        """Parity of 1s in the state's defining suffix."""
        suffix_patterns = machine.get_state_suffixes()
        suffixes = suffix_patterns.get(state, [])

        if not suffixes:
            return 0

        # Use the first (longest) suffix
        suffix = suffixes[0]
        count_ones = suffix.count('1')
        return count_ones % 2

    @staticmethod
    def _mod3(state: str, machine) -> int:
        """Length of defining suffix mod 3."""
        suffix_patterns = machine.get_state_suffixes()
        suffixes = suffix_patterns.get(state, [])

        if not suffixes:
            return 0

        # Use the first (longest) suffix
        suffix = suffixes[0]
        return len(suffix) % 3

    @staticmethod
    def _stationary_tertile(state: str, machine) -> int:
        """Stationary distribution mass tertiles."""
        # For finite state machines, we can compute stationary distribution
        # This is a simplified version - in practice you'd solve πP = π

        # Use emission entropy as proxy for "importance" in stationary dist
        # States with more balanced emissions tend to have higher stationary mass
        emissions = machine.emissions[state]
        balance = 1.0 - abs(emissions.get('0', 0.5) - emissions.get('1', 0.5))

        # Tertiles across all states
        all_balances = []
        for s in machine.states:
            em = machine.emissions[s]
            bal = 1.0 - abs(em.get('0', 0.5) - em.get('1', 0.5))
            all_balances.append(bal)

        tertile_bounds = np.percentile(all_balances, [33.33, 66.67])
        if balance <= tertile_bounds[0]:
            return 0  # low
        elif balance <= tertile_bounds[1]:
            return 1  # medium
        else:
            return 2  # high

    @staticmethod
    def _out_degree_class(state: str, machine) -> int:
        """Number of distinct outgoing transitions."""
        transitions = machine.transitions
        outgoing = set()

        for symbol in machine.alphabet:
            next_state = transitions.get((state, symbol))
            if next_state:
                outgoing.add(next_state)

        return len(outgoing)

    @staticmethod
    def _self_loop_flag(state: str, machine) -> bool:
        """Whether state has any self-loops."""
        transitions = machine.transitions

        for symbol in machine.alphabet:
            next_state = transitions.get((state, symbol))
            if next_state == state:
                return True

        return False

    @staticmethod
    def get_primitive_info() -> Dict[str, Dict[str, Any]]:
        """Get metadata about each primitive factor."""
        return {
            'entropy_tertile': {
                'type': 'categorical',
                'classes': ['low', 'medium', 'high'],
                'description': 'Emission entropy tertiles'
            },
            'emission_bias_bucket': {
                'type': 'categorical',
                'classes': ['strong_0', 'weak_0', 'balanced', 'weak_1', 'strong_1'],
                'description': 'Bias towards 0 vs 1 emissions'
            },
            'community_id': {
                'type': 'categorical',
                'classes': ['recurrent', 'transient', 'balanced'],
                'description': 'Natural state communities'
            },
            'parity_mod2': {
                'type': 'categorical',
                'classes': [0, 1],
                'description': 'Parity of 1s in suffix (mod 2)'
            },
            'mod3': {
                'type': 'categorical',
                'classes': [0, 1, 2],
                'description': 'Suffix length mod 3'
            },
            'stationary_tertile': {
                'type': 'categorical',
                'classes': ['low', 'medium', 'high'],
                'description': 'Stationary distribution mass tertiles'
            },
            'out_degree_class': {
                'type': 'numerical',
                'range': [1, 2],  # For binary alphabet
                'description': 'Number of distinct outgoing transitions'
            },
            'self_loop_flag': {
                'type': 'categorical',
                'classes': [False, True],
                'description': 'Presence of self-loops'
            }
        }
