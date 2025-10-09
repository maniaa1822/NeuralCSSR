"""
Data extraction for entanglement analysis.

Handles extraction of histories, ε-states, and primitive factors from machine and data.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import math

from .primitives import PrimitiveFactors


class DataExtractor:
    """
    Extracts histories, ε-states, and primitive factors from machine specifications.

    Handles:
    - History extraction from sequences
    - ε-state labeling using ground truth
    - Primitive factor computation
    - Balanced sampling across states
    """

    def __init__(self, machine):
        """
        Initialize with a machine specification.

        Args:
            machine: Machine instance (e.g., SevenStateHumanMachine)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.machine = machine
        self.states = machine.states
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}

        # Precompute primitive factors for all states
        self._primitive_factors = self._compute_primitive_factors()

    def _compute_primitive_factors(self) -> Dict[str, Dict[str, Any]]:
        """Compute primitive factors for all states once."""
        primitives = {}

        for state in self.states:
            primitives[state] = PrimitiveFactors.compute_all(state, self.machine)

        return primitives

    def extract_histories_and_labels(self, sequences: List[np.ndarray],
                                   min_length: int = 2,
                                   max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract valid histories and their labels from sequences.

        Args:
            sequences: List of symbol sequences (0s and 1s)
            min_length: Minimum history length to consider
            max_length: Maximum history length (None for no limit)

        Returns:
            Tuple of:
            - histories: (N, max_L) array of padded histories
            - state_labels: (N,) array of state indices
            - primitive_labels: Dict of primitive_name -> (N,) arrays
        """
        histories = []
        state_labels = []
        primitive_labels = defaultdict(list)

        for seq in sequences:
            seq = np.array(seq)
            if max_length is None:
                max_length = len(seq)

            # Extract all valid histories from this sequence
            for start in range(len(seq)):
                for length in range(min_length, min(max_length + 1, len(seq) - start + 1)):
                    history = seq[start:start + length]

                    # Get ground truth state
                    state = self.machine.get_gt_state(history)
                    if state is None:
                        continue

                    # Store history padded to max_length with 0s, RIGHT-aligned.
                    # Rationale: downstream FeatureExtractor captures the final
                    # token representation at position -1 (output[:, -1, :]).
                    # Right-aligning ensures that the last position corresponds
                    # to the actual last symbol of each history rather than a
                    # padding 0, avoiding degenerate, uninformative activations.
                    padded_history = np.zeros(max_length, dtype=int)
                    padded_history[-length:] = history
                    histories.append(padded_history)

                    # Store state label
                    state_labels.append(self.state_to_idx[state])

                    # Store primitive labels
                    for prim_name, prim_value in self._primitive_factors[state].items():
                        primitive_labels[prim_name].append(prim_value)

        # Convert to arrays
        histories = np.array(histories)
        state_labels = np.array(state_labels)

        # Convert primitive labels to arrays and handle categorical encoding
        primitive_arrays = {}
        for prim_name, values in primitive_labels.items():
            # For categorical primitives, convert to indices
            if isinstance(values[0], str):
                unique_vals = sorted(set(values))
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                primitive_arrays[prim_name] = np.array([val_to_idx[v] for v in values])
            else:
                primitive_arrays[prim_name] = np.array(values)

        self.logger.debug(
            "Extracted %d histories (min_len=%d, max_len=%s)",
            len(histories), min_length, max_length if max_length is not None else 'auto'
        )

        return histories, state_labels, primitive_arrays

    def get_balanced_sample(self, sequences: List[np.ndarray],
                          n_per_state: int = 1000,
                          min_length: int = 2) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract balanced sample with equal representation across ε-states.

        Args:
            sequences: Source sequences
            n_per_state: Samples per state
            min_length: Minimum history length

        Returns:
            Same format as extract_histories_and_labels
        """
        # First extract all possible histories
        all_histories, all_states, all_primitives = self.extract_histories_and_labels(
            sequences, min_length=min_length
        )

        self.logger.debug(
            "Total candidate histories: %d", len(all_histories)
        )

        # Group by state
        state_indices = defaultdict(list)
        for i, state_idx in enumerate(all_states):
            state_indices[state_idx].append(i)

        # Sample equally from each state
        selected_indices = []
        min_samples = min(len(indices) for indices in state_indices.values())
        target_samples = min(n_per_state, min_samples)
        self.logger.debug(
            "Target per-state sample=%d (available min=%d)",
            n_per_state, min_samples
        )

        for state_idx, indices in state_indices.items():
            # Random sample without replacement
            np.random.shuffle(indices)
            selected_indices.extend(indices[:target_samples])

        # Extract selected samples
        histories = all_histories[selected_indices]
        state_labels = all_states[selected_indices]
        primitive_labels = {k: v[selected_indices] for k, v in all_primitives.items()}

        self.logger.info(
            "Balanced sample size=%d (per-state≈%d)",
            len(histories), target_samples
        )

        return histories, state_labels, primitive_labels

    def get_state_successors(self, state: str) -> Dict[str, str]:
        """Get successor states for each symbol from given state."""
        successors = {}
        for symbol in self.machine.alphabet:
            next_state = self.machine.transitions.get((state, symbol))
            if next_state:
                successors[symbol] = next_state
        return successors

    def get_reachability_info(self, max_steps: int = 5) -> Dict[str, Dict[str, List[str]]]:
        """Precompute reachability information for k-step extrapolation."""
        reachability = {}

        for start_state in self.states:
            reachability[start_state] = {}
            current_states = {start_state}

            for k in range(1, max_steps + 1):
                next_states = set()
                for state in current_states:
                    for symbol in self.machine.alphabet:
                        next_state = self.machine.transitions.get((state, symbol))
                        if next_state:
                            next_states.add(next_state)
                reachability[start_state][f"k_{k}"] = sorted(list(next_states))
                current_states = next_states

        return reachability
