"""
Base Machine class for unified machine specifications.

All machine definitions inherit from this base class and implement
the required properties and methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class Machine(ABC):
    """Abstract base class for all epsilon machine specifications.

    A Machine encapsulates:
    - Structural properties (states, alphabet, transitions)
    - Emission probabilities
    - Ground truth state mapping
    - Theoretical properties (entropy, memory length)
    - Pipeline integration (paths, training configs)
    """

    # ===== Identity (must be defined by subclasses) =====

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine identifier (e.g., 'seven_state_human')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Seven-State Human')."""
        pass

    # ===== Structure (must be defined by subclasses) =====

    @property
    @abstractmethod
    def states(self) -> List[str]:
        """List of state names (e.g., ['bb', 'aaa', ...])."""
        pass

    @property
    @abstractmethod
    def alphabet(self) -> List[str]:
        """Symbol alphabet (e.g., ['0', '1'])."""
        pass

    @property
    @abstractmethod
    def start_state(self) -> str:
        """Initial state name."""
        pass

    @property
    @abstractmethod
    def transitions(self) -> Dict[Tuple[str, str], str]:
        """Transition function: (state, symbol) -> next_state.

        Example:
            {('bb', '0'): 'ba', ('bb', '1'): 'bb', ...}
        """
        pass

    @property
    @abstractmethod
    def emissions(self) -> Dict[str, Dict[str, float]]:
        """Emission probabilities: state -> {symbol: probability}.

        Example:
            {'bb': {'0': 0.9375, '1': 0.0625}, ...}
        """
        pass

    # ===== Memory Properties (must be defined by subclasses) =====

    @property
    @abstractmethod
    def memory_type(self) -> str:
        """'finite' or 'infinite'."""
        pass

    @property
    @abstractmethod
    def memory_length(self) -> Optional[int]:
        """Maximum history length needed to determine state (None if infinite)."""
        pass

    # ===== Ground Truth Methods (must be implemented by subclasses) =====

    @abstractmethod
    def get_gt_state(self, history: np.ndarray) -> Optional[str]:
        """Map a history to its ground truth causal state.

        Args:
            history: Array of symbols (e.g., np.array([0, 1, 0, 1]))

        Returns:
            State name (e.g., 'ba'), or None if state cannot be determined
        """
        pass

    def get_state_suffixes(self) -> Dict[str, List[str]]:
        """Get minimal suffix patterns that define each state.

        Returns:
            Dict mapping state names to their defining suffix patterns.
            Empty list means state is defined by other criteria.

        Example:
            {'bb': ['11'], 'aaa': ['000'], ...}
        """
        return {state: [] for state in self.states}

    # ===== Theoretical Properties =====

    def compute_theoretical_entropy(self) -> float:
        """Compute theoretical entropy rate H(X) for this machine.

        Uses stationary distribution via power iteration:
        H(X) = -Σ π[s] * Σ P(x|s) log P(x|s)

        Returns:
            Entropy rate in nats (use / np.log(2) for bits)
        """
        n = len(self.states)
        state_to_idx = {s: i for i, s in enumerate(self.states)}

        # Build transition matrix weighted by emission probabilities
        T = np.zeros((n, n))
        for (s, x), s_next in self.transitions.items():
            i, j = state_to_idx[s], state_to_idx[s_next]
            p_x = self.emissions[s][x]
            T[i, j] += p_x

        # Find stationary distribution via power iteration
        pi = np.ones(n) / n
        for _ in range(10000):
            pi_new = T.T @ pi
            if np.allclose(pi, pi_new, atol=1e-15):
                break
            pi = pi_new

        # Normalize
        pi = pi / pi.sum()

        # Compute entropy H = -Σ π[s] * Σ P(x|s) log P(x|s)
        H = 0.0
        for i, state in enumerate(self.states):
            for symbol, prob in self.emissions[state].items():
                if prob > 0:
                    H -= pi[i] * prob * np.log(prob)

        return float(H)

    # ===== Derived Properties =====

    @property
    def num_states(self) -> int:
        """Number of states in the machine."""
        return len(self.states)

    @property
    def alphabet_size(self) -> int:
        """Size of the symbol alphabet."""
        return len(self.alphabet)

    # ===== Pipeline Integration =====

    def get_dataset_path(self, base_dir: Path, filename_only: bool = False) -> Path:
        """Get path to dataset files for this machine.

        Args:
            base_dir: Base directory (e.g., experiments/datasets)
            filename_only: If True, return just the filename stem

        Returns:
            Path to dataset directory or filename
        """
        if filename_only:
            return Path(self.name)
        return base_dir / self.name / f"{self.name}.dat"

    def get_model_path(self, base_dir: Path, variant: str = "char") -> Path:
        """Get path to trained model checkpoint for this machine.

        Args:
            base_dir: Base directory (e.g., nanoGPT/)
            variant: Model variant suffix (e.g., 'char', 'char_large')

        Returns:
            Path to model checkpoint
        """
        if variant:
            out_dir = f"out-{self.name}-{variant}"
        else:
            out_dir = f"out-{self.name}"
        return base_dir / out_dir / "ckpt.pt"

    def get_training_config(self, variant: str = "char") -> Dict:
        """Get recommended training hyperparameters for this machine.

        Args:
            variant: Config variant (e.g., 'char', 'char_large')

        Returns:
            Dict of training hyperparameters
        """
        # Default configuration (can be overridden by subclasses)
        config = {
            'dataset': self.name,
            'out_dir': f'out-{self.name}-{variant}',
            'batch_size': 64,
            'block_size': max(64, self.memory_length or 64),
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 128,
            'dropout': 0.0 if self.memory_type == 'finite' else 0.1,
            'learning_rate': 1e-3,
            'max_iters': 5000,
        }
        return config

    # ===== Validation =====

    def validate(self) -> bool:
        """Validate machine specification (unifilarity, stochasticity, etc.).

        Returns:
            True if machine is valid

        Raises:
            ValueError: If machine specification is invalid
        """
        # Check emissions sum to 1.0 for each state
        for state, emission_probs in self.emissions.items():
            total = sum(emission_probs.values())
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(
                    f"Emissions for state '{state}' sum to {total}, not 1.0"
                )

        # Check all transitions are defined
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transitions:
                    # Check if this is a forbidden transition (emission prob = 0)
                    if symbol in self.emissions[state] and self.emissions[state][symbol] == 0:
                        continue
                    raise ValueError(
                        f"Missing transition for state='{state}', symbol='{symbol}'"
                    )

        # Check transitions point to valid states
        for (state, symbol), next_state in self.transitions.items():
            if next_state not in self.states:
                raise ValueError(
                    f"Transition ({state}, {symbol}) points to unknown state '{next_state}'"
                )

        return True

    # ===== String Representation =====

    def __repr__(self) -> str:
        return (
            f"<Machine '{self.name}': "
            f"{self.num_states} states, "
            f"{self.alphabet_size} symbols, "
            f"memory={self.memory_type}>"
        )

    def summary(self) -> str:
        """Generate a detailed summary of this machine."""
        lines = [
            f"Machine: {self.display_name} ({self.name})",
            f"States: {self.num_states} ({', '.join(self.states)})",
            f"Alphabet: {self.alphabet}",
            f"Memory: {self.memory_type}",
        ]
        if self.memory_length is not None:
            lines.append(f"Memory length: {self.memory_length}")

        try:
            entropy = self.compute_theoretical_entropy()
            lines.append(f"Theoretical entropy: {entropy:.4f} nats ({entropy/np.log(2):.4f} bits)")
        except Exception as e:
            lines.append(f"Theoretical entropy: (computation failed: {e})")

        return '\n'.join(lines)