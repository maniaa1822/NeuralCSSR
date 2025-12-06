"""
Phoneme Machine specification.

A 12-symbol machine with 4 causal states based on vowel/consonant phonotactics.
The alphabet consists of 3 vowels and 9 consonants (stops, fricatives, nasals),
and states are determined by the classes of the last two emitted symbols.

States:
- VV: Two vowels in a row (hiatus) → strongly favor consonants
- VC: Vowel then consonant (coda) → favor vowels for new syllable
- CV: Consonant then vowel (onset) → balanced  
- CC: Two consonants (cluster) → strongly favor vowels to break cluster

Theoretical entropy rate: 2.221 nats (3.205 bits)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .base import Machine
from . import register_machine


# Symbol definitions
VOWELS = [0, 1, 2]      # a, e, i
STOPS = [3, 4, 5]       # p, t, k
FRICATIVES = [6, 7, 8]  # f, s, h
NASALS = [9, 10, 11]    # m, n, ŋ
CONSONANTS = STOPS + FRICATIVES + NASALS  # all 9 consonants

# Symbol names for display
SYMBOL_NAMES = ['a', 'e', 'i', 'p', 't', 'k', 'f', 's', 'h', 'm', 'n', 'ŋ']


def is_vowel(symbol: int) -> bool:
    """Check if symbol is a vowel."""
    return symbol in VOWELS


def symbol_class(symbol: int) -> str:
    """Return 'V' for vowel, 'C' for consonant."""
    return 'V' if is_vowel(symbol) else 'C'


@register_machine("phoneme_machine")
class PhonemeMachine(Machine):
    """12-symbol phoneme machine with 4 causal states (VV, VC, CV, CC)."""

    # ===== Identity =====

    @property
    def name(self) -> str:
        return "phoneme_machine"

    @property
    def display_name(self) -> str:
        return "Phoneme Machine"

    # ===== Structure =====

    @property
    def states(self) -> List[str]:
        return ["VV", "VC", "CV", "CC"]

    @property
    def alphabet(self) -> List[str]:
        return SYMBOL_NAMES.copy()

    @property
    def start_state(self) -> str:
        # Start in CV (as if preceded by consonant then vowel)
        return "CV"

    @property
    def transitions(self) -> Dict[Tuple[str, str], str]:
        """Deterministic transitions based on emitted symbol class."""
        trans = {}
        for state in self.states:
            for sym_id, sym_name in enumerate(SYMBOL_NAMES):
                # New state = last char of current state + class of emitted symbol
                new_last_two = state[-1] + symbol_class(sym_id)
                trans[(state, sym_name)] = new_last_two
        return trans

    @property
    def emissions(self) -> Dict[str, Dict[str, float]]:
        """Emission probabilities reflecting phonotactic constraints.
        
        VV: P(V)=0.10, P(C)=0.90 - avoid hiatus (VVV)
        VC: P(V)=0.60, P(C)=0.40 - new syllable onset likely
        CV: P(V)=0.25, P(C)=0.75 - balanced, slight C preference
        CC: P(V)=0.90, P(C)=0.10 - break consonant cluster
        """
        p_vowel = {
            "VV": 0.10,
            "VC": 0.60,
            "CV": 0.25,
            "CC": 0.90,
        }
        
        emissions = {}
        for state in self.states:
            pv = p_vowel[state]
            pc = 1.0 - pv
            
            probs = {}
            for sym_id, sym_name in enumerate(SYMBOL_NAMES):
                if is_vowel(sym_id):
                    probs[sym_name] = pv / len(VOWELS)  # uniform within class
                else:
                    probs[sym_name] = pc / len(CONSONANTS)
            emissions[state] = probs
        
        return emissions

    # ===== Memory Properties =====

    @property
    def memory_type(self) -> str:
        return "finite"

    @property
    def memory_length(self) -> Optional[int]:
        return 2  # Need last two symbols to determine state

    # ===== Ground Truth Methods =====

    def get_gt_state(self, history: np.ndarray) -> Optional[str]:
        """Determine ground truth state from last two symbols.
        
        Args:
            history: Array of symbol IDs (integers 0-11)
            
        Returns:
            State name ('VV', 'VC', 'CV', 'CC'), or None if history too short
        """
        if len(history) < 2:
            return None
        
        last_two = history[-2:]
        class_str = symbol_class(int(last_two[0])) + symbol_class(int(last_two[1]))
        return class_str

    def get_state_suffixes(self) -> Dict[str, List[str]]:
        """Get patterns that define each state.
        
        States are defined by the classes of last two symbols, not specific
        symbol values, so we return empty lists (states are class-based).
        """
        return {
            'VV': [],  # Any two vowels
            'VC': [],  # Vowel then consonant
            'CV': [],  # Consonant then vowel  
            'CC': [],  # Any two consonants
        }

    # ===== Training Configuration =====

    def get_training_config(self, variant: str = "char") -> Dict:
        """Recommended training config for phoneme machine."""
        config = super().get_training_config(variant)

        config.update({
            'block_size': 64,
            'n_layer': 4,
            'n_head': 4,
            'n_embd': 128,
            'dropout': 0.1,
            'max_iters': 5000,
            'batch_size': 64,
            'vocab_size': 12,
        })

        return config

    # ===== Theoretical Properties =====

    def compute_theoretical_entropy(self) -> float:
        """Compute theoretical entropy rate for this machine.
        
        Uses stationary distribution and per-state entropy:
        H = Σ π[s] * H(emission|s)
        
        Returns:
            Entropy rate in nats (2.221 for this machine)
        """
        # Stationary distribution (computed analytically)
        pi = {
            'VV': 5/49,
            'VC': 18/49,
            'CV': 18/49,
            'CC': 8/49,
        }
        
        # Per-state entropy
        emissions = self.emissions
        H_per_state = {}
        
        for state, probs in emissions.items():
            H = 0.0
            for sym, p in probs.items():
                if p > 0:
                    H -= p * np.log(p)
            H_per_state[state] = H
        
        # Weighted sum
        H_total = sum(pi[s] * H_per_state[s] for s in self.states)
        return H_total

    def get_optimal_loss(self) -> Dict[str, float]:
        """Return optimal loss in various units."""
        H_nats = self.compute_theoretical_entropy()
        return {
            'nats': H_nats,
            'bits': H_nats / np.log(2),
            'uniform_baseline_nats': np.log(12),
            'uniform_baseline_bits': np.log(12) / np.log(2),
        }
