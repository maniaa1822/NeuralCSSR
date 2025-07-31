"""
TransCSSR-compatible CSSR implementation.

This module implements the CSSR algorithm following the reference transCSSR implementation
to ensure compatibility and correctness.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
import copy
import math


class JointHistory:
    """Represents a joint input-output history (X_history, Y_history)."""
    
    def __init__(self, x_history: str, y_history: str):
        self.x_history = x_history
        self.y_history = y_history
        
    def __eq__(self, other):
        if not isinstance(other, JointHistory):
            return False
        return self.x_history == other.x_history and self.y_history == other.y_history
    
    def __hash__(self):
        return hash((self.x_history, self.y_history))
    
    def __repr__(self):
        return f"JointHistory('{self.x_history}', '{self.y_history}')"
    
    def __str__(self):
        return f"({self.x_history}, {self.y_history})"
    
    def extend(self, x_symbol: str, y_symbol: str, max_length: int) -> 'JointHistory':
        """Extend history with new symbols, keeping max_length."""
        if len(self.x_history) >= max_length:
            new_x = self.x_history[1:] + x_symbol
        else:
            new_x = self.x_history + x_symbol
            
        if len(self.y_history) >= max_length:
            new_y = self.y_history[1:] + y_symbol
        else:
            new_y = self.y_history + y_symbol
            
        return JointHistory(new_x, new_y)


class CausalState:
    """Represents a causal state in the epsilon-transducer."""
    
    def __init__(self, state_id: int):
        self.state_id = state_id
        self.histories: Set[JointHistory] = set()
        self.morph: List[int] = []  # Counts for each emission symbol
        
    def add_history(self, history: JointHistory):
        """Add a history to this state."""
        self.histories.add(history)
        
    def remove_history(self, history: JointHistory):
        """Remove a history from this state."""
        self.histories.discard(history)
        
    def get_emission_probabilities(self, x_symbols: List[str], y_symbols: List[str]) -> List[float]:
        """Get emission probabilities for all (x,y) combinations."""
        if not self.morph or sum(self.morph) == 0:
            return [0.0] * len(self.morph)
        
        total = sum(self.morph)
        return [count / total for count in self.morph]
    
    def __repr__(self):
        return f"CausalState({self.state_id}, {len(self.histories)} histories)"


class TransCSSRCompatibleCSSR:
    """CSSR implementation compatible with transCSSR reference."""
    
    def __init__(self, x_symbols: List[str], y_symbols: List[str], 
                 significance_level: float = 0.001, test_type: str = 'chi2'):
        self.x_symbols = x_symbols
        self.y_symbols = y_symbols
        self.significance_level = significance_level
        self.test_type = test_type
        
        # Create all possible emission symbols (x, y) combinations
        self.emission_symbols = []
        for x in x_symbols:
            for y in y_symbols:
                self.emission_symbols.append((x, y))
        
        # Data structures
        self.word_lookup_marg: Dict[JointHistory, int] = defaultdict(int)
        self.word_lookup_fut: Dict[Tuple[JointHistory, str], int] = defaultdict(int)
        
        # CSSR state structures
        self.epsilon: Dict[JointHistory, int] = {}  # history -> state_id
        self.invepsilon: Dict[int, Set[JointHistory]] = defaultdict(set)  # state_id -> histories
        self.morph_by_state: Dict[int, List[int]] = {}  # state_id -> emission counts
        
        self.num_states = 0
        
    def load_data(self, string_x: str, string_y: str, max_length: int):
        """Load data and compute predictive distributions like transCSSR."""
        print(f"Loading data: X={len(string_x)}, Y={len(string_y)}, L_max={max_length}")
        
        if len(string_x) != len(string_y):
            raise ValueError("Input and output strings must have same length")
        
        # Clear existing data
        self.word_lookup_marg.clear()
        self.word_lookup_fut.clear()
        
        # Build word lookup tables
        for t in range(len(string_y)):
            for history_len in range(1, max_length + 1):
                if t >= history_len:
                    # Create joint history
                    x_hist = string_x[t-history_len:t]
                    y_hist = string_y[t-history_len:t]
                    history = JointHistory(x_hist, y_hist)
                    
                    # Count marginal (past) occurrences
                    self.word_lookup_marg[history] += 1
                    
                    # Count future occurrences (past + next symbol)
                    if t < len(string_y):
                        next_y = string_y[t]
                        self.word_lookup_fut[(history, next_y)] += 1
        
        print(f"Loaded {len(self.word_lookup_marg)} unique histories")
        print(f"Loaded {len(self.word_lookup_fut)} history-future pairs")
    
    def chi_square_test(self, morph1: List[int], morph2: List[int]) -> Tuple[bool, float]:
        """Chi-square test following transCSSR implementation exactly."""
        if len(morph1) != len(morph2):
            return False, 1.0
        
        # Convert to numpy arrays
        array_morph1 = np.array(morph1)
        array_morph2 = np.array(morph2)
        
        chi_squared_statistic = 0.0
        df_x = 0
        
        # Process by "row" (input symbol) - this is the transCSSR logic
        for row_ind in range(len(self.x_symbols)):
            # Extract counts for this input symbol (FIX: was using len(axs), should be len(ays))
            start_idx = row_ind * len(self.y_symbols)
            end_idx = start_idx + len(self.y_symbols)
            
            tmp_morph1 = array_morph1[start_idx:end_idx]
            tmp_morph2 = array_morph2[start_idx:end_idx]
            
            if np.sum(tmp_morph1) == 0 or np.sum(tmp_morph2) == 0:
                continue
            
            df_x += 1  # This row is non-zero in the stochastic matrix
            n1 = np.sum(tmp_morph1)
            n2 = np.sum(tmp_morph2)
            
            ns = [n1, n2]
            
            # Compute expected frequencies under null hypothesis
            theta0 = (tmp_morph1 + tmp_morph2) / float(n1 + n2)
            
            contingency_table = np.vstack((tmp_morph1, tmp_morph2))
            
            # Compute chi-square statistic exactly like transCSSR
            for array_ind in range(len(tmp_morph1)):
                for sample_ind in range(2):
                    denominator = ns[sample_ind] * theta0[array_ind]
                    if denominator == 0:
                        continue
                    else:
                        numerator = np.power(contingency_table[sample_ind, array_ind] - ns[sample_ind] * theta0[array_ind], 2)
                        chi_squared_statistic += numerator / float(denominator)
        
        # Degrees of freedom
        df = df_x * (len(self.y_symbols) - 1)
        
        if df <= 0:
            return False, 1.0
        
        # Test significance and p-value exactly like transCSSR
        is_significant = chi_squared_statistic > stats.chi2.ppf(1 - self.significance_level, df)
        p_value = 1 - stats.chi2.cdf(chi_squared_statistic, df)
        
        return is_significant, p_value
    
    def initialize_states(self):
        """Initialize causal states following transCSSR: start with null history."""
        print("Initializing causal states...")
        
        self.epsilon.clear()
        self.invepsilon.clear()
        self.morph_by_state.clear()
        self.num_states = 0
        
        # Initialize with null histories (empty strings) like transCSSR
        null_history = JointHistory('', '')
        
        # Create first state with null history
        state_id = 0
        self.epsilon[null_history] = state_id
        self.invepsilon[state_id].add(null_history)
        
        # Initialize morph for null history
        self.morph_by_state[state_id] = [0] * len(self.emission_symbols)
        
        # Populate morph with counts for each emission symbol
        for emit_idx, (x_emit, y_emit) in enumerate(self.emission_symbols):
            future_key = (null_history, y_emit)
            count = self.word_lookup_fut.get(future_key, 0)
            self.morph_by_state[state_id][emit_idx] = count
        
        self.num_states = 1
        print(f"Initialized with 1 state (null history)")
    
    def determinization_step(self, max_length: int) -> bool:
        """Determinization step: split states based on transition differences."""
        print("Running determinization step...")
        
        has_changes = False
        states_to_process = list(self.invepsilon.keys())
        
        for state_id in states_to_process:
            if state_id not in self.invepsilon:
                continue
                
            histories = list(self.invepsilon[state_id])
            
            if len(histories) <= 1:
                continue
            
            # Check for transition differences
            for emit_x, emit_y in self.emission_symbols:
                # Find transitions for each history
                transitions = {}
                for history in histories:
                    # Create next history
                    next_history = history.extend(emit_x, emit_y, max_length)
                    next_state = self.epsilon.get(next_history, -1)
                    
                    if next_state not in transitions:
                        transitions[next_state] = []
                    transitions[next_state].append(history)
                
                # If we have different transitions, split the state
                if len(transitions) > 1:
                    print(f"  Splitting state {state_id} on emission ({emit_x}, {emit_y})")
                    
                    # Keep first transition group in original state
                    first_transition = list(transitions.keys())[0]
                    keep_histories = transitions[first_transition]
                    
                    # Create new states for other transition groups
                    for trans_state, split_histories in list(transitions.items())[1:]:
                        if trans_state != first_transition:
                            new_state_id = self.num_states
                            self.num_states += 1
                            
                            # Move histories to new state
                            for hist in split_histories:
                                self.invepsilon[state_id].remove(hist)
                                self.invepsilon[new_state_id].add(hist)
                                self.epsilon[hist] = new_state_id
                            
                            # Initialize morph for new state
                            self.morph_by_state[new_state_id] = [0] * len(self.emission_symbols)
                            for emit_idx, (x_e, y_e) in enumerate(self.emission_symbols):
                                count = 0
                                for hist in self.invepsilon[new_state_id]:
                                    future_key = (hist, y_e)
                                    count += self.word_lookup_fut.get(future_key, 0)
                                self.morph_by_state[new_state_id][emit_idx] = count
                            
                            has_changes = True
                            print(f"    Created new state {new_state_id} with {len(split_histories)} histories")
                    
                    # Update original state morph
                    for emit_idx, (x_e, y_e) in enumerate(self.emission_symbols):
                        count = 0
                        for hist in self.invepsilon[state_id]:
                            future_key = (hist, y_e)
                            count += self.word_lookup_fut.get(future_key, 0)
                        self.morph_by_state[state_id][emit_idx] = count
                    
                    break  # Only split once per state per iteration
        
        return has_changes
    
    def homogenization_step(self) -> bool:
        """Homogenization step: merge states with statistically similar emissions."""
        print("Running homogenization step...")
        
        has_changes = False
        state_ids = list(self.invepsilon.keys())
        
        for i in range(len(state_ids)):
            if state_ids[i] not in self.invepsilon:
                continue
                
            for j in range(i + 1, len(state_ids)):
                if state_ids[j] not in self.invepsilon:
                    continue
                
                # Test if states are statistically equivalent
                morph1 = self.morph_by_state[state_ids[i]]
                morph2 = self.morph_by_state[state_ids[j]]
                
                is_different, p_value = self.chi_square_test(morph1, morph2)
                
                if not is_different:  # States are equivalent, merge them
                    print(f"  Merging states {state_ids[i]} and {state_ids[j]} (p={p_value:.4f})")
                    
                    # Merge histories
                    for hist in self.invepsilon[state_ids[j]]:
                        self.epsilon[hist] = state_ids[i]
                        self.invepsilon[state_ids[i]].add(hist)
                    
                    # Merge morphs
                    for k in range(len(self.emission_symbols)):
                        self.morph_by_state[state_ids[i]][k] += self.morph_by_state[state_ids[j]][k]
                    
                    # Remove merged state
                    del self.invepsilon[state_ids[j]]
                    del self.morph_by_state[state_ids[j]]
                    
                    has_changes = True
                    break
            
            if has_changes:
                break
        
        return has_changes
    
    def run_cssr(self, string_x: str, string_y: str, max_length: int) -> bool:
        """Run the complete CSSR algorithm following transCSSR structure."""
        print(f"Running TransCSSR-compatible CSSR algorithm...")
        print(f"Data: X={len(string_x)}, Y={len(string_y)}, L_max={max_length}")
        
        # Load data
        self.load_data(string_x, string_y, max_length)
        
        # Initialize states with null history
        self.initialize_states()
        
        # Main CSSR loop: grow histories incrementally by length
        print("\nHomogenizing (growing histories by length)...")
        for L_cur in range(0, max_length):
            print(f"\n--- Processing length {L_cur} ---")
            
            # Get current states (copy to avoid modification during iteration)
            current_states = list(self.invepsilon.keys())
            
            for state_id in current_states:
                if state_id not in self.invepsilon:
                    continue
                    
                # Get histories for this state (copy to avoid modification)
                current_histories = list(self.invepsilon[state_id])
                
                for history in current_histories:
                    # Only process histories of current length L_cur
                    if len(history.x_history) != L_cur:
                        continue
                        
                    # Try to grow this history in all possible ways
                    self.grow_history(history, state_id, L_cur)
            
            print(f"States after length {L_cur}: {len(self.invepsilon)}")
        
        print(f"CSSR algorithm completed. Final states: {len(self.invepsilon)}")
        return True
    
    def get_results(self) -> Dict[str, Any]:
        """Get results summary."""
        return {
            'num_states': len(self.invepsilon),
            'states': {
                state_id: {
                    'histories': [str(h) for h in histories],
                    'morph': morph
                }
                for state_id, histories in self.invepsilon.items()
                for morph in [self.morph_by_state.get(state_id, [])]
            },
            'convergence': True
        }