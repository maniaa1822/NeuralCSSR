"""Classical CSSR algorithm that works with generated dataset format."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import torch
from ..data.dataset_generation import EpsilonMachineDataset
from .statistical_tests import StatisticalTests


@dataclass
class CSSRCausalState:
    """Represents a causal state discovered by CSSR."""
    state_id: int
    histories: Set[str]  # Set of histories belonging to this state
    next_symbol_distribution: Dict[str, int]  # Counts of next symbols
    total_count: int
    
    def get_probability_distribution(self) -> Dict[str, float]:
        """Get P(next_symbol | causal_state)."""
        if self.total_count == 0:
            return {}
        return {symbol: count / self.total_count 
                for symbol, count in self.next_symbol_distribution.items()}


class ClassicalCSSR:
    """Classical CSSR algorithm that works with EpsilonMachineDataset."""
    
    def __init__(
        self,
        significance_level: float = 0.05,
        min_count: int = 5,
        test_type: str = "chi_square"
    ):
        self.significance_level = significance_level
        self.min_count = min_count
        self.test_type = test_type
        
        self.statistical_tests = StatisticalTests(significance_level)
        
        # Data structures
        self.history_data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.history_counts: Dict[str, int] = defaultdict(int)
        self.alphabet: Set[str] = set()
        
        # CSSR results
        self.causal_states: List[CSSRCausalState] = []
        self.history_to_state: Dict[str, int] = {}
        
    def load_from_dataset(self, dataset: EpsilonMachineDataset):
        """Load data from an EpsilonMachineDataset."""
        print(f"Loading data from dataset with {len(dataset)} examples...")
        
        for i in range(len(dataset)):
            example = dataset.examples[i]  # Get raw example, not tensor version
            
            history = example['history']
            target = example['target']
            
            # Convert history list to string for hashing
            history_str = ''.join(history) if history else ''
            
            # Record this history -> next_symbol observation
            self.history_data[history_str][target] += 1
            self.history_counts[history_str] += 1
            self.alphabet.add(target)
            
            # Also add all symbols in history to alphabet
            for symbol in history:
                self.alphabet.add(symbol)
        
        self.alphabet = sorted(list(self.alphabet))
        print(f"Loaded {len(self.history_data)} unique histories")
        print(f"Alphabet: {self.alphabet}")
        print(f"Total observations: {sum(self.history_counts.values())}")
    
    def load_from_raw_data(self, data_list, metadata):
        """Load data from raw data format (list of dicts)."""
        print(f"Loading data from raw format with {len(data_list)} examples...")
        
        for item in data_list:
            history = item['raw_history']
            target = item['raw_target']
            
            # Convert history list to string for hashing
            history_str = ''.join(history) if history else ''
            
            # Record this history -> next_symbol observation
            self.history_data[history_str][target] += 1
            self.history_counts[history_str] += 1
            self.alphabet.add(target)
            
            # Also add all symbols in history to alphabet
            for symbol in history:
                self.alphabet.add(symbol)
        
        self.alphabet = sorted(list(self.alphabet))
        print(f"Loaded {len(self.history_data)} unique histories")
        print(f"Alphabet: {self.alphabet}")
        print(f"Total observations: {sum(self.history_counts.values())}")
        
    def get_sufficient_histories(self) -> Set[str]:
        """Get histories with sufficient observation counts."""
        return {hist for hist, count in self.history_counts.items() 
                if count >= self.min_count}
    
    def histories_are_equivalent(self, hist1: str, hist2: str) -> Tuple[bool, Dict]:
        """Test if two histories have equivalent next-symbol distributions."""
        if (self.history_counts[hist1] < self.min_count or 
            self.history_counts[hist2] < self.min_count):
            return False, {'reason': 'insufficient_data'}
        
        dist1 = self.history_data[hist1]
        dist2 = self.history_data[hist2]
        
        # Use statistical test to compare distributions
        are_different, test_results = self.statistical_tests.sufficient_statistics_test(
            dist1, dist2, self.test_type
        )
        
        # Equivalent means NOT significantly different
        return not are_different, test_results
    
    def initialize_states(self):
        """Initialize causal states with single-symbol histories."""
        print("Initializing causal states...")
        
        self.causal_states = []
        self.history_to_state = {}
        
        sufficient_histories = self.get_sufficient_histories()
        single_symbol_histories = {h for h in sufficient_histories if len(h) == 1}
        
        state_id = 0
        for history in sorted(single_symbol_histories):
            # Create new causal state
            state = CSSRCausalState(
                state_id=state_id,
                histories={history},
                next_symbol_distribution=dict(self.history_data[history]),
                total_count=self.history_counts[history]
            )
            
            self.causal_states.append(state)
            self.history_to_state[history] = state_id
            state_id += 1
        
        print(f"Initialized {len(self.causal_states)} states from single-symbol histories")
    
    def merge_equivalent_states(self) -> bool:
        """Merge causal states that are statistically equivalent."""
        changes_made = False
        
        # Compare all pairs of states
        for i in range(len(self.causal_states)):
            for j in range(i + 1, len(self.causal_states)):
                if not self.causal_states[i].histories or not self.causal_states[j].histories:
                    continue  # Skip empty states
                
                # Get representative histories from each state
                hist1 = next(iter(self.causal_states[i].histories))
                hist2 = next(iter(self.causal_states[j].histories))
                
                are_equivalent, _ = self.histories_are_equivalent(hist1, hist2)
                
                if are_equivalent:
                    # Merge state j into state i
                    state_i = self.causal_states[i]
                    state_j = self.causal_states[j]
                    
                    # Transfer all histories
                    for hist in state_j.histories:
                        state_i.histories.add(hist)
                        self.history_to_state[hist] = i
                        
                        # Update distribution
                        for symbol, count in self.history_data[hist].items():
                            state_i.next_symbol_distribution[symbol] = (
                                state_i.next_symbol_distribution.get(symbol, 0) + count
                            )
                        state_i.total_count += self.history_counts[hist]
                    
                    # Clear the merged state
                    state_j.histories.clear()
                    changes_made = True
        
        # Remove empty states
        self.causal_states = [s for s in self.causal_states if s.histories]
        
        # Update state IDs
        for new_id, state in enumerate(self.causal_states):
            old_id = state.state_id
            state.state_id = new_id
            
            # Update history mappings
            for hist in state.histories:
                self.history_to_state[hist] = new_id
        
        return changes_made
    
    def assign_longer_histories(self) -> bool:
        """Assign longer histories to existing causal states or create new ones."""
        changes_made = False
        sufficient_histories = self.get_sufficient_histories()
        
        # Group histories by length
        by_length = defaultdict(list)
        for hist in sufficient_histories:
            by_length[len(hist)].append(hist)
        
        # Process histories in order of increasing length
        for length in sorted(by_length.keys()):
            if length == 1:
                continue  # Already processed
            
            for history in by_length[length]:
                if history in self.history_to_state:
                    continue  # Already assigned
                
                # Try to assign to existing state
                assigned = False
                
                for state_idx, state in enumerate(self.causal_states):
                    if not state.histories:
                        continue
                    
                    # Test against a representative history from this state
                    representative = next(iter(state.histories))
                    are_equivalent, _ = self.histories_are_equivalent(history, representative)
                    
                    if are_equivalent:
                        # Add to this state
                        state.histories.add(history)
                        self.history_to_state[history] = state_idx
                        
                        # Update distribution
                        for symbol, count in self.history_data[history].items():
                            state.next_symbol_distribution[symbol] = (
                                state.next_symbol_distribution.get(symbol, 0) + count
                            )
                        state.total_count += self.history_counts[history]
                        
                        assigned = True
                        changes_made = True
                        break
                
                if not assigned:
                    # Create new state
                    new_state = CSSRCausalState(
                        state_id=len(self.causal_states),
                        histories={history},
                        next_symbol_distribution=dict(self.history_data[history]),
                        total_count=self.history_counts[history]
                    )
                    
                    self.causal_states.append(new_state)
                    self.history_to_state[history] = new_state.state_id
                    changes_made = True
        
        return changes_made
    
    def run_cssr(self, max_iterations: int = 20) -> bool:
        """Run the complete CSSR algorithm."""
        print("Starting CSSR algorithm...")
        
        # Step 1: Initialize with single-symbol histories
        self.initialize_states()
        
        # Step 2: Iteratively refine states
        for iteration in range(max_iterations):
            print(f"\nCSSR iteration {iteration + 1}")
            
            changes_made = False
            
            # Merge equivalent states
            print("  Merging equivalent states...")
            if self.merge_equivalent_states():
                changes_made = True
                print(f"    After merging: {len(self.causal_states)} states")
            
            # Assign longer histories
            print("  Assigning longer histories...")
            if self.assign_longer_histories():
                changes_made = True
                print(f"    After assignment: {len(self.causal_states)} states")
            
            if not changes_made:
                print(f"  Converged after {iteration + 1} iterations")
                return True
        
        print(f"  Stopped after {max_iterations} iterations (may not have converged)")
        return False
    
    def evaluate_against_ground_truth(self, dataset: EpsilonMachineDataset) -> Dict[str, float]:
        """Evaluate CSSR results against ground truth from dataset."""
        if not self.causal_states:
            return {'error': 'CSSR not yet run'}
        
        correct_predictions = 0
        total_predictions = 0
        
        # Test on a subset of the dataset
        for i in range(min(1000, len(dataset))):  # Sample to avoid expensive computation
            example = dataset.examples[i]
            history_str = ''.join(example['history']) if example['history'] else ''
            true_target = example['target']
            
            if history_str in self.history_to_state:
                # Get predicted distribution
                state_id = self.history_to_state[history_str]
                state = self.causal_states[state_id]
                pred_dist = state.get_probability_distribution()
                
                # Predict most likely next symbol
                if pred_dist:
                    predicted_target = max(pred_dist.keys(), key=pred_dist.get)
                    if predicted_target == true_target:
                        correct_predictions += 1
                
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'num_states_discovered': len(self.causal_states),
            'coverage': len(self.history_to_state) / len(self.history_data),
            'predictions_made': total_predictions
        }
    
    def evaluate_against_raw_data(self, data_list) -> Dict[str, float]:
        """Evaluate CSSR results against ground truth from raw data."""
        if not self.causal_states:
            return {'error': 'CSSR not yet run'}
        
        correct_predictions = 0
        total_predictions = 0
        
        # Test on a subset of the data
        for i, item in enumerate(data_list[:1000]):  # Sample to avoid expensive computation
            history = item['raw_history']
            true_target = item['raw_target']
            history_str = ''.join(history) if history else ''
            
            if history_str in self.history_to_state:
                # Get predicted distribution
                state_id = self.history_to_state[history_str]
                state = self.causal_states[state_id]
                pred_dist = state.get_probability_distribution()
                
                # Predict most likely next symbol
                if pred_dist:
                    predicted_target = max(pred_dist.keys(), key=pred_dist.get)
                    if predicted_target == true_target:
                        correct_predictions += 1
                
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'num_states_discovered': len(self.causal_states),
            'coverage': len(self.history_to_state) / len(self.history_data),
            'predictions_made': total_predictions
        }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of CSSR results."""
        return {
            'num_causal_states': len(self.causal_states),
            'total_histories_assigned': len(self.history_to_state),
            'total_histories_observed': len(self.history_data),
            'coverage': len(self.history_to_state) / len(self.history_data) if self.history_data else 0,
            'causal_states': [
                {
                    'state_id': state.state_id,
                    'num_histories': len(state.histories),
                    'total_observations': state.total_count,
                    'next_symbol_distribution': state.get_probability_distribution(),
                    'sample_histories': list(state.histories)[:5]  # Show a few examples
                }
                for state in self.causal_states
            ]
        }