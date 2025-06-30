"""
Sequence processor for Neural CSSR dataset generation.

This module handles raw sequence generation from epsilon-machines with
complete state trajectory tracking and transition metadata.
"""

from typing import Dict, List, Any, Tuple
import random
import numpy as np
from ..core.epsilon_machine import EpsilonMachine


class SequenceProcessor:
    """
    Processes epsilon-machines to generate sequences with rich metadata.
    
    Handles sequence generation with complete state tracking, transition logging,
    and statistical metadata computation for downstream analysis.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize sequence processor.
        
        Args:
            seed: Random seed for reproducible sequence generation
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
    def process_machine_sequences(
        self, 
        machine: EpsilonMachine, 
        num_sequences: int,
        length_distribution: Tuple[int, int],
        track_states: bool = True
    ) -> Dict[str, Any]:
        """
        Generate sequences from a single machine with complete metadata.
        
        Args:
            machine: EpsilonMachine to generate sequences from
            num_sequences: Number of sequences to generate
            length_distribution: (min_length, max_length) for sequence lengths
            track_states: Whether to track state trajectories
            
        Returns:
            Dictionary containing:
            - 'sequences': List[List[str]] - Raw symbol sequences
            - 'state_trajectories': List[List[str]] - State at each step
            - 'transition_log': List[List[Dict]] - Detailed transition info
            - 'generation_metadata': Dict - Statistics and parameters
        """
        sequences = []
        state_trajectories = []
        transition_log = []
        
        min_length, max_length = length_distribution
        
        for seq_idx in range(num_sequences):
            # Generate random sequence length
            sequence_length = random.randint(min_length, max_length)
            
            # Generate sequence with tracking
            if track_states:
                seq_data = self._generate_sequence_with_tracking(
                    machine, sequence_length, seq_idx
                )
                sequences.append(seq_data['sequence'])
                state_trajectories.append(seq_data['states'])
                transition_log.append(seq_data['transitions'])
            else:
                # Faster generation without tracking
                sequence = machine.generate_sequence(sequence_length)
                sequences.append(sequence)
                state_trajectories.append([])
                transition_log.append([])
                
        # Compute generation metadata
        generation_metadata = self._compute_generation_metadata(
            sequences, state_trajectories, transition_log, machine
        )
        
        return {
            'sequences': sequences,
            'state_trajectories': state_trajectories,
            'transition_log': transition_log,
            'generation_metadata': generation_metadata
        }
        
    def _generate_sequence_with_tracking(
        self, 
        machine: EpsilonMachine, 
        length: int, 
        sequence_id: int
    ) -> Dict[str, Any]:
        """
        Generate single sequence with complete state and transition tracking.
        
        Args:
            machine: EpsilonMachine to use
            length: Sequence length
            sequence_id: Unique sequence identifier
            
        Returns:
            Dictionary with sequence, states, and transition details
        """
        sequence = []
        states = []
        transitions = []
        
        # Start from initial state
        current_state = machine.start_state
        states.append(current_state)
        
        for step in range(length):
            # Get available transitions from current state
            available_transitions = []
            for symbol in machine.alphabet:
                key = (current_state, symbol)
                if key in machine.transitions:
                    for next_state, prob in machine.transitions[key]:
                        available_transitions.append({
                            'symbol': symbol,
                            'next_state': next_state,
                            'probability': prob,
                            'transition_key': key
                        })
                        
            if not available_transitions:
                # No available transitions - shouldn't happen in valid machines
                break
                
            # Sample transition based on probabilities
            transition_probs = [t['probability'] for t in available_transitions]
            chosen_idx = np.random.choice(len(available_transitions), p=transition_probs)
            chosen_transition = available_transitions[chosen_idx]
            
            # Record transition details
            transition_info = {
                'step': step,
                'from_state': current_state,
                'symbol': chosen_transition['symbol'],
                'to_state': chosen_transition['next_state'],
                'probability': chosen_transition['probability'],
                'num_available': len(available_transitions),
                'sequence_id': sequence_id
            }
            
            # Update sequence and state
            sequence.append(chosen_transition['symbol'])
            current_state = chosen_transition['next_state']
            states.append(current_state)
            transitions.append(transition_info)
            
        return {
            'sequence': sequence,
            'states': states,
            'transitions': transitions
        }
        
    def _compute_generation_metadata(
        self, 
        sequences: List[List[str]], 
        state_trajectories: List[List[str]], 
        transition_log: List[List[Dict]], 
        machine: EpsilonMachine
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metadata about the generated sequences.
        
        Args:
            sequences: Generated sequences
            state_trajectories: State trajectories for each sequence
            transition_log: Transition details for each sequence
            machine: Source machine
            
        Returns:
            Dictionary of generation metadata
        """
        if not sequences:
            return {}
            
        # Basic sequence statistics
        lengths = [len(seq) for seq in sequences]
        total_symbols = sum(lengths)
        
        # Symbol frequency analysis
        symbol_counts = {}
        for seq in sequences:
            for symbol in seq:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                
        symbol_frequencies = {
            symbol: count / total_symbols 
            for symbol, count in symbol_counts.items()
        }
        
        # State usage analysis (if tracking enabled)
        state_usage = {}
        if state_trajectories and state_trajectories[0]:
            for trajectory in state_trajectories:
                for state in trajectory:
                    state_usage[state] = state_usage.get(state, 0) + 1
                    
            total_state_steps = sum(state_usage.values())
            state_frequencies = {
                state: count / total_state_steps 
                for state, count in state_usage.items()
            }
        else:
            state_frequencies = {}
            
        # Transition analysis
        transition_stats = self._analyze_transitions(transition_log)
        
        # Sequence diversity metrics
        unique_sequences = len(set(''.join(seq) for seq in sequences))
        diversity_ratio = unique_sequences / len(sequences)
        
        # Length statistics
        length_stats = {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'median_length': np.median(lengths)
        }
        
        return {
            'num_sequences': len(sequences),
            'total_symbols': total_symbols,
            'unique_sequences': unique_sequences,
            'diversity_ratio': diversity_ratio,
            'length_statistics': length_stats,
            'symbol_frequencies': symbol_frequencies,
            'state_frequencies': state_frequencies,
            'transition_statistics': transition_stats,
            'machine_properties': {
                'num_states': len(machine.states),
                'alphabet_size': len(machine.alphabet),
                'is_topological': machine.is_topological()
            }
        }
        
    def _analyze_transitions(self, transition_log: List[List[Dict]]) -> Dict[str, Any]:
        """
        Analyze transition patterns in the generated sequences.
        
        Args:
            transition_log: List of transition logs for each sequence
            
        Returns:
            Dictionary of transition statistics
        """
        if not transition_log or not transition_log[0]:
            return {}
            
        # Flatten all transitions
        all_transitions = []
        for seq_transitions in transition_log:
            all_transitions.extend(seq_transitions)
            
        if not all_transitions:
            return {}
            
        # Transition type counts
        transition_counts = {}
        for trans in all_transitions:
            key = (trans['from_state'], trans['symbol'], trans['to_state'])
            transition_counts[key] = transition_counts.get(key, 0) + 1
            
        total_transitions = len(all_transitions)
        transition_frequencies = {
            f"{key[0]}-{key[1]}->{key[2]}": count / total_transitions
            for key, count in transition_counts.items()
        }
        
        # State transition statistics
        state_transitions = {}
        for trans in all_transitions:
            key = (trans['from_state'], trans['to_state'])
            state_transitions[key] = state_transitions.get(key, 0) + 1
            
        # Probability distribution analysis
        prob_values = [trans['probability'] for trans in all_transitions]
        prob_stats = {
            'mean_probability': np.mean(prob_values),
            'std_probability': np.std(prob_values),
            'min_probability': min(prob_values),
            'max_probability': max(prob_values)
        }
        
        return {
            'total_transitions': total_transitions,
            'unique_transition_types': len(transition_counts),
            'transition_frequencies': transition_frequencies,
            'state_transition_counts': {
                f"{key[0]}->{key[1]}": count 
                for key, count in state_transitions.items()
            },
            'probability_statistics': prob_stats
        }
        
    def create_classical_cssr_format(self, sequences: List[List[str]]) -> List[str]:
        """
        Convert sequences to classical CSSR format (plain text).
        
        Args:
            sequences: List of symbol sequences
            
        Returns:
            List of strings, one sequence per line
        """
        return [''.join(seq) for seq in sequences]
        
    def export_sequences_to_file(self, sequences: List[List[str]], filepath: str) -> None:
        """
        Export sequences to text file for classical CSSR.
        
        Args:
            sequences: List of symbol sequences
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            for seq in sequences:
                f.write(''.join(seq) + '\n')
                
    def compute_sequence_quality_metrics(self, sequences_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute quality metrics for generated sequences.
        
        Args:
            sequences_data: Output from process_machine_sequences
            
        Returns:
            Dictionary of quality metrics
        """
        sequences = sequences_data['sequences']
        metadata = sequences_data['generation_metadata']
        
        if not sequences:
            return {'quality_score': 0.0, 'issues': ['No sequences generated']}
            
        issues = []
        quality_components = {}
        
        # Check sequence length distribution
        lengths = [len(seq) for seq in sequences]
        if max(lengths) - min(lengths) < 2:
            issues.append('Insufficient length diversity')
        quality_components['length_diversity'] = (max(lengths) - min(lengths)) / max(lengths)
        
        # Check symbol distribution balance
        symbol_freqs = metadata.get('symbol_frequencies', {})
        if symbol_freqs:
            freq_values = list(symbol_freqs.values())
            expected_freq = 1.0 / len(symbol_freqs)
            balance_score = 1.0 - max(abs(f - expected_freq) for f in freq_values) / expected_freq
            quality_components['symbol_balance'] = balance_score
            
            if balance_score < 0.8:
                issues.append('Unbalanced symbol distribution')
        else:
            quality_components['symbol_balance'] = 0.0
            issues.append('No symbol frequency data')
            
        # Check state coverage (if available)
        state_freqs = metadata.get('state_frequencies', {})
        if state_freqs:
            # All states should appear at least once
            coverage_score = len(state_freqs) / metadata['machine_properties']['num_states']
            quality_components['state_coverage'] = coverage_score
            
            if coverage_score < 1.0:
                issues.append('Incomplete state coverage')
        else:
            quality_components['state_coverage'] = 1.0  # Assume OK if not tracking
            
        # Overall quality score
        quality_score = np.mean(list(quality_components.values()))
        
        return {
            'quality_score': quality_score,
            'components': quality_components,
            'issues': issues,
            'recommendations': self._generate_recommendations(quality_components, issues)
        }
        
    def _generate_recommendations(self, quality_components: Dict[str, float], 
                                issues: List[str]) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        if quality_components.get('length_diversity', 1.0) < 0.5:
            recommendations.append('Increase sequence length range for better diversity')
            
        if quality_components.get('symbol_balance', 1.0) < 0.8:
            recommendations.append('Check machine transition probabilities for symbol balance')
            
        if quality_components.get('state_coverage', 1.0) < 1.0:
            recommendations.append('Generate more sequences to ensure all states are visited')
            
        if not recommendations:
            recommendations.append('Sequence quality is good')
            
        return recommendations