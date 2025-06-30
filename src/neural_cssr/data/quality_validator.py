"""
Dataset quality validator for Neural CSSR.

This module validates dataset quality through coverage analysis,
distribution checks, and consistency verification.
"""

from typing import Dict, List, Any
import numpy as np
from collections import Counter, defaultdict


class DatasetQualityValidator:
    """
    Validates dataset quality and generates quality reports.
    
    Ensures adequate statistical coverage, balanced distributions,
    and consistency between theoretical and empirical measures.
    """
    
    def __init__(self):
        """Initialize quality validator."""
        pass
        
    def validate_coverage(self, sequences_data: Dict[str, Any], 
                         quality_spec: Any) -> Dict[str, Any]:
        """
        Ensure adequate statistical coverage across states and transitions.
        
        Args:
            sequences_data: Dictionary containing sequences and metadata
            quality_spec: Quality specification with thresholds
            
        Returns:
            Dictionary with coverage validation results
        """
        metadata = sequences_data.get('metadata', [])
        
        if not metadata:
            return {
                'overall_score': 0.0,
                'issues': ['No metadata available for coverage analysis'],
                'state_coverage': {},
                'transition_coverage': {}
            }
            
        # State coverage analysis
        state_coverage = self._validate_state_coverage(metadata, quality_spec)
        
        # Transition coverage analysis  
        transition_coverage = self._validate_transition_coverage(metadata, quality_spec)
        
        # Length coverage analysis
        length_coverage = self._validate_length_coverage(sequences_data, quality_spec)
        
        # Compute overall coverage score
        coverage_scores = [
            state_coverage.get('score', 0.0),
            transition_coverage.get('score', 0.0),
            length_coverage.get('score', 0.0)
        ]
        overall_score = np.mean(coverage_scores)
        
        # Collect all issues
        all_issues = []
        all_issues.extend(state_coverage.get('issues', []))
        all_issues.extend(transition_coverage.get('issues', []))
        all_issues.extend(length_coverage.get('issues', []))
        
        return {
            'overall_score': overall_score,
            'issues': all_issues,
            'state_coverage': state_coverage,
            'transition_coverage': transition_coverage,
            'length_coverage': length_coverage,
            'recommendations': self._generate_coverage_recommendations(
                state_coverage, transition_coverage, length_coverage
            )
        }
        
    def _validate_state_coverage(self, metadata: List[Dict], quality_spec: Any) -> Dict[str, Any]:
        """Validate causal state coverage."""
        min_coverage = getattr(quality_spec, 'min_state_coverage', 50)
        
        # Count state appearances across all sequences
        state_counts = defaultdict(int)
        machine_states = defaultdict(set)
        
        for seq_meta in metadata:
            machine_id = seq_meta.get('machine_id', -1)
            state_trajectory = seq_meta.get('state_trajectory', [])
            
            # Track states for this machine
            for state in state_trajectory:
                if state:  # Skip empty states
                    state_counts[(machine_id, state)] += 1
                    machine_states[machine_id].add(state)
                    
        # Check coverage for each machine
        coverage_results = {}
        issues = []
        
        for machine_id, states in machine_states.items():
            machine_coverage = {}
            
            for state in states:
                count = state_counts[(machine_id, state)]
                machine_coverage[state] = count
                
                if count < min_coverage:
                    issues.append(
                        f"Machine {machine_id}, state {state}: {count} appearances "
                        f"(minimum: {min_coverage})"
                    )
                    
            coverage_results[machine_id] = machine_coverage
            
        # Compute coverage score
        total_state_instances = len(state_counts)
        adequate_coverage = sum(
            1 for count in state_counts.values() 
            if count >= min_coverage
        )
        
        coverage_score = adequate_coverage / total_state_instances if total_state_instances > 0 else 0.0
        
        return {
            'score': coverage_score,
            'issues': issues,
            'state_counts': dict(state_counts),
            'machine_states': {k: list(v) for k, v in machine_states.items()},
            'adequate_coverage_ratio': coverage_score,
            'min_threshold': min_coverage
        }
        
    def _validate_transition_coverage(self, metadata: List[Dict], quality_spec: Any) -> Dict[str, Any]:
        """Validate transition coverage."""
        min_coverage = getattr(quality_spec, 'min_transition_coverage', 20)
        
        # Count transition appearances
        transition_counts = defaultdict(int)
        
        for seq_meta in metadata:
            transition_log = seq_meta.get('transition_log', [])
            
            for transition in transition_log:
                if isinstance(transition, dict):
                    from_state = transition.get('from_state', '')
                    symbol = transition.get('symbol', '')
                    to_state = transition.get('to_state', '')
                    
                    if from_state and symbol and to_state:
                        key = (from_state, symbol, to_state)
                        transition_counts[key] += 1
                        
        # Check coverage
        issues = []
        adequate_transitions = 0
        
        for transition, count in transition_counts.items():
            if count < min_coverage:
                issues.append(
                    f"Transition {transition[0]}-{transition[1]}->{transition[2]}: "
                    f"{count} appearances (minimum: {min_coverage})"
                )
            else:
                adequate_transitions += 1
                
        # Compute score
        total_transitions = len(transition_counts)
        coverage_score = adequate_transitions / total_transitions if total_transitions > 0 else 0.0
        
        return {
            'score': coverage_score,
            'issues': issues,
            'transition_counts': dict(transition_counts),
            'adequate_coverage_ratio': coverage_score,
            'min_threshold': min_coverage
        }
        
    def _validate_length_coverage(self, sequences_data: Dict[str, Any], quality_spec: Any) -> Dict[str, Any]:
        """Validate sequence length coverage."""
        diversity_threshold = getattr(quality_spec, 'length_diversity_threshold', 0.3)
        
        sequences = sequences_data.get('sequences', [])
        lengths = [len(seq) for seq in sequences]
        
        if not lengths:
            return {
                'score': 0.0,
                'issues': ['No sequences available for length analysis'],
                'length_distribution': {}
            }
            
        # Analyze length distribution
        length_counts = Counter(lengths)
        unique_lengths = len(length_counts)
        total_sequences = len(lengths)
        
        # Diversity score: ratio of unique lengths to reasonable range
        length_range = max(lengths) - min(lengths) + 1
        diversity_score = unique_lengths / length_range if length_range > 0 else 0.0
        
        issues = []
        if diversity_score < diversity_threshold:
            issues.append(
                f"Length diversity {diversity_score:.3f} below threshold {diversity_threshold}"
            )
            
        # Check for reasonable distribution
        most_common_count = length_counts.most_common(1)[0][1]
        if most_common_count / total_sequences > 0.8:
            issues.append("Length distribution heavily skewed to single length")
            
        return {
            'score': min(1.0, diversity_score / diversity_threshold),
            'issues': issues,
            'length_distribution': dict(length_counts),
            'diversity_score': diversity_score,
            'diversity_threshold': diversity_threshold
        }
        
    def _generate_coverage_recommendations(self, state_cov: Dict, trans_cov: Dict, 
                                         length_cov: Dict) -> List[str]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []
        
        if state_cov.get('score', 0) < 0.8:
            recommendations.append(
                "Increase number of sequences per machine to improve state coverage"
            )
            
        if trans_cov.get('score', 0) < 0.8:
            recommendations.append(
                "Generate longer sequences to ensure adequate transition coverage"
            )
            
        if length_cov.get('score', 0) < 0.8:
            recommendations.append(
                "Increase sequence length diversity for better dataset balance"
            )
            
        if not recommendations:
            recommendations.append("Coverage metrics are adequate")
            
        return recommendations
        
    def validate_distributions(self, sequences_data: Dict[str, Any], 
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check statistical consistency between theoretical and empirical measures.
        
        Args:
            sequences_data: Sequences and metadata
            metadata: Computed statistical metadata
            
        Returns:
            Dictionary with distribution validation results
        """
        # Theoretical vs empirical entropy validation
        entropy_validation = self._validate_entropy_consistency(sequences_data, metadata)
        
        # Symbol distribution validation
        symbol_validation = self._validate_symbol_distributions(sequences_data)
        
        # Machine distribution validation
        machine_validation = self._validate_machine_distributions(sequences_data)
        
        # Compute overall distribution score
        validation_scores = [
            entropy_validation.get('score', 0.0),
            symbol_validation.get('score', 0.0),
            machine_validation.get('score', 0.0)
        ]
        overall_score = np.mean(validation_scores)
        
        # Collect issues
        all_issues = []
        all_issues.extend(entropy_validation.get('issues', []))
        all_issues.extend(symbol_validation.get('issues', []))
        all_issues.extend(machine_validation.get('issues', []))
        
        return {
            'overall_score': overall_score,
            'issues': all_issues,
            'entropy_validation': entropy_validation,
            'symbol_validation': symbol_validation,
            'machine_validation': machine_validation
        }
        
    def _validate_entropy_consistency(self, sequences_data: Dict, metadata: Dict) -> Dict[str, Any]:
        """Validate entropy rate consistency."""
        issues = []
        
        # Get theoretical and empirical entropy measures
        info_measures = metadata.get('information_measures', {})
        entropy_measures = info_measures.get('entropy_measures', {})
        machine_measures = info_measures.get('machine_measures', {})
        
        if not entropy_measures or not machine_measures:
            return {
                'score': 0.5,  # Neutral score when validation impossible
                'issues': ['Insufficient entropy data for validation'],
                'comparisons': {}
            }
            
        # Compare theoretical vs empirical for each machine
        comparisons = {}
        consistency_scores = []
        
        for machine_id, machine_data in machine_measures.items():
            theoretical = machine_data.get('theoretical_entropy_rate', 0.0)
            empirical = machine_data.get('empirical_entropy_rate', 0.0)
            
            if theoretical > 0:
                relative_error = abs(theoretical - empirical) / theoretical
                consistency_scores.append(1.0 - min(1.0, relative_error))
                
                comparisons[machine_id] = {
                    'theoretical': theoretical,
                    'empirical': empirical,
                    'relative_error': relative_error
                }
                
                if relative_error > 0.2:  # 20% tolerance
                    issues.append(
                        f"Machine {machine_id}: Large entropy rate discrepancy "
                        f"(theoretical: {theoretical:.3f}, empirical: {empirical:.3f})"
                    )
                    
        consistency_score = np.mean(consistency_scores) if consistency_scores else 0.5
        
        return {
            'score': consistency_score,
            'issues': issues,
            'comparisons': comparisons,
            'mean_consistency': consistency_score
        }
        
    def _validate_symbol_distributions(self, sequences_data: Dict) -> Dict[str, Any]:
        """Validate symbol frequency distributions."""
        sequences = sequences_data.get('sequences', [])
        alphabet = sequences_data.get('alphabet', [])
        
        if not sequences or not alphabet:
            return {
                'score': 0.0,
                'issues': ['No sequences or alphabet for symbol validation'],
                'symbol_frequencies': {}
            }
            
        # Count symbol frequencies
        symbol_counts = defaultdict(int)
        total_symbols = 0
        
        for seq in sequences:
            for symbol in seq:
                symbol_counts[symbol] += 1
                total_symbols += 1
                
        symbol_frequencies = {
            symbol: count / total_symbols 
            for symbol, count in symbol_counts.items()
        }
        
        # Check for balance (for topological machines, expect uniform distribution)
        expected_freq = 1.0 / len(alphabet)
        balance_score = 1.0
        issues = []
        
        for symbol in alphabet:
            actual_freq = symbol_frequencies.get(symbol, 0.0)
            deviation = abs(actual_freq - expected_freq)
            
            if deviation > expected_freq * 0.3:  # 30% tolerance
                issues.append(
                    f"Symbol '{symbol}': frequency {actual_freq:.3f} "
                    f"deviates from expected {expected_freq:.3f}"
                )
                balance_score *= (1.0 - deviation / expected_freq)
                
        return {
            'score': max(0.0, balance_score),
            'issues': issues,
            'symbol_frequencies': symbol_frequencies,
            'expected_frequency': expected_freq,
            'balance_score': balance_score
        }
        
    def _validate_machine_distributions(self, sequences_data: Dict) -> Dict[str, Any]:
        """Validate machine sampling distributions."""
        metadata = sequences_data.get('metadata', [])
        
        if not metadata:
            return {
                'score': 0.0,
                'issues': ['No metadata for machine distribution validation'],
                'machine_counts': {}
            }
            
        # Count sequences per machine
        machine_counts = defaultdict(int)
        for seq_meta in metadata:
            machine_id = seq_meta.get('machine_id', -1)
            machine_counts[machine_id] += 1
            
        # Check for reasonable distribution
        total_sequences = sum(machine_counts.values())
        machine_frequencies = {
            machine_id: count / total_sequences
            for machine_id, count in machine_counts.items()
        }
        
        # Check for extreme imbalances
        issues = []
        balance_score = 1.0
        
        if machine_frequencies:
            max_freq = max(machine_frequencies.values())
            min_freq = min(machine_frequencies.values())
            
            if max_freq / min_freq > 10.0:  # 10x imbalance threshold
                issues.append(
                    f"Extreme machine imbalance: max frequency {max_freq:.3f}, "
                    f"min frequency {min_freq:.3f}"
                )
                balance_score *= 0.5
                
        return {
            'score': balance_score,
            'issues': issues,
            'machine_counts': dict(machine_counts),
            'machine_frequencies': machine_frequencies,
            'balance_score': balance_score
        }