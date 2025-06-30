"""
CSSR Execution Engine for Classical Analysis

Executes classical CSSR with comprehensive parameter exploration
and detailed result extraction.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from collections import defaultdict, Counter

from ..classical.cssr import ClassicalCSSR


class CSSRExecutionEngine:
    """Execute classical CSSR with comprehensive parameter exploration."""
    
    def __init__(self, sequences: List[str]):
        """
        Initialize execution engine with sequences.
        
        Args:
            sequences: List of sequence strings for CSSR analysis
        """
        self.sequences = sequences
        self.results = {}
        
        # Compute basic sequence statistics
        self.sequence_stats = self._compute_sequence_stats()
        
    def run_single_analysis(self, max_length: int = 10, 
                          significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Run CSSR with single parameter set.
        
        Args:
            max_length: Maximum history length
            significance_level: Statistical significance level
            
        Returns:
            Single analysis results
        """
        print(f"Running single CSSR analysis (L={max_length}, α={significance_level})")
        
        start_time = time.time()
        
        # Run CSSR
        cssr = ClassicalCSSR(significance_level=significance_level)
        
        # Convert sequences to CSSR format
        data_list = []
        for seq in self.sequences:
            for i in range(len(seq) - 1):
                history = list(seq[:i+1])  # Everything up to position i
                target = seq[i+1]          # Next symbol
                data_list.append({
                    'raw_history': history,
                    'raw_target': target
                })
        
        cssr.load_from_raw_data(data_list, metadata={})
        converged = cssr.run_cssr(
            max_iterations=20,
            max_history_length=max_length
        )
        
        runtime = time.time() - start_time
        
        # Extract detailed results
        result = {
            'parameters': {
                'max_length': max_length,
                'significance_level': significance_level
            },
            'execution_info': {
                'converged': converged,
                'runtime_seconds': runtime,
                'sequence_count': len(self.sequences),
                'total_observations': sum(len(seq) for seq in self.sequences)
            },
            'discovered_structure': {
                'num_states': len(cssr.causal_states),
                'states': self._extract_detailed_state_info(cssr.causal_states),
                'transitions': self._extract_transition_info(cssr.causal_states),
                'alphabet': list(cssr.alphabet) if hasattr(cssr, 'alphabet') else self._get_alphabet()
            },
            'computational_stats': {
                'total_histories': len(cssr.history_counts),
                'max_history_length_used': max([len(h) for h in cssr.history_counts.keys()]) if cssr.history_counts else 0,
                'total_statistical_tests': getattr(cssr, 'total_tests_performed', 0)
            }
        }
        
        # Compute prediction performance
        result['prediction_performance'] = self._compute_prediction_performance(cssr)
        
        return result
        
    def run_parameter_sweep(self, 
                          max_lengths: Optional[List[int]] = None,
                          significance_levels: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Run CSSR across multiple parameter combinations.
        
        Args:
            max_lengths: List of maximum history lengths to try
            significance_levels: List of significance levels to try
            
        Returns:
            Dictionary containing:
            {
                'parameter_results': {...},     # Results for each (L, α) pair
                'best_parameters': {...},       # Optimal parameter selection
                'convergence_analysis': {...},  # Convergence behavior
                'parameter_sensitivity': {...}  # Sensitivity analysis
            }
        """
        if max_lengths is None:
            max_lengths = [6, 8, 10, 12]
        if significance_levels is None:
            significance_levels = [0.001, 0.01, 0.05, 0.1]
        
        print(f"Running CSSR parameter sweep: {len(max_lengths)} × {len(significance_levels)} = {len(max_lengths) * len(significance_levels)} combinations")
        
        results = {}
        
        for max_length in max_lengths:
            for significance_level in significance_levels:
                param_key = f"L{max_length}_alpha{significance_level}"
                
                print(f"  Running {param_key}...")
                
                try:
                    result = self.run_single_analysis(max_length, significance_level)
                    results[param_key] = result
                    
                    # Print quick summary
                    print(f"    States: {result['discovered_structure']['num_states']}, "
                          f"Converged: {result['execution_info']['converged']}, "
                          f"Time: {result['execution_info']['runtime_seconds']:.2f}s")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    results[param_key] = {
                        'parameters': {'max_length': max_length, 'significance_level': significance_level},
                        'error': str(e)
                    }
        
        # Analyze results
        analysis_results = {
            'parameter_results': results,
            'best_parameters': self._select_best_parameters(results),
            'convergence_analysis': self._analyze_convergence(results),
            'parameter_sensitivity': self._analyze_parameter_sensitivity(results),
            'summary_statistics': self._compute_summary_statistics(results)
        }
        
        return analysis_results
    
    def _compute_sequence_stats(self) -> Dict[str, Any]:
        """Compute basic statistics about input sequences."""
        lengths = [len(seq) for seq in self.sequences]
        all_symbols = ''.join(self.sequences)
        symbol_counts = Counter(all_symbols)
        
        return {
            'num_sequences': len(self.sequences),
            'total_length': sum(lengths),
            'length_stats': {
                'mean': np.mean(lengths),
                'min': min(lengths),
                'max': max(lengths),
                'std': np.std(lengths)
            },
            'alphabet': sorted(symbol_counts.keys()),
            'symbol_frequencies': dict(symbol_counts),
            'entropy_estimate': self._estimate_entropy(symbol_counts)
        }
    
    def _estimate_entropy(self, symbol_counts: Counter) -> float:
        """Estimate entropy from symbol frequencies."""
        total = sum(symbol_counts.values())
        entropy = 0.0
        for count in symbol_counts.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        return entropy
    
    def _get_alphabet(self) -> List[str]:
        """Extract alphabet from sequences."""
        symbols = set()
        for seq in self.sequences:
            symbols.update(seq)
        return sorted(list(symbols))
    
    def _extract_detailed_state_info(self, causal_states: List) -> Dict[str, Any]:
        """Extract detailed information about discovered states."""
        states_info = {}
        
        for i, state in enumerate(causal_states):
            state_id = f"State_{i}"
            
            # Get symbol distribution
            symbol_dist = {}
            if hasattr(state, 'next_symbol_distribution'):
                symbol_dist = dict(state.next_symbol_distribution)
            elif hasattr(state, 'emission_probabilities'):
                symbol_dist = dict(state.emission_probabilities)
            
            # Compute statistics
            total_count = getattr(state, 'total_count', 0)
            if total_count == 0 and symbol_dist:
                total_count = sum(symbol_dist.values())
            
            state_info = {
                'state_index': i,
                'histories': list(getattr(state, 'histories', [])),
                'history_count': len(getattr(state, 'histories', [])),
                'total_observations': total_count,
                'symbol_distribution': symbol_dist,
                'entropy': self._compute_state_entropy(symbol_dist),
                'prediction_probabilities': self._normalize_distribution(symbol_dist)
            }
            
            states_info[state_id] = state_info
        
        return states_info
    
    def _extract_transition_info(self, causal_states: List) -> Dict[str, Any]:
        """Extract transition information between states."""
        transitions = {}
        transition_matrix = {}
        
        # Build transition information
        for i, state in enumerate(causal_states):
            state_id = f"State_{i}"
            
            # Get transitions if available
            state_transitions = {}
            if hasattr(state, 'transitions'):
                for symbol, next_state_idx in state.transitions.items():
                    state_transitions[symbol] = f"State_{next_state_idx}"
            
            transitions[state_id] = state_transitions
            
            # Build transition matrix row
            if hasattr(state, 'next_symbol_distribution'):
                transition_matrix[state_id] = dict(state.next_symbol_distribution)
        
        return {
            'state_transitions': transitions,
            'transition_matrix': transition_matrix,
            'num_transitions': sum(len(t) for t in transitions.values())
        }
    
    def _compute_state_entropy(self, symbol_dist: Dict[str, int]) -> float:
        """Compute entropy of state's emission distribution."""
        if not symbol_dist:
            return 0.0
            
        total = sum(symbol_dist.values())
        if total == 0:
            return 0.0
            
        entropy = 0.0
        for count in symbol_dist.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _normalize_distribution(self, symbol_dist: Dict[str, int]) -> Dict[str, float]:
        """Normalize symbol distribution to probabilities."""
        if not symbol_dist:
            return {}
            
        total = sum(symbol_dist.values())
        if total == 0:
            return {}
            
        return {symbol: count / total for symbol, count in symbol_dist.items()}
    
    def _compute_prediction_performance(self, cssr) -> Dict[str, float]:
        """Compute prediction performance metrics."""
        # This is a simplified implementation
        # In practice, you'd want to compute actual cross-entropy, perplexity, etc.
        
        performance = {
            'states_used': len(cssr.causal_states),
            'avg_state_entropy': 0.0,
            'total_entropy': 0.0
        }
        
        # Compute average entropy across states
        entropies = []
        for state in cssr.causal_states:
            if hasattr(state, 'next_symbol_distribution'):
                entropy = self._compute_state_entropy(dict(state.next_symbol_distribution))
                entropies.append(entropy)
        
        if entropies:
            performance['avg_state_entropy'] = np.mean(entropies)
            performance['total_entropy'] = sum(entropies)
        
        return performance
    
    def _select_best_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select best parameters based on multiple criteria."""
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        # Criteria for best parameters
        best_by_convergence = None
        best_by_structure = None
        best_by_entropy = None
        
        # Find best converged result
        converged_results = {k: v for k, v in valid_results.items() 
                           if v.get('execution_info', {}).get('converged', False)}
        
        if converged_results:
            # Best by structure quality (prefer reasonable state counts - not just minimum)
            # Sort by state count and pick a result that's neither too simple nor too complex
            sorted_results = sorted(converged_results.items(), 
                                  key=lambda x: x[1]['discovered_structure']['num_states'])
            
            # Choose the median result or one that's close to expected complexity
            if len(sorted_results) >= 3:
                # For multiple results, prefer middle-complexity result
                best_by_structure = sorted_results[len(sorted_results) // 2]
            elif len(sorted_results) == 2:
                # For two results, prefer the higher-complexity one (often more accurate)
                best_by_structure = sorted_results[1]
            else:
                # Single result
                best_by_structure = sorted_results[0]
            
            # Best by entropy (lower is better for prediction)
            entropy_results = [(k, v) for k, v in converged_results.items() 
                             if v.get('prediction_performance', {}).get('avg_state_entropy', float('inf')) < float('inf')]
            
            if entropy_results:
                best_by_entropy = min(entropy_results, 
                                    key=lambda x: x[1]['prediction_performance']['avg_state_entropy'])
        
        # Overall best (prioritize convergence, then balanced complexity)
        if best_by_structure:
            overall_best = best_by_structure
        elif valid_results:
            # If no converged results, pick one with reasonable state count (not minimum)
            sorted_valid = sorted(valid_results.items(), 
                                key=lambda x: x[1]['discovered_structure']['num_states'])
            if len(sorted_valid) >= 2:
                overall_best = sorted_valid[len(sorted_valid) // 2]
            else:
                overall_best = sorted_valid[0]
        else:
            overall_best = None
        
        return {
            'overall_best': {
                'parameter_key': overall_best[0] if overall_best else None,
                'parameters': overall_best[1]['parameters'] if overall_best else None,
                'reason': 'converged_with_fewest_states' if best_by_structure else 'fewest_states_no_convergence'
            },
            'best_by_structure': {
                'parameter_key': best_by_structure[0] if best_by_structure else None,
                'parameters': best_by_structure[1]['parameters'] if best_by_structure else None
            },
            'best_by_entropy': {
                'parameter_key': best_by_entropy[0] if best_by_entropy else None,
                'parameters': best_by_entropy[1]['parameters'] if best_by_entropy else None
            },
            'total_valid_results': len(valid_results),
            'total_converged_results': len(converged_results) if converged_results else 0
        }
    
    def _analyze_convergence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence behavior across parameters."""
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        convergence_analysis = {
            'total_runs': len(results),
            'valid_runs': len(valid_results),
            'converged_runs': 0,
            'convergence_rate': 0.0,
            'convergence_by_max_length': defaultdict(lambda: {'total': 0, 'converged': 0}),
            'convergence_by_significance': defaultdict(lambda: {'total': 0, 'converged': 0})
        }
        
        for param_key, result in valid_results.items():
            params = result.get('parameters', {})
            max_length = params.get('max_length', 0)
            significance = params.get('significance_level', 0.0)
            converged = result.get('execution_info', {}).get('converged', False)
            
            if converged:
                convergence_analysis['converged_runs'] += 1
            
            # Track by max_length
            convergence_analysis['convergence_by_max_length'][max_length]['total'] += 1
            if converged:
                convergence_analysis['convergence_by_max_length'][max_length]['converged'] += 1
            
            # Track by significance level
            convergence_analysis['convergence_by_significance'][significance]['total'] += 1
            if converged:
                convergence_analysis['convergence_by_significance'][significance]['converged'] += 1
        
        # Compute convergence rates
        if valid_results:
            convergence_analysis['convergence_rate'] = convergence_analysis['converged_runs'] / len(valid_results)
        
        # Compute rates by parameter
        for max_length, stats in convergence_analysis['convergence_by_max_length'].items():
            stats['rate'] = stats['converged'] / stats['total'] if stats['total'] > 0 else 0.0
        
        for significance, stats in convergence_analysis['convergence_by_significance'].items():
            stats['rate'] = stats['converged'] / stats['total'] if stats['total'] > 0 else 0.0
        
        return convergence_analysis
    
    def _analyze_parameter_sensitivity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensitivity to parameter choices."""
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        # Group results by parameter values
        by_max_length = defaultdict(list)
        by_significance = defaultdict(list)
        
        for param_key, result in valid_results.items():
            params = result.get('parameters', {})
            max_length = params.get('max_length', 0)
            significance = params.get('significance_level', 0.0)
            num_states = result.get('discovered_structure', {}).get('num_states', 0)
            
            by_max_length[max_length].append({
                'param_key': param_key,
                'significance': significance,
                'num_states': num_states,
                'converged': result.get('execution_info', {}).get('converged', False)
            })
            
            by_significance[significance].append({
                'param_key': param_key,
                'max_length': max_length,
                'num_states': num_states,
                'converged': result.get('execution_info', {}).get('converged', False)
            })
        
        # Compute sensitivity metrics
        sensitivity_analysis = {
            'max_length_sensitivity': {},
            'significance_sensitivity': {},
            'parameter_stability': {}
        }
        
        # Max length sensitivity
        for max_length, results_list in by_max_length.items():
            state_counts = [r['num_states'] for r in results_list]
            sensitivity_analysis['max_length_sensitivity'][max_length] = {
                'mean_states': np.mean(state_counts),
                'std_states': np.std(state_counts),
                'min_states': min(state_counts),
                'max_states': max(state_counts),
                'convergence_rate': np.mean([r['converged'] for r in results_list])
            }
        
        # Significance level sensitivity
        for significance, results_list in by_significance.items():
            state_counts = [r['num_states'] for r in results_list]
            sensitivity_analysis['significance_sensitivity'][significance] = {
                'mean_states': np.mean(state_counts),
                'std_states': np.std(state_counts),
                'min_states': min(state_counts),
                'max_states': max(state_counts),
                'convergence_rate': np.mean([r['converged'] for r in results_list])
            }
        
        return sensitivity_analysis
    
    def _compute_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        # Extract key metrics
        state_counts = []
        runtimes = []
        entropies = []
        convergence_flags = []
        
        for result in valid_results.values():
            state_counts.append(result.get('discovered_structure', {}).get('num_states', 0))
            runtimes.append(result.get('execution_info', {}).get('runtime_seconds', 0))
            entropies.append(result.get('prediction_performance', {}).get('avg_state_entropy', 0))
            convergence_flags.append(result.get('execution_info', {}).get('converged', False))
        
        summary = {
            'total_parameter_combinations': len(results),
            'successful_runs': len(valid_results),
            'success_rate': len(valid_results) / len(results) if results else 0.0,
            'convergence_rate': np.mean(convergence_flags) if convergence_flags else 0.0,
            
            'state_count_stats': {
                'mean': np.mean(state_counts) if state_counts else 0.0,
                'std': np.std(state_counts) if state_counts else 0.0,
                'min': min(state_counts) if state_counts else 0,
                'max': max(state_counts) if state_counts else 0
            },
            
            'runtime_stats': {
                'mean': np.mean(runtimes) if runtimes else 0.0,
                'std': np.std(runtimes) if runtimes else 0.0,
                'min': min(runtimes) if runtimes else 0.0,
                'max': max(runtimes) if runtimes else 0.0,
                'total': sum(runtimes) if runtimes else 0.0
            },
            
            'entropy_stats': {
                'mean': np.mean(entropies) if entropies else 0.0,
                'std': np.std(entropies) if entropies else 0.0,
                'min': min(entropies) if entropies else 0.0,
                'max': max(entropies) if entropies else 0.0
            }
        }
        
        return summary