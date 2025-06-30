"""
Performance Analyzer for Classical CSSR Analysis

Comprehensive performance analysis including scaling behavior,
baseline comparisons, and parameter sensitivity analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter
import json


class PerformanceAnalyzer:
    """Comprehensive performance analysis and baseline comparison."""
    
    def analyze_scaling_behavior(self, cssr_results: Dict, metadata: Dict) -> Dict[str, Any]:
        """
        Analyze how CSSR performance scales with dataset properties.
        
        Args:
            cssr_results: Results from CSSR execution
            metadata: Dataset metadata with complexity and statistics
            
        Returns:
            Dictionary containing:
            {
                'parameter_sensitivity': {...},     # Sensitivity to L and α
                'complexity_scaling': {...},        # Performance vs machine complexity  
                'sample_size_scaling': {...},       # Performance vs dataset size
                'convergence_analysis': {...}       # Convergence behavior analysis
            }
        """
        analysis = {}
        
        # Parameter sensitivity analysis
        analysis['parameter_sensitivity'] = self._analyze_parameter_sensitivity(cssr_results)
        
        # Complexity scaling analysis
        machine_complexities = metadata.get('complexity_metrics', {})
        analysis['complexity_scaling'] = self._analyze_complexity_scaling(cssr_results, machine_complexities)
        
        # Sample size effects
        sequence_stats = metadata.get('sequence_statistics', {})
        analysis['sample_size_scaling'] = self._analyze_sample_scaling(cssr_results, sequence_stats)
        
        # Convergence behavior
        analysis['convergence_analysis'] = self._analyze_convergence_behavior(cssr_results)
        
        # Runtime scaling
        analysis['runtime_scaling'] = self._analyze_runtime_scaling(cssr_results)
        
        return analysis
    
    def compare_against_baselines(self, cssr_results: Dict, metadata: Dict) -> Dict[str, Any]:
        """
        Compare CSSR performance against various baselines.
        
        Args:
            cssr_results: Results from CSSR execution
            metadata: Dataset metadata with baseline information
            
        Returns:
            Dictionary containing:
            {
                'vs_random_baseline': {...},        # vs random prediction
                'vs_empirical_baselines': {...},    # vs n-gram models
                'vs_theoretical_optimal': {...},    # vs known optimal performance
                'relative_performance': {...}       # Overall performance assessment
            }
        """
        baselines = metadata.get('comparative_baselines', {})
        best_cssr = self._get_best_cssr_result(cssr_results)
        
        if not best_cssr:
            return {'error': 'No valid CSSR results for baseline comparison'}
        
        comparison = {}
        
        # Random baseline comparison
        if 'random_baselines' in baselines:
            comparison['vs_random_baseline'] = self._compare_vs_random(best_cssr, baselines['random_baselines'])
        
        # Empirical baselines comparison  
        if 'empirical_baselines' in baselines:
            comparison['vs_empirical_baselines'] = self._compare_vs_empirical(best_cssr, baselines['empirical_baselines'])
        
        # Theoretical optimal comparison
        if 'optimal_baselines' in baselines:
            comparison['vs_theoretical_optimal'] = self._compare_vs_optimal(best_cssr, baselines['optimal_baselines'])
        
        # Information theory baselines
        if 'information_measures' in metadata:
            comparison['vs_information_theory'] = self._compare_vs_information_theory(best_cssr, metadata['information_measures'])
        
        # Overall relative performance
        comparison['relative_performance'] = self._compute_relative_performance(comparison)
        
        return comparison
    
    def _get_best_cssr_result(self, cssr_results: Dict) -> Optional[Dict]:
        """Extract the best CSSR result from results."""
        
        if 'parameter_results' in cssr_results:
            # Parameter sweep results - get the best one
            best_params = cssr_results.get('best_parameters', {})
            overall_best = best_params.get('overall_best', {})
            param_key = overall_best.get('parameter_key')
            
            if param_key and param_key in cssr_results['parameter_results']:
                return cssr_results['parameter_results'][param_key]
            
            # Fallback: find first converged result
            for result in cssr_results['parameter_results'].values():
                if result.get('execution_info', {}).get('converged', False):
                    return result
            
            # Last resort: return any valid result
            valid_results = [r for r in cssr_results['parameter_results'].values() 
                           if 'error' not in r]
            if valid_results:
                return valid_results[0]
        
        elif 'discovered_structure' in cssr_results:
            # Single result
            return cssr_results
        
        return None
    
    def _analyze_parameter_sensitivity(self, cssr_results: Dict) -> Dict[str, Any]:
        """Analyze sensitivity to parameter choices."""
        
        if 'parameter_results' not in cssr_results:
            return {'error': 'No parameter sweep results available'}
        
        results = cssr_results['parameter_results']
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid parameter results to analyze'}
        
        # Group results by parameter values
        by_max_length = defaultdict(list)
        by_significance = defaultdict(list)
        
        for param_key, result in valid_results.items():
            params = result.get('parameters', {})
            max_length = params.get('max_length', 0)
            significance = params.get('significance_level', 0.0)
            
            num_states = result.get('discovered_structure', {}).get('num_states', 0)
            converged = result.get('execution_info', {}).get('converged', False)
            runtime = result.get('execution_info', {}).get('runtime_seconds', 0.0)
            entropy = result.get('prediction_performance', {}).get('avg_state_entropy', 0.0)
            
            metrics = {
                'param_key': param_key,
                'num_states': num_states,
                'converged': converged,
                'runtime': runtime,
                'entropy': entropy
            }
            
            by_max_length[max_length].append(metrics)
            by_significance[significance].append(metrics)
        
        sensitivity_analysis = {
            'max_length_effects': self._analyze_parameter_effects(by_max_length, 'max_length'),
            'significance_effects': self._analyze_parameter_effects(by_significance, 'significance_level'),
            'interaction_effects': self._analyze_parameter_interactions(valid_results),
            'stability_metrics': self._compute_parameter_stability(valid_results)
        }
        
        return sensitivity_analysis
    
    def _analyze_parameter_effects(self, grouped_results: Dict, param_name: str) -> Dict[str, Any]:
        """Analyze effects of a single parameter."""
        
        effects = {}
        
        for param_value, results_list in grouped_results.items():
            if not results_list:
                continue
            
            state_counts = [r['num_states'] for r in results_list]
            runtimes = [r['runtime'] for r in results_list]
            entropies = [r['entropy'] for r in results_list if r['entropy'] > 0]
            convergence_rate = np.mean([r['converged'] for r in results_list])
            
            effects[param_value] = {
                'num_runs': len(results_list),
                'convergence_rate': convergence_rate,
                'state_count_stats': {
                    'mean': np.mean(state_counts),
                    'std': np.std(state_counts),
                    'min': min(state_counts),
                    'max': max(state_counts)
                },
                'runtime_stats': {
                    'mean': np.mean(runtimes),
                    'std': np.std(runtimes),
                    'min': min(runtimes),
                    'max': max(runtimes)
                },
                'entropy_stats': {
                    'mean': np.mean(entropies) if entropies else 0.0,
                    'std': np.std(entropies) if entropies else 0.0,
                    'count': len(entropies)
                }
            }
        
        # Compute trends
        param_values = sorted(effects.keys())
        if len(param_values) > 1:
            mean_states = [effects[pv]['state_count_stats']['mean'] for pv in param_values]
            mean_runtimes = [effects[pv]['runtime_stats']['mean'] for pv in param_values]
            convergence_rates = [effects[pv]['convergence_rate'] for pv in param_values]
            
            effects['trends'] = {
                'state_count_trend': self._compute_trend(param_values, mean_states),
                'runtime_trend': self._compute_trend(param_values, mean_runtimes),
                'convergence_trend': self._compute_trend(param_values, convergence_rates)
            }
        
        return effects
    
    def _compute_trend(self, x_values: List, y_values: List) -> str:
        """Compute trend direction (increasing, decreasing, stable)."""
        
        if len(x_values) < 2 or len(y_values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend analysis
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        
        if np.isnan(correlation):
            return 'no_trend'
        elif correlation > 0.3:
            return 'increasing'
        elif correlation < -0.3:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_parameter_interactions(self, valid_results: Dict) -> Dict[str, Any]:
        """Analyze interactions between parameters."""
        
        interactions = {}
        
        # Create interaction matrix
        param_combinations = {}
        for param_key, result in valid_results.items():
            params = result.get('parameters', {})
            max_length = params.get('max_length', 0)
            significance = params.get('significance_level', 0.0)
            
            combo_key = f"{max_length}_{significance}"
            param_combinations[combo_key] = {
                'max_length': max_length,
                'significance': significance,
                'num_states': result.get('discovered_structure', {}).get('num_states', 0),
                'converged': result.get('execution_info', {}).get('converged', False),
                'runtime': result.get('execution_info', {}).get('runtime_seconds', 0.0)
            }
        
        if len(param_combinations) > 1:
            # Analyze which parameter combinations work best
            converged_combos = {k: v for k, v in param_combinations.items() if v['converged']}
            
            interactions['best_combinations'] = []
            if converged_combos:
                # Sort by fewest states (simpler is better)
                sorted_combos = sorted(converged_combos.items(), 
                                     key=lambda x: x[1]['num_states'])
                interactions['best_combinations'] = [
                    {
                        'max_length': combo['max_length'],
                        'significance': combo['significance'],
                        'num_states': combo['num_states'],
                        'runtime': combo['runtime']
                    }
                    for _, combo in sorted_combos[:3]  # Top 3
                ]
            
            interactions['total_combinations'] = len(param_combinations)
            interactions['converged_combinations'] = len(converged_combos)
            interactions['convergence_rate'] = len(converged_combos) / len(param_combinations)
        
        return interactions
    
    def _compute_parameter_stability(self, valid_results: Dict) -> Dict[str, Any]:
        """Compute stability metrics across parameter choices."""
        
        state_counts = []
        runtimes = []
        entropies = []
        
        for result in valid_results.values():
            state_counts.append(result.get('discovered_structure', {}).get('num_states', 0))
            runtimes.append(result.get('execution_info', {}).get('runtime_seconds', 0.0))
            entropy = result.get('prediction_performance', {}).get('avg_state_entropy', 0.0)
            if entropy > 0:
                entropies.append(entropy)
        
        stability = {}
        
        if state_counts:
            stability['state_count_stability'] = {
                'coefficient_of_variation': np.std(state_counts) / np.mean(state_counts) if np.mean(state_counts) > 0 else 0.0,
                'range_ratio': (max(state_counts) - min(state_counts)) / np.mean(state_counts) if np.mean(state_counts) > 0 else 0.0
            }
        
        if runtimes:
            stability['runtime_stability'] = {
                'coefficient_of_variation': np.std(runtimes) / np.mean(runtimes) if np.mean(runtimes) > 0 else 0.0,
                'range_ratio': (max(runtimes) - min(runtimes)) / np.mean(runtimes) if np.mean(runtimes) > 0 else 0.0
            }
        
        return stability
    
    def _analyze_complexity_scaling(self, cssr_results: Dict, machine_complexities: Dict) -> Dict[str, Any]:
        """Analyze performance vs machine complexity."""
        
        if not machine_complexities:
            return {'error': 'No complexity metrics available'}
        
        best_cssr = self._get_best_cssr_result(cssr_results)
        if not best_cssr:
            return {'error': 'No valid CSSR results'}
        
        discovered_states = best_cssr.get('discovered_structure', {}).get('num_states', 0)
        
        # Get complexity measures
        complexity_measures = {}
        if 'statistical_complexity' in machine_complexities:
            complexity_measures['statistical_complexity'] = machine_complexities['statistical_complexity']
        if 'topological_entropy' in machine_complexities:
            complexity_measures['topological_entropy'] = machine_complexities['topological_entropy']
        if 'machine_state_counts' in machine_complexities:
            true_states = machine_complexities['machine_state_counts']
            if isinstance(true_states, list):
                complexity_measures['avg_true_states'] = np.mean(true_states)
                complexity_measures['max_true_states'] = max(true_states)
        
        scaling_analysis = {
            'discovered_vs_true_states': {},
            'complexity_efficiency': {},
            'scaling_metrics': complexity_measures
        }
        
        # Compare discovered vs true states
        if 'avg_true_states' in complexity_measures:
            true_states = complexity_measures['avg_true_states']
            scaling_analysis['discovered_vs_true_states'] = {
                'true_states': true_states,
                'discovered_states': discovered_states,
                'ratio': discovered_states / true_states if true_states > 0 else 0.0,
                'difference': discovered_states - true_states,
                'accuracy': 1.0 - abs(discovered_states - true_states) / true_states if true_states > 0 else 0.0
            }
        
        return scaling_analysis
    
    def _analyze_sample_scaling(self, cssr_results: Dict, sequence_stats: Dict) -> Dict[str, Any]:
        """Analyze performance vs dataset size."""
        
        if not sequence_stats:
            return {'error': 'No sequence statistics available'}
        
        best_cssr = self._get_best_cssr_result(cssr_results)
        if not best_cssr:
            return {'error': 'No valid CSSR results'}
        
        # Get dataset size metrics
        total_length = sequence_stats.get('total_length', 0)
        num_sequences = sequence_stats.get('num_sequences', 0)
        
        # Get CSSR performance metrics
        converged = best_cssr.get('execution_info', {}).get('converged', False)
        runtime = best_cssr.get('execution_info', {}).get('runtime_seconds', 0.0)
        num_states = best_cssr.get('discovered_structure', {}).get('num_states', 0)
        
        scaling_analysis = {
            'dataset_size': {
                'total_symbols': total_length,
                'num_sequences': num_sequences,
                'avg_sequence_length': total_length / num_sequences if num_sequences > 0 else 0.0
            },
            'sample_efficiency': {
                'converged': converged,
                'states_per_symbol': num_states / total_length if total_length > 0 else 0.0,
                'runtime_per_symbol': runtime / total_length if total_length > 0 else 0.0,
                'symbols_per_state': total_length / num_states if num_states > 0 else 0.0
            }
        }
        
        # Estimate sample complexity
        if converged and num_states > 0:
            # Rough estimate of sample complexity
            estimated_min_samples = num_states * 100  # Rule of thumb: 100 samples per state
            scaling_analysis['sample_efficiency']['efficiency_ratio'] = estimated_min_samples / total_length if total_length > 0 else 0.0
        
        return scaling_analysis
    
    def _analyze_convergence_behavior(self, cssr_results: Dict) -> Dict[str, Any]:
        """Analyze convergence behavior patterns."""
        
        if 'parameter_results' not in cssr_results:
            return {'error': 'No parameter sweep results for convergence analysis'}
        
        # Use the convergence analysis from parameter sweep
        if 'convergence_analysis' in cssr_results:
            return cssr_results['convergence_analysis']
        
        # Fallback: compute basic convergence stats
        results = cssr_results['parameter_results']
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        converged_count = sum(1 for r in valid_results.values() 
                            if r.get('execution_info', {}).get('converged', False))
        
        return {
            'total_runs': len(valid_results),
            'converged_runs': converged_count,
            'convergence_rate': converged_count / len(valid_results) if valid_results else 0.0
        }
    
    def _analyze_runtime_scaling(self, cssr_results: Dict) -> Dict[str, Any]:
        """Analyze runtime scaling behavior."""
        
        if 'parameter_results' not in cssr_results:
            single_result = self._get_best_cssr_result(cssr_results)
            if single_result:
                runtime = single_result.get('execution_info', {}).get('runtime_seconds', 0.0)
                return {'single_run_runtime': runtime}
            return {'error': 'No runtime data available'}
        
        results = cssr_results['parameter_results']
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        runtime_analysis = {}
        
        # Group by parameters to analyze scaling
        by_max_length = defaultdict(list)
        by_significance = defaultdict(list)
        
        for result in valid_results.values():
            params = result.get('parameters', {})
            runtime = result.get('execution_info', {}).get('runtime_seconds', 0.0)
            
            by_max_length[params.get('max_length', 0)].append(runtime)
            by_significance[params.get('significance_level', 0.0)].append(runtime)
        
        # Analyze runtime vs max_length
        if len(by_max_length) > 1:
            max_lengths = sorted(by_max_length.keys())
            avg_runtimes = [np.mean(by_max_length[ml]) for ml in max_lengths]
            
            runtime_analysis['max_length_scaling'] = {
                'max_lengths': max_lengths,
                'avg_runtimes': avg_runtimes,
                'scaling_trend': self._compute_trend(max_lengths, avg_runtimes)
            }
        
        # Overall runtime statistics
        all_runtimes = [r.get('execution_info', {}).get('runtime_seconds', 0.0) 
                       for r in valid_results.values()]
        
        if all_runtimes:
            runtime_analysis['overall_stats'] = {
                'mean_runtime': np.mean(all_runtimes),
                'std_runtime': np.std(all_runtimes),
                'min_runtime': min(all_runtimes),
                'max_runtime': max(all_runtimes),
                'total_runtime': sum(all_runtimes)
            }
        
        return runtime_analysis
    
    def _compare_vs_random(self, cssr_result: Dict, random_baselines: Dict) -> Dict[str, Any]:
        """Compare CSSR vs random baseline."""
        
        # Get CSSR entropy
        cssr_entropy = cssr_result.get('prediction_performance', {}).get('avg_state_entropy', 0.0)
        
        # Get random baseline entropy
        random_entropy = random_baselines.get('random_entropy', 1.0)  # Default for binary
        
        if random_entropy > 0:
            improvement_factor = random_entropy / cssr_entropy if cssr_entropy > 0 else float('inf')
        else:
            improvement_factor = 1.0
        
        return {
            'cssr_entropy': cssr_entropy,
            'random_entropy': random_entropy,
            'improvement_factor': improvement_factor,
            'bits_saved': random_entropy - cssr_entropy
        }
    
    def _compare_vs_empirical(self, cssr_result: Dict, empirical_baselines: Dict) -> Dict[str, Any]:
        """Compare CSSR vs empirical baselines (n-grams, etc.)."""
        
        cssr_entropy = cssr_result.get('prediction_performance', {}).get('avg_state_entropy', 0.0)
        
        comparisons = {}
        
        # Compare vs different n-gram models
        for baseline_name, baseline_info in empirical_baselines.items():
            if isinstance(baseline_info, dict) and 'entropy' in baseline_info:
                baseline_entropy = baseline_info['entropy']
                
                if baseline_entropy > 0:
                    improvement = baseline_entropy / cssr_entropy if cssr_entropy > 0 else float('inf')
                else:
                    improvement = 1.0
                
                comparisons[baseline_name] = {
                    'baseline_entropy': baseline_entropy,
                    'improvement_factor': improvement,
                    'bits_saved': baseline_entropy - cssr_entropy
                }
        
        return comparisons
    
    def _compare_vs_optimal(self, cssr_result: Dict, optimal_baselines: Dict) -> Dict[str, Any]:
        """Compare CSSR vs theoretical optimal."""
        
        cssr_entropy = cssr_result.get('prediction_performance', {}).get('avg_state_entropy', 0.0)
        
        optimal_entropy = optimal_baselines.get('optimal_entropy', 0.5)  # Placeholder
        
        if optimal_entropy > 0:
            efficiency = optimal_entropy / cssr_entropy if cssr_entropy > 0 else 0.0
        else:
            efficiency = 1.0 if cssr_entropy == 0 else 0.0
        
        return {
            'cssr_entropy': cssr_entropy,
            'optimal_entropy': optimal_entropy,
            'efficiency': efficiency,
            'excess_bits': cssr_entropy - optimal_entropy
        }
    
    def _compare_vs_information_theory(self, cssr_result: Dict, info_measures: Dict) -> Dict[str, Any]:
        """Compare CSSR vs information-theoretic measures."""
        
        cssr_entropy = cssr_result.get('prediction_performance', {}).get('avg_state_entropy', 0.0)
        
        comparisons = {}
        
        # Compare vs empirical entropy
        if 'empirical_entropy' in info_measures:
            emp_entropy = info_measures['empirical_entropy']
            comparisons['vs_empirical_entropy'] = {
                'empirical_entropy': emp_entropy,
                'ratio': cssr_entropy / emp_entropy if emp_entropy > 0 else 0.0
            }
        
        # Compare vs conditional entropy
        if 'conditional_entropy' in info_measures:
            cond_entropy = info_measures['conditional_entropy']
            comparisons['vs_conditional_entropy'] = {
                'conditional_entropy': cond_entropy,
                'ratio': cssr_entropy / cond_entropy if cond_entropy > 0 else 0.0
            }
        
        return comparisons
    
    def _compute_relative_performance(self, comparison: Dict) -> Dict[str, Any]:
        """Compute overall relative performance assessment."""
        
        performance_scores = []
        
        # Score vs random baseline
        if 'vs_random_baseline' in comparison:
            random_improvement = comparison['vs_random_baseline'].get('improvement_factor', 1.0)
            # Cap at 10 to avoid extreme scores
            random_score = min(random_improvement / 2.0, 5.0) / 5.0  # Normalize to 0-1
            performance_scores.append(('random', random_score))
        
        # Score vs empirical baselines
        if 'vs_empirical_baselines' in comparison:
            empirical_improvements = []
            for baseline_comp in comparison['vs_empirical_baselines'].values():
                if isinstance(baseline_comp, dict):
                    improvement = baseline_comp.get('improvement_factor', 1.0)
                    empirical_improvements.append(improvement)
            
            if empirical_improvements:
                avg_improvement = np.mean(empirical_improvements)
                empirical_score = min(avg_improvement / 1.5, 1.0)  # Normalize
                performance_scores.append(('empirical', empirical_score))
        
        # Score vs optimal
        if 'vs_theoretical_optimal' in comparison:
            efficiency = comparison['vs_theoretical_optimal'].get('efficiency', 0.0)
            optimal_score = min(efficiency, 1.0)  # Already 0-1
            performance_scores.append(('optimal', optimal_score))
        
        # Compute overall score
        if performance_scores:
            overall_score = np.mean([score for _, score in performance_scores])
            
            # Determine performance category
            if overall_score >= 0.8:
                category = 'excellent'
            elif overall_score >= 0.6:
                category = 'good'
            elif overall_score >= 0.4:
                category = 'fair'
            else:
                category = 'poor'
        else:
            overall_score = 0.0
            category = 'unknown'
        
        return {
            'overall_score': overall_score,
            'performance_category': category,
            'component_scores': dict(performance_scores),
            'num_comparisons': len(performance_scores)
        }