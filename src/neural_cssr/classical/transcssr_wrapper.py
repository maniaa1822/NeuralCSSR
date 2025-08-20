"""
TransCSSR Wrapper for Analysis Pipeline

Integrates the reference transCSSR implementation into our analysis framework.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np
from collections import defaultdict
import math

# Add transCSSR to path
transcssr_path = Path(__file__).parent.parent.parent.parent / 'transCSSR'
sys.path.insert(0, str(transcssr_path))

from transCSSR import estimate_predictive_distributions, run_transCSSR
import itertools


class TransCSSRWrapper:
    """Wrapper for transCSSR reference implementation."""
    
    def __init__(self, significance_level: float = 0.001, test_type: str = 'chi2'):
        """
        Initialize transCSSR wrapper.
        
        Args:
            significance_level: Statistical significance level for chi-square tests
        """
        self.significance_level = significance_level
        self.test_type = test_type  # 'chi2' or 'G'
        self.axs = ['0']  # Input alphabet (usually just '0' for single input)
        self.ays = ['0', '1']  # Output alphabet (binary)
        
        # Create emission symbols (all (x,y) pairs)
        self.e_symbols = list(itertools.product(self.axs, self.ays))
        
    def run_cssr(self, string_x: str, string_y: str, max_length: int = 9) -> Dict[str, Any]:
        """
        Run transCSSR algorithm on input/output sequences.
        
        Args:
            string_x: Input sequence string
            string_y: Output sequence string  
            max_length: Maximum history length (L_max)
            
        Returns:
            Analysis results dictionary
        """
        print(f"Running transCSSR: L_max={max_length}, α={self.significance_level}")
        
        start_time = time.time()
        
        # Estimate predictive distributions
        word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(
            string_x, string_y, max_length
        )
        
        # Run transCSSR algorithm
        epsilon, invepsilon, morph_by_state = run_transCSSR(
            word_lookup_marg, word_lookup_fut, max_length, 
            self.axs, self.ays, self.e_symbols, 
            '', '',  # Xt_name, Yt_name
            alpha=self.significance_level,
            test_type=self.test_type
        )
        
        runtime = time.time() - start_time
        
        # Extract results
        num_states = len(invepsilon)
        
        print(f"transCSSR found {num_states} states in {runtime:.2f} seconds")
        
        # Format results to match our analysis framework
        results = {
            'discovered_structure': {
                'num_states': num_states,
                'states': self._format_states(invepsilon, morph_by_state),
                'epsilon_mapping': self._format_epsilon_mapping(epsilon),
                'transitions': self._extract_transitions(epsilon, invepsilon, morph_by_state)
            },
            'execution_info': {
                'converged': True,  # transCSSR always converges
                'runtime_seconds': runtime,
                'algorithm': 'transCSSR',
                'parameters': {
                    'max_length': max_length,
                    'significance_level': self.significance_level
                }
            },
            'raw_results': {
                'epsilon': self._format_epsilon_mapping(epsilon),  # Convert to JSON-serializable format
                'invepsilon': {str(k): [str(h) for h in v] for k, v in invepsilon.items()},
                'morph_by_state': {str(k): v for k, v in morph_by_state.items()},
                'word_lookup_marg': None,  # These are too large for JSON
                'word_lookup_fut': None
            }
        }
        
        return results

    def run_cssr_neural(self, provider, string_x: str, string_y: str,
                        max_length: int = 9, pseudo_count_scale: int = 100,
                        prob_clip: float = 1e-3, temperature: float = 1.0,
                        mix_empirical: float = 0.0) -> Dict[str, Any]:
        """
        Run transCSSR using neural probabilities instead of empirical counts.
        
        Builds word lookups (marginal and future) by reusing observed marginal
        counts from the data and distributing them across future symbols using
        neural probabilities. This preserves sample sizes and avoids fabricating
        unobserved histories.
        
        Args:
            provider: Object exposing predict_next_distribution(history: str) -> {symbol: prob}
            string_x, string_y: Input/output sequences (Y is the observed process)
            max_length: L_max
            pseudo_count_scale: total pseudo-count mass per (xpast, ypast)
        """
        from collections import Counter

        axs = self.axs
        ays = self.ays

        # Reuse observed marginals and futures from data
        observed_marg, observed_fut = estimate_predictive_distributions(string_x, string_y, max_length)
        word_lookup_marg = Counter(observed_marg)
        word_lookup_fut = Counter()

        # For each observed (xpast+ax, ypast), distribute its count by neural probs
        for (xpax, ypast), c_x in word_lookup_marg.items():
            # Predict distribution over next Y given ypast
            probs = provider.predict_next_distribution(ypast)
            # Clip and renormalize to avoid zeros and extreme confidence
            p0 = max(prob_clip, min(1.0 - prob_clip, probs['0']))
            p1 = max(prob_clip, min(1.0 - prob_clip, probs['1']))
            s = p0 + p1
            p0, p1 = p0 / s, p1 / s
            # Temperature smoothing on probabilities (power transform)
            if temperature and temperature != 1.0:
                gamma = 1.0 / float(temperature)
                p0_t = p0 ** gamma
                p1_t = p1 ** gamma
                st = p0_t + p1_t
                p0, p1 = p0_t / st, p1_t / st
            # Optional mixing with empirical counts for this (xpast, ypast)
            if mix_empirical and mix_empirical > 0.0:
                c0_emp = observed_fut.get((xpax, ypast + '0'), 0)
                c1_emp = observed_fut.get((xpax, ypast + '1'), 0)
                if c_x > 0:
                    pe0 = c0_emp / float(c_x)
                    pe1 = c1_emp / float(c_x)
                else:
                    pe0 = pe1 = 0.5
                # Mix and renormalize
                lam = float(mix_empirical)
                p0 = (1 - lam) * p0 + lam * pe0
                p1 = (1 - lam) * p1 + lam * pe1
                st2 = p0 + p1
                if st2 > 0:
                    p0, p1 = p0 / st2, p1 / st2
            # Initial rounding
            raw_counts = {'0': p0 * c_x, '1': p1 * c_x}
            rounded = {ay: int(math.floor(v)) for ay, v in raw_counts.items()}
            # Fix rounding to preserve total
            delta = int(c_x - sum(rounded.values()))
            if delta > 0:
                # Assign remainder to largest fractional parts
                fracs = sorted([(raw_counts[ay] - rounded[ay], ay) for ay in ays], reverse=True)
                for i in range(delta):
                    rounded[fracs[i % len(ays)][1]] += 1
            elif delta < 0:
                # Remove from smallest fractional parts
                fracs = sorted([(raw_counts[ay] - rounded[ay], ay) for ay in ays])
                for i in range(-delta):
                    rounded[fracs[i % len(ays)][1]] = max(0, rounded[fracs[i % len(ays)][1]] - 1)
            for ay in ays:
                if rounded[ay] < 0:
                    rounded[ay] = 0
                word_lookup_fut[(xpax, ypast + ay)] += rounded[ay]

        # Run reference algorithm with neural-derived counts
        epsilon, invepsilon, morph_by_state = run_transCSSR(
            word_lookup_marg, word_lookup_fut, max_length,
            axs, ays, self.e_symbols, '', '', alpha=self.significance_level, test_type=self.test_type
        )

        # Package results as in run_cssr
        num_states = len(invepsilon)
        results = {
            'discovered_structure': {
                'num_states': num_states,
                'states': self._format_states(invepsilon, morph_by_state),
                'epsilon_mapping': self._format_epsilon_mapping(epsilon),
                'transitions': self._extract_transitions(epsilon, invepsilon, morph_by_state)
            },
            'execution_info': {
                'converged': True,
                'algorithm': 'transCSSR-neural',
                'parameters': {
                    'max_length': max_length,
                    'significance_level': self.significance_level,
                    'pseudo_count_scale': pseudo_count_scale,
                    'test_type': self.test_type,
                    'prob_clip': prob_clip,
                    'temperature': temperature,
                    'mix_empirical': mix_empirical
                }
            }
        }

        return results
    
    def run_parameter_sweep(self, string_x: str, string_y: str, 
                          max_lengths: List[int] = None,
                          significance_levels: List[float] = None) -> Dict[str, Any]:
        """
        Run parameter sweep analysis.
        
        Args:
            string_x: Input sequence string
            string_y: Output sequence string
            max_lengths: List of L_max values to test
            significance_levels: List of alpha values to test
            
        Returns:
            Parameter sweep results
        """
        if max_lengths is None:
            max_lengths = [6, 8, 10, 12]
        if significance_levels is None:
            significance_levels = [0.001, 0.01, 0.05, 0.1]
        
        print(f"Running transCSSR parameter sweep:")
        print(f"  L_max values: {max_lengths}")
        print(f"  α values: {significance_levels}")
        
        results = {}
        best_result = None
        best_score = -1
        
        for max_length in max_lengths:
            for sig_level in significance_levels:
                param_key = f"L{max_length}_alpha{sig_level}"
                
                # Temporarily set significance level
                original_sig = self.significance_level
                self.significance_level = sig_level
                
                try:
                    result = self.run_cssr(string_x, string_y, max_length)
                    results[param_key] = result
                    
                    # Simple scoring: prefer more states (up to a reasonable limit)
                    num_states = result['discovered_structure']['num_states']
                    score = min(num_states, 100)  # Cap at 100 to avoid runaway
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'parameter_key': param_key,
                            'parameters': {
                                'max_length': max_length,
                                'significance_level': sig_level
                            },
                            'score': score,
                            'num_states': num_states
                        }
                        
                except Exception as e:
                    print(f"  Error with {param_key}: {e}")
                    results[param_key] = {'error': str(e)}
                
                finally:
                    # Restore original significance level
                    self.significance_level = original_sig
        
        return {
            'parameter_results': results,
            'best_parameters': {
                'overall_best': best_result
            },
            'parameter_space': {
                'max_lengths': max_lengths,
                'significance_levels': significance_levels
            }
        }
    
    def _format_states(self, invepsilon: Dict, morph_by_state: Dict) -> Dict[str, Any]:
        """Format state information for analysis framework."""
        states = {}
        
        for state_id, histories in invepsilon.items():
            # Convert histories to string format
            history_strs = []
            for hist in histories:
                if isinstance(hist, tuple):
                    history_strs.append(f"({hist[0]}, {hist[1]})")
                else:
                    history_strs.append(str(hist))
            
            # Get emission probabilities
            morph = morph_by_state.get(state_id, [])
            total_count = sum(morph) if morph else 0
            
            if total_count > 0:
                emission_probs = [count / total_count for count in morph]
            else:
                emission_probs = [0.0] * len(self.e_symbols)
            
            states[f"State_{state_id}"] = {
                'id': state_id,
                'histories': history_strs,
                'emission_counts': morph,
                'emission_probabilities': emission_probs,
                'total_count': total_count
            }
        
        return states
    
    def _format_epsilon_mapping(self, epsilon: Dict) -> Dict[str, int]:
        """Format epsilon mapping (history -> state) for analysis."""
        epsilon_formatted = {}
        
        for hist, state_id in epsilon.items():
            if isinstance(hist, tuple):
                hist_str = f"({hist[0]}, {hist[1]})"
            else:
                hist_str = str(hist)
            epsilon_formatted[hist_str] = state_id
        
        return epsilon_formatted
    
    def _extract_transitions(self, epsilon: Dict, invepsilon: Dict, 
                           morph_by_state: Dict) -> Dict[str, Dict]:
        """Extract state transition structure."""
        transitions = {}
        
        for state_id in invepsilon.keys():
            transitions[f"State_{state_id}"] = {
                'outgoing': {},
                'emission_distribution': {}
            }
            
            # Get emission distribution
            morph = morph_by_state.get(state_id, [])
            total = sum(morph) if morph else 0
            
            for i, (x_sym, y_sym) in enumerate(self.e_symbols):
                count = morph[i] if i < len(morph) else 0
                prob = count / total if total > 0 else 0.0
                
                transitions[f"State_{state_id}"]["emission_distribution"][f"({x_sym},{y_sym})"] = {
                    'count': count,
                    'probability': prob
                }
        
        return transitions


def print_morph_by_states(morph_by_state: Dict, axs: List[str], ays: List[str], 
                         e_symbols: List[Tuple[str, str]]):
    """Print morphs by state in readable format (transCSSR compatibility)."""
    print("\\nState emission distributions:")
    
    for state_id, morph in morph_by_state.items():
        print(f"State {state_id}:")
        
        total_count = sum(morph)
        if total_count == 0:
            print("  No emissions")
            continue
        
        for i, (x_sym, y_sym) in enumerate(e_symbols):
            count = morph[i] if i < len(morph) else 0
            prob = count / total_count if total_count > 0 else 0.0
            print(f"  ({x_sym},{y_sym}): {count} ({prob:.3f})")


if __name__ == "__main__":
    # Test with our converted biased dataset
    print("Testing transCSSR wrapper...")
    
    # Load test data
    test_file = Path(__file__).parent.parent.parent.parent / "transcssr_biased_exp.dat"
    if test_file.exists():
        with open(test_file, 'r') as f:
            data = f.read().strip()
        
        string_x = '0' * len(data)
        string_y = data
        
        # Run test
        wrapper = TransCSSRWrapper()
        result = wrapper.run_cssr(string_x, string_y, max_length=9)
        
        print(f"Found {result['discovered_structure']['num_states']} states")
        print_morph_by_states(
            result['raw_results']['morph_by_state'],
            wrapper.axs, wrapper.ays, wrapper.e_symbols
        )
    else:
        print("Test data not found. Run conversion script first.")