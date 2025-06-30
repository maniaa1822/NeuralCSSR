"""
Evaluation baselines computer for Neural CSSR.

This module computes baseline performance metrics including random baselines,
empirical n-gram models, and theoretical optimal performance estimates.
"""

from typing import Dict, List, Any
import math
import numpy as np
from collections import Counter, defaultdict
from ..core.epsilon_machine import EpsilonMachine


class BaselineComputer:
    """
    Computes comprehensive baseline metrics for Neural CSSR evaluation.
    
    Provides random baselines, empirical n-gram model performance,
    and theoretical optimal performance estimates.
    """
    
    def __init__(self):
        """Initialize baseline computer."""
        pass
        
    def compute_random_baselines(self, alphabet_size: int, num_states: int) -> Dict[str, float]:
        """
        Compute theoretical random performance baselines.
        
        Args:
            alphabet_size: Size of symbol alphabet
            num_states: Maximum number of states in dataset
            
        Returns:
            Dictionary of random baseline metrics
        """
        # Random prediction cross-entropy
        random_cross_entropy = math.log2(alphabet_size)
        
        # Random state prediction accuracy
        random_state_accuracy = 1.0 / num_states if num_states > 0 else 0.0
        
        # Random perplexity
        random_perplexity = alphabet_size
        
        # Random bits per symbol
        random_bits_per_symbol = math.log2(alphabet_size)
        
        return {
            'random_cross_entropy': random_cross_entropy,
            'random_state_accuracy': random_state_accuracy,
            'random_perplexity': random_perplexity,
            'random_bits_per_symbol': random_bits_per_symbol,
            'alphabet_size': alphabet_size,
            'num_states': num_states
        }
        
    def compute_empirical_baselines(self, sequences: List[str]) -> Dict[str, Any]:
        """
        Compute n-gram model baselines from empirical data.
        
        Args:
            sequences: List of sequence strings or lists
            
        Returns:
            Dictionary of empirical baseline metrics
        """
        if not sequences:
            return {}
            
        # Convert sequences to strings if they are lists
        if sequences and isinstance(sequences[0], list):
            sequences = [''.join(seq) for seq in sequences]
            
        # Concatenate all sequences for model building
        all_text = ''.join(sequences)
        
        if not all_text:
            return {}
            
        baselines = {}
        
        # Compute n-gram models for n = 1, 2, 3, 4
        for n in range(1, 5):
            ngram_metrics = self._compute_ngram_baseline(all_text, n)
            baselines[f'{n}_gram_model'] = ngram_metrics
            
        # Frequency-based baseline
        baselines['frequency_baseline'] = self._compute_frequency_baseline(all_text)
        
        # Entropy-based theoretical limits
        baselines['entropy_limits'] = self._compute_entropy_limits(all_text)
        
        return baselines
        
    def _compute_ngram_baseline(self, text: str, n: int) -> Dict[str, float]:
        """Compute n-gram model baseline metrics."""
        if len(text) < n:
            return {
                'cross_entropy': float('inf'),
                'perplexity': float('inf'),
                'accuracy': 0.0,
                'coverage': 0.0
            }
            
        # Build n-gram model
        if n == 1:
            # Unigram model
            symbol_counts = Counter(text)
            total_symbols = len(text)
            
            # Compute cross-entropy
            cross_entropy = 0.0
            for symbol in text:
                prob = symbol_counts[symbol] / total_symbols
                cross_entropy -= math.log2(prob)
            cross_entropy /= len(text)
            
            # Accuracy: probability of most frequent symbol
            most_frequent_count = symbol_counts.most_common(1)[0][1]
            accuracy = most_frequent_count / total_symbols
            
        else:
            # Higher-order n-gram model
            ngrams = {}
            contexts = {}
            
            # Build context and n-gram counts
            for i in range(len(text) - n + 1):
                context = text[i:i+n-1]
                symbol = text[i+n-1]
                ngram = text[i:i+n]
                
                if context not in contexts:
                    contexts[context] = Counter()
                contexts[context][symbol] += 1
                
                if ngram not in ngrams:
                    ngrams[ngram] = 0
                ngrams[ngram] += 1
                
            # Compute cross-entropy and accuracy
            cross_entropy = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(n-1, len(text)):
                context = text[i-n+1:i]
                actual_symbol = text[i]
                
                if context in contexts:
                    context_counts = contexts[context]
                    total_context = sum(context_counts.values())
                    
                    if actual_symbol in context_counts:
                        prob = context_counts[actual_symbol] / total_context
                        cross_entropy -= math.log2(prob)
                        
                        # Check if this was the most likely prediction
                        most_likely_symbol = context_counts.most_common(1)[0][0]
                        if actual_symbol == most_likely_symbol:
                            correct_predictions += 1
                    else:
                        # Unseen n-gram, use uniform distribution
                        alphabet_size = len(set(text))
                        cross_entropy -= math.log2(1.0 / alphabet_size)
                        
                    total_predictions += 1
                    
            if total_predictions > 0:
                cross_entropy /= total_predictions
                accuracy = correct_predictions / total_predictions
            else:
                cross_entropy = float('inf')
                accuracy = 0.0
                
        # Coverage: fraction of possible n-grams seen
        alphabet_size = len(set(text))
        possible_ngrams = alphabet_size ** n
        actual_ngrams = len(Counter(text[i:i+n] for i in range(len(text) - n + 1)))
        coverage = actual_ngrams / possible_ngrams
        
        return {
            'cross_entropy': cross_entropy,
            'perplexity': 2 ** cross_entropy,
            'accuracy': accuracy,
            'coverage': coverage,
            'n': n
        }
        
    def _compute_frequency_baseline(self, text: str) -> Dict[str, float]:
        """Compute simple frequency-based baseline."""
        if not text:
            return {}
            
        symbol_counts = Counter(text)
        total_symbols = len(text)
        
        # Always predict most frequent symbol
        most_frequent_symbol, most_frequent_count = symbol_counts.most_common(1)[0]
        accuracy = most_frequent_count / total_symbols
        
        # Cross-entropy if always predicting most frequent
        prob_most_frequent = most_frequent_count / total_symbols
        cross_entropy = -math.log2(prob_most_frequent)
        
        return {
            'accuracy': accuracy,
            'cross_entropy': cross_entropy,
            'perplexity': 2 ** cross_entropy,
            'most_frequent_symbol': most_frequent_symbol,
            'most_frequent_probability': prob_most_frequent
        }
        
    def _compute_entropy_limits(self, text: str) -> Dict[str, float]:
        """Compute entropy-based theoretical limits."""
        if not text:
            return {}
            
        # Empirical entropy (unigram)
        symbol_counts = Counter(text)
        total_symbols = len(text)
        empirical_entropy = -sum(
            (count / total_symbols) * math.log2(count / total_symbols)
            for count in symbol_counts.values()
        )
        
        # Conditional entropy estimates
        conditional_entropies = {}
        for n in range(2, 5):
            if len(text) >= n:
                conditional_entropy = self._estimate_conditional_entropy(text, n)
                conditional_entropies[f'conditional_entropy_{n-1}'] = conditional_entropy
                
        return {
            'empirical_entropy': empirical_entropy,
            'alphabet_size': len(symbol_counts),
            'max_possible_entropy': math.log2(len(symbol_counts)),
            'entropy_efficiency': empirical_entropy / math.log2(len(symbol_counts)),
            **conditional_entropies
        }
        
    def _estimate_conditional_entropy(self, text: str, n: int) -> float:
        """Estimate conditional entropy H(X_n | X_1...X_{n-1})."""
        if len(text) < n:
            return 0.0
            
        contexts = defaultdict(Counter)
        
        # Build context -> next symbol counts
        for i in range(len(text) - n + 1):
            context = text[i:i+n-1]
            next_symbol = text[i+n-1]
            contexts[context][next_symbol] += 1
            
        # Compute conditional entropy
        total_entropy = 0.0
        total_count = 0
        
        for context, symbol_counts in contexts.items():
            context_total = sum(symbol_counts.values())
            context_entropy = -sum(
                (count / context_total) * math.log2(count / context_total)
                for count in symbol_counts.values()
            )
            total_entropy += context_entropy * context_total
            total_count += context_total
            
        return total_entropy / total_count if total_count > 0 else 0.0
        
    def _compute_bayes_optimal_accuracy(self, machine: EpsilonMachine) -> float:
        """
        Compute the Bayes optimal next-symbol prediction accuracy for a machine.
        
        This represents the theoretical upper bound for next-symbol prediction
        given perfect knowledge of the current causal state.
        
        Bayes optimal accuracy = Σ_s P(s) * max_σ P(σ|s)
        
        Where:
        - s ranges over causal states
        - P(s) is the stationary probability of state s  
        - P(σ|s) is the probability of emitting symbol σ from state s
        - max_σ P(σ|s) is the probability of the most likely symbol in state s
        
        Args:
            machine: Epsilon machine to analyze
            
        Returns:
            Bayes optimal accuracy (0.0 to 1.0)
        """
        try:
            # Get stationary distribution of states
            stationary_probs = machine.get_stationary_distribution()
            
            if not stationary_probs:
                # Fallback: assume uniform distribution over states
                stationary_probs = {state: 1.0/len(machine.states) for state in machine.states}
            
            total_bayes_accuracy = 0.0
            
            # For each state, find the maximum transition probability
            for state in machine.states:
                state_prob = stationary_probs.get(state, 0.0)
                
                if state_prob == 0.0:
                    continue
                    
                # Collect all transition probabilities from this state
                symbol_probs = {}
                
                for symbol in machine.alphabet:
                    key = (state, symbol)
                    if key in machine.transitions:
                        # Get the probability of this transition
                        # transitions[key] = [(next_state, probability), ...]
                        transition_prob = machine.transitions[key][0][1]
                        symbol_probs[symbol] = transition_prob
                    else:
                        symbol_probs[symbol] = 0.0
                
                # Find the maximum probability symbol from this state
                if symbol_probs:
                    max_symbol_prob = max(symbol_probs.values())
                else:
                    # Fallback: uniform over alphabet
                    max_symbol_prob = 1.0 / len(machine.alphabet)
                
                # Add this state's contribution to overall Bayes accuracy
                total_bayes_accuracy += state_prob * max_symbol_prob
                
            return total_bayes_accuracy
            
        except Exception as e:
            # Fallback: assume uniform transitions
            return 1.0 / len(machine.alphabet)
    
    def estimate_classical_performance(self, machines: List[EpsilonMachine]) -> Dict[str, Any]:
        """
        Estimate theoretical optimal performance with perfect causal state recovery.
        
        Args:
            machines: List of epsilon machines in dataset
            
        Returns:
            Dictionary of theoretical optimal performance metrics
        """
        if not machines:
            return {}
            
        # Perfect causal state recovery accuracy
        perfect_state_accuracy = 1.0
        
        # Optimal cross-entropy: weighted average of machine entropy rates
        entropy_rates = []
        complexities = []
        bayes_accuracies = []
        
        for machine in machines:
            try:
                # For topological machines, entropy rate = log2(alphabet_size)
                if machine.is_topological():
                    entropy_rate = math.log2(len(machine.alphabet))
                else:
                    # Would need full computation for non-topological
                    entropy_rate = math.log2(len(machine.alphabet))  # Approximation
                    
                entropy_rates.append(entropy_rate)
                complexities.append(machine.get_statistical_complexity())
                
                # Compute Bayes optimal accuracy for this machine
                bayes_accuracy = self._compute_bayes_optimal_accuracy(machine)
                bayes_accuracies.append(bayes_accuracy)
                
            except Exception:
                # Fallback values
                entropy_rates.append(1.0)
                complexities.append(1.0)
                bayes_accuracies.append(0.5)  # Random baseline for binary
                
        # Weighted averages
        if entropy_rates:
            optimal_cross_entropy = np.mean(entropy_rates)
            mean_complexity = np.mean(complexities)
            mean_bayes_accuracy = np.mean(bayes_accuracies)
        else:
            optimal_cross_entropy = 1.0
            mean_complexity = 1.0
            mean_bayes_accuracy = 0.5
            
        # Sample complexity estimates (heuristic)
        sample_complexity_estimates = self._estimate_sample_complexity(machines)
        
        # Computational complexity estimates
        computational_estimates = self._estimate_computational_complexity(machines)
        
        return {
            'perfect_state_accuracy': perfect_state_accuracy,
            'optimal_cross_entropy': optimal_cross_entropy,
            'optimal_perplexity': 2 ** optimal_cross_entropy,
            'bayes_optimal_accuracy': mean_bayes_accuracy,
            'bayes_accuracy_distribution': {
                'mean': np.mean(bayes_accuracies),
                'std': np.std(bayes_accuracies),
                'min': min(bayes_accuracies),
                'max': max(bayes_accuracies),
                'by_machine': bayes_accuracies
            },
            'mean_statistical_complexity': mean_complexity,
            'sample_complexity_estimates': sample_complexity_estimates,
            'computational_complexity_estimates': computational_estimates,
            'num_machines': len(machines),
            'entropy_rate_distribution': {
                'mean': np.mean(entropy_rates),
                'std': np.std(entropy_rates),
                'min': min(entropy_rates),
                'max': max(entropy_rates)
            }
        }
        
    def _estimate_sample_complexity(self, machines: List[EpsilonMachine]) -> Dict[str, Any]:
        """Estimate sample complexity for classical CSSR convergence."""
        estimates = []
        
        for machine in machines:
            num_states = len(machine.states)
            alphabet_size = len(machine.alphabet)
            
            # Heuristic: O(states^2 * alphabet_size * log(confidence))
            # This is a rough estimate based on PAC learning theory
            base_complexity = num_states ** 2 * alphabet_size
            confidence_factor = math.log(20)  # 95% confidence approximation
            
            estimate = base_complexity * confidence_factor
            estimates.append(estimate)
            
        return {
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates),
            'min_estimate': min(estimates),
            'max_estimate': max(estimates),
            'estimates_by_machine': estimates
        }
        
    def _estimate_computational_complexity(self, machines: List[EpsilonMachine]) -> Dict[str, Any]:
        """Estimate computational complexity for classical CSSR."""
        complexities = []
        
        for machine in machines:
            num_states = len(machine.states)
            alphabet_size = len(machine.alphabet)
            
            # Classical CSSR complexity is roughly O(L * |A|^L * |S|)
            # where L is max history length, |A| is alphabet size, |S| is states
            max_history = 10  # Typical maximum
            
            complexity = max_history * (alphabet_size ** max_history) * num_states
            complexities.append(complexity)
            
        return {
            'mean_complexity': np.mean(complexities),
            'std_complexity': np.std(complexities),
            'min_complexity': min(complexities),
            'max_complexity': max(complexities),
            'complexities_by_machine': complexities
        }
        
    def compare_performance_to_baselines(self, model_metrics: Dict[str, float], 
                                       baselines: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare model performance to computed baselines.
        
        Args:
            model_metrics: Dictionary of model performance metrics
            baselines: Dictionary of baseline metrics
            
        Returns:
            Dictionary of performance comparisons
        """
        comparisons = {}
        
        # Compare to random baseline
        random_baselines = baselines.get('random_baselines', {})
        if 'cross_entropy' in model_metrics and 'random_cross_entropy' in random_baselines:
            model_ce = model_metrics['cross_entropy']
            random_ce = random_baselines['random_cross_entropy']
            improvement_over_random = (random_ce - model_ce) / random_ce
            comparisons['improvement_over_random'] = improvement_over_random
            
        # Compare to n-gram baselines
        empirical_baselines = baselines.get('empirical_baselines', {})
        for n in range(1, 5):
            ngram_key = f'{n}_gram_model'
            if ngram_key in empirical_baselines:
                ngram_metrics = empirical_baselines[ngram_key]
                if 'cross_entropy' in model_metrics and 'cross_entropy' in ngram_metrics:
                    model_ce = model_metrics['cross_entropy']
                    ngram_ce = ngram_metrics['cross_entropy']
                    if ngram_ce > 0:
                        improvement = (ngram_ce - model_ce) / ngram_ce
                        comparisons[f'improvement_over_{n}gram'] = improvement
                        
        # Compare to theoretical optimal
        optimal_baselines = baselines.get('optimal_baselines', {})
        if 'cross_entropy' in model_metrics and 'optimal_cross_entropy' in optimal_baselines:
            model_ce = model_metrics['cross_entropy']
            optimal_ce = optimal_baselines['optimal_cross_entropy']
            if optimal_ce > 0:
                efficiency = optimal_ce / model_ce  # Closer to 1.0 is better
                comparisons['efficiency_ratio'] = efficiency
                gap_to_optimal = model_ce - optimal_ce
                comparisons['gap_to_optimal'] = gap_to_optimal
                
        return comparisons