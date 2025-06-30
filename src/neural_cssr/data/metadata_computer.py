"""
Statistical metadata computer for Neural CSSR datasets.

This module computes comprehensive information-theoretic measures,
complexity metrics, and sequence statistics for dataset analysis.
"""

from typing import Dict, List, Any
import numpy as np
from collections import Counter, defaultdict
import math
from itertools import combinations
from ..core.epsilon_machine import EpsilonMachine


class StatisticalMetadataComputer:
    """
    Computes comprehensive statistical metadata for Neural CSSR datasets.
    
    Provides information-theoretic measures, complexity metrics, and
    sequence statistics for rigorous dataset analysis.
    """
    
    def __init__(self):
        """Initialize metadata computer."""
        pass
        
    def compute_information_measures(self, sequences_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive information-theoretic measures.
        
        Args:
            sequences_data: Dictionary containing sequences and metadata
            
        Returns:
            Dictionary of information-theoretic measures
        """
        sequences = sequences_data['sequences']
        metadata = sequences_data.get('metadata', [])
        
        if not sequences:
            return {}
            
        # Convert sequences to strings for analysis
        if sequences and isinstance(sequences[0], list):
            sequence_strings = [''.join(seq) for seq in sequences]
        else:
            sequence_strings = sequences
        
        # Basic entropy measures
        entropy_measures = self._compute_entropy_measures(sequence_strings)
        
        # Complexity measures
        complexity_measures = self._compute_complexity_measures(sequence_strings)
        
        # Machine-level measures (if metadata available)
        machine_measures = self._compute_machine_level_measures(metadata)
        
        # Predictive information
        predictive_info = self._compute_predictive_information(sequence_strings)
        
        return {
            'entropy_measures': entropy_measures,
            'complexity_measures': complexity_measures,
            'machine_measures': machine_measures,
            'predictive_information': predictive_info,
            'dataset_level': {
                'total_sequences': len(sequences),
                'total_symbols': sum(len(seq) for seq in sequences),
                'unique_sequences': len(set(sequence_strings)),
                'diversity_ratio': len(set(sequence_strings)) / len(sequences)
            }
        }
        
    def _compute_entropy_measures(self, sequences: List[str]) -> Dict[str, float]:
        """Compute various entropy measures."""
        if not sequences:
            return {}
            
        # Concatenate all sequences for symbol counting
        all_text = ''.join(sequences)
        
        # Unigram entropy
        symbol_counts = Counter(all_text)
        total_symbols = len(all_text)
        symbol_probs = [count / total_symbols for count in symbol_counts.values()]
        unigram_entropy = -sum(p * math.log2(p) for p in symbol_probs if p > 0)
        
        # Bigram entropy
        bigrams = [all_text[i:i+2] for i in range(len(all_text)-1)]
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        bigram_entropy = 0.0
        if total_bigrams > 0:
            bigram_probs = [count / total_bigrams for count in bigram_counts.values()]
            bigram_entropy = -sum(p * math.log2(p) for p in bigram_probs if p > 0)
        
        # Trigram entropy
        trigrams = [all_text[i:i+3] for i in range(len(all_text)-2)]
        trigram_counts = Counter(trigrams)
        total_trigrams = len(trigrams)
        trigram_entropy = 0.0
        if total_trigrams > 0:
            trigram_probs = [count / total_trigrams for count in trigram_counts.values()]
            trigram_entropy = -sum(p * math.log2(p) for p in trigram_probs if p > 0)
        
        # Conditional entropies
        conditional_entropy_1 = bigram_entropy - unigram_entropy if bigram_entropy > unigram_entropy else 0
        conditional_entropy_2 = trigram_entropy - bigram_entropy if trigram_entropy > bigram_entropy else 0
        
        return {
            'unigram_entropy': unigram_entropy,
            'bigram_entropy': bigram_entropy,
            'trigram_entropy': trigram_entropy,
            'conditional_entropy_1': conditional_entropy_1,
            'conditional_entropy_2': conditional_entropy_2,
            'entropy_rate_estimate': conditional_entropy_1  # Approximation
        }
        
    def _compute_complexity_measures(self, sequences: List[str]) -> Dict[str, Any]:
        """Compute complexity measures for sequences."""
        if not sequences:
            return {}
            
        # Lempel-Ziv complexity (simplified)
        lz_complexities = [self._lempel_ziv_complexity(seq) for seq in sequences]
        
        # Compression ratios (simple dictionary compression)
        compression_ratios = [self._compute_compression_ratio(seq) for seq in sequences]
        
        # Effective alphabet sizes
        effective_alphabets = [len(set(seq)) for seq in sequences]
        
        return {
            'lempel_ziv_complexity': {
                'mean': np.mean(lz_complexities),
                'std': np.std(lz_complexities),
                'min': min(lz_complexities),
                'max': max(lz_complexities)
            },
            'compression_ratio': {
                'mean': np.mean(compression_ratios),
                'std': np.std(compression_ratios),
                'min': min(compression_ratios),
                'max': max(compression_ratios)
            },
            'effective_alphabet_size': {
                'mean': np.mean(effective_alphabets),
                'std': np.std(effective_alphabets),
                'min': min(effective_alphabets),
                'max': max(effective_alphabets)
            }
        }
        
    def _lempel_ziv_complexity(self, sequence: str) -> float:
        """
        Compute simplified Lempel-Ziv complexity.
        
        Args:
            sequence: Input sequence string
            
        Returns:
            Normalized LZ complexity
        """
        if not sequence:
            return 0.0
            
        # Simplified LZ77-style complexity
        dictionary = set()
        i = 0
        complexity = 0
        
        while i < len(sequence):
            # Find longest match in dictionary
            match_length = 0
            for length in range(1, len(sequence) - i + 1):
                substring = sequence[i:i + length]
                if substring in dictionary:
                    match_length = length
                else:
                    break
                    
            # Add new pattern to dictionary
            if i + match_length < len(sequence):
                new_pattern = sequence[i:i + match_length + 1]
                dictionary.add(new_pattern)
                i += match_length + 1
            else:
                i += match_length
                
            complexity += 1
            
        # Normalize by sequence length
        return complexity / len(sequence) if len(sequence) > 0 else 0.0
        
    def _compute_compression_ratio(self, sequence: str) -> float:
        """
        Compute simple compression ratio using dictionary compression.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Compression ratio (compressed_size / original_size)
        """
        if not sequence:
            return 1.0
            
        # Count unique substrings of various lengths
        unique_patterns = set()
        for length in range(1, min(6, len(sequence) + 1)):  # Up to length 5
            for i in range(len(sequence) - length + 1):
                unique_patterns.add(sequence[i:i + length])
                
        # Estimate compressed size (very simplified)
        estimated_compressed = len(unique_patterns) + len(sequence) // 4
        return min(1.0, estimated_compressed / len(sequence))
        
    def _compute_machine_level_measures(self, metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute machine-level information measures."""
        if not metadata:
            return {}
            
        # Group by machine
        machine_data = defaultdict(list)
        for seq_meta in metadata:
            machine_id = seq_meta.get('machine_id', -1)
            machine_data[machine_id].append(seq_meta)
            
        machine_measures = {}
        for machine_id, sequences in machine_data.items():
            if not sequences:
                continue
                
            # Extract machine properties
            machine_props = sequences[0].get('machine_properties', {})
            
            # Compute empirical entropy rate for this machine
            machine_sequences = []
            for seq_meta in sequences:
                if 'sequence' in seq_meta:
                    machine_sequences.append(''.join(seq_meta['sequence']))
                    
            if machine_sequences:
                entropy_measures = self._compute_entropy_measures(machine_sequences)
                
                machine_measures[str(machine_id)] = {
                    'theoretical_properties': machine_props,
                    'empirical_entropy_measures': entropy_measures,
                    'num_sequences': len(machine_sequences),
                    'statistical_complexity': machine_props.get('statistical_complexity', 0.0),
                    'theoretical_entropy_rate': machine_props.get('entropy_rate', 0.0),
                    'empirical_entropy_rate': entropy_measures.get('entropy_rate_estimate', 0.0)
                }
                
        return machine_measures
        
    def _compute_predictive_information(self, sequences: List[str]) -> Dict[str, float]:
        """Compute predictive information measures."""
        if not sequences:
            return {}
            
        # Compute mutual information between past and future
        # Simplified implementation using n-gram statistics
        
        all_text = ''.join(sequences)
        
        # Compute mutual information for different history lengths
        mutual_infos = {}
        for history_length in [1, 2, 3]:
            if len(all_text) <= history_length:
                continue
                
            # Extract (history, future) pairs
            pairs = []
            for i in range(len(all_text) - history_length):
                history = all_text[i:i + history_length]
                future = all_text[i + history_length]
                pairs.append((history, future))
                
            if pairs:
                mi = self._compute_mutual_information(pairs)
                mutual_infos[f'mi_history_{history_length}'] = mi
                
        return mutual_infos
        
    def _compute_mutual_information(self, pairs: List[tuple]) -> float:
        """Compute mutual information from (x, y) pairs."""
        if not pairs:
            return 0.0
            
        # Count joint and marginal frequencies
        joint_counts = Counter(pairs)
        x_counts = Counter([x for x, y in pairs])
        y_counts = Counter([y for x, y in pairs])
        
        total = len(pairs)
        mi = 0.0
        
        for (x, y), joint_count in joint_counts.items():
            p_xy = joint_count / total
            p_x = x_counts[x] / total
            p_y = y_counts[y] / total
            
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log2(p_xy / (p_x * p_y))
                
        return mi
        
    def compute_complexity_metrics(self, machines: List[EpsilonMachine]) -> Dict[str, Any]:
        """
        Analyze structural and learning complexity of machines.
        
        Args:
            machines: List of epsilon machines
            
        Returns:
            Dictionary of complexity metrics
        """
        if not machines:
            return {}
            
        # Structural complexity measures
        state_counts = [len(machine.states) for machine in machines]
        transition_counts = [
            sum(len(transitions) for transitions in machine.transitions.values())
            for machine in machines
        ]
        
        # Topological measures
        strongly_connected = [
            self._is_strongly_connected(machine) for machine in machines
        ]
        
        # Cycle analysis
        cycle_info = [self._analyze_cycles(machine) for machine in machines]
        
        # Statistical complexity
        stat_complexities = []
        entropy_rates = []
        for machine in machines:
            try:
                stat_complexities.append(machine.get_statistical_complexity())
                # Compute entropy rate (simplified)
                if machine.is_topological():
                    entropy_rate = math.log2(len(machine.alphabet))
                else:
                    entropy_rate = 0.0  # Would need full computation
                entropy_rates.append(entropy_rate)
            except:
                stat_complexities.append(0.0)
                entropy_rates.append(0.0)
                
        return {
            'structural_measures': {
                'state_count_distribution': {
                    'mean': np.mean(state_counts),
                    'std': np.std(state_counts),
                    'min': min(state_counts),
                    'max': max(state_counts),
                    'counts': Counter(state_counts)
                },
                'transition_count_distribution': {
                    'mean': np.mean(transition_counts),
                    'std': np.std(transition_counts),
                    'min': min(transition_counts),
                    'max': max(transition_counts)
                },
                'strongly_connected_ratio': sum(strongly_connected) / len(strongly_connected)
            },
            'cycle_analysis': {
                'average_shortest_cycle': np.mean([c['shortest_cycle'] for c in cycle_info]),
                'cycle_complexity_distribution': Counter([c['num_cycles'] for c in cycle_info])
            },
            'learning_complexity': {
                'statistical_complexity_distribution': {
                    'mean': np.mean(stat_complexities),
                    'std': np.std(stat_complexities),
                    'min': min(stat_complexities),
                    'max': max(stat_complexities)
                },
                'entropy_rate_distribution': {
                    'mean': np.mean(entropy_rates),
                    'std': np.std(entropy_rates),
                    'min': min(entropy_rates),
                    'max': max(entropy_rates)
                }
            }
        }
        
    def _is_strongly_connected(self, machine: EpsilonMachine) -> bool:
        """Check if machine is strongly connected."""
        if not machine.states:
            return False
            
        # Simple reachability check
        visited = set()
        start_state = next(iter(machine.states))
        to_visit = [start_state]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # Find reachable states
            for symbol in machine.alphabet:
                key = (current, symbol)
                if key in machine.transitions:
                    for next_state, _ in machine.transitions[key]:
                        if next_state not in visited:
                            to_visit.append(next_state)
                            
        return len(visited) == len(machine.states)
        
    def _analyze_cycles(self, machine: EpsilonMachine) -> Dict[str, Any]:
        """Analyze cycle structure of machine."""
        # Simplified cycle detection
        cycles_found = []
        
        for start_state in machine.states:
            # Try to find cycles from this state
            visited = {start_state: 0}
            current = start_state
            step = 0
            
            # Follow first available transition
            while step < 20:  # Limit search depth
                next_states = []
                for symbol in machine.alphabet:
                    key = (current, symbol)
                    if key in machine.transitions:
                        for next_state, _ in machine.transitions[key]:
                            next_states.append(next_state)
                            break
                        break
                        
                if not next_states:
                    break
                    
                next_state = next_states[0]
                step += 1
                
                if next_state in visited:
                    # Found cycle
                    cycle_length = step - visited[next_state]
                    cycles_found.append(cycle_length)
                    break
                    
                visited[next_state] = step
                current = next_state
                
        return {
            'num_cycles': len(cycles_found),
            'shortest_cycle': min(cycles_found) if cycles_found else 0,
            'cycle_lengths': cycles_found
        }
        
    def compute_sequence_statistics(self, sequences: List[Any]) -> Dict[str, Any]:
        """
        Detailed sequence-level analysis.
        
        Args:
            sequences: List of sequence strings or lists
            
        Returns:
            Dictionary of sequence statistics
        """
        if not sequences:
            return {}
            
        # Convert sequences to strings if they are lists
        if sequences and isinstance(sequences[0], list):
            sequences = [''.join(seq) for seq in sequences]
            
        # Length analysis
        lengths = [len(seq) for seq in sequences]
        length_stats = {
            'count': len(sequences),
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'median': np.median(lengths),
            'distribution': Counter(lengths)
        }
        
        # N-gram analysis
        ngram_stats = {}
        for n in range(1, 7):  # 1-grams to 6-grams
            ngram_stats[f'{n}_gram'] = self._analyze_ngrams(sequences, n)
            
        # Run length analysis
        run_lengths = self._analyze_run_lengths(sequences)
        
        # Autocorrelation analysis
        autocorr = self._compute_autocorrelation(sequences)
        
        # Periodicity detection
        periodicity = self._detect_periodicity(sequences)
        
        return {
            'length_statistics': length_stats,
            'ngram_analysis': ngram_stats,
            'run_length_analysis': run_lengths,
            'autocorrelation': autocorr,
            'periodicity_analysis': periodicity
        }
        
    def _analyze_ngrams(self, sequences: List[str], n: int) -> Dict[str, Any]:
        """Analyze n-gram patterns in sequences."""
        all_ngrams = []
        for seq in sequences:
            if len(seq) >= n:
                for i in range(len(seq) - n + 1):
                    all_ngrams.append(seq[i:i+n])
                    
        if not all_ngrams:
            return {}
            
        ngram_counts = Counter(all_ngrams)
        total_ngrams = len(all_ngrams)
        
        return {
            'total_count': total_ngrams,
            'unique_count': len(ngram_counts),
            'diversity_ratio': len(ngram_counts) / total_ngrams,
            'most_frequent': ngram_counts.most_common(10),
            'entropy': -sum(
                (count / total_ngrams) * math.log2(count / total_ngrams)
                for count in ngram_counts.values()
            )
        }
        
    def _analyze_run_lengths(self, sequences: List[str]) -> Dict[str, Any]:
        """Analyze run lengths (consecutive identical symbols)."""
        all_run_lengths = []
        
        for seq in sequences:
            if not seq:
                continue
                
            current_char = seq[0]
            current_run = 1
            
            for char in seq[1:]:
                if char == current_char:
                    current_run += 1
                else:
                    all_run_lengths.append(current_run)
                    current_char = char
                    current_run = 1
                    
            all_run_lengths.append(current_run)
            
        if not all_run_lengths:
            return {}
            
        return {
            'mean_run_length': np.mean(all_run_lengths),
            'std_run_length': np.std(all_run_lengths),
            'max_run_length': max(all_run_lengths),
            'run_length_distribution': Counter(all_run_lengths)
        }
        
    def _compute_autocorrelation(self, sequences: List[str]) -> Dict[str, float]:
        """Compute autocorrelation functions for sequences."""
        if not sequences:
            return {}
            
        # Convert sequences to numerical for autocorrelation
        autocorr_results = {}
        
        for seq in sequences[:10]:  # Limit to first 10 sequences
            if len(seq) < 4:
                continue
                
            # Convert to numbers
            char_to_num = {char: i for i, char in enumerate(set(seq))}
            num_seq = [char_to_num[char] for char in seq]
            
            # Compute autocorrelation for lags 1-5
            for lag in range(1, min(6, len(num_seq))):
                corr = np.corrcoef(num_seq[:-lag], num_seq[lag:])[0, 1]
                if not np.isnan(corr):
                    key = f'lag_{lag}'
                    if key not in autocorr_results:
                        autocorr_results[key] = []
                    autocorr_results[key].append(corr)
                    
        # Average across sequences
        return {
            key: np.mean(values) for key, values in autocorr_results.items()
        }
        
    def _detect_periodicity(self, sequences: List[str]) -> Dict[str, Any]:
        """Detect periodic patterns in sequences."""
        periodicity_scores = []
        
        for seq in sequences[:20]:  # Limit analysis
            if len(seq) < 8:
                continue
                
            # Check for periods 2-6
            max_score = 0
            best_period = 0
            
            for period in range(2, min(7, len(seq) // 2)):
                score = 0
                comparisons = 0
                
                for i in range(len(seq) - period):
                    if seq[i] == seq[i + period]:
                        score += 1
                    comparisons += 1
                    
                if comparisons > 0:
                    period_score = score / comparisons
                    if period_score > max_score:
                        max_score = period_score
                        best_period = period
                        
            periodicity_scores.append({
                'max_periodicity': max_score,
                'best_period': best_period
            })
            
        if not periodicity_scores:
            return {}
            
        return {
            'mean_periodicity': np.mean([p['max_periodicity'] for p in periodicity_scores]),
            'period_distribution': Counter([p['best_period'] for p in periodicity_scores]),
            'highly_periodic_sequences': sum(
                1 for p in periodicity_scores if p['max_periodicity'] > 0.7
            )
        }