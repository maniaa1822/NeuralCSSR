#!/usr/bin/env python3
"""
CSSR-Enhanced FSM Extraction from Sliding Window Transformers

This combines classical CSSR principles with neural trajectory dynamics:
1. Build suffix tree from sequences (classical CSSR)
2. Augment with neural hidden state representations
3. Test equivalence using both statistical tests AND neural similarity
4. Extract FSM with proper probability preservation
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from extract_fsm_sliding_window import SlidingWindowFSMExtractor


class CSSREnhancedExtractor(SlidingWindowFSMExtractor):
    """CSSR-enhanced FSM extraction using suffix trees + neural representations."""
    
    def __init__(self, model, device, num_states=7, max_suffix_length=10, 
                 significance_level=0.001, min_suffix_count=10):
        super().__init__(model, device, num_states)
        self.max_suffix_length = max_suffix_length
        self.significance_level = significance_level
        self.neural_threshold = 5.0  # Default neural threshold
        self.min_suffix_count = min_suffix_count  # For filtering sparse suffixes
        
        # CSSR-specific storage
        self.suffix_tree = {}
        self.suffix_futures = {}
        self.suffix_hidden_states = {}
        self.causal_states = []
        
    def build_suffix_tree(self, sequences: List[List[str]], 
                         hidden_states: List[torch.Tensor]) -> None:
        """Build suffix tree augmented with hidden state information."""
        print("🌳 Building suffix tree with neural augmentation...")
        
        self.suffix_tree = {}
        self.suffix_futures = defaultdict(lambda: defaultdict(int))
        self.suffix_hidden_states = defaultdict(list)
        
        for seq_idx, (sequence, hidden_seq) in enumerate(zip(sequences, hidden_states)):
            # Convert to string sequence for suffix processing
            seq_str = ''.join(map(str, sequence))
            
            # Extract all suffixes up to max length
            for i in range(len(sequence)):
                for length in range(1, min(self.max_suffix_length + 1, len(sequence) - i + 1)):
                    # Get suffix
                    suffix_start = max(0, i - length + 1)
                    suffix = seq_str[suffix_start:i+1]
                    
                    # Record suffix occurrence
                    if suffix not in self.suffix_tree:
                        self.suffix_tree[suffix] = {
                            'count': 0,
                            'contexts': [],
                            'positions': []
                        }
                    
                    self.suffix_tree[suffix]['count'] += 1
                    self.suffix_tree[suffix]['contexts'].append(seq_idx)
                    self.suffix_tree[suffix]['positions'].append(i)
                    
                    # Record future symbol if available
                    if i + 1 < len(sequence):
                        next_symbol = str(sequence[i + 1])
                        self.suffix_futures[suffix][next_symbol] += 1
                    
                    # Record hidden state at this position
                    if i < hidden_seq.shape[0]:
                        self.suffix_hidden_states[suffix].append(hidden_seq[i].cpu().numpy())
        
        # Filter suffixes by minimum count (dynamic based on max_suffix_length)
        min_count = getattr(self, 'min_suffix_count', 10)
        pre_filter_count = len(self.suffix_tree)
        
        # Analyze suffix distribution before filtering
        length_distribution = {}
        for suffix in self.suffix_tree.keys():
            length = len(suffix)
            if length not in length_distribution:
                length_distribution[length] = {'count': 0, 'total_obs': 0, 'suffixes': []}
            length_distribution[length]['count'] += 1
            length_distribution[length]['total_obs'] += self.suffix_tree[suffix]['count']
            length_distribution[length]['suffixes'].append(suffix)
        
        print(f"📊 Pre-filter suffix analysis:")
        for length in sorted(length_distribution.keys()):
            stats = length_distribution[length]
            avg_obs = stats['total_obs'] / stats['count'] if stats['count'] > 0 else 0
            print(f"   Length {length}: {stats['count']} suffixes, avg {avg_obs:.1f} observations each")
        
        filtered_suffixes = {
            suffix: data for suffix, data in self.suffix_tree.items() 
            if data['count'] >= min_count
        }
        
        # Analyze what was filtered out
        filtered_out = pre_filter_count - len(filtered_suffixes)
        if filtered_out > 0:
            print(f"⚠️  Filtered out {filtered_out} suffixes with < {min_count} observations")
            
            # Count by length what was filtered
            filtered_by_length = {}
            for suffix, data in self.suffix_tree.items():
                if data['count'] < min_count:
                    length = len(suffix)
                    filtered_by_length[length] = filtered_by_length.get(length, 0) + 1
            
            if filtered_by_length:
                print(f"   Filtered by length: {dict(sorted(filtered_by_length.items()))}")
        
        self.suffix_tree = filtered_suffixes
        
        print(f"✅ Built suffix tree with {len(self.suffix_tree)} suffixes (was {pre_filter_count})")
        print(f"   Suffix lengths: {sorted(set(len(s) for s in self.suffix_tree.keys()))}")
    
    def test_suffix_equivalence(self, suffix1: str, suffix2: str) -> Dict:
        """Test if two suffixes are equivalent using both statistical and neural tests."""
        
        # Classical CSSR: Chi-square test on future distributions
        futures1 = self.suffix_futures[suffix1]
        futures2 = self.suffix_futures[suffix2]
        
        # Get all possible future symbols
        all_symbols = set(futures1.keys()) | set(futures2.keys())
        
        if len(all_symbols) < 2:
            # Not enough data for chi-square test
            classical_equivalent = True
            chi2_pvalue = 1.0
        else:
            # Build contingency table
            observed = []
            for symbol in sorted(all_symbols):
                observed.append([futures1.get(symbol, 0), futures2.get(symbol, 0)])
            
            observed = np.array(observed)
            
            # Check if we have sufficient data for reliable chi-square test
            total_obs = np.sum(observed)
            min_expected = 5  # Standard chi-square requirement
            
            # Perform chi-square test
            try:
                chi2_stat, chi2_pvalue, _, expected = chi2_contingency(observed)
                
                # Check if chi-square assumptions are violated (expected frequencies too low)
                if np.any(expected < min_expected) and total_obs < 20:
                    # Classical CSSR struggles here - insufficient data for reliable test
                    classical_equivalent = True  # Default to equivalent when data is sparse
                    chi2_pvalue = 1.0  # Indicate unreliable test
                else:
                    classical_equivalent = chi2_pvalue > self.significance_level
                    
            except (ValueError, RuntimeWarning):
                # Not enough data or other statistical issues
                classical_equivalent = True
                chi2_pvalue = 1.0
        
        # Neural test: Compare hidden state distributions
        hidden1 = self.suffix_hidden_states[suffix1]
        hidden2 = self.suffix_hidden_states[suffix2]
        
        if len(hidden1) == 0 or len(hidden2) == 0:
            neural_equivalent = True
            neural_distance = 0.0
        else:
            # Compare mean hidden states
            mean1 = np.mean(hidden1, axis=0)
            mean2 = np.mean(hidden2, axis=0)
            neural_distance = np.linalg.norm(mean1 - mean2)
            
            # Threshold for neural equivalence (tunable parameter)
            neural_threshold = getattr(self, 'neural_threshold', 5.0)
            neural_equivalent = neural_distance < neural_threshold
        
        return {
            'classical_equivalent': classical_equivalent,
            'neural_equivalent': neural_equivalent,
            'combined_equivalent': classical_equivalent and neural_equivalent,
            'chi2_pvalue': chi2_pvalue,
            'neural_distance': neural_distance,
            'suffix1_count': self.suffix_tree[suffix1]['count'],
            'suffix2_count': self.suffix_tree[suffix2]['count']
        }
    
    def find_equivalent_suffixes(self) -> List[Set[str]]:
        """Find groups of equivalent suffixes using both classical and neural tests."""
        print("🔍 Finding equivalent suffixes using CSSR + neural tests...")
        
        suffixes = list(self.suffix_tree.keys())
        equivalence_groups = []
        processed = set()
        
        # Logging counters
        neural_discriminant_count = 0
        classical_discriminant_count = 0
        both_agree_merge = 0
        both_agree_separate = 0
        
        for i, suffix1 in enumerate(suffixes):
            if suffix1 in processed:
                continue
            
            # Start new equivalence group
            equiv_group = {suffix1}
            processed.add(suffix1)
            
            # Test against all remaining suffixes
            for j in range(i + 1, len(suffixes)):
                suffix2 = suffixes[j]
                if suffix2 in processed:
                    continue
                
                # Test equivalence
                test_result = self.test_suffix_equivalence(suffix1, suffix2)
                
                # Log when neural vs classical disagree
                if test_result['classical_equivalent'] != test_result['neural_equivalent']:
                    if test_result['neural_equivalent'] and not test_result['classical_equivalent']:
                        # Neural says merge, classical says separate - neural is discriminant
                        neural_discriminant_count += 1
                        print(f"   🧠 NEURAL DISCRIMINANT: '{suffix1[:8]}...' ↔ '{suffix2[:8]}...' → MERGE")
                        print(f"      Classical: χ²={test_result['chi2_pvalue']:.4f} (separate), Neural: d={test_result['neural_distance']:.2f} (merge)")
                    elif test_result['classical_equivalent'] and not test_result['neural_equivalent']:
                        # Classical says merge, neural says separate - classical is discriminant  
                        classical_discriminant_count += 1
                        print(f"   📊 CLASSICAL DISCRIMINANT: '{suffix1[:8]}...' ↔ '{suffix2[:8]}...' → SEPARATE")
                        print(f"      Classical: χ²={test_result['chi2_pvalue']:.4f} (merge), Neural: d={test_result['neural_distance']:.2f} (separate)")
                elif test_result['classical_equivalent'] and test_result['neural_equivalent']:
                    both_agree_merge += 1
                else:
                    both_agree_separate += 1
                
                if test_result['combined_equivalent']:
                    equiv_group.add(suffix2)
                    processed.add(suffix2)
            
            equivalence_groups.append(equiv_group)
        
        print(f"\n📊 EQUIVALENCE TEST RESULTS:")
        print(f"   🧠 Neural discriminant (overruled classical): {neural_discriminant_count}")
        print(f"   📊 Classical discriminant (overruled neural): {classical_discriminant_count}")
        print(f"   🤝 Both agreed to merge: {both_agree_merge}")
        print(f"   🚫 Both agreed to separate: {both_agree_separate}")
        print(f"\n✅ Found {len(equivalence_groups)} equivalence groups")
        for i, group in enumerate(equivalence_groups):
            print(f"   Group {i}: {len(group)} suffixes, examples: {list(group)[:3]}")
        
        return equivalence_groups
    
    def build_causal_states(self, equivalence_groups: List[Set[str]]) -> None:
        """Build causal states from equivalent suffix groups."""
        print("🏗️  Building causal states from equivalent suffixes...")
        
        self.causal_states = []
        
        for group_idx, suffix_group in enumerate(equivalence_groups):
            # Compute merged statistics for this causal state
            merged_futures = defaultdict(int)
            merged_hidden_states = []
            total_count = 0
            
            for suffix in suffix_group:
                # Merge future distributions
                for symbol, count in self.suffix_futures[suffix].items():
                    merged_futures[symbol] += count
                
                # Merge hidden states
                merged_hidden_states.extend(self.suffix_hidden_states[suffix])
                
                # Add to total count
                total_count += self.suffix_tree[suffix]['count']
            
            # Normalize future distribution
            future_probs = {}
            if sum(merged_futures.values()) > 0:
                total_futures = sum(merged_futures.values())
                future_probs = {
                    symbol: count / total_futures 
                    for symbol, count in merged_futures.items()
                }
            
            # Compute representative hidden state
            if merged_hidden_states:
                representative_hidden = np.mean(merged_hidden_states, axis=0)
            else:
                representative_hidden = np.zeros(self.model.d_model)
            
            causal_state = {
                'id': f'CS_{group_idx}',
                'suffixes': suffix_group,
                'future_probabilities': future_probs,
                'representative_hidden': representative_hidden,
                'count': total_count,
                'size': len(suffix_group)
            }
            
            self.causal_states.append(causal_state)
        
        print(f"✅ Built {len(self.causal_states)} causal states")
        
        # Skip K-means fallback - show raw CSSR+neural results
        if hasattr(self, 'num_states') and self.num_states and len(self.causal_states) > self.num_states * 2:
            print(f"ℹ️  CSSR+neural produced {len(self.causal_states)} states (target: {self.num_states})")
            print(f"   Ratio: {len(self.causal_states) / self.num_states:.1f}x target → K-means fallback DISABLED")
            print(f"   Keeping all {len(self.causal_states)} naturally discovered states")
        elif not hasattr(self, 'num_states') or not self.num_states:
            # Natural discovery mode - no forced merging
            print(f"✅ Natural discovery mode: keeping all {len(self.causal_states)} discovered states")
        else:
            print(f"✅ CSSR+neural produced {len(self.causal_states)} states (within acceptable range for target: {getattr(self, 'num_states', 'unspecified')})")
    
    def merge_similar_causal_states(self, causal_states: List[Dict], target_states: int) -> List[Dict]:
        """Merge similar causal states using hidden state similarity."""
        if len(causal_states) <= target_states:
            return causal_states
        
        print(f"\n🔧 K-MEANS FALLBACK ACTIVATED")
        print(f"   Initial states: {len(causal_states)} → Target: {target_states}")
        print(f"   Reason: CSSR+neural produced too many states, applying K-means clustering")
        
        # Use K-means clustering on representative hidden states
        hidden_reps = np.array([state['representative_hidden'] for state in causal_states])
        
        print(f"   📊 Clustering {len(hidden_reps)} states in {hidden_reps.shape[1]}-dim space")
        
        kmeans = KMeans(n_clusters=target_states, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(hidden_reps)
        
        # Log cluster assignments
        from collections import Counter
        cluster_sizes = Counter(cluster_labels)
        print(f"   🎯 K-means cluster sizes: {dict(cluster_sizes)}")
        
        # Compute silhouette score for clustering quality
        if len(set(cluster_labels)) > 1:
            from sklearn.metrics import silhouette_score
            sil_score = silhouette_score(hidden_reps, cluster_labels)
            print(f"   📈 Clustering quality (silhouette): {sil_score:.3f}")
        
        # Merge states within each cluster
        merged_states = []
        for cluster_id in range(target_states):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_indices:
                continue
            
            print(f"   🔀 Cluster {cluster_id}: merging {len(cluster_indices)} states")
            
            # Log which states are being merged
            merged_state_ids = [causal_states[idx]['id'] for idx in cluster_indices]
            print(f"      Merging: {merged_state_ids}")
            
            # Merge all states in this cluster
            merged_suffixes = set()
            merged_futures = defaultdict(int)
            merged_hidden_states = []
            total_count = 0
            
            for idx in cluster_indices:
                state = causal_states[idx]
                merged_suffixes.update(state['suffixes'])
                
                for symbol, prob in state['future_probabilities'].items():
                    merged_futures[symbol] += prob * state['count']
                
                merged_hidden_states.append(state['representative_hidden'])
                total_count += state['count']
            
            # Normalize merged future probabilities
            if total_count > 0:
                merged_future_probs = {
                    symbol: count / total_count 
                    for symbol, count in merged_futures.items()
                }
            else:
                merged_future_probs = {}
            
            merged_state = {
                'id': f'MCS_{cluster_id}',
                'suffixes': merged_suffixes,
                'future_probabilities': merged_future_probs,
                'representative_hidden': np.mean(merged_hidden_states, axis=0),
                'count': total_count,
                'size': len(merged_suffixes)
            }
            
            merged_states.append(merged_state)
            
            # Log final merged state info
            print(f"      → Final state MCS_{cluster_id}: {len(merged_suffixes)} suffixes, count={total_count}")
            print(f"         Emission: P(0)={merged_future_probs.get('0', 0):.3f}, P(1)={merged_future_probs.get('1', 0):.3f}")
        
        print(f"\n✅ K-means fallback complete: {len(causal_states)} → {len(merged_states)} states")
        return merged_states
    
    def build_epsilon_machine(self, sequences: List[List[int]]) -> Dict:
        """Build epsilon machine from causal states with proper probability preservation."""
        print("⚙️  Building epsilon machine from causal states...")
        
        # Build transition matrix between causal states
        state_transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # For each causal state, determine transitions to other states
        for from_state_idx, from_state in enumerate(self.causal_states):
            from_id = from_state['id']
            
            # Count transitions by looking at suffix extensions
            transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # <-- FIXED
            
            for suffix in from_state['suffixes']:
                # Look at all occurrences of this suffix
                for context_idx in self.suffix_tree[suffix]['contexts']:
                    for pos in self.suffix_tree[suffix]['positions']:
                        # Look at actual next symbols in the data instead of trying all symbols
                        if context_idx < len(sequences) and pos + 1 < len(sequences[context_idx]):
                            next_symbol = str(sequences[context_idx][pos + 1])
                            extended_suffix = suffix + next_symbol
                            
                            # Find which causal state contains this extended suffix
                            for to_state_idx, to_state in enumerate(self.causal_states):
                                if extended_suffix in to_state['suffixes']:
                                    transition_counts[from_id][next_symbol][self.causal_states[to_state_idx]['id']] += 1
                                    break
            
            # Normalize to probabilities
            for symbol in ['0', '1']:
                total_transitions = sum(transition_counts[from_id][symbol].values())
                if total_transitions > 0:
                    for to_state_id, count in transition_counts[from_id][symbol].items():
                        state_transitions[from_id][symbol][to_state_id] = count / total_transitions
        
        # Build final epsilon machine structure
        epsilon_machine = {
            'num_states': len(self.causal_states),
            'alphabet': ['0', '1'],
            'states': [state['id'] for state in self.causal_states],
            'transitions': dict(state_transitions),
            'causal_state_info': {
                state['id']: {
                    'suffixes': list(state['suffixes']),
                    'future_probabilities': state['future_probabilities'],
                    'count': state['count'],
                    'size': state['size']
                }
                for state in self.causal_states
            }
        }
        
        print(f"✅ Epsilon machine built with {len(self.causal_states)} states")
        return epsilon_machine
    
    def extract_cssr_enhanced_fsm(self, data_path: Path, output_dir: Path,
                                 max_sequences: int = 1000, chunk_size: int = 25) -> Dict:
        """Complete CSSR-enhanced extraction pipeline."""
        print("🚀 Starting CSSR-enhanced FSM extraction")
        print("=" * 70)
        
        # Step 1: Extract hidden states (from parent class)
        self.extract_hidden_states(data_path, max_sequences, chunk_size)
        
        # Step 2: Build suffix tree with neural augmentation
        sequences = [[int(token) for token in seq] for seq in self.tokens]
        self.build_suffix_tree(sequences, self.hidden_states)
        
        # Step 3: Find equivalent suffixes using both classical and neural tests
        equivalence_groups = self.find_equivalent_suffixes()
        
        # Step 4: Build causal states
        self.build_causal_states(equivalence_groups)
        
        # Step 5: Build epsilon machine
        epsilon_machine = self.build_epsilon_machine(sequences)
        
        # Step 6: Save results
        self.save_cssr_results(epsilon_machine, output_dir)
        
        print("=" * 70)
        print("🎉 CSSR-enhanced extraction complete!")
        
        return epsilon_machine
    
    def save_cssr_results(self, epsilon_machine: Dict, output_dir: Path) -> None:
        """Save CSSR-enhanced extraction results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        
        results = {
            'extraction_method': 'cssr_enhanced_trajectory_dynamics',
            'model_info': {
                'num_causal_states': len(self.causal_states),
                'num_suffixes_processed': len(self.suffix_tree),
                'max_suffix_length': self.max_suffix_length,
                'significance_level': self.significance_level
            },
            'epsilon_machine': epsilon_machine,
            'suffix_statistics': {
                suffix: {
                    'count': data['count'],
                    'future_distribution': dict(self.suffix_futures[suffix])
                }
                for suffix, data in self.suffix_tree.items()
            }
        }
        
        output_file = output_dir / 'cssr_enhanced_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ CSSR-enhanced results saved to {output_file}")


def main():
    import argparse
    from extract_fsm_sliding_window import load_model_from_checkpoint
    
    parser = argparse.ArgumentParser(description="CSSR-enhanced FSM extraction")
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, 
                       default=Path('domain_machines/seven_state_human/seven_state_human/seven_state_human.dat'))
    parser.add_argument('--output', type=Path, default=Path('results/cssr_enhanced_extraction'))
    parser.add_argument('--max-sequences', type=int, default=500)
    parser.add_argument('--max-suffix-length', type=int, default=8)
    parser.add_argument('--significance', type=float, default=0.001)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    extractor = CSSREnhancedExtractor(
        model, device, 
        max_suffix_length=args.max_suffix_length,
        significance_level=args.significance
    )
    
    epsilon_machine = extractor.extract_cssr_enhanced_fsm(
        args.data, args.output, args.max_sequences
    )
    
    return epsilon_machine


if __name__ == '__main__':
    main()