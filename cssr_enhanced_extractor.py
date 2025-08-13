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
from sklearn.metrics import silhouette_score, mutual_info_score

# Optional canonical epsilon machine (unifilar) post-processing
try:
    from canonical_epsilon_machine import (
        build_canonical_epsilon_machine, CanonicalBuildConfig
    )
except ImportError:
    build_canonical_epsilon_machine = None
    CanonicalBuildConfig = None

from extract_fsm_sliding_window import SlidingWindowFSMExtractor


class CSSREnhancedExtractor(SlidingWindowFSMExtractor):
    """CSSR-enhanced FSM extraction using suffix trees + neural representations."""
    
    def __init__(self, model, device, num_states=None, max_suffix_length=10, 
                 significance_level=0.001, min_suffix_count=10,
                 use_neural_test=True, use_classical_test=True,
                 use_information_theoretic_threshold=False, neural_threshold=5.0,
                 threshold_method='silhouette', use_emission_based_merging=False, 
                 emission_similarity_threshold=0.05, verbose_discriminant=False):
        super().__init__(model, device, num_states)
        self.max_suffix_length = max_suffix_length
        self.significance_level = significance_level
        self.neural_threshold = neural_threshold  # Default neural threshold
        self.threshold_method = threshold_method  # Method for computing threshold
        self.min_suffix_count = min_suffix_count  # For filtering sparse suffixes
        self.kl_threshold = 0.001  # Default KL-divergence threshold (very small)
        self.kl_steps = 3  # Number of forward steps for multi-step KL-divergence
        self.kl_weights = [1.0, 0.8, 0.6]  # Weights for each step (decreasing importance)
        self.use_neural_test = use_neural_test
        self.use_classical_test = use_classical_test
        self.use_information_theoretic_threshold = use_information_theoretic_threshold
        self.use_emission_based_merging = use_emission_based_merging
        self.emission_similarity_threshold = emission_similarity_threshold
        self.verbose_discriminant = verbose_discriminant
        
        # CSSR-specific storage
        self.suffix_tree = {}
        self.suffix_futures = {}
        self.suffix_hidden_states = {}
        self.suffix_logits = {}  # Store transformer logits for KL-divergence testing
        self.causal_states = []
        
    def extract_hidden_states_and_logits(self, data_path: Path, max_sequences: int = 1000, 
                                        chunk_size: int = 25) -> tuple:
        """Extract both hidden states and logits from sliding window model."""
        print(f"🔍 Extracting hidden states and logits from {data_path}")
        print(f"Using sliding windows of size {chunk_size}")
        
        # Load data with sliding windows (same as training)
        from models.data_utils import SequenceDataset
        dataset = SequenceDataset(data_path, chunk_size=chunk_size, sliding_window=True)
        
        # Limit sequences for computational efficiency
        num_sequences = min(len(dataset), max_sequences)
        print(f"Loaded {data_path.name}: {len(dataset.sequences):,} symbols → {len(dataset)} sliding windows of size {chunk_size}")
        print(f"Processing {num_sequences} sequences (of {len(dataset)} available)")
        
        sequences = []
        hidden_states = []
        logits_list = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(num_sequences):
                if i % 100 == 0:
                    print(f"  Processing sequence {i}/{num_sequences}")
                
                # Get sequence and convert to tensor
                seq = dataset[i]
                input_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Forward pass to get logits (what we need for KL-divergence)
                logits_seq = self.model(input_tensor)
                
                # For KL-divergence testing, we don't need intermediate hidden states
                # We'll create dummy hidden states to maintain compatibility
                seq_len = logits_seq.shape[1]
                d_model = getattr(self.model, 'd_model', 64)  # Default size
                hidden_seq = torch.zeros(1, seq_len, d_model, device=self.device)
                
                # Remove batch dimension and store
                if hidden_seq is not None:
                    hidden_states.append(hidden_seq.squeeze(0))
                if logits_seq is not None:
                    logits_list.append(logits_seq.squeeze(0))
                sequences.append(seq)
        
        print(f"✅ Extracted {len(sequences)} sequences")
        return sequences, hidden_states, logits_list
    
    def build_suffix_tree(self, sequences: List[List[str]], 
                         hidden_states: List[torch.Tensor], 
                         logits_list: List[torch.Tensor] = None) -> None:
        """Build suffix tree augmented with hidden state and logit information."""
        print("🌳 Building suffix tree with neural augmentation...")
        
        self.suffix_tree = {}
        self.suffix_futures = defaultdict(lambda: defaultdict(int))
        self.suffix_hidden_states = defaultdict(list)
        self.suffix_logits = defaultdict(list)
        
        # Handle case where logits_list might be None
        if logits_list is None:
            logits_list = [None] * len(sequences)
        
        for seq_idx, (sequence, hidden_seq, logits_seq) in enumerate(zip(sequences, hidden_states, logits_list)):
            # Convert tensor to string sequence for suffix processing
            if torch.is_tensor(sequence):
                seq_str = ''.join(map(str, sequence.tolist()))
            else:
                seq_str = ''.join(map(str, sequence))
            
            # Extract all suffixes up to max length
            for i in range(len(sequence)):
                # Only consider valid suffix lengths that end at position i
                max_len_at_pos = min(self.max_suffix_length, i + 1)
                for length in range(1, max_len_at_pos + 1):
                    # Get suffix ending at i with given length
                    suffix_start = i - length + 1
                    suffix = seq_str[suffix_start:i + 1]
                    
                    # Record suffix occurrence
                    if suffix not in self.suffix_tree:
                        self.suffix_tree[suffix] = {
                            'count': 0,
                            'contexts': [],
                            'positions': []
                        }
                    
                    self.suffix_tree[suffix]['count'] += 1
                    # Store full 25-token context for proper transformer inference
                    full_context = sequence.tolist() if torch.is_tensor(sequence) else sequence
                    self.suffix_tree[suffix]['contexts'].append(full_context)
                    self.suffix_tree[suffix]['positions'].append(i)
                    
                    # Record future symbol if available
                    if i + 1 < len(sequence):
                        next_symbol = str(int(sequence[i + 1]))  # Ensure consistent string format
                        self.suffix_futures[suffix][next_symbol] += 1
                    
                    # Record hidden state at this position
                    if i < hidden_seq.shape[0]:
                        self.suffix_hidden_states[suffix].append(hidden_seq[i].cpu().numpy())
                    
                    # Record logits at this position for KL-divergence testing
                    if logits_seq is not None and i < logits_seq.shape[0]:
                        self.suffix_logits[suffix].append(logits_seq[i].cpu().numpy())
        
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
        
        # Compute information-theoretic threshold if enabled
        if self.use_information_theoretic_threshold:
            threshold_method = getattr(self, 'threshold_method', 'silhouette')
            if threshold_method == 'mutual_info':
                self._compute_mutual_info_threshold()
            elif threshold_method == 'plateau':
                self._compute_plateau_threshold()
            elif threshold_method == 'kl_divergence':
                self._compute_kl_divergence_threshold()
            else:
                self._compute_information_theoretic_threshold()
    
    def _compute_information_theoretic_threshold(self) -> None:
        """Compute optimal neural threshold using silhouette-based method."""
        print("🧮 Computing silhouette-based neural threshold...")
        
        # Collect all pairwise neural distances and hidden states
        suffixes = list(self.suffix_tree.keys())
        
        print(f"   Computing distances for {len(suffixes)} suffixes...")
        
        # Collect all hidden state representations
        hidden_representations = []
        suffix_list = []
        
        for suffix in suffixes:
            hidden_states = self.suffix_hidden_states.get(suffix, [])
            if len(hidden_states) > 0:
                # Use mean representation for each suffix
                mean_hidden = np.mean(hidden_states, axis=0)
                hidden_representations.append(mean_hidden)
                suffix_list.append(suffix)
        
        if len(hidden_representations) < 10:
            print("⚠️  Not enough suffix representations, keeping default threshold")
            return
        
        hidden_representations = np.array(hidden_representations)
        
        # Test different thresholds using silhouette method
        distances = []
        for i in range(len(hidden_representations)):
            for j in range(i + 1, len(hidden_representations)):
                dist = np.linalg.norm(hidden_representations[i] - hidden_representations[j])
                distances.append(dist)
        
        distances = np.array(distances)
        min_dist, max_dist = np.min(distances), np.max(distances)
        
        # Test range of thresholds
        test_thresholds = np.linspace(min_dist, max_dist, 50)
        best_threshold = self.neural_threshold
        best_silhouette = -1
        
        print(f"   Testing thresholds from {min_dist:.3f} to {max_dist:.3f}")
        
        for threshold in test_thresholds:
            # Create clusters based on threshold
            cluster_labels = []
            current_cluster = 0
            processed = set()
            
            for i, suffix1 in enumerate(suffix_list):
                if i in processed:
                    continue
                
                cluster = [i]
                processed.add(i)
                
                for j in range(i + 1, len(suffix_list)):
                    if j in processed:
                        continue
                    
                    # Compute distance
                    dist = np.linalg.norm(hidden_representations[i] - hidden_representations[j])
                    if dist < threshold:
                        cluster.append(j)
                        processed.add(j)
                
                # Assign cluster labels
                for idx in cluster:
                    if len(cluster_labels) <= idx:
                        cluster_labels.extend([-1] * (idx - len(cluster_labels) + 1))
                    cluster_labels[idx] = current_cluster
                
                current_cluster += 1
            
            # Ensure all points have labels
            while len(cluster_labels) < len(suffix_list):
                cluster_labels.append(current_cluster)
                current_cluster += 1
            
            # Check if we have valid clustering (more than 1 cluster, less than n_samples)
            n_clusters = len(set(cluster_labels))
            if n_clusters > 1 and n_clusters < len(suffix_list):
                try:
                    silhouette = silhouette_score(hidden_representations, cluster_labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_threshold = threshold
                except:
                    continue
        
        if best_silhouette > -1:
            print(f"🎯 Silhouette-based threshold: {best_threshold:.3f} (silhouette: {best_silhouette:.3f})")
            print(f"   Previous threshold: {self.neural_threshold:.3f}")
            self.neural_threshold = best_threshold
        else:
            print("⚠️  Could not find optimal threshold with silhouette method, keeping default")
    
    def _compute_mutual_info_threshold(self) -> None:
        """Compute optimal neural threshold using mutual information with classical equivalences."""
        print("🧮 Computing mutual information-based neural threshold...")
        
        # Collect all pairwise neural distances and hidden states
        suffixes = list(self.suffix_tree.keys())
        
        if len(suffixes) < 10:
            print("⚠️  Too few suffixes for threshold optimization, keeping default")
            return
            
        # Collect all hidden state representations
        hidden_representations = []
        suffix_list = []
        
        for suffix in suffixes:
            hidden_states = self.suffix_hidden_states.get(suffix, [])
            if len(hidden_states) > 0:
                # Use mean representation for each suffix
                mean_hidden = np.mean(hidden_states, axis=0)
                hidden_representations.append(mean_hidden)
                suffix_list.append(suffix)
        
        if len(hidden_representations) < 10:
            print("⚠️  Not enough suffix representations, keeping default threshold")
            return
            
        hidden_representations = np.array(hidden_representations)
        
        # Test different thresholds and calculate mutual information with classical equivalences
        distances = []
        for i in range(len(hidden_representations)):
            for j in range(i + 1, len(hidden_representations)):
                dist = np.linalg.norm(hidden_representations[i] - hidden_representations[j])
                distances.append(dist)
        
        if not distances:
            return
            
        min_dist, max_dist = min(distances), max(distances)
        threshold_range = np.linspace(min_dist * 1.1, max_dist * 0.9, 30)
        
        # Build classical equivalence ground truth
        classical_equivalences = []
        
        for i, suffix1 in enumerate(suffix_list):
            for j, suffix2 in enumerate(suffix_list[i+1:], i+1):
                # Test classical equivalence
                if self.use_classical_test:
                    futures1 = self.suffix_futures.get(suffix1, [])
                    futures2 = self.suffix_futures.get(suffix2, [])
                    
                    if len(futures1) >= 5 and len(futures2) >= 5:
                        # Simple chi-square test
                        from collections import Counter
                        counter1 = Counter(futures1)
                        counter2 = Counter(futures2)
                        
                        all_symbols = set(counter1.keys()) | set(counter2.keys())
                        if len(all_symbols) > 1:
                            observed = []
                            for symbol in sorted(all_symbols):
                                observed.append([counter1.get(symbol, 0), counter2.get(symbol, 0)])
                            
                            try:
                                from scipy.stats import chi2_contingency
                                chi2_stat, chi2_pvalue = chi2_contingency(observed)
                                classical_equivalent = chi2_pvalue > self.significance_level
                            except:
                                classical_equivalent = False
                        else:
                            classical_equivalent = True
                    else:
                        classical_equivalent = False
                else:
                    # If no classical test, use heuristic based on suffix similarity
                    classical_equivalent = len(set(suffix1[-3:]) & set(suffix2[-3:])) > 1
                    
                classical_equivalences.append(1 if classical_equivalent else 0)
        
        print(f"   Testing {len(threshold_range)} thresholds on {len(classical_equivalences)} suffix pairs")
        
        # Calculate MI curve for different thresholds
        mi_curve = []
        
        for threshold in threshold_range:
            # Get neural equivalences at this threshold
            neural_equivalences = []
            
            pair_idx = 0
            for i, suffix1 in enumerate(suffix_list):
                for j, suffix2 in enumerate(suffix_list[i+1:], i+1):
                    neural_dist = np.linalg.norm(hidden_representations[i] - hidden_representations[j])
                    neural_equivalent = 1 if neural_dist < threshold else 0
                    neural_equivalences.append(neural_equivalent)
                    pair_idx += 1
            
            # Calculate mutual information between neural and classical equivalences
            if len(neural_equivalences) == len(classical_equivalences) and len(set(neural_equivalences)) > 1:
                try:
                    mi = self._compute_mutual_info(classical_equivalences, neural_equivalences)
                    mi_curve.append((threshold, mi))
                except:
                    mi_curve.append((threshold, 0.0))
            else:
                mi_curve.append((threshold, 0.0))
        
        if not mi_curve:
            print("⚠️  Could not compute MI curve, keeping default threshold")
            return
        
        # Find optimal threshold using elbow point detection
        optimal_threshold = find_elbow_point(mi_curve, method='max_curvature')
        
        # Log analysis results
        max_mi = max(point[1] for point in mi_curve)
        optimal_mi = next((mi for thresh, mi in mi_curve if abs(thresh - optimal_threshold) < 1e-6), max_mi)
        
        print(f"📊 Mutual Information threshold analysis:")
        print(f"   Neural distance range: [{min_dist:.3f}, {max_dist:.3f}]")
        print(f"   Tested {len(mi_curve)} thresholds")
        print(f"   Maximum MI: {max_mi:.4f}")
        print(f"   Optimal threshold: {optimal_threshold:.3f} (MI: {optimal_mi:.4f})")
        print(f"   Previous threshold: {self.neural_threshold:.3f}")
        
        if optimal_threshold > 0:
            self.neural_threshold = optimal_threshold
        else:
            print("⚠️  Could not find valid threshold, keeping default")
    
    def _compute_plateau_threshold(self) -> None:
        """Compute optimal neural threshold using plateau detection method."""
        print("🧮 Computing plateau-based neural threshold...")
        
        # Collect all pairwise neural distances and hidden states
        suffixes = list(self.suffix_tree.keys())
        
        if len(suffixes) < 10:
            print("⚠️  Too few suffixes for threshold optimization, keeping default")
            return
            
        # Collect all hidden state representations
        hidden_representations = []
        suffix_list = []
        
        for suffix in suffixes:
            hidden_states = self.suffix_hidden_states.get(suffix, [])
            if len(hidden_states) > 0:
                # Use mean representation for each suffix
                mean_hidden = np.mean(hidden_states, axis=0)
                hidden_representations.append(mean_hidden)
                suffix_list.append(suffix)
        
        if len(hidden_representations) < 10:
            print("⚠️  Not enough suffix representations, keeping default threshold")
            return
            
        hidden_representations = np.array(hidden_representations)
        
        # Calculate distances
        distances = []
        for i in range(len(hidden_representations)):
            for j in range(i + 1, len(hidden_representations)):
                dist = np.linalg.norm(hidden_representations[i] - hidden_representations[j])
                distances.append(dist)
        
        distances = np.array(distances)
        min_dist, max_dist = np.min(distances), np.max(distances)
        
        # Test range of thresholds and count merges
        test_thresholds = np.linspace(min_dist, max_dist, 50)
        merge_counts = []
        
        print(f"   Testing thresholds from {min_dist:.3f} to {max_dist:.3f}")
        
        for threshold in test_thresholds:
            # Count how many merges would occur at this threshold
            merge_count = 0
            for i in range(len(hidden_representations)):
                for j in range(i + 1, len(hidden_representations)):
                    dist = np.linalg.norm(hidden_representations[i] - hidden_representations[j])
                    if dist < threshold:
                        merge_count += 1
            merge_counts.append(merge_count)
        
        # Find plateau using derivatives
        merge_counts = np.array(merge_counts)
        if len(merge_counts) < 3:
            print("⚠️  Not enough data points for plateau detection, keeping default")
            return
        
        # Calculate first derivative (rate of change)
        first_deriv = np.gradient(merge_counts)
        
        # Find plateau point using elbow detection on merge curve
        merge_curve = [(test_thresholds[i], merge_counts[i]) for i in range(len(test_thresholds))]
        optimal_threshold = find_elbow_point(merge_curve, method='plateau')
        
        # Calculate merge statistics
        optimal_idx = np.argmin(np.abs(test_thresholds - optimal_threshold))
        optimal_merges = merge_counts[optimal_idx]
        total_pairs = len(suffix_list) * (len(suffix_list) - 1) // 2
        
        print(f"📊 Plateau threshold analysis:")
        print(f"   Neural distance range: [{min_dist:.3f}, {max_dist:.3f}]")
        print(f"   Optimal threshold: {optimal_threshold:.3f}")
        print(f"   Merges at optimal: {optimal_merges}/{total_pairs} ({100*optimal_merges/total_pairs:.1f}%)")
        print(f"   Previous threshold: {self.neural_threshold:.3f}")
        
        if optimal_threshold > 0:
            self.neural_threshold = optimal_threshold
        else:
            print("⚠️  Could not find valid threshold, keeping default")
    
    def _compute_kl_divergence_threshold(self) -> None:
        """Compute optimal KL-divergence threshold using transformer logits."""
        print("🧮 Computing KL-divergence threshold based on transformer logits...")
        
        # Collect all pairwise KL divergences
        suffixes = list(self.suffix_tree.keys())
        
        if len(suffixes) < 10:
            print("⚠️  Too few suffixes for KL threshold optimization, keeping default")
            return
        
        # Filter suffixes that have logit information
        suffixes_with_logits = [s for s in suffixes if s in self.suffix_logits and len(self.suffix_logits[s]) > 0]
        
        if len(suffixes_with_logits) < 10:
            print("⚠️  Too few suffixes with logit information, keeping default")
            return
        
        print(f"   Computing KL divergences for {len(suffixes_with_logits)} suffixes...")
        
        # Compute average logit distributions for each suffix
        suffix_distributions = {}
        for suffix in suffixes_with_logits:
            logit_arrays = self.suffix_logits[suffix]
            if len(logit_arrays) > 0:
                # Average logits across all occurrences
                mean_logits = np.mean(logit_arrays, axis=0)
                # Convert to probabilities using softmax
                probs = self._softmax(mean_logits)
                suffix_distributions[suffix] = probs
        
        # Calculate pairwise KL divergences
        kl_divergences = []
        suffix_pairs = []
        
        for i, suffix1 in enumerate(suffixes_with_logits):
            for j, suffix2 in enumerate(suffixes_with_logits[i+1:], i+1):
                if suffix1 in suffix_distributions and suffix2 in suffix_distributions:
                    kl_div = self._kl_divergence(suffix_distributions[suffix1], suffix_distributions[suffix2])
                    kl_divergences.append(kl_div)
                    suffix_pairs.append((suffix1, suffix2))
        
        if not kl_divergences:
            print("⚠️  Could not compute KL divergences, keeping default threshold")
            return
        
        kl_divergences = np.array(kl_divergences)
        
        # Use elbow method to find optimal threshold
        # Sort KL divergences and find the elbow point
        sorted_kl = np.sort(kl_divergences)
        
        # Create curve for elbow detection
        percentiles = np.linspace(0, 100, 50)
        kl_curve = [(np.percentile(sorted_kl, p), p) for p in percentiles]
        
        # Find elbow point - this will be our threshold
        optimal_kl_threshold = find_elbow_point(kl_curve, method='max_curvature')
        
        # Statistics
        min_kl, max_kl, mean_kl = np.min(kl_divergences), np.max(kl_divergences), np.mean(kl_divergences)
        std_kl = np.std(kl_divergences)
        
        print(f"📊 KL-divergence threshold analysis:")
        print(f"   KL divergence range: [{min_kl:.6f}, {max_kl:.6f}]")
        print(f"   Mean KL divergence: {mean_kl:.6f} ± {std_kl:.6f}")
        print(f"   Tested {len(kl_divergences)} suffix pairs")
        print(f"   Optimal KL threshold: {optimal_kl_threshold:.6f}")
        print(f"   Previous neural threshold: {self.neural_threshold:.3f}")
        
        # Store the KL threshold for use in equivalence testing
        self.kl_threshold = optimal_kl_threshold
        print(f"✅ KL-divergence threshold set to {self.kl_threshold:.6f}")
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from logits."""
        # Subtract max for numerical stability
        logits_stable = logits - np.max(logits)
        exp_logits = np.exp(logits_stable)
        return exp_logits / np.sum(exp_logits)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence KL(P||Q) between two probability distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        p_safe = np.clip(p, epsilon, 1.0)
        q_safe = np.clip(q, epsilon, 1.0)
        
        # KL(P||Q) = sum(P * log(P/Q))
        return np.sum(p_safe * np.log(p_safe / q_safe))
    
    def _compute_multi_step_predictions(self, suffix: str, k_steps: int = 3) -> List[np.ndarray]:
        """Compute k-step ahead predictions from a suffix using transformer with full context."""
        if suffix not in self.suffix_logits or len(self.suffix_logits[suffix]) == 0:
            # Return uniform distributions if no data
            vocab_size = 3  # Binary + padding
            return [np.ones(vocab_size) / vocab_size for _ in range(k_steps)]
        
        # CRITICAL FIX: Use full training context instead of short suffix
        # Get full context windows where this suffix appears
        suffix_contexts = self.suffix_tree[suffix].get('contexts', [])
        if not suffix_contexts:
            # Fallback to mean logits if no contexts available
            mean_logits = np.mean(self.suffix_logits[suffix], axis=0)
            return [self._softmax(mean_logits) for _ in range(k_steps)]
        
        # Use first available full context (25 tokens) for prediction
        full_context = suffix_contexts[0]  # This should be a 25-token sequence
        
        # Convert to tensor with proper length (up to 25 tokens for training compatibility)
        if len(full_context) > 25:
            context_tokens = full_context[-25:]  # Last 25 tokens
        else:
            context_tokens = full_context
            
        # Roll forward k steps using transformer with full context
        predictions = []
        current_input = torch.tensor(context_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            for step in range(k_steps):
                # Get next-token logits
                logits = self.model(current_input)
                
                # Extract logits for last position
                if len(logits.shape) == 3:  # [batch, seq, vocab]
                    step_logits = logits[0, -1, :].cpu().numpy()
                else:  # [batch, vocab] 
                    step_logits = logits[0, :].cpu().numpy()
                
                # Convert to probabilities and store
                step_probs = self._softmax(step_logits)
                predictions.append(step_probs)
                
                # Sample next token for continuation (or use argmax for deterministic)
                next_token = np.argmax(step_probs)
                
                # Update input sequence for next step
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)
                
                # Keep sequence length manageable
                if current_input.shape[1] > 25:
                    current_input = current_input[:, -25:]
        
        return predictions
    
    def _multi_step_kl_divergence(self, suffix1: str, suffix2: str, k_steps: int = None, weights: List[float] = None) -> float:
        """Compute multi-step KL divergence between two suffixes."""
        if k_steps is None:
            k_steps = self.kl_steps
        if weights is None:
            weights = self.kl_weights[:k_steps]
        
        # Ensure we have enough weights
        while len(weights) < k_steps:
            weights.append(weights[-1] * 0.8)  # Exponential decay
        
        # Get multi-step predictions for both suffixes
        preds1 = self._compute_multi_step_predictions(suffix1, k_steps)
        preds2 = self._compute_multi_step_predictions(suffix2, k_steps)
        
        # Compute weighted sum of KL divergences at each step
        total_kl = 0.0
        total_weight = 0.0
        
        for step in range(k_steps):
            if step < len(preds1) and step < len(preds2):
                # Symmetric KL divergence at this step
                kl_step = 0.5 * (
                    self._kl_divergence(preds1[step], preds2[step]) +
                    self._kl_divergence(preds2[step], preds1[step])
                )
                
                weight = weights[step] if step < len(weights) else weights[-1]
                total_kl += weight * kl_step
                total_weight += weight
        
        # Normalize by total weight
        return total_kl / total_weight if total_weight > 0 else float('inf')
    
    def test_suffix_equivalence(self, suffix1: str, suffix2: str) -> Dict:
        """Test if two suffixes are equivalent using both statistical and neural tests."""
        
        # Classical CSSR: Chi-square test on future distributions (only if enabled)
        if self.use_classical_test:
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
        else:
            # Classical test disabled - always equivalent
            classical_equivalent = True
            chi2_pvalue = 1.0
        
        # Neural test: Choose between hidden state comparison or KL-divergence
        if self.use_neural_test:
            # Check if using KL-divergence method
            if getattr(self, 'threshold_method', 'silhouette') == 'kl_divergence':
                # KL-divergence test using transformer logits
                logits1 = self.suffix_logits.get(suffix1, [])
                logits2 = self.suffix_logits.get(suffix2, [])
                
                if len(logits1) == 0 or len(logits2) == 0:
                    neural_equivalent = True
                    neural_distance = 0.0
                else:
                    # Compute average probability distributions
                    mean_logits1 = np.mean(logits1, axis=0)
                    mean_logits2 = np.mean(logits2, axis=0)
                    
                    probs1 = self._softmax(mean_logits1)
                    probs2 = self._softmax(mean_logits2)
                    
                    # Use multi-step KL divergence for richer temporal comparison
                    kl_div = self._multi_step_kl_divergence(suffix1, suffix2)
                    neural_distance = kl_div
                    
                    # Use KL threshold
                    kl_threshold = getattr(self, 'kl_threshold', 0.001)
                    neural_equivalent = kl_div < kl_threshold
            else:
                # Standard hidden state comparison
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
        else:
            # Neural test disabled - always equivalent
            neural_equivalent = True
            neural_distance = 0.0
        
        # Determine combined equivalence based on enabled tests
        if self.use_classical_test and self.use_neural_test:
            # Both tests enabled: both must agree
            combined_equivalent = classical_equivalent and neural_equivalent
        elif self.use_classical_test and not self.use_neural_test:
            # Classical only
            combined_equivalent = classical_equivalent
        elif not self.use_classical_test and self.use_neural_test:
            # Neural only
            combined_equivalent = neural_equivalent
        else:
            # Neither test enabled - always equivalent (shouldn't happen)
            combined_equivalent = True
            
        return {
            'classical_equivalent': classical_equivalent,
            'neural_equivalent': neural_equivalent,
            'combined_equivalent': combined_equivalent,
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
                
                # Log when neural vs classical disagree (only if verbose mode enabled)
                if self.verbose_discriminant and test_result['classical_equivalent'] != test_result['neural_equivalent']:
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
                elif test_result['classical_equivalent'] != test_result['neural_equivalent']:
                    # Still count disagreements even when not printing details
                    if test_result['neural_equivalent'] and not test_result['classical_equivalent']:
                        neural_discriminant_count += 1
                    elif test_result['classical_equivalent'] and not test_result['neural_equivalent']:
                        classical_discriminant_count += 1
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
        
        # Apply emission-based merging if enabled
        if self.use_emission_based_merging:
            print(f"🔄 Applying emission-based merging (threshold: {self.emission_similarity_threshold})")
            self.causal_states = self._merge_by_emission_similarity(self.causal_states)
            print(f"✅ After emission-based merging: {len(self.causal_states)} states")
        
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
    
    def _merge_by_emission_similarity(self, causal_states: List[Dict]) -> List[Dict]:
        """Merge causal states with similar emission patterns."""
        print(f"   🔍 Testing emission similarities among {len(causal_states)} states...")
        
        merged_states = []
        processed = set()
        merge_count = 0
        
        for i, state1 in enumerate(causal_states):
            if i in processed:
                continue
                
            # Start a new merged group with this state
            states_to_merge = [state1]
            processed.add(i)
            
            # Find states with similar emission patterns
            for j, state2 in enumerate(causal_states[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Calculate emission pattern similarity
                p1 = state1['future_probabilities']
                p2 = state2['future_probabilities']
                
                # Get common symbols
                symbols = set(p1.keys()) | set(p2.keys())
                if not symbols:
                    continue
                
                # Calculate maximum absolute difference
                max_diff = 0.0
                for symbol in symbols:
                    diff = abs(p1.get(symbol, 0.0) - p2.get(symbol, 0.0))
                    max_diff = max(max_diff, diff)
                
                # Merge if within threshold
                if max_diff <= self.emission_similarity_threshold:
                    states_to_merge.append(state2)
                    processed.add(j)
                    print(f"      Merging {state1['id']} with {state2['id']} (max_diff: {max_diff:.3f})")
                    merge_count += 1
            
            # Create merged state
            if len(states_to_merge) == 1:
                # No merging needed
                merged_states.append(state1)
            else:
                # Merge multiple states
                merged_suffixes = set()
                merged_futures = defaultdict(float)
                merged_hidden_states = []
                total_count = 0
                
                for state in states_to_merge:
                    merged_suffixes.update(state['suffixes'])
                    
                    # Weight by count for proper averaging
                    for symbol, prob in state['future_probabilities'].items():
                        merged_futures[symbol] += prob * state['count']
                    
                    merged_hidden_states.append(state['representative_hidden'])
                    total_count += state['count']
                
                # Normalize probabilities
                if total_count > 0:
                    merged_future_probs = {
                        symbol: count / total_count 
                        for symbol, count in merged_futures.items()
                    }
                else:
                    merged_future_probs = {}
                
                merged_state = {
                    'id': f'MES_{len(merged_states)}',  # MES = Merged Emission State
                    'suffixes': merged_suffixes,
                    'future_probabilities': merged_future_probs,
                    'representative_hidden': np.mean(merged_hidden_states, axis=0),
                    'count': total_count,
                    'size': len(merged_suffixes)
                }
                
                merged_states.append(merged_state)
                
                # Log the merge
                state_ids = [s['id'] for s in states_to_merge]
                print(f"   ➡️  Created {merged_state['id']} from {state_ids}")
                print(f"      Final emission: P(0)={merged_future_probs.get('0', 0):.3f}, P(1)={merged_future_probs.get('1', 0):.3f}")
        
        print(f"   📊 Emission-based merging: {len(causal_states)} → {len(merged_states)} states ({merge_count} merges)")
        return merged_states
    
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
        
        # Build suffix-to-state lookup for fast transitions
        print("🔍 Building suffix-to-state lookup...")
        suffix_to_state = {}
        for state in self.causal_states:
            for suffix in state['suffixes']:
                suffix_to_state[suffix] = state['id']
        print(f"✅ Indexed {len(suffix_to_state)} suffixes across {len(self.causal_states)} states")
        
        # Build transition matrix between causal states
        state_transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # For each causal state, determine transitions to other states
        print("🔄 Computing state transitions...")
        for from_state_idx, from_state in enumerate(self.causal_states):
            if from_state_idx % 4 == 0:
                print(f"   Processing state {from_state_idx}/{len(self.causal_states)}")
            
            from_id = from_state['id']
            
            # Count transitions by looking at suffix extensions
            transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            
            for suffix in from_state['suffixes']:
                # Limit occurrences per suffix to speed up computation
                contexts = self.suffix_tree[suffix]['contexts']
                positions = self.suffix_tree[suffix]['positions']
                
                # Sample occurrences if there are too many (keep first 100 for stability)
                max_occurrences = 100
                if len(contexts) > max_occurrences:
                    sample_indices = list(range(min(max_occurrences, len(contexts))))
                    contexts = [contexts[i] for i in sample_indices]
                    positions = [positions[i] for i in sample_indices]
                
                # Look at sampled occurrences of this suffix
                for context_sequence, pos in zip(contexts, positions):
                    # context_sequence is now the full sequence, pos is position within it
                    if pos + 1 < len(context_sequence):
                        next_symbol = str(int(context_sequence[pos + 1]))  # Ensure consistent format
                        extended_suffix = suffix + next_symbol
                        
                        # Fast lookup using prebuilt index
                        if extended_suffix in suffix_to_state:
                            to_state_id = suffix_to_state[extended_suffix]
                            transition_counts[from_id][next_symbol][to_state_id] += 1
            
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
                                 max_sequences: int = 1000, chunk_size: int = 25,
                                 build_canonical: bool = True,
                                 canonical_config: Dict = None,
                                 canonical_as_primary: bool = True) -> Dict:
        """Complete CSSR-enhanced extraction pipeline."""
        print("🚀 Starting CSSR-enhanced FSM extraction")
        print("=" * 70)
        
        # Step 1: Extract hidden states and logits (if using KL-divergence method)
        if getattr(self, 'threshold_method', 'silhouette') == 'kl_divergence':
            sequences, hidden_states, logits_list = self.extract_hidden_states_and_logits(data_path, max_sequences, chunk_size)
            # Store the sequences directly
            self.hidden_states = hidden_states
            # Convert sequences to the format expected by build_suffix_tree
            processed_sequences = sequences
        else:
            # Standard extraction (from parent class)
            self.extract_hidden_states(data_path, max_sequences, chunk_size)
            logits_list = None
            # Convert tokens to sequences
            processed_sequences = [[int(token) for token in seq] for seq in self.tokens]
        
        # Step 2: Build suffix tree with neural augmentation
        self.build_suffix_tree(processed_sequences, self.hidden_states, logits_list)
        
        # Step 3: Find equivalent suffixes using both classical and neural tests
        equivalence_groups = self.find_equivalent_suffixes()
        
        # Step 4: Build causal states
        self.build_causal_states(equivalence_groups)
        
        # Step 5: Build (possibly non-unifilar) empirical epsilon machine
        # NOTE: original code passed an undefined variable in some branches; use processed_sequences.
        epsilon_machine = self.build_epsilon_machine(processed_sequences)

        # Step 5b: Optionally build canonical unifilar version
        canonical_machine = None
        if build_canonical and build_canonical_epsilon_machine is not None:
            try:
                cfg = None
                if canonical_config and CanonicalBuildConfig is not None:
                    # Filter only valid CanonicalBuildConfig fields
                    valid_fields = {f.name for f in CanonicalBuildConfig.__dataclass_fields__.values()}
                    filtered = {k: v for k, v in canonical_config.items() if k in valid_fields}
                    cfg = CanonicalBuildConfig(**filtered)
                canonical_machine = build_canonical_epsilon_machine(self, config=cfg)
                epsilon_machine['canonical_version'] = {
                    'num_states': canonical_machine['num_states'],
                    'start_state': canonical_machine.get('start_state'),
                    'validation': canonical_machine.get('validation', {}),
                }
            except Exception as e:
                print(f"⚠️  Failed to build canonical epsilon machine: {e}")
        
        # Step 6: Save results (include canonical if built)
        self.save_cssr_results(
            epsilon_machine,
            output_dir,
            canonical_machine=canonical_machine,
            canonical_as_primary=canonical_as_primary
        )
        
        print("=" * 70)
        print("🎉 CSSR-enhanced extraction complete!")
        
        return epsilon_machine
    
    def save_cssr_results(self, epsilon_machine: Dict, output_dir: Path,
                          canonical_machine: Dict | None = None,
                          canonical_as_primary: bool = True) -> None:
        """Save CSSR-enhanced extraction results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        
        # Promote canonical machine as primary if requested and available
        if canonical_as_primary and canonical_machine is not None:
            primary_machine = canonical_machine
            secondary_machine = epsilon_machine
            primary_label = 'canonical'
        else:
            primary_machine = epsilon_machine
            secondary_machine = canonical_machine
            primary_label = 'empirical'

        results = {
            'extraction_method': 'cssr_enhanced_trajectory_dynamics',
            'primary_machine_type': primary_label,
            'model_info': {
                'num_causal_states': len(self.causal_states),
                'num_suffixes_processed': len(self.suffix_tree),
                'max_suffix_length': self.max_suffix_length,
                'significance_level': self.significance_level
            },
            'epsilon_machine': primary_machine,  # canonical if available & promoted
            'raw_empirical_machine': epsilon_machine if primary_machine is not epsilon_machine else None,
            'canonical_epsilon_machine': canonical_machine,
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


def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """Calculate KL divergence between two probability distributions."""
    if not p or not q:
        return 0.0
    
    # Get all symbols from both distributions
    all_symbols = set(p.keys()) | set(q.keys())
    
    # Convert to arrays with smoothing for numerical stability
    p_probs = np.array([p.get(symbol, epsilon) for symbol in sorted(all_symbols)])
    q_probs = np.array([q.get(symbol, epsilon) for symbol in sorted(all_symbols)])
    
    # Normalize to ensure they sum to 1
    p_probs = p_probs / np.sum(p_probs)
    q_probs = q_probs / np.sum(q_probs)
    
    # Calculate KL divergence: KL(P||Q) = sum(P * log(P/Q))
    return np.sum(p_probs * np.log(p_probs / q_probs))


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate mutual information between two binary arrays."""
    # Convert to discrete bins for mutual_info_score
    x_discrete = x.astype(int)
    y_discrete = y.astype(int)
    
    return mutual_info_score(x_discrete, y_discrete)


def find_elbow_point(curve: List[Tuple[float, float]], method: str = 'max_curvature') -> float:
    """Find elbow point in a curve using maximum curvature method."""
    if len(curve) < 3:
        return curve[-1][0] if curve else 0.0
    
    thresholds = np.array([point[0] for point in curve])
    values = np.array([point[1] for point in curve])
    
    if method == 'max_curvature':
        # Calculate second derivative (curvature)
        if len(values) < 3:
            return thresholds[np.argmax(values)]
        
        # Smooth the curve and find maximum curvature
        first_deriv = np.gradient(values)
        second_deriv = np.gradient(first_deriv)
        
        # Find point of maximum curvature (absolute value)
        max_curvature_idx = np.argmax(np.abs(second_deriv))
        return thresholds[max_curvature_idx]
    
    elif method == 'plateau':
        # Find where the curve plateaus (derivative approaches zero)
        if len(values) < 2:
            return thresholds[-1]
        
        first_deriv = np.gradient(values)
        # Find where derivative is closest to zero after initial increase
        min_deriv_idx = np.argmin(np.abs(first_deriv[len(first_deriv)//3:]))
        return thresholds[min_deriv_idx + len(first_deriv)//3]
    
    else:
        # Default: return point of maximum MI
        max_mi_idx = np.argmax(values)
        return thresholds[max_mi_idx]


def information_theoretic_threshold(suffix_distances: Dict[Tuple[str, str], float], 
                                  suffix_futures: Dict[str, Dict[str, int]],
                                  num_thresholds: int = 100,
                                  future_similarity_threshold: float = 0.1) -> float:
    """Derive threshold from mutual information between neural distance and future divergence.
    
    Args:
        suffix_distances: Dictionary mapping (suffix1, suffix2) -> neural_distance
        suffix_futures: Dictionary mapping suffix -> {symbol: count}
        num_thresholds: Number of threshold points to test
        future_similarity_threshold: KL divergence threshold for considering futures similar
    
    Returns:
        Optimal neural distance threshold that maximizes mutual information
    """
    if not suffix_distances or not suffix_futures:
        return 5.0  # Default fallback
    
    # Convert future counts to probabilities
    suffix_future_probs = {}
    for suffix, future_counts in suffix_futures.items():
        total_count = sum(future_counts.values())
        if total_count > 0:
            suffix_future_probs[suffix] = {
                symbol: count / total_count 
                for symbol, count in future_counts.items()
            }
        else:
            suffix_future_probs[suffix] = {}
    
    # Compute KL divergence between future distributions and collect neural distances
    future_divergences = []
    neural_distances = []
    
    for (s1, s2), neural_dist in suffix_distances.items():
        if s1 in suffix_future_probs and s2 in suffix_future_probs:
            kl_div = kl_divergence(suffix_future_probs[s1], suffix_future_probs[s2])
            future_divergences.append(kl_div)
            neural_distances.append(neural_dist)
    
    if len(neural_distances) < 10:
        print(f"⚠️  Warning: Only {len(neural_distances)} suffix pairs for threshold estimation")
        return 5.0  # Default fallback
    
    neural_distances = np.array(neural_distances)
    future_divergences = np.array(future_divergences)
    
    # Test range of thresholds
    min_dist = np.min(neural_distances)
    max_dist = np.max(neural_distances)
    thresholds = np.linspace(min_dist, max_dist, num_thresholds)
    
    mi_curve = []
    
    for threshold in thresholds:
        # Binary indicators: neural similarity and future similarity
        neural_similar = (neural_distances < threshold).astype(int)
        future_similar = (future_divergences < future_similarity_threshold).astype(int)
        
        # Calculate mutual information between these binary variables
        try:
            mi = mutual_information(neural_similar, future_similar)
            mi_curve.append((threshold, mi))
        except Exception:
            # Skip problematic thresholds
            mi_curve.append((threshold, 0.0))
    
    if not mi_curve:
        return 5.0  # Default fallback
    
    # Find optimal threshold using elbow point detection
    optimal_threshold = find_elbow_point(mi_curve, method='max_curvature')
    
    # Log analysis results
    max_mi = max(point[1] for point in mi_curve)
    optimal_mi = next((mi for thresh, mi in mi_curve if abs(thresh - optimal_threshold) < 1e-6), max_mi)
    
    print(f"📊 Information-theoretic threshold analysis:")
    print(f"   Neural distance range: [{min_dist:.3f}, {max_dist:.3f}]")
    print(f"   Tested {len(mi_curve)} thresholds")
    print(f"   Maximum MI: {max_mi:.4f}")
    print(f"   Optimal threshold: {optimal_threshold:.3f} (MI: {optimal_mi:.4f})")
    print(f"   {np.sum(neural_distances < optimal_threshold)} of {len(neural_distances)} pairs below threshold")
    
    return optimal_threshold


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
    parser.add_argument('--use-neural', action='store_true', default=True, help='Use neural component')
    parser.add_argument('--no-neural', action='store_true', help='Disable neural component')
    parser.add_argument('--use-classical', action='store_true', default=True, help='Use classical component')
    parser.add_argument('--no-classical', action='store_true', help='Disable classical component')
    parser.add_argument('--neural-threshold', type=float, default=5.0, help='Neural distance threshold')
    parser.add_argument('--use-information-theoretic-threshold', action='store_true', 
                       help='Use information-theoretic method to determine neural threshold')
    parser.add_argument('--threshold-method', type=str, default='silhouette', 
                       choices=['silhouette', 'mutual_info', 'plateau', 'kl_divergence'], 
                       help='Method for computing optimal threshold: silhouette, mutual_info, plateau, or kl_divergence')
    parser.add_argument('--kl-steps', type=int, default=3,
                       help='Number of forward prediction steps for multi-step KL divergence')
    parser.add_argument('--kl-weights', type=float, nargs='+', default=[1.0, 0.8, 0.6],
                       help='Weights for each step in multi-step KL divergence')
    parser.add_argument('--use-emission-based-merging', action='store_true',
                       help='Merge states with similar emission patterns')
    parser.add_argument('--emission-similarity-threshold', type=float, default=0.05,
                       help='Maximum difference in emission probabilities for merging')
    parser.add_argument('--verbose-discriminant', action='store_true',
                       help='Show detailed discriminant messages when classical and neural tests disagree')
    # Canonical epsilon machine options
    parser.add_argument('--no-canonical', action='store_true',
                        help='Disable canonical epsilon machine build (enabled by default)')
    parser.add_argument('--empirical-primary', action='store_true',
                        help='Keep empirical machine as primary in results JSON (default promotes canonical)')
    parser.add_argument('--canonical-smoothing', type=float, default=0.0,
                        help='Additive smoothing for canonical emission probabilities')
    parser.add_argument('--canonical-no-self-loop', action='store_true',
                        help='Disable automatic self-loop on missing symbol (omit transition)')
    parser.add_argument('--canonical-start', type=str, default='largest_count',
                        choices=['largest_count', 'first'],
                        help='Strategy to choose start state for canonical machine')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Determine component flags
    use_neural = args.use_neural and not args.no_neural
    use_classical = args.use_classical and not args.no_classical
    
    extractor = CSSREnhancedExtractor(
        model, device, 
        max_suffix_length=args.max_suffix_length,
        significance_level=args.significance,
        use_neural_test=use_neural,
        use_classical_test=use_classical,
        neural_threshold=args.neural_threshold,
        use_information_theoretic_threshold=args.use_information_theoretic_threshold,
        threshold_method=args.threshold_method,
        use_emission_based_merging=args.use_emission_based_merging,
        emission_similarity_threshold=args.emission_similarity_threshold,
        verbose_discriminant=args.verbose_discriminant
    )
    
    # Set multi-step KL parameters if using KL divergence method
    if args.threshold_method == 'kl_divergence':
        extractor.kl_steps = args.kl_steps
        extractor.kl_weights = args.kl_weights
    
    build_canonical = not args.no_canonical
    canonical_cfg = None
    if build_canonical and CanonicalBuildConfig is not None:
        canonical_cfg = {
            'smoothing': args.canonical_smoothing,
            'missing_symbol_self_loop': (not args.canonical_no_self_loop),
            'choose_start': args.canonical_start,
            'verbose': True
        }
    epsilon_machine = extractor.extract_cssr_enhanced_fsm(
        args.data, args.output, args.max_sequences,
        build_canonical=build_canonical,
        canonical_config=canonical_cfg,
        canonical_as_primary=(not args.empirical_primary)
    )
    
    return epsilon_machine


if __name__ == '__main__':
    main()