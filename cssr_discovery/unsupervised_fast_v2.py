"""
Ultra-efficient unsupervised epsilon machine discovery with adaptive algorithms.

This implements an optimized version with:
- Vectorized operations and efficient caching
- Automatic hyperparameter selection via silhouette analysis
- Backward stability for minimal suffix detection (DEFAULT: ENABLED)
- Emission-stratified sampling with deduplication (from original)
- Parallel processing where beneficial
- Early convergence detection
- Fully unsupervised (no ground truth dependencies)

COMMAND EXAMPLES:

# Seven-state human with backward stability (recommended - all defaults)
python cssr_discovery/unsupervised_fast_v2.py --preset seven_state_human_char_large \
  --n_samples 200 --L 5 --output_json results/seven_state.json

# Even process with custom backward stability tolerance
python cssr_discovery/unsupervised_fast_v2.py --preset even_process \
  --n_samples 150 --L 6 --tolerance_bits 1e-3 --min_suffix_len 1 \
  --output_json results/even.json

# Golden mean with fast mode (quick discovery)
python cssr_discovery/unsupervised_fast_v2.py --preset golden_mean \
  --n_samples 50 --L 3 --fast_mode --output_json results/golden.json

# Disable backward stability (not recommended)
python cssr_discovery/unsupervised_fast_v2.py --preset seven_state_human_char_large \
  --disable_backward_stability --n_samples 150 --output_json results/no_bs.json
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import json
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import from existing modules
from js_metrics import (
    get_next_token_distribution,
    js_divergence,
    js_divergence_conditional_k,
    get_kstep_distribution,
    extract_subsequences
)
from state_mapping import get_gt_state
from calibration import fit_platt_params


def js_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JS divergence in bits (base-2) for backward stability checks."""
    import math
    m = (p + q) / 2.0

    def kl_bits(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 1e-12:
                s += ai * math.log(ai / max(bi, 1e-12), 2)
        return s

    return 0.5 * (kl_bits(p, m) + kl_bits(q, m))


def find_minimal_suffix(history: np.ndarray, model, platt_params: Optional[dict] = None,
                       tolerance_bits: float = 1e-3, min_suffix_len: int = 2) -> Tuple[np.ndarray, int, float]:
    """
    Find the shortest suffix of history that preserves emission distribution.

    Backward stability test: Start with full history, try progressively shorter suffixes
    until emission changes beyond tolerance. This identifies minimal causal states.

    Args:
        history: The full history context
        model: Neural model for emission estimation
        platt_params: Calibration parameters
        tolerance_bits: JS divergence tolerance in bits (default: 1e-3)
        min_suffix_len: Minimum suffix length to consider (default: 2)

    Returns:
        (minimal_suffix, minimal_length, js_to_full): Shortest stable suffix
    """
    if len(history) <= min_suffix_len:
        return history, len(history), 0.0

    # Get emission distribution for full history
    full_probs = get_next_token_distribution(model, history, platt_params)

    # Test progressively shorter suffixes (backward stability)
    for suffix_len in range(len(history) - 1, min_suffix_len - 1, -1):
        suffix = history[-suffix_len:]
        suffix_probs = get_next_token_distribution(model, suffix, platt_params)

        # Check if suffix preserves emission within tolerance
        js_div = js_bits(full_probs, suffix_probs)
        if js_div <= tolerance_bits:
            # This suffix is stable - can drop earlier tokens
            continue
        else:
            # Suffix too short - need one more token
            minimal_suffix = history[-(suffix_len + 1):]
            minimal_len = suffix_len + 1
            minimal_probs = get_next_token_distribution(model, minimal_suffix, platt_params)
            js_to_full = js_bits(full_probs, minimal_probs)
            return minimal_suffix, minimal_len, js_to_full

    # If we get here, even min_suffix_len suffix is stable
    minimal_suffix = history[-min_suffix_len:]
    minimal_probs = get_next_token_distribution(model, minimal_suffix, platt_params)
    js_to_full = js_bits(full_probs, minimal_probs)
    return minimal_suffix, min_suffix_len, js_to_full


@dataclass
class OptimizedClusteringResult:
    """Results from optimized unsupervised clustering."""
    clusters: List[List[np.ndarray]]
    cluster_representatives: List[np.ndarray]
    cluster_emission_sigs: List[str]
    cluster_sizes: List[int]
    total_histories_sampled: int
    emission_cache_hits: int
    emission_cache_misses: int
    kstep_cache_hits: int
    kstep_cache_misses: int
    optimal_threshold: float
    silhouette_score: float
    convergence_iterations: int
    algorithm_metadata: Dict


class UnifiedCache:
    """High-performance unified cache for both emission and k-step distributions."""

    def __init__(self):
        self.emission_cache = {}
        self.kstep_cache = {}
        self.emission_hits = 0
        self.emission_misses = 0
        self.kstep_hits = 0
        self.kstep_misses = 0

    def _get_key(self, hist: np.ndarray, suffix: str = "") -> str:
        """Create cache key from history."""
        return f"{tuple(hist)}{suffix}"

    def get_emission(self, model, hist: np.ndarray,
                     platt_params: Optional[dict] = None) -> np.ndarray:
        """Get emission distribution with caching."""
        key = self._get_key(hist, f"_emit_{id(platt_params)}")
        if key in self.emission_cache:
            self.emission_hits += 1
            return self.emission_cache[key]

        dist = get_next_token_distribution(model, hist, platt_params)
        self.emission_cache[key] = dist
        self.emission_misses += 1
        return dist

    def get_kstep(self, model, hist: np.ndarray, k: int,
                  platt_params: Optional[dict] = None) -> np.ndarray:
        """Get k-step distribution with caching."""
        key = self._get_key(hist, f"_k{k}_{id(platt_params)}")
        if key in self.kstep_cache:
            self.kstep_hits += 1
            return self.kstep_cache[key]

        dist = get_kstep_distribution(model, hist, k, platt_params)
        self.kstep_cache[key] = dist
        self.kstep_misses += 1
        return dist

    def batch_get_emissions(self, model, histories: List[np.ndarray],
                           platt_params: Optional[dict] = None) -> np.ndarray:
        """Batch get emission distributions (2D array)."""
        return np.array([self.get_emission(model, h, platt_params)
                        for h in histories])


def random_history_sampling(data: np.ndarray, L: int, n_samples: int,
                           seed: Optional[int] = None) -> List[np.ndarray]:
    """Sample histories randomly from the data."""
    if seed is not None:
        np.random.seed(seed)
    all_histories = extract_subsequences(data, L)
    if len(all_histories) <= n_samples:
        return all_histories
    indices = np.random.choice(len(all_histories), size=n_samples, replace=False)
    return [all_histories[i] for i in indices]


def emission_stratified_sampling(data: np.ndarray, L: int, model, n_samples: int,
                                platt_params: Optional[dict] = None, n_strata: int = 5,
                                seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Sample histories by stratifying based on emission probabilities.

    This is the proven sampling strategy from unsupervised_fast_original.py that
    ensures good coverage across the emission probability space with deduplication.
    """
    if seed is not None:
        np.random.seed(seed)

    all_histories = extract_subsequences(data, L)
    sample_size = min(len(all_histories), 2000)
    sample_indices = np.random.choice(len(all_histories), size=sample_size, replace=False)

    # Compute emissions for sample
    emissions = []
    sampled_histories = []
    for idx in sample_indices:
        hist = all_histories[idx]
        probs = get_next_token_distribution(model, hist, platt_params)
        emissions.append(float(probs[0]))
        sampled_histories.append(hist)

    # Create strata
    emissions = np.array(emissions)
    quantiles = np.linspace(0, 1, n_strata + 1)
    boundaries = np.quantile(emissions, quantiles)

    # Sample from each stratum with deduplication
    selected_histories = []
    per_stratum = n_samples // n_strata
    extra = n_samples % n_strata

    for i in range(n_strata):
        lower = boundaries[i]
        upper = boundaries[i + 1] if i < n_strata - 1 else 1.0
        in_stratum = [(hist, em) for hist, em in zip(sampled_histories, emissions)
                     if lower <= em <= upper]
        stratum_size = per_stratum + (1 if i < extra else 0)

        # Deduplicate contexts first (convert to tuples for hashing)
        unique_contexts = {}
        for hist, em in in_stratum:
            hist_key = tuple(hist)
            if hist_key not in unique_contexts:
                unique_contexts[hist_key] = (hist, em)

        unique_list = list(unique_contexts.values())

        if len(unique_list) >= stratum_size:
            # Sample from unique contexts only
            stratum_indices = np.random.choice(len(unique_list), size=stratum_size, replace=False)
            selected_histories.extend([unique_list[idx][0] for idx in stratum_indices])
        else:
            # Not enough unique contexts, take all unique and fill with repeats
            selected_histories.extend([hist for hist, _ in unique_list])
            remaining = stratum_size - len(unique_list)
            if remaining > 0:
                repeat_indices = np.random.choice(len(unique_list), size=remaining, replace=True)
                selected_histories.extend([unique_list[idx][0] for idx in repeat_indices])

    return selected_histories


def apply_backward_stability(histories: List[np.ndarray], model, cache: UnifiedCache,
                             platt_params: Optional[dict] = None,
                             tolerance_bits: float = 1e-3,
                             min_suffix_len: int = 2) -> Tuple[List[np.ndarray], Dict]:
    """
    Apply backward stability to find minimal suffixes for all histories.

    This identifies the shortest suffix of each history that preserves the emission
    distribution, which is critical for discovering minimal causal states.

    Args:
        histories: List of full histories
        model: Neural model
        cache: Unified cache for efficiency
        platt_params: Calibration parameters
        tolerance_bits: JS divergence tolerance in bits
        min_suffix_len: Minimum suffix length

    Returns:
        (minimal_histories, stats): Minimal suffix histories and statistics
    """
    print(f"\n=== Backward Stability Analysis ===")
    print(f"Testing {len(histories)} histories for minimal suffixes...")

    minimal_histories = []
    suffix_length_counts = defaultdict(int)
    total_js_deviation = 0.0

    for hist in histories:
        minimal_suffix, minimal_len, js_to_full = find_minimal_suffix(
            hist, model, platt_params, tolerance_bits, min_suffix_len
        )
        minimal_histories.append(minimal_suffix)
        suffix_length_counts[minimal_len] += 1
        total_js_deviation += js_to_full

    # Remove duplicates that emerge from suffix reduction
    unique_minimal = []
    seen = set()
    for hist in minimal_histories:
        key = tuple(hist)
        if key not in seen:
            unique_minimal.append(hist)
            seen.add(key)

    stats = {
        'original_count': len(histories),
        'unique_minimal_count': len(unique_minimal),
        'deduplication_ratio': len(unique_minimal) / len(histories),
        'suffix_length_distribution': dict(suffix_length_counts),
        'avg_js_deviation': total_js_deviation / len(histories)
    }

    print(f"Results: {len(histories)} → {len(unique_minimal)} unique minimal suffixes")
    print(f"Deduplication ratio: {stats['deduplication_ratio']:.3f}")
    print(f"Suffix length distribution: {dict(suffix_length_counts)}")
    print(f"Avg JS deviation: {stats['avg_js_deviation']:.6f} bits")

    return unique_minimal, stats


def compute_emission_distance_matrix(histories: List[np.ndarray], cache: UnifiedCache,
                                    model, platt_params: Optional[dict] = None) -> np.ndarray:
    """
    Compute pairwise JS divergence matrix efficiently using vectorization.
    Returns condensed distance matrix suitable for scipy hierarchical clustering.
    """
    n = len(histories)

    # Batch get all emission distributions
    emissions = cache.batch_get_emissions(model, histories, platt_params)

    # Compute JS divergence using vectorized operations
    def js_divergence_vectorized(P: np.ndarray) -> np.ndarray:
        """Compute pairwise JS divergences for rows of P."""
        n = P.shape[0]
        distances = []

        for i in range(n):
            for j in range(i + 1, n):
                p, q = P[i], P[j]
                m = 0.5 * (p + q)
                # Use scipy's entropy for efficiency
                from scipy.stats import entropy
                js = 0.5 * (entropy(p, m) + entropy(q, m))
                distances.append(js)

        return np.array(distances)

    return js_divergence_vectorized(emissions)


def adaptive_threshold_selection(distance_matrix: np.ndarray,
                                 histories: List[np.ndarray],
                                 min_clusters: int = 2,
                                 max_clusters: int = 20) -> Tuple[float, int]:
    """
    Select optimal threshold using silhouette analysis.
    Returns (optimal_threshold, optimal_n_clusters).
    """
    # Convert condensed to square form for silhouette computation
    D_square = squareform(distance_matrix)

    best_score = -1
    best_n_clusters = min_clusters
    best_threshold = None

    # Try different numbers of clusters
    linkage_matrix = linkage(distance_matrix, method='average')

    for n_clusters in range(min_clusters, min(max_clusters + 1, len(histories))):
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Check if we have at least 2 clusters with at least 1 point each
        if len(np.unique(labels)) < 2:
            continue

        try:
            score = silhouette_score(D_square, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

                # Find threshold that gives this number of clusters
                # Use the maximum distance within clusters as threshold
                cluster_distances = []
                for cluster_id in np.unique(labels):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    if len(cluster_indices) > 1:
                        for i, idx_i in enumerate(cluster_indices):
                            for idx_j in cluster_indices[i+1:]:
                                if idx_i < idx_j:
                                    # Find position in condensed matrix
                                    pos = idx_i * len(histories) - idx_i * (idx_i + 1) // 2 + idx_j - idx_i - 1
                                    cluster_distances.append(distance_matrix[pos])

                best_threshold = max(cluster_distances) if cluster_distances else 0.01
        except:
            continue

    # Fallback if silhouette fails
    if best_threshold is None:
        best_threshold = np.percentile(distance_matrix, 25)

    return best_threshold, best_n_clusters


def efficient_emission_clustering(histories: List[np.ndarray], cache: UnifiedCache,
                                  model, platt_params: Optional[dict] = None,
                                  stage_a_threshold: float = 0.001) -> Tuple[Dict, float]:
    """
    Efficient Stage A: Agglomerative clustering with fixed threshold (from original).

    This uses the proven algorithm from unsupervised_fast_original.py:
    - Start with singleton clusters
    - Repeatedly merge closest clusters by centroid JS divergence
    - Stop when minimum JS exceeds threshold

    Returns:
        (emission_groups, threshold_used)
    """
    print(f"\n=== Stage A: Agglomerative Emission Clustering ===")
    print(f"Threshold: {stage_a_threshold}")

    # Compute emission probabilities for all histories
    emission_data = []
    for hist in histories:
        probs = cache.get_emission(model, hist, platt_params)
        emission_data.append((hist, probs))

    # Start with each history as its own cluster
    emission_clusters = [{'histories': [hist], 'centroid': probs.copy()}
                        for hist, probs in emission_data]
    print(f"Starting with {len(emission_clusters)} singleton clusters")

    # Agglomerative clustering: repeatedly merge closest clusters
    iteration = 0
    while len(emission_clusters) > 1:
        iteration += 1
        min_js = float('inf')
        merge_i, merge_j = -1, -1

        # Find closest pair of clusters based on centroid JS divergence
        for i in range(len(emission_clusters)):
            for j in range(i + 1, len(emission_clusters)):
                centroid_i = emission_clusters[i]['centroid']
                centroid_j = emission_clusters[j]['centroid']
                js = js_divergence(centroid_i, centroid_j)
                if js < min_js:
                    min_js = js
                    merge_i, merge_j = i, j

        # Stop if minimum JS exceeds threshold
        if min_js > stage_a_threshold:
            print(f"Iteration {iteration}: min JS = {min_js:.6f} > {stage_a_threshold}, stopping")
            break

        # Merge the closest clusters
        cluster_i = emission_clusters[merge_i]
        cluster_j = emission_clusters[merge_j]

        # Combine histories
        merged_histories = cluster_i['histories'] + cluster_j['histories']

        # Recompute centroid as mean emission over all members
        all_emissions = cache.batch_get_emissions(model, merged_histories, platt_params)
        mean_emission = np.mean(all_emissions, axis=0)
        # Renormalize to ensure it's a proper probability distribution
        mean_emission = mean_emission / mean_emission.sum()

        # Create merged cluster
        merged_cluster = {
            'histories': merged_histories,
            'centroid': mean_emission
        }

        # Remove old clusters and add merged one
        # Remove in reverse order to avoid index issues
        emission_clusters.pop(max(merge_i, merge_j))
        emission_clusters.pop(min(merge_i, merge_j))
        emission_clusters.append(merged_cluster)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: {len(emission_clusters)} clusters, last merge JS = {min_js:.6f}")

    print(f"Converged to {len(emission_clusters)} emission groups after {iteration} iterations")

    # Convert to dictionary format
    emission_groups = {}
    for i, cluster in enumerate(emission_clusters):
        sig = f"cluster_{i}"
        emission_groups[sig] = cluster['histories']
        centroid = cluster['centroid']
        print(f" {sig}: {len(cluster['histories'])} histories, P(0)={centroid[0]:.4f}")

    return emission_groups, stage_a_threshold


def smart_representative_selection(histories: List[np.ndarray], cache: UnifiedCache,
                                   model, platt_params: Optional[dict],
                                   max_representatives: int = 10) -> List[np.ndarray]:
    """
    K-means++ style selection of representatives for maximum coverage.
    """
    if len(histories) <= max_representatives:
        return histories

    representatives = []
    emissions = cache.batch_get_emissions(model, histories, platt_params)

    # First representative: most central (closest to mean)
    mean_emission = np.mean(emissions, axis=0)
    distances_to_mean = np.array([np.linalg.norm(e - mean_emission) for e in emissions])
    first_idx = np.argmin(distances_to_mean)
    representatives.append(histories[first_idx])
    selected_indices = {first_idx}

    # Remaining representatives: k-means++ style (maximize minimum distance)
    for _ in range(max_representatives - 1):
        if len(selected_indices) >= len(histories):
            break

        # Compute minimum distance to existing representatives
        min_distances = []
        for i, hist in enumerate(histories):
            if i in selected_indices:
                min_distances.append(-1)
                continue

            min_dist = float('inf')
            for rep in representatives:
                dist = js_divergence(emissions[i],
                                    cache.get_emission(model, rep, platt_params))
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)

        # Select point with maximum minimum distance
        next_idx = np.argmax(min_distances)
        if min_distances[next_idx] > 0:
            representatives.append(histories[next_idx])
            selected_indices.add(next_idx)

    return representatives


def efficient_conditional_refinement(group_histories: List[np.ndarray], cache: UnifiedCache,
                                    model, k_refine: int, platt_params: Optional[dict],
                                    threshold: float, max_representatives: int = 10) -> List[List[np.ndarray]]:
    """
    Efficient Stage B: Refine using representatives and conditional JS.
    """
    if len(group_histories) <= 1:
        return [group_histories]

    # Select smart representatives
    representatives = smart_representative_selection(
        group_histories, cache, model, platt_params, max_representatives
    )

    if len(representatives) <= 1:
        return [group_histories]

    # Cluster representatives using conditional JS
    n_reps = len(representatives)
    rep_distances = []

    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            try:
                js_cond = js_divergence_conditional_k(
                    model, representatives[i], representatives[j], k_refine, platt_params
                )
                rep_distances.append(js_cond)
            except:
                rep_distances.append(0.0)

    if not rep_distances:
        return [group_histories]

    rep_distances = np.array(rep_distances)

    # Hierarchical clustering of representatives
    rep_linkage = linkage(rep_distances, method='average')
    rep_labels = fcluster(rep_linkage, threshold, criterion='distance')

    # Assign all histories to nearest representative cluster
    rep_clusters = defaultdict(list)
    for rep, label in zip(representatives, rep_labels):
        rep_clusters[label].append(rep)

    # Assign remaining histories to closest representative
    final_clusters = [[] for _ in rep_clusters]
    rep_cluster_list = list(rep_clusters.items())

    for hist in group_histories:
        best_cluster = 0
        min_dist = float('inf')

        hist_emission = cache.get_emission(model, hist, platt_params)

        for cluster_idx, (label, cluster_reps) in enumerate(rep_cluster_list):
            for rep in cluster_reps:
                rep_emission = cache.get_emission(model, rep, platt_params)
                dist = js_divergence(hist_emission, rep_emission)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_idx

        final_clusters[best_cluster].append(hist)

    return [c for c in final_clusters if c]


def optimized_state_remerging(clusters: List[List[np.ndarray]], cache: UnifiedCache,
                              model, platt_params: Optional[dict], k_refine: int,
                              emission_threshold: float = 1e-4,
                              rollout_threshold: float = 5e-4,
                              parallel: bool = False) -> List[List[np.ndarray]]:
    """
    Efficient Stage C: Remerge functionally identical states.
    Uses representatives and early stopping.
    """
    if len(clusters) <= 1:
        return clusters

    print(f"\n=== Stage C: Efficient State Remerging ===")
    print(f"Checking {len(clusters)} states...")

    representatives = [cluster[0] for cluster in clusters]
    merged = [False] * len(clusters)
    remerged_clusters = []

    for i in range(len(clusters)):
        if merged[i]:
            continue

        current_cluster = clusters[i].copy()
        current_rep = representatives[i]

        # Check potential merges
        for j in range(i + 1, len(clusters)):
            if merged[j]:
                continue

            rep_j = representatives[j]

            # Quick emission check first (cheaper)
            emission_i = cache.get_emission(model, current_rep, platt_params)
            emission_j = cache.get_emission(model, rep_j, platt_params)
            emission_js = js_divergence(emission_i, emission_j)

            if emission_js >= emission_threshold:
                continue

            # More expensive rollout check
            try:
                rollout_js = js_divergence_conditional_k(
                    model, current_rep, rep_j, k_refine, platt_params
                )
            except:
                continue

            if rollout_js < rollout_threshold:
                print(f"  Merging states {i}↔{j}: emission={emission_js:.6f}, rollout={rollout_js:.6f}")
                current_cluster.extend(clusters[j])
                merged[j] = True

        remerged_clusters.append(current_cluster)

    print(f"Merged {len(clusters)} → {len(remerged_clusters)} states")
    return remerged_clusters


def ultrafast_unsupervised_discovery(histories: List[np.ndarray], model,
                                     k_refine: int = 4,
                                     platt_params: Optional[dict] = None,
                                     stage_a_threshold: float = 0.001,
                                     stage_b_threshold: Optional[float] = None,
                                     enable_remerging: bool = True,
                                     backward_stability: bool = True,
                                     tolerance_bits: float = 1e-3,
                                     min_suffix_len: int = 2,
                                     max_representatives: int = 10,
                                     parallel: bool = False) -> OptimizedClusteringResult:
    """
    Ultra-efficient fully unsupervised epsilon machine discovery.

    Key optimizations:
    - Backward stability (DEFAULT: ON) - finds minimal suffixes
    - Unified caching system with emission and k-step caching
    - Vectorized distance computations
    - Automatic threshold selection via silhouette analysis
    - Smart representative selection (k-means++ style)
    - Early convergence detection
    - Optional parallelization

    Args:
        histories: List of history arrays to cluster
        model: Neural network model for probability estimation
        k_refine: Steps for k-step rollout refinement (default: 4)
        platt_params: Platt calibration parameters (optional)
        stage_a_threshold: Stage A agglomerative clustering threshold (default: 0.001)
        stage_b_threshold: Stage B conditional JS threshold (default: 0.8 * stage_a_threshold)
        enable_remerging: Enable Stage C state remerging (default: True)
        backward_stability: Enable backward stability (default: True, RECOMMENDED)
        tolerance_bits: JS tolerance for backward stability in bits (default: 1e-3)
        min_suffix_len: Minimum suffix length for backward stability (default: 2)
        max_representatives: Max representatives per group (default: 10)
        parallel: Enable parallel processing (default: False)

    Returns:
        OptimizedClusteringResult with discovered states and performance metrics
    """
    if not histories:
        return OptimizedClusteringResult(
            [], [], [], [], 0, 0, 0, 0, 0, 0.0, 0.0, 0, {}
        )

    print(f"\n{'='*70}")
    print("ULTRA-FAST UNSUPERVISED EPSILON MACHINE DISCOVERY")
    print(f"{'='*70}")
    print(f"Histories: {len(histories)}, Parallel: {parallel}")
    print(f"Backward stability: {backward_stability}")
    print(f"Stage A threshold: {stage_a_threshold}")

    cache = UnifiedCache()

    # Stage 0: Backward Stability (optional, default enabled)
    backward_stability_stats = None
    if backward_stability:
        histories, backward_stability_stats = apply_backward_stability(
            histories, model, cache, platt_params, tolerance_bits, min_suffix_len
        )

    # Stage A: Agglomerative emission clustering
    emission_groups, _ = efficient_emission_clustering(
        histories, cache, model, platt_params, stage_a_threshold
    )
    sil_score = 0.0  # Not computed in agglomerative version

    # Stage B: Conditional refinement with smart representatives
    print(f"\n=== Stage B: Conditional Refinement (k={k_refine}) ===")
    final_clusters = []

    if stage_b_threshold is None:
        # Default Stage B threshold based on Stage A
        stage_b_threshold = stage_a_threshold * 0.8

    for sig, group_histories in emission_groups.items():
        if len(group_histories) == 1:
            final_clusters.append(group_histories)
            print(f" {sig}: singleton")
            continue

        print(f" Refining {sig} ({len(group_histories)} histories)...")
        refined = efficient_conditional_refinement(
            group_histories, cache, model, k_refine, platt_params,
            stage_b_threshold, max_representatives
        )
        final_clusters.extend(refined)
        print(f"  → {len(refined)} sub-clusters")

    # Stage C: State remerging (optional)
    if enable_remerging and len(final_clusters) > 1:
        final_clusters = optimized_state_remerging(
            final_clusters, cache, model, platt_params, k_refine,
            emission_threshold=1e-4, rollout_threshold=5e-4, parallel=parallel
        )

    # Prepare result
    cluster_representatives = [cluster[0] for cluster in final_clusters]
    cluster_emission_sigs = []
    cluster_sizes = []

    for cluster in final_clusters:
        repr_hist = cluster[0]
        probs = cache.get_emission(model, repr_hist, platt_params)
        sig = f"{probs[0]:.3f}_{probs[1]:.3f}"
        cluster_emission_sigs.append(sig)
        cluster_sizes.append(len(cluster))

    metadata = {
        'stage_a_threshold': stage_a_threshold,
        'stage_b_threshold': stage_b_threshold,
        'parallel': parallel,
        'max_representatives': max_representatives,
        'k_refine': k_refine,
        'backward_stability': backward_stability,
        'backward_stability_stats': backward_stability_stats
    }

    result = OptimizedClusteringResult(
        clusters=final_clusters,
        cluster_representatives=cluster_representatives,
        cluster_emission_sigs=cluster_emission_sigs,
        cluster_sizes=cluster_sizes,
        total_histories_sampled=len(histories),
        emission_cache_hits=cache.emission_hits,
        emission_cache_misses=cache.emission_misses,
        kstep_cache_hits=cache.kstep_hits,
        kstep_cache_misses=cache.kstep_misses,
        optimal_threshold=stage_a_threshold,
        silhouette_score=sil_score,
        convergence_iterations=1,
        algorithm_metadata=metadata
    )

    print(f"\n{'='*70}")
    print(f"DISCOVERED {len(final_clusters)} EPSILON MACHINE STATES")
    print(f"{'='*70}")
    print(f"Cluster sizes: {cluster_sizes}")
    print(f"Silhouette score: {sil_score:.4f}")
    print(f"Emission cache: {cache.emission_hits} hits / {cache.emission_misses} misses")
    print(f"K-step cache: {cache.kstep_hits} hits / {cache.kstep_misses} misses")

    return result


def compute_epsilon_machine_loss(clustering_result: OptimizedClusteringResult, model, data: np.ndarray,
                                L: int, platt_params: Optional[dict] = None) -> Dict:
    """
    Compute the negative log-likelihood (loss) that the discovered epsilon machine would achieve.
    For each discovered state (cluster), compute the emission probabilities from the model.
    Then evaluate the likelihood of the data under this epsilon machine.
    """
    clusters = clustering_result.clusters
    if not clusters:
        return {'total_loss': float('inf'), 'avg_loss_per_symbol': float('inf'), 'num_predictions': 0}

    print(f"\n{'='*70}")
    print("EPSILON MACHINE LOSS")
    print(f"{'='*70}")

    # Step 1: Learn emission probabilities for each discovered state
    state_emissions = {}
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        # Use representative history to compute emissions for this state
        repr_hist = cluster[0]
        probs = get_next_token_distribution(model, repr_hist, platt_params)
        state_emissions[i] = probs
        print(f"State {i}: P(0)={probs[0]:.4f}, P(1)={probs[1]:.4f} (from {len(cluster)} histories)")

    # Step 2: Define state mapping function based on discovered clusters
    def get_discovered_state(history: np.ndarray) -> Optional[int]:
        """Map a history to the discovered state based on suffix matching with representatives."""
        if len(history) < 2:
            return None

        # Try to match the history against each cluster's representative suffix
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            repr_hist = cluster[0]
            repr_len = len(repr_hist)

            if len(history) >= repr_len:
                history_suffix = history[-repr_len:]
                if np.array_equal(history_suffix, repr_hist):
                    return i

        # Fuzzy matching with shortest representatives first
        best_state = None
        min_distance = float('inf')

        cluster_indices = list(range(len(clusters)))
        cluster_indices.sort(key=lambda idx: len(clusters[idx][0]) if clusters[idx] else float('inf'))

        for i in cluster_indices:
            cluster = clusters[i]
            if not cluster:
                continue
            repr_hist = cluster[0]
            repr_len = len(repr_hist)

            if len(history) >= repr_len:
                history_suffix = history[-repr_len:]
                distance = np.sum(history_suffix != repr_hist)
                if distance < min_distance:
                    min_distance = distance
                    best_state = i

        return best_state if min_distance <= 1 else None

    # Step 3: Compute negative log-likelihood on the dataset
    total_loss = 0.0
    num_predictions = 0
    print(f"Evaluating epsilon machine on {len(data)} symbols...")

    for i in range(L, len(data)):
        context = data[i-L:i]
        next_symbol = int(data[i])

        state = get_discovered_state(context)
        if state is None or state not in state_emissions:
            continue

        emission_probs = state_emissions[state]
        prob_next = float(emission_probs[next_symbol])

        if prob_next > 1e-12:
            total_loss -= np.log(prob_next)
            num_predictions += 1

    avg_loss_per_symbol = total_loss / num_predictions if num_predictions > 0 else float('inf')
    avg_loss_per_symbol_bits = avg_loss_per_symbol / np.log(2)

    print(f"Total predictions: {num_predictions}")
    print(f"Average loss: {avg_loss_per_symbol:.4f} nats ({avg_loss_per_symbol_bits:.4f} bits)")

    return {
        'total_loss': total_loss,
        'avg_loss_per_symbol': avg_loss_per_symbol,
        'avg_loss_per_symbol_bits': avg_loss_per_symbol_bits,
        'num_predictions': num_predictions,
        'num_states': len([c for c in clusters if c])
    }


def evaluate_clustering_against_gt(clustering_result: OptimizedClusteringResult,
                                   preset: str) -> Dict:
    """Evaluate discovered clusters against ground truth (if available)."""
    clusters = clustering_result.clusters
    cluster_analysis = []

    for i, cluster in enumerate(clusters):
        gt_states = []
        hist_strs = []

        for hist in cluster:
            try:
                gt_state = get_gt_state(hist, preset)
                if gt_state:
                    gt_states.append(gt_state)
            except:
                pass
            hist_strs.append(''.join(map(str, hist)))

        gt_counts = {}
        for gt in gt_states:
            gt_counts[gt] = gt_counts.get(gt, 0) + 1

        purity = max(gt_counts.values()) / len(gt_states) if gt_states else 0.0
        dominant_gt = max(gt_counts.keys(), key=gt_counts.get) if gt_counts else None

        cluster_info = {
            'cluster_id': i,
            'size': len(cluster),
            'gt_state_counts': gt_counts,
            'purity': purity,
            'dominant_gt_state': dominant_gt,
            'emission_signature': clustering_result.cluster_emission_sigs[i],
            'sample_histories': hist_strs[:5]
        }
        cluster_analysis.append(cluster_info)

    total_histories = sum(len(cluster) for cluster in clusters)
    weighted_purity = sum(info['purity'] * info['size'] for info in cluster_analysis) / total_histories \
        if total_histories > 0 else 0.0

    return {
        'num_clusters_discovered': len(clusters),
        'total_histories': total_histories,
        'weighted_purity': weighted_purity,
        'cluster_analysis': cluster_analysis,
        'silhouette_score': clustering_result.silhouette_score,
        'optimal_threshold': clustering_result.optimal_threshold
    }


def main():
    """Main function for ultra-fast unsupervised epsilon machine discovery."""
    parser = argparse.ArgumentParser(
        description='Ultra-efficient unsupervised epsilon machine discovery'
    )
    parser.add_argument('--preset', type=str,
                       choices=['seven_state_human', 'seven_state_human_100k',
                               'seven_state_human_large', 'seven_state_human_char_large',
                               'sevestateold', 'even_process', 'golden_mean'],
                       default='seven_state_human_large')
    parser.add_argument('--model_ckpt', type=str, help='Override model checkpoint path')
    parser.add_argument('--data', type=str, help='Override dataset path')
    parser.add_argument('--L', type=int, default=5, help='History length')
    parser.add_argument('--k_refine', type=int, default=4, help='k for conditional JS')
    parser.add_argument('--n_samples', type=int, default=150,
                       help='Number of histories to sample')
    parser.add_argument('--sampling_strategy', type=str,
                       choices=['emission_stratified', 'random', 'exhaustive'],
                       default='emission_stratified')

    # Threshold parameters
    parser.add_argument('--stage_a_threshold', type=float, default=0.001,
                       help='Stage A agglomerative clustering threshold (default: 0.001)')
    parser.add_argument('--stage_b_threshold', type=float,
                       help='Stage B conditional JS threshold (default: 0.8 * stage_a_threshold)')

    # Performance options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Fast mode: reduced representatives, quicker convergence')
    parser.add_argument('--max_representatives', type=int, default=10,
                       help='Max representatives per emission group')

    # State remerging
    parser.add_argument('--enable_remerging', action='store_true', default=True,
                       help='Enable state remerging (default: True)')
    parser.add_argument('--disable_remerging', action='store_true',
                       help='Disable state remerging')

    # Backward stability (default: enabled)
    parser.add_argument('--backward_stability', action='store_true', default=True,
                       help='Enable backward stability for minimal suffix detection (default: True)')
    parser.add_argument('--disable_backward_stability', action='store_true',
                       help='Disable backward stability')
    parser.add_argument('--tolerance_bits', type=float, default=1e-3,
                       help='JS divergence tolerance in bits for backward stability (default: 1e-3)')
    parser.add_argument('--min_suffix_len', type=int, default=2,
                       help='Minimum suffix length for backward stability (default: 2)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip_calibration', action='store_true',
                       help='Skip Platt calibration (use raw neural probabilities)')
    parser.add_argument('--output_json', type=str, help='Save results to JSON')

    args = parser.parse_args()

    # Handle flags
    if args.disable_remerging:
        args.enable_remerging = False
    if args.disable_backward_stability:
        args.backward_stability = False
    if args.fast_mode:
        args.max_representatives = 5
        args.n_samples = min(args.n_samples, 100)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char_50k' / 'ckpt.pt'
    default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'

    # Preset-specific paths
    preset_paths = {
        'seven_state_human_100k': (
            repo_root / 'nanoGPT' / 'out-seven-state-human-char_100k' / 'ckpt.pt',
            default_data
        ),
        'seven_state_human_large': (
            repo_root / 'nanoGPT' / 'out-seven-state-char_large' / 'ckpt.pt',
            default_data
        ),
        'seven_state_human_char_large': (
            repo_root / 'nanoGPT' / 'out-seven-state-human-char-large' / 'ckpt.pt',
            default_data
        ),
        'sevestateold': (
            repo_root / 'nanoGPT' / 'out-sevestateold-char' / 'ckpt.pt',
            repo_root / 'experiments' / 'datasets' / 'sevestateold' / 'sevestateold.dat'
        ),
        'even_process': (
            repo_root / 'nanoGPT' / 'out-even-process-char' / 'ckpt.pt',
            repo_root / 'experiments' / 'datasets' / 'even_process' / 'even_process.dat'
        ),
        'golden_mean': (
            repo_root / 'nanoGPT' / 'out-golden-mean-char' / 'ckpt.pt',
            repo_root / 'experiments' / 'datasets' / 'golden_mean' / 'golden_mean.dat'
        ),
    }

    if args.preset in preset_paths:
        default_model, default_data = preset_paths[args.preset]

    model_ckpt = Path(args.model_ckpt) if args.model_ckpt else default_model
    data_path = Path(args.data) if args.data else default_data

    # Add repo root to path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from transcssr_baseline.transcssr_neural_runner import _load_nano_gpt_model, load_binary_string

    print(f"{'='*70}")
    print("ULTRA-FAST UNSUPERVISED EPSILON MACHINE DISCOVERY")
    print(f"{'='*70}")
    print(f"Model: {model_ckpt}")
    print(f"Data: {data_path}")
    print(f"Sampling: {args.sampling_strategy} (n={args.n_samples}, L={args.L})")
    print(f"Stage A threshold: {args.stage_a_threshold}, Parallel: {args.parallel}")
    print(f"Fast mode: {args.fast_mode}")

    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, block_size = _load_nano_gpt_model(model_ckpt, device)
    s = load_binary_string(data_path)
    data = np.array([int(c) for c in s], dtype=np.int64)
    print(f"Data length: {len(data)}, Device: {device}")

    # Fit Platt calibration
    if args.skip_calibration:
        print("\nSkipping Platt calibration (using raw probabilities)")
        platt = None
    else:
        print("\nFitting Platt calibration...")
        platt = fit_platt_params(data, model, L_max=args.L, min_count=5)
        if platt:
            print(f"Platt params: a={platt['a']:.4f}, b={platt['b']:.4f}")

    # Sample histories
    print(f"\nSampling histories ({args.sampling_strategy})...")
    if args.sampling_strategy == 'emission_stratified':
        sampled_histories = emission_stratified_sampling(
            data, args.L, model, args.n_samples, platt, n_strata=5, seed=args.seed
        )
    elif args.sampling_strategy == 'random':
        sampled_histories = random_history_sampling(data, args.L, args.n_samples, args.seed)
    elif args.sampling_strategy == 'exhaustive':
        sampled_histories = []
        for i in range(2**args.L):
            binary_str = format(i, f'0{args.L}b')
            history = np.array([int(c) for c in binary_str], dtype=np.int64)
            sampled_histories.append(history)

    print(f"Sampled {len(sampled_histories)} histories")

    # Run ultra-fast discovery
    clustering_result = ultrafast_unsupervised_discovery(
        sampled_histories, model, args.k_refine, platt,
        stage_a_threshold=args.stage_a_threshold,
        stage_b_threshold=args.stage_b_threshold,
        enable_remerging=args.enable_remerging,
        backward_stability=args.backward_stability,
        tolerance_bits=args.tolerance_bits,
        min_suffix_len=args.min_suffix_len,
        max_representatives=args.max_representatives,
        parallel=args.parallel
    )

    # Compute epsilon machine loss
    loss_results = compute_epsilon_machine_loss(clustering_result, model, data, args.L, platt)

    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")

    evaluation = evaluate_clustering_against_gt(clustering_result, args.preset)
    print(f"Discovered {evaluation['num_clusters_discovered']} states")
    if evaluation['weighted_purity'] > 0:
        print(f"Weighted purity: {evaluation['weighted_purity']:.3f}")
    print(f"Silhouette score: {evaluation['silhouette_score']:.4f}")

    for cluster_info in evaluation['cluster_analysis']:
        cid = cluster_info['cluster_id']
        size = cluster_info['size']
        sig = cluster_info['emission_signature']
        dominant = cluster_info['dominant_gt_state']
        print(f"\nCluster {cid}: {size} histories, emission={sig}")
        if dominant:
            print(f"  Dominant GT: {dominant}, purity={cluster_info['purity']:.3f}")
        print(f"  Sample: {cluster_info['sample_histories'][:3]}")

    # Save results
    if args.output_json:
        results = {
            'parameters': vars(args),
            'platt_params': platt,
            'clustering_result': {
                'num_clusters': len(clustering_result.clusters),
                'cluster_sizes': clustering_result.cluster_sizes,
                'emission_signatures': clustering_result.cluster_emission_sigs,
                'silhouette_score': clustering_result.silhouette_score,
                'optimal_threshold': clustering_result.optimal_threshold,
                'emission_cache_performance': {
                    'hits': clustering_result.emission_cache_hits,
                    'misses': clustering_result.emission_cache_misses,
                    'hit_rate': clustering_result.emission_cache_hits /
                               max(1, clustering_result.emission_cache_hits + clustering_result.emission_cache_misses)
                },
                'kstep_cache_performance': {
                    'hits': clustering_result.kstep_cache_hits,
                    'misses': clustering_result.kstep_cache_misses,
                    'hit_rate': clustering_result.kstep_cache_hits /
                               max(1, clustering_result.kstep_cache_hits + clustering_result.kstep_cache_misses)
                },
                'algorithm_metadata': clustering_result.algorithm_metadata
            },
            'epsilon_machine_loss': loss_results,
            'evaluation': evaluation
        }

        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()
