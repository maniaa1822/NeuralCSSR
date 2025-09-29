"""
Fast unsupervised two-stage JS divergence analysis for epsilon machine recovery.
This implements an optimized version that uses efficient clustering algorithms
and caching to avoid the O(n²) computational bottleneck.

COMMAND EXAMPLES:

# Seven-state human with backward stability (finds minimal suffixes)
python cssr_discovery/unsupervised_fast_original.py --preset seven_state_human_char_large \
  --backward_stability --tolerance_bits 1e-3 --min_suffix_len 2 \
  --stage_a_threshold 0.001 --n_samples 100 --L 5

# Even process with backward stability (should find length-1 suffixes)
python cssr_discovery/unsupervised_fast_original.py --preset even_process \
  --backward_stability --tolerance_bits 1e-3 --min_suffix_len 1 \
  --stage_a_threshold 0.001 --n_samples 100 --L 5

# Standard clustering without backward stability
python cssr_discovery/unsupervised_fast_original.py --preset seven_state_human_char_large \
  --stage_a_threshold 0.001 --stage_b_threshold 0.001 \
  --n_samples 100 --L 5 --k_refine 4
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

# Import from existing js_analysis modules
from js_metrics import (
    get_next_token_distribution,
    js_divergence,
    js_divergence_conditional_k,
    get_kstep_distribution,
    extract_subsequences
)
from state_mapping import get_gt_state
from calibration import fit_platt_params


@dataclass
class FastClusteringResult:
    """Results from fast unsupervised clustering."""
    clusters: List[List[np.ndarray]]
    cluster_representatives: List[np.ndarray]
    cluster_emission_sigs: List[str]
    total_histories_sampled: int
    cache_hits: int
    cache_misses: int


class KStepCache:
    """Cache for k-step distributions to avoid recomputation."""

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_key(self, hist: np.ndarray, k: int) -> str:
        """Create cache key from history and k."""
        return f"{tuple(hist)}_{k}"

    def get(self, model, hist: np.ndarray, k: int, platt_params: Optional[dict] = None) -> np.ndarray:
        """Get k-step distribution with caching."""
        key = self.get_key(hist, k)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        # Compute and cache
        dist = get_kstep_distribution(model, hist, k, platt_params)
        self.cache[key] = dist
        self.misses += 1
        return dist


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
    """Sample histories by stratifying based on emission probabilities."""
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


def js_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JS divergence in bits."""
    import math
    m = (p + q) / 2.0

    def kl_bits(a, b):
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 1e-12:  # avoid log(0)
                s += ai * math.log(ai / bi, 2)
        return s

    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def find_minimal_suffix_per_history(history: np.ndarray, model, platt_params: Optional[dict] = None,
                                   tolerance_bits: float = 1e-3, Lmin: int = 2) -> tuple:
    """
    Find the shortest suffix of history that preserves emission distribution.

    Backward stability test: Start with full history, try progressively shorter suffixes
    until emission changes beyond tolerance.

    Args:
        history: The full history context
        model: Neural model for emission estimation
        platt_params: Calibration parameters
        tolerance_bits: JS divergence tolerance in bits for suffix stability
        Lmin: Minimum suffix length to consider

    Returns:
        (minimal_suffix, minimal_length, js_to_full): Shortest stable suffix
    """
    if len(history) <= Lmin:
        return history, len(history), 0.0

    # Get emission distribution for full history
    full_probs = get_next_token_distribution(model, history, platt_params)

    # Test progressively shorter suffixes (backward stability)
    for suffix_len in range(len(history) - 1, Lmin - 1, -1):
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

    # If we get here, even Lmin suffix is stable
    minimal_suffix = history[-Lmin:]
    minimal_probs = get_next_token_distribution(model, minimal_suffix, platt_params)
    js_to_full = js_bits(full_probs, minimal_probs)
    return minimal_suffix, Lmin, js_to_full


def backward_stability_refinement_per_bucket(bucket_histories: List[np.ndarray],
                                            model, platt_params: Optional[dict] = None,
                                            tolerance_bits: float = 1e-3,
                                            Lmin: int = 2) -> List[Dict]:
    """
    Apply backward stability test to each history in a Stage A bucket.

    For each history h, find the shortest suffix h[-ℓ*:] such that shortening further
    changes P(next|⋅) beyond tolerance. Keep the minimal suffix and drop earlier tokens.

    Args:
        bucket_histories: List of histories in this emission bucket
        model: Neural model for emission estimation
        platt_params: Calibration parameters
        tolerance_bits: JS divergence tolerance in bits
        Lmin: Minimum suffix length to consider

    Returns:
        List of refined contexts with minimal suffixes
    """
    if not bucket_histories:
        return []

    refined_contexts = []

    for hist in bucket_histories:
        minimal_suffix, minimal_len, js_to_full = find_minimal_suffix_per_history(
            hist, model, platt_params, tolerance_bits, Lmin
        )

        hist_key = ''.join(map(str, hist))
        minimal_key = ''.join(map(str, minimal_suffix))

        # Get emission probabilities for the minimal suffix
        minimal_probs = get_next_token_distribution(model, minimal_suffix, platt_params)

        refined_contexts.append({
            "original": hist,
            "original_key": hist_key,
            "minimal_suffix": minimal_suffix,
            "minimal_key": minimal_key,
            "original_length": len(hist),
            "minimal_length": minimal_len,
            "tokens_dropped": len(hist) - minimal_len,
            "p_next": minimal_probs,
            "js_to_full": js_to_full,
            "is_minimal": minimal_len < len(hist)
        })

    # Group by minimal suffix (contexts that reduce to same minimal suffix)
    from collections import defaultdict
    grouped = defaultdict(list)
    for ctx in refined_contexts:
        grouped[ctx["minimal_key"]].append(ctx)

    # Create final clusters grouped by minimal suffix
    final_clusters = []
    for minimal_key, contexts in grouped.items():
        if not contexts:
            continue

        rep_context = contexts[0]  # Use first as representative

        final_clusters.append({
            "rep": rep_context["minimal_key"],
            "rep_array": rep_context["minimal_suffix"],
            "p_next": rep_context["p_next"],
            "size": len(contexts),
            "contexts": contexts,
            "original_lengths": [ctx["original_length"] for ctx in contexts],
            "minimal_length": rep_context["minimal_length"],
            "avg_tokens_dropped": sum(ctx["tokens_dropped"] for ctx in contexts) / len(contexts),
            "max_js_to_full": max(ctx["js_to_full"] for ctx in contexts)
        })

    # Sort by minimal suffix for consistent output
    final_clusters.sort(key=lambda x: (x["minimal_length"], x["rep"]))

    return final_clusters


def compute_emission_signature(probs: np.ndarray, precision: int = 2) -> str:
    """Create signature string from emission probabilities."""
    p0, p1 = float(probs[0]), float(probs[1])
    p0_rounded = round(p0, precision)
    p1_rounded = round(p1, precision)
    return f"{p0_rounded:.{precision}f}_{p1_rounded:.{precision}f}"


def fast_conditional_js_with_cache(cache: KStepCache, model, h1: np.ndarray, h2: np.ndarray,
                                  k: int, platt_params: Optional[dict] = None) -> float:
    """Compute conditional JS using cached k-step distributions."""
    if k < 2:
        raise ValueError("Conditional JS requires k >= 2")

    # Get cached k-step distributions
    p = cache.get(model, h1, k, platt_params)
    q = cache.get(model, h2, k, platt_params)

    # Split by first bit
    m = 1 << (k - 1)
    p0, p1 = p[:m], p[m:]
    q0, q1 = q[:m], q[m:]

    # Marginal probabilities
    w0p, w1p = float(p0.sum()), float(p1.sum())
    w0q, w1q = float(q0.sum()), float(q1.sum())

    # Normalize conditional distributions
    p0 = p0 / max(w0p, 1e-12)
    p1 = p1 / max(w1p, 1e-12)
    q0 = q0 / max(w0q, 1e-12)
    q1 = q1 / max(w1q, 1e-12)

    # Average weights
    w0_bar = 0.5 * (w0p + w0q)
    w1_bar = 1.0 - w0_bar

    # Conditional JS
    from scipy.stats import entropy
    js_cond = (w0_bar * (0.5 * (entropy(p0, 0.5 * (p0 + q0)) + entropy(q0, 0.5 * (p0 + q0)))) +
               w1_bar * (0.5 * (entropy(p1, 0.5 * (p1 + q1)) + entropy(q1, 0.5 * (p1 + q1)))))
    return float(js_cond)


def stage_c_remerge_states(clusters: List[List[np.ndarray]], representatives: List[np.ndarray],
                          emission_sigs: List[str], model, platt_params: Optional[dict],
                          k_refine: int, emission_threshold: float = 1e-4,
                          rollout_threshold: float = 5e-4) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[str]]:
    """
    Stage C: Remerge states that are functionally identical.

    This solves the infinite memory problem by detecting when multiple discovered states
    have identical behavior in both emissions and multi-step rollouts.

    Args:
        clusters: List of state clusters (each is a list of histories)
        representatives: Representative history for each cluster
        emission_sigs: Emission signature for each cluster
        model: Neural model for computing rollouts
        platt_params: Platt calibration parameters
        k_refine: k-step rollout horizon
        emission_threshold: JS threshold for emission similarity
        rollout_threshold: JS threshold for rollout similarity

    Returns:
        Tuple of (remerged_clusters, remerged_representatives, remerged_emission_sigs)
    """
    if not clusters:
        return clusters, representatives, emission_sigs

    print(f"Stage C: Checking {len(clusters)} states for remerging...")

    # Track which clusters have been merged
    merged_indices = set()
    remerged_clusters = []
    remerged_representatives = []
    remerged_emission_sigs = []

    for i, cluster_i in enumerate(clusters):
        if i in merged_indices:
            continue

        # Start a new merged group with cluster i
        merged_cluster = cluster_i.copy()
        merged_repr = representatives[i]
        merged_sig = emission_sigs[i]

        # Check all remaining clusters for merge candidates
        for j in range(i + 1, len(clusters)):
            if j in merged_indices:
                continue

            cluster_j = clusters[j]
            repr_j = representatives[j]

            # Compute emission JS divergence
            emission_i = get_next_token_distribution(model, merged_repr, platt_params)
            emission_j = get_next_token_distribution(model, repr_j, platt_params)
            emission_js = js_divergence(emission_i, emission_j)

            # Compute multi-step rollout JS divergence
            rollout_js = js_divergence_conditional_k(model, merged_repr, repr_j, k_refine, platt_params)

            print(f"  States {i}↔{j}: emission_js={emission_js:.6f}, rollout_js={rollout_js:.6f}")

            # If both similarities are below threshold, merge
            if emission_js < emission_threshold and rollout_js < rollout_threshold:
                print(f"    → Merging states {i} and {j}")
                merged_cluster.extend(cluster_j)
                merged_indices.add(j)

                # Update representative (use the one with more histories)
                if len(cluster_j) > len(cluster_i):
                    merged_repr = repr_j
                    merged_sig = emission_sigs[j]

        remerged_clusters.append(merged_cluster)
        remerged_representatives.append(merged_repr)
        remerged_emission_sigs.append(merged_sig)

    print(f"Stage C: Merged {len(clusters)} → {len(remerged_clusters)} states")

    return remerged_clusters, remerged_representatives, remerged_emission_sigs


def efficient_two_stage_clustering(histories: List[np.ndarray], model, k_refine: int = 3,
                                  platt_params: Optional[dict] = None, stage_a_threshold: float = 0.025,
                                  stage_b_threshold: float = 0.02, emission_precision: int = 2,
                                  max_representatives: int = 5, backward_stability: bool = False,
                                  tolerance_bits: float = 1e-3, min_suffix_len: int = 2,
                                  enable_remerging: bool = True, remerge_emission_threshold: float = 1e-4,
                                  remerge_rollout_threshold: float = 5e-4) -> FastClusteringResult:
    """
    Efficient two-stage clustering using representatives and caching.

    Stage A: Distance-based grouping by JS divergence on emission vectors
    Stage B: Refine using conditional JS between representatives only (O(k))
    """
    if not histories:
        return FastClusteringResult([], [], [], 0, 0, 0)

    print(f"\nEfficient two-stage clustering: {len(histories)} histories")
    print(f"Stage A: distance-based emission clustering, threshold={stage_a_threshold}")
    print(f"Stage B: conditional JS with k={k_refine}, max_representatives={max_representatives}")

    cache = KStepCache()

    # Stage A: Distance-based clustering by JS divergence on emission vectors
    print("\n=== Stage A: Distance-based emission clustering ===")

    # Compute emission probabilities for all histories
    emission_data = []
    for hist in histories:
        probs = get_next_token_distribution(model, hist, platt_params)
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
            print(f" Iteration {iteration}: min JS = {min_js:.6f} > {stage_a_threshold}, stopping")
            break

        # Merge the closest clusters
        cluster_i = emission_clusters[merge_i]
        cluster_j = emission_clusters[merge_j]

        # Combine histories
        merged_histories = cluster_i['histories'] + cluster_j['histories']

        # Recompute centroid as mean emission over all members
        all_emissions = []
        for hist in merged_histories:
            probs = get_next_token_distribution(model, hist, platt_params)
            all_emissions.append(probs)

        # Mean emission centroid
        mean_emission = np.mean(all_emissions, axis=0)
        # Renormalize to ensure it's a proper probability distribution
        mean_emission = mean_emission / mean_emission.sum()

        # Create merged cluster
        merged_cluster = {
            'histories': merged_histories,
            'centroid': mean_emission
        }

        print(f" Iteration {iteration}: merging clusters {merge_i},{merge_j} "
              f"(sizes {len(cluster_i['histories'])},{len(cluster_j['histories'])}) "
              f"JS = {min_js:.6f}")

        # Replace cluster_i with merged, remove cluster_j
        emission_clusters[merge_i] = merged_cluster
        del emission_clusters[merge_j]

    print(f"Stage A: Converged to {len(emission_clusters)} emission groups")

    # Convert to the format expected by Stage B
    emission_groups = {}
    for i, cluster in enumerate(emission_clusters):
        centroid = cluster['centroid']
        sig = f"cluster_{i}_P0={centroid[0]:.3f}_P1={centroid[1]:.3f}"
        emission_groups[sig] = cluster['histories']
        history_strings = [''.join(map(str, hist)) for hist in cluster['histories']]
        print(f" {sig}: {len(cluster['histories'])} histories")
        print(f"   {history_strings}")

    print(f"Stage A: Found {len(emission_groups)} emission groups")

    # Optional: Backward stability refinement per bucket
    if backward_stability:
        print(f"\n=== Backward Stability Refinement (tolerance={tolerance_bits} bits) ===")

        refined_emission_groups = {}
        total_contexts_before = 0
        total_contexts_after = 0
        total_tokens_dropped = 0

        for sig, group_histories in emission_groups.items():
            total_contexts_before += len(group_histories)
            print(f"\nApplying backward stability to {sig} ({len(group_histories)} contexts)...")

            refined_clusters = backward_stability_refinement_per_bucket(
                group_histories, model, platt_params, tolerance_bits, min_suffix_len
            )

            # Report results for this bucket
            for i, cluster in enumerate(refined_clusters):
                rep = cluster["rep"]
                size = cluster["size"]
                min_len = cluster["minimal_length"]
                avg_dropped = cluster["avg_tokens_dropped"]
                max_js = cluster["max_js_to_full"]

                print(f"  Minimal context {i+1}: '{rep}' (len={min_len})")
                print(f"    {size} contexts → avg {avg_dropped:.1f} tokens dropped")
                print(f"    max JS to full: {max_js:.6f} bits")

                # Show length distribution
                orig_lens = cluster["original_lengths"]
                len_counts = {}
                for L in orig_lens:
                    len_counts[L] = len_counts.get(L, 0) + 1
                print(f"    original lengths: {len_counts}")

                # Create new signature for the refined cluster
                p0, p1 = cluster["p_next"]
                new_sig = f"{rep}_P0={p0:.3f}_P1={p1:.3f}"

                # Convert back to list of numpy arrays for Stage B compatibility
                cluster_histories = [ctx["minimal_suffix"] for ctx in cluster["contexts"]]
                refined_emission_groups[new_sig] = cluster_histories

                total_contexts_after += 1
                total_tokens_dropped += sum(ctx["tokens_dropped"] for ctx in cluster["contexts"])

        print(f"\nBackward stability summary:")
        print(f"  Before: {total_contexts_before} contexts")
        print(f"  After: {total_contexts_after} minimal contexts")
        print(f"  Total tokens dropped: {total_tokens_dropped}")
        print(f"  Avg tokens dropped per original context: {total_tokens_dropped/total_contexts_before:.1f}")

        # Use refined groups for Stage B
        emission_groups = refined_emission_groups
        print(f"Stage A+: Using {len(emission_groups)} refined minimal contexts for Stage B")

    # Stage B: Efficient refinement using representatives
    print(f"\n=== Stage B: Refine with conditional JS (k={k_refine}) ===")

    final_clusters = []
    for sig, group_histories in emission_groups.items():
        if len(group_histories) == 1:
            final_clusters.append(group_histories)
            print(f" Group {sig}: singleton, no refinement needed")
            continue

        print(f" Refining group {sig} with {len(group_histories)} histories...")

        # Select representatives (first N unique histories)
        representatives = []
        seen_patterns = set()
        for hist in group_histories:
            pattern = tuple(hist)
            if pattern not in seen_patterns and len(representatives) < max_representatives:
                representatives.append(hist)
                seen_patterns.add(pattern)

        if len(representatives) <= 1:
            # All histories are identical
            final_clusters.append(group_histories)
            print(f" All histories identical, keeping as single cluster")
            continue

        print(f" Using {len(representatives)} representatives for clustering")

        # Cluster representatives using conditional JS
        rep_clusters = [[rep] for rep in representatives]

        # Merge representatives that are similar
        while len(rep_clusters) > 1:
            min_js = float('inf')
            merge_i, merge_j = -1, -1

            # Find closest pair of representative clusters
            for i in range(len(rep_clusters)):
                for j in range(i + 1, len(rep_clusters)):
                    h1, h2 = rep_clusters[i][0], rep_clusters[j][0]
                    try:
                        js_cond = fast_conditional_js_with_cache(cache, model, h1, h2,
                                                               k_refine, platt_params)
                        if js_cond < min_js:
                            min_js = js_cond
                            merge_i, merge_j = i, j
                    except Exception as e:
                        print(f" Warning: Conditional JS failed: {e}")
                        continue

            # Stop if minimum JS exceeds threshold
            if min_js > stage_b_threshold:
                print(f" Min conditional JS = {min_js:.6f} > {stage_b_threshold}, stopping")
                break

            # Merge the closest clusters
            print(f" Merging representatives (JS_cond = {min_js:.6f})")
            rep_clusters[merge_i].extend(rep_clusters[merge_j])
            del rep_clusters[merge_j]

        print(f" Representative clustering produced {len(rep_clusters)} clusters")

        # Assign all histories to representative clusters
        for rep_cluster in rep_clusters:
            # Each rep cluster becomes a final cluster
            # For now, just use the representative as the cluster
            # In a more sophisticated approach, we'd assign all similar histories
            final_clusters.append([rep_cluster[0]])  # Just the representative for now

        # Add remaining histories to the closest representative cluster
        remaining_histories = [h for h in group_histories
                              if not any(np.array_equal(h, rep) for rep in representatives)]
        if remaining_histories:
            print(f" Assigning {len(remaining_histories)} remaining histories to closest representatives")
            # For simplicity, assign to first representative cluster
            if final_clusters:
                final_clusters[-len(rep_clusters)].extend(remaining_histories)

    # Create result
    cluster_representatives = [cluster[0] for cluster in final_clusters]
    cluster_emission_sigs = []
    for cluster in final_clusters:
        repr_hist = cluster[0]
        probs = get_next_token_distribution(model, repr_hist, platt_params)
        sig = compute_emission_signature(probs, emission_precision)
        cluster_emission_sigs.append(sig)

    # Stage C: Optional state remerging for infinite memory processes
    if enable_remerging:
        print(f"\n=== Stage C: State Remerging ===")
        print(f"Before remerging: {len(final_clusters)} states")

        remerged_clusters, remerged_representatives, remerged_emission_sigs = stage_c_remerge_states(
            final_clusters, cluster_representatives, cluster_emission_sigs,
            model, platt_params, k_refine,
            emission_threshold=remerge_emission_threshold,
            rollout_threshold=remerge_rollout_threshold
        )

        final_clusters = remerged_clusters
        cluster_representatives = remerged_representatives
        cluster_emission_sigs = remerged_emission_sigs

        print(f"After remerging: {len(final_clusters)} states")

    result = FastClusteringResult(
        clusters=final_clusters,
        cluster_representatives=cluster_representatives,
        cluster_emission_sigs=cluster_emission_sigs,
        total_histories_sampled=len(histories),
        cache_hits=cache.hits,
        cache_misses=cache.misses
    )

    print(f"\nFinal result: {len(final_clusters)} epsilon machine states")
    print(f"Cache performance: {cache.hits} hits, {cache.misses} misses")

    return result


def compute_epsilon_machine_loss(clustering_result: FastClusteringResult, model, data: np.ndarray,
                                L: int, platt_params: Optional[dict] = None) -> Dict:
    """
    Compute the negative log-likelihood (loss) that the discovered epsilon machine would achieve.
    For each discovered state (cluster), compute the emission probabilities from the model.
    Then evaluate the likelihood of the data under this epsilon machine.
    """
    clusters = clustering_result.clusters
    if not clusters:
        return {'total_loss': float('inf'), 'avg_loss_per_symbol': float('inf'), 'num_predictions': 0}

    print(f"\n=== Computing Epsilon Machine Loss ===")

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
        if len(history) < 2:  # Need at least length 2 for shortest suffix
            return None

        # Try to match the history against each cluster's representative suffix
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            repr_hist = cluster[0]  # Representative (minimal suffix)
            repr_len = len(repr_hist)

            # Check if history is long enough to contain this representative suffix
            if len(history) >= repr_len:
                # Get the suffix of history that matches representative length
                history_suffix = history[-repr_len:]

                # Exact match with representative suffix
                if np.array_equal(history_suffix, repr_hist):
                    return i

        # If no exact suffix match found, try fuzzy matching with shortest representatives first
        best_state = None
        min_distance = float('inf')

        # Sort clusters by representative length (shorter first for precedence)
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
                # Hamming distance for fuzzy matching
                distance = np.sum(history_suffix != repr_hist)
                if distance < min_distance:
                    min_distance = distance
                    best_state = i

        return best_state if min_distance <= 1 else None  # Allow up to 1 bit difference

    # Step 3: Compute negative log-likelihood on the dataset
    total_loss = 0.0
    num_predictions = 0
    print(f"Evaluating epsilon machine on {len(data)} symbols...")

    for i in range(L, len(data)):
        # Start from position L to have enough context
        # Get context of length L
        context = data[i-L:i]
        next_symbol = int(data[i])

        # Map context to discovered state
        state = get_discovered_state(context)
        if state is None or state not in state_emissions:
            continue

        # Get emission probability for next symbol in this state
        emission_probs = state_emissions[state]
        prob_next = float(emission_probs[next_symbol])

        # Add negative log-likelihood (avoid log(0))
        if prob_next > 1e-12:
            total_loss -= np.log(prob_next)
            num_predictions += 1

    avg_loss_per_symbol = total_loss / num_predictions if num_predictions > 0 else float('inf')
    avg_loss_per_symbol_bits = avg_loss_per_symbol / np.log(2)  # Convert to bits

    print(f"Total predictions: {num_predictions}")
    print(f"Average loss: {avg_loss_per_symbol:.4f} nats ({avg_loss_per_symbol_bits:.4f} bits)")

    return {
        'total_loss': total_loss,
        'avg_loss_per_symbol': avg_loss_per_symbol,
        'avg_loss_per_symbol_bits': avg_loss_per_symbol_bits,
        'num_predictions': num_predictions,
        'num_states': len([c for c in clusters if c])
    }


def evaluate_clustering_against_gt(clustering_result: FastClusteringResult, preset: str) -> Dict:
    """Evaluate discovered clusters against ground truth states."""
    clusters = clustering_result.clusters
    cluster_analysis = []

    for i, cluster in enumerate(clusters):
        gt_states = []
        hist_strs = []
        for hist in cluster:
            gt_state = get_gt_state(hist, preset)
            if gt_state:
                gt_states.append(gt_state)
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

    # Compute ground truth state coverage
    all_gt_states = set()
    for info in cluster_analysis:
        all_gt_states.update(info['gt_state_counts'].keys())

    # Expected states for different presets
    expected_gt_states = {
        'seven_state_human': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'seven_state_human_large': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'seven_state_human_char_large': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'sevestateold': {'bb', 'aaa', 'aaab', 'ba', 'bab', 'baab', 'baa'},
        'golden_mean': {'A', 'B'},
        'even_process': {'E', 'O'}
    }
    expected_states = expected_gt_states.get(preset, set())
    gt_coverage = len(all_gt_states & expected_states) / len(expected_states) \
        if expected_states else 0.0

    return {
        'num_clusters_discovered': len(clusters),
        'total_histories': total_histories,
        'weighted_purity': weighted_purity,
        'gt_state_coverage': gt_coverage,
        'discovered_gt_states': sorted(list(all_gt_states)),
        'expected_gt_states': sorted(list(expected_states)),
        'cluster_analysis': cluster_analysis,
        'cache_hits': clustering_result.cache_hits,
        'cache_misses': clustering_result.cache_misses
    }


def main():
    """Main function for fast unsupervised epsilon machine discovery."""
    parser = argparse.ArgumentParser(description='Fast unsupervised two-stage epsilon machine discovery')
    parser.add_argument('--preset', type=str,
                       choices=['seven_state_human', 'seven_state_human_100k',
                               'seven_state_human_large', 'seven_state_human_char_large', 'sevestateold', 'even_process'],
                       default='seven_state_human_large')
    parser.add_argument('--model_ckpt', type=str, help='Override model checkpoint path')
    parser.add_argument('--data', type=str, help='Override dataset path')
    parser.add_argument('--L', type=int, default=5, help='History length')
    parser.add_argument('--k_refine', type=int, default=4, help='k for conditional JS in Stage B')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of histories to sample')
    parser.add_argument('--sampling_strategy', type=str, choices=['random', 'emission_stratified'],
                       default='emission_stratified', help='History sampling strategy')
    parser.add_argument('--stage_a_threshold', type=float, default=0.001, help='Stage A JS threshold')
    parser.add_argument('--stage_b_threshold', type=float, default=0.001,
                       help='Stage B conditional JS threshold')
    parser.add_argument('--emission_precision', type=int, default=2,
                       help='Decimal places for emission grouping')
    parser.add_argument('--max_representatives', type=int, default=10,
                       help='Max representatives per emission group')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_json', type=str, help='Save results to JSON')

    # Backward stability options
    parser.add_argument('--backward_stability', action='store_true',
                       help='Apply backward stability test to find minimal suffixes within Stage A buckets')
    parser.add_argument('--tolerance_bits', type=float, default=1e-3,
                       help='JS divergence tolerance in bits for backward stability test')
    parser.add_argument('--min_suffix_len', type=int, default=2,
                       help='Minimum suffix length to consider')

    # State remerging options
    parser.add_argument('--enable_remerging', action='store_true', default=True,
                       help='Enable Stage C state remerging for infinite memory processes')
    parser.add_argument('--disable_remerging', action='store_true',
                       help='Disable Stage C state remerging (overrides --enable_remerging)')
    parser.add_argument('--remerge_emission_threshold', type=float, default=1e-4,
                       help='JS divergence threshold for emission similarity in remerging')
    parser.add_argument('--remerge_rollout_threshold', type=float, default=5e-4,
                       help='JS divergence threshold for rollout similarity in remerging')

    args = parser.parse_args()

    # Handle remerging flag logic
    if args.disable_remerging:
        args.enable_remerging = False

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[1]  # cssr_discovery is in repo root
    default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char_50k' / 'ckpt.pt'
    default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'

    if args.preset == 'seven_state_human_100k':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char_100k' / 'ckpt.pt'
    elif args.preset == 'seven_state_human_large':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-char_large' / 'ckpt.pt'
    elif args.preset == 'seven_state_human_char_large':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-human-char-large' / 'ckpt.pt'
    elif args.preset == 'sevestateold':
        default_model = repo_root / 'nanoGPT' / 'out-sevestateold-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'sevestateold' / 'sevestateold.dat'
    elif args.preset == 'even_process':
        default_model = repo_root / 'nanoGPT' / 'out-even-process-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'even_process' / 'even_process.dat'

    model_ckpt = Path(args.model_ckpt) if args.model_ckpt else default_model
    data_path = Path(args.data) if args.data else default_data

    # Add repo root to path
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from transcssr_neural_runner import _load_nano_gpt_model, load_binary_string

    print(f"Fast Unsupervised Epsilon Machine Discovery")
    print(f"Model: {model_ckpt}")
    print(f"Sampling: {args.sampling_strategy} ({args.n_samples} histories, L={args.L})")
    print(f"Refinement: k={args.k_refine}, max_reps={args.max_representatives}")

    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, block_size = _load_nano_gpt_model(model_ckpt, device)
    s = load_binary_string(data_path)
    data = np.array([int(c) for c in s], dtype=np.int64)
    print(f"Data length: {len(data)}")

    # Fit Platt calibration
    print("\nFitting Platt calibration...")
    platt = fit_platt_params(data, model, L_max=args.L, min_count=5)
    if platt:
        print(f"Platt params: {platt}")

    # Sample histories
    print(f"\nSampling histories using {args.sampling_strategy} strategy...")
    if args.sampling_strategy == 'random':
        sampled_histories = random_history_sampling(data, args.L, args.n_samples, args.seed)
    elif args.sampling_strategy == 'emission_stratified':
        sampled_histories = emission_stratified_sampling(data, args.L, model, args.n_samples,
                                                       platt, n_strata=5, seed=args.seed)
    print(f"Sampled {len(sampled_histories)} histories")

    # Run efficient clustering
    print(f"\n{'='*70}")
    print("FAST UNSUPERVISED TWO-STAGE EPSILON MACHINE DISCOVERY")
    print(f"{'='*70}")

    clustering_result = efficient_two_stage_clustering(
        sampled_histories, model, args.k_refine, platt,
        args.stage_a_threshold, args.stage_b_threshold,
        args.emission_precision, args.max_representatives,
        args.backward_stability, args.tolerance_bits, args.min_suffix_len,
        args.enable_remerging, args.remerge_emission_threshold, args.remerge_rollout_threshold
    )

    # Evaluate against ground truth
    print(f"\n{'='*70}")
    print("EVALUATION AGAINST GROUND TRUTH")
    print(f"{'='*70}")

    evaluation = evaluate_clustering_against_gt(clustering_result, args.preset)
    print(f"Discovered {evaluation['num_clusters_discovered']} epsilon machine states:")
    print(f"Overall weighted purity: {evaluation['weighted_purity']:.3f}")
    print(f"Ground truth state coverage: {evaluation['gt_state_coverage']:.3f}")
    print(f"Expected GT states: {evaluation['expected_gt_states']}")
    print(f"Discovered GT states: {evaluation['discovered_gt_states']}")
    print(f"Cache performance: {evaluation['cache_hits']} hits, {evaluation['cache_misses']} misses")

    for cluster_info in evaluation['cluster_analysis']:
        cid = cluster_info['cluster_id']
        size = cluster_info['size']
        purity = cluster_info['purity']
        dominant = cluster_info['dominant_gt_state']
        emission_sig = cluster_info['emission_signature']
        gt_counts = cluster_info['gt_state_counts']
        print(f"\nCluster {cid}: {size} histories (purity={purity:.3f})")
        print(f" Emission: {emission_sig}, Dominant GT: {dominant}")
        print(f" GT distribution: {gt_counts}")
        print(f" Sample: {cluster_info['sample_histories']}")

    # Expected states based on preset
    expected_states = {
        'seven_state_human': 7,
        'seven_state_human_100k': 7,
        'seven_state_human_large': 7,
        'seven_state_human_char_large': 7,
        'sevestateold': 7,
        'golden_mean': 2,
        'even_process': 2
    }
    expected = expected_states.get(args.preset, 'unknown')
    print(f"\nExpected: {expected} states → Discovered: {len(clustering_result.clusters)} states")

    # Compute epsilon machine loss
    print(f"\n{'='*70}")
    print("EPSILON MACHINE LOSS EVALUATION")
    print(f"{'='*70}")

    loss_results = compute_epsilon_machine_loss(clustering_result, model, data, args.L, platt)
    print(f"\nDiscovered Epsilon Machine Performance:")
    print(f" States: {loss_results['num_states']}")
    print(f" Average loss: {loss_results['avg_loss_per_symbol_bits']:.4f} bits/symbol")
    print(f" Total predictions: {loss_results['num_predictions']}")

    # Save results
    if args.output_json:
        results = {
            'parameters': vars(args),
            'platt_params': platt,
            'clustering_result': {
                'num_clusters': len(clustering_result.clusters),
                'clusters': [
                    {
                        'cluster_id': i,
                        'size': len(cluster),
                        'histories': [h.tolist() for h in cluster],
                        'history_strings': [''.join(map(str, h)) for h in cluster],
                        'emission_signature': clustering_result.cluster_emission_sigs[i]
                    }
                    for i, cluster in enumerate(clustering_result.clusters)
                ],
                'cache_hits': clustering_result.cache_hits,
                'cache_misses': clustering_result.cache_misses
            },
            'evaluation': evaluation,
            'loss_evaluation': loss_results
        }
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()