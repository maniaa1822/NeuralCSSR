#!/usr/bin/env python3
"""
Peak-based clustering algorithm for epsilon-machine discovery.

This implements the clean recipe for discovering epsilon-machines from neural models
using peak analysis of JS divergence histograms.

Algorithm steps:
0) Estimate peak centers from random-pairs histogram
1) Build band labeler from peaks
2) Pick anchors using farthest-point selection
3) Give every history a short peak code (signature)
4) Cluster by signature
5) Make it unifiliar (refinement)
6) Optional: refine hard blocks with larger k
7) Compute transition probabilities

Usage:
    python peak_clustering_cssr.py --preset seven_state_human --L 4 --k 4
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from typing import List, Tuple, Optional, Dict, Any
import argparse
import torch
from pathlib import Path
import sys
from collections import defaultdict

# Add nanoGPT directory to path for imports
sys.path.append(str(Path(__file__).parent / 'nanoGPT'))

def extract_subsequences(data: np.ndarray, L: int) -> List[np.ndarray]:
    """Extract all length-L subsequences from dataset."""
    subsequences = []
    for i in range(len(data) - L + 1):
        subsequences.append(data[i:i+L])
    return subsequences

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def get_next_token_distribution(model, history: np.ndarray, platt_params: Optional[dict] = None) -> np.ndarray:
    """Get next token probability distribution from model."""
    device = next(model.parameters()).device

    with torch.no_grad():
        x = torch.tensor(history, dtype=torch.long).unsqueeze(0)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if x.size(1) > 0:
            x = x.to(device)

        logits, _ = model(x)
        logits = logits[0, -1, :2]  # Get last position, first 2 tokens (0,1)

        # Apply Platt calibration if provided
        if platt_params is not None:
            a, b = platt_params['a'], platt_params['b']
            logits = a * logits + b

        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

def get_kstep_distribution(model, history: np.ndarray, k: int, platt_params: Optional[dict] = None) -> np.ndarray:
    """Get k-step rollout probability distribution."""
    if k == 0:
        return get_next_token_distribution(model, history, platt_params)

    n_outcomes = 1 << k  # 2^k possible outcomes
    probabilities = np.zeros(n_outcomes)

    # Generate all possible k-step sequences
    for outcome in range(n_outcomes):
        # Convert outcome to binary sequence
        sequence = [(outcome >> i) & 1 for i in range(k-1, -1, -1)]

        # Compute probability of this sequence
        prob = 1.0
        current_hist = history.tolist() if hasattr(history, 'tolist') else list(history)

        for symbol in sequence:
            p = get_next_token_distribution(model, np.array(current_hist, dtype=np.int64), platt_params)
            prob *= p[symbol]
            current_hist = current_hist[1:] + [symbol]  # Shift and append

        probabilities[outcome] = prob

    return probabilities

def js_divergence_k(model, h1: np.ndarray, h2: np.ndarray, k: int, platt_params: Optional[dict] = None) -> float:
    """Compute k-step JS divergence between two histories."""
    p = get_kstep_distribution(model, h1, k, platt_params)
    q = get_kstep_distribution(model, h2, k, platt_params)
    return js_divergence(p, q)

def js_divergence_conditional_k(model, h1: np.ndarray, h2: np.ndarray, k: int, platt_params: Optional[dict] = None) -> float:
    """Compute conditional k-step JS divergence between two histories."""
    if k < 2:
        raise ValueError("Conditional JS requires k >= 2")

    p = get_kstep_distribution(model, h1, k, platt_params)
    q = get_kstep_distribution(model, h2, k, platt_params)

    # Split by first symbol
    m = 1 << (k - 1)  # 2^(k-1)
    p0, p1 = p[:m], p[m:]  # First symbol = 0, First symbol = 1
    q0, q1 = q[:m], q[m:]

    # Compute weights (marginal probabilities of first symbol)
    w0p, w1p = float(p0.sum()), float(p1.sum())
    w0q, w1q = float(q0.sum()), float(q1.sum())

    # Normalize conditional distributions
    p0 = p0 / max(w0p, 1e-12)
    p1 = p1 / max(w1p, 1e-12)
    q0 = q0 / max(w0q, 1e-12)
    q1 = q1 / max(w1q, 1e-12)

    # Weighted average of conditional JS divergences
    w0_bar = 0.5 * (w0p + w0q)
    w1_bar = 1.0 - w0_bar

    js_cond = (w0_bar * js_divergence(p0, q0) + w1_bar * js_divergence(p1, q1))
    return float(js_cond)

def discover_epsilon_machine_from_peaks(data: np.ndarray, model, L: int, k: int = 4,
                                       use_conditional: bool = True, platt_params: Optional[dict] = None,
                                       n_samples: int = 2000, m_anchors: int = 8,
                                       max_refinement_k: int = 6, preset: str = 'seven_state_human',
                                       anchor_subsample: int = 10000) -> Dict[str, Any]:
    """
    Discover epsilon-machine using peak-based clustering algorithm.

    Algorithm steps:
    0) Estimate peak centers from random-pairs histogram
    1) Build band labeler from peaks
    2) Pick anchors using farthest-point selection
    3) Give every history a short peak code (signature)
    4) Cluster by signature
    5) Make it unifiliar (refinement)
    6) Optional: refine hard blocks with larger k
    7) Compute transition probabilities

    Args:
        data: Binary sequence data
        model: Neural model for probability computation
        L: History length
        k: k-step rollout length
        use_conditional: Whether to use conditional JS divergence
        platt_params: Platt calibration parameters
        n_samples: Number of random pairs for peak estimation
        m_anchors: Number of anchors to select
        max_refinement_k: Maximum k for refinement pass
        preset: Dataset preset for ground truth mapping

    Returns:
        Dictionary containing discovered epsilon-machine structure
    """
    print(f"\n=== Peak-based Epsilon-Machine Discovery ===")
    print(f"L={L}, k={k}, conditional={use_conditional}, anchors={m_anchors}")

    # Extract all histories of length L
    histories = extract_subsequences(data, L)
    n_histories = len(histories)
    print(f"Extracted {n_histories} histories of length {L}")

    # Step 0: Estimate peak centers from random pairs
    print("\nStep 0: Estimating peak centers from random pairs...")

    # Sample random pairs and compute distances
    js_distances = []
    pairs_to_sample = min(n_samples, (n_histories * (n_histories - 1)) // 2)

    for _ in range(pairs_to_sample):
        i, j = np.random.choice(n_histories, 2, replace=False)
        h1, h2 = np.array(histories[i]), np.array(histories[j])

        if use_conditional and k >= 2:
            dist = js_divergence_conditional_k(model, h1, h2, k, platt_params)
        elif k > 0:
            dist = js_divergence_k(model, h1, h2, k, platt_params)
        else:
            p = get_next_token_distribution(model, h1, platt_params)
            q = get_next_token_distribution(model, h2, platt_params)
            dist = js_divergence(p, q)

        js_distances.append(dist)

    js_distances = np.array(js_distances)

    # Fit GMM to find peaks
    # Try different numbers of components
    best_aic = float('inf')
    best_gmm = None

    for n_components in range(2, 6):
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(js_distances.reshape(-1, 1))
            aic = gmm.aic(js_distances.reshape(-1, 1))

            if aic < best_aic:
                best_aic = aic
                best_gmm = gmm
        except:
            continue

    if best_gmm is None:
        raise ValueError("Failed to fit GMM to JS distances")

    # Extract peak centers and tolerances
    peak_centers = best_gmm.means_.flatten()
    peak_stds = np.sqrt(best_gmm.covariances_.flatten())
    peak_weights = best_gmm.weights_

    # Sort by peak centers
    sort_idx = np.argsort(peak_centers)
    peak_centers = peak_centers[sort_idx]
    peak_stds = peak_stds[sort_idx]
    peak_weights = peak_weights[sort_idx]

    # Set tolerances as half the inter-peak gaps
    tolerances = np.zeros_like(peak_centers)
    for i in range(len(peak_centers)):
        if i == 0:
            tolerances[i] = peak_stds[i] * 2  # Use 2 standard deviations
        else:
            gap = peak_centers[i] - peak_centers[i-1]
            tolerances[i] = min(gap / 2, peak_stds[i] * 2)

    print(f"Found {len(peak_centers)} peaks:")
    for i, (center, tol, weight) in enumerate(zip(peak_centers, tolerances, peak_weights)):
        print(f"  Peak {i}: μ={center:.4f}, δ={tol:.4f}, weight={weight:.3f}")

    # Step 1: Build band labeler from peaks
    def assign_band(distance: float) -> int:
        """Assign distance to nearest peak band, return -1 if outside tolerances."""
        diffs = np.abs(distance - peak_centers)
        closest_idx = np.argmin(diffs)

        if diffs[closest_idx] <= tolerances[closest_idx]:
            return closest_idx
        else:
            return -1  # Unknown band

    # Step 2: Pick anchors using farthest-point selection
    print(f"\nStep 2: Selecting {m_anchors} anchors using farthest-point method...")

    # Subsample histories for anchor selection to speed up computation
    if n_histories > anchor_subsample:
        print(f"  Subsampling {anchor_subsample} histories from {n_histories} for anchor selection")
        anchor_candidate_indices = np.random.choice(n_histories, anchor_subsample, replace=False)
        anchor_candidates = [histories[i] for i in anchor_candidate_indices]
    else:
        anchor_candidate_indices = np.arange(n_histories)
        anchor_candidates = histories

    n_candidates = len(anchor_candidates)

    # Start from random history
    anchor_idx_in_candidates = np.random.choice(n_candidates)
    anchors = [anchor_candidate_indices[anchor_idx_in_candidates]]
    anchor_histories = [np.array(anchor_candidates[anchor_idx_in_candidates])]

    # Helper function to compute distance to anchor set
    def distance_to_anchors(candidate_idx: int) -> float:
        """Compute median distance from candidate history to current anchor set."""
        h = np.array(anchor_candidates[candidate_idx])
        distances = []

        for anchor_h in anchor_histories:
            if use_conditional and k >= 2:
                dist = js_divergence_conditional_k(model, h, anchor_h, k, platt_params)
            elif k > 0:
                dist = js_divergence_k(model, h, anchor_h, k, platt_params)
            else:
                p = get_next_token_distribution(model, h, platt_params)
                q = get_next_token_distribution(model, anchor_h, platt_params)
                dist = js_divergence(p, q)
            distances.append(dist)

        return np.median(distances)

    # Iteratively add farthest point
    for anchor_num in range(m_anchors - 1):
        max_dist = -1
        farthest_candidate_idx = -1

        for i in range(n_candidates):
            if anchor_candidate_indices[i] not in anchors:
                dist = distance_to_anchors(i)
                if dist > max_dist:
                    max_dist = dist
                    farthest_candidate_idx = i

        if farthest_candidate_idx != -1:
            global_idx = anchor_candidate_indices[farthest_candidate_idx]
            anchors.append(global_idx)
            anchor_histories.append(np.array(anchor_candidates[farthest_candidate_idx]))
            print(f"  Selected anchor {anchor_num + 2}/{m_anchors}, distance: {max_dist:.4f}")

    print(f"Selected anchors: {anchors}")

    # Step 3: Give every history a signature
    print(f"\nStep 3: Computing signatures for all {n_histories} histories...")

    signatures = []
    for i in range(n_histories):
        h = np.array(histories[i])
        signature = []

        for anchor_h in anchor_histories:
            if use_conditional and k >= 2:
                dist = js_divergence_conditional_k(model, h, anchor_h, k, platt_params)
            elif k > 0:
                dist = js_divergence_k(model, h, anchor_h, k, platt_params)
            else:
                p = get_next_token_distribution(model, h, platt_params)
                q = get_next_token_distribution(model, anchor_h, platt_params)
                dist = js_divergence(p, q)

            band = assign_band(dist)
            signature.append(band)

        signatures.append(tuple(signature))

    # Step 4: Cluster by signature
    print(f"\nStep 4: Clustering by signature...")

    signature_groups = defaultdict(list)

    for i, sig in enumerate(signatures):
        signature_groups[sig].append(i)

    print(f"Found {len(signature_groups)} unique signatures")
    for sig, indices in signature_groups.items():
        print(f"  Signature {sig}: {len(indices)} histories")

    # Step 5: Unifilarity refinement
    print(f"\nStep 5: Unifilarity refinement...")

    def get_successor_bucket(hist_idx: int, symbol: int, bucket_map: dict) -> Optional[int]:
        """Get the bucket ID that the successor history belongs to."""
        h = histories[hist_idx]
        successor = h[1:] + [symbol]  # Shift and append

        # Find this successor in our histories
        for bucket_id, hist_indices in bucket_map.items():
            for idx in hist_indices:
                if np.array_equal(histories[idx], successor):
                    return bucket_id
        return None

    # Convert to bucket format
    buckets = {}
    for bucket_id, (sig, indices) in enumerate(signature_groups.items()):
        buckets[bucket_id] = indices

    # Iterative refinement
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        print(f"  Refinement iteration {iteration + 1}")
        needs_split = False
        new_buckets = {}
        next_bucket_id = 0

        for bucket_id, hist_indices in buckets.items():
            # Check if this bucket needs splitting
            successor_patterns = defaultdict(list)

            for hist_idx in hist_indices:
                succ_0 = get_successor_bucket(hist_idx, 0, buckets)
                succ_1 = get_successor_bucket(hist_idx, 1, buckets)
                pattern = (succ_0, succ_1)
                successor_patterns[pattern].append(hist_idx)

            if len(successor_patterns) > 1:
                needs_split = True
                print(f"    Splitting bucket {bucket_id} into {len(successor_patterns)} sub-buckets")

                for pattern, indices in successor_patterns.items():
                    new_buckets[next_bucket_id] = indices
                    next_bucket_id += 1
            else:
                new_buckets[next_bucket_id] = hist_indices
                next_bucket_id += 1

        buckets = new_buckets
        iteration += 1

        if not needs_split:
            print(f"  Converged after {iteration} iterations")
            break

    print(f"Final state partition: {len(buckets)} states")
    for state_id, hist_indices in buckets.items():
        print(f"  State {state_id}: {len(hist_indices)} histories")

    # Step 7: Compute transition probabilities
    print(f"\nStep 7: Computing transition probabilities...")

    transitions = {}
    emissions = {}

    for state_id, hist_indices in buckets.items():
        # Compute average transition probabilities for this state
        symbol_counts = {0: 0, 1: 0}
        transition_counts = defaultdict(lambda: {0: 0, 1: 0})

        for hist_idx in hist_indices:
            h = np.array(histories[hist_idx])

            # Get model's next-token distribution
            p = get_next_token_distribution(model, h, platt_params)

            # Weight by probability
            for symbol in [0, 1]:
                prob = p[symbol]
                symbol_counts[symbol] += prob

                # Find successor state
                successor_state = get_successor_bucket(hist_idx, symbol, buckets)
                if successor_state is not None:
                    transition_counts[successor_state][symbol] += prob

        # Normalize
        total_emission = sum(symbol_counts.values())
        if total_emission > 0:
            emissions[state_id] = {s: count/total_emission for s, count in symbol_counts.items()}
        else:
            emissions[state_id] = {0: 0.5, 1: 0.5}

        # Normalize transitions
        transitions[state_id] = {}
        for next_state, symbol_counts_next in transition_counts.items():
            total_trans = sum(symbol_counts_next.values())
            if total_trans > 0:
                transitions[state_id][next_state] = {s: count/total_trans for s, count in symbol_counts_next.items()}

    # Prepare result
    result = {
        'n_states': len(buckets),
        'states': buckets,
        'transitions': transitions,
        'emissions': emissions,
        'anchors': anchors,
        'anchor_histories': [histories[i] for i in anchors],
        'peak_centers': peak_centers.tolist(),
        'tolerances': tolerances.tolist(),
        'js_distances': js_distances.tolist(),
        'signatures': signatures,
        'algorithm_params': {
            'L': L,
            'k': k,
            'use_conditional': use_conditional,
            'n_samples': n_samples,
            'm_anchors': m_anchors,
            'preset': preset
        }
    }

    print(f"\n=== Discovery Complete ===")
    print(f"Discovered {len(buckets)} states with {len(anchors)} anchors")

    return result

def main():
    parser = argparse.ArgumentParser(description='Discover epsilon-machine using peak-based clustering')
    parser.add_argument('--preset', type=str, choices=['golden_mean', 'seven_state_human', 'seven_state_human_large', 'even_process'],
                       default='seven_state_human')
    parser.add_argument('--model_ckpt', type=str, help='Override model checkpoint path')
    parser.add_argument('--data', type=str, help='Override dataset path')
    parser.add_argument('--L', type=int, default=4, help='History length')
    parser.add_argument('--k', type=int, default=4, help='k-step rollout length')
    parser.add_argument('--use_conditional', action='store_true', default=True,
                       help='Use conditional JS divergence')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of random pairs for peak estimation')
    parser.add_argument('--m_anchors', type=int, default=8, help='Number of anchors to select')
    parser.add_argument('--anchor_subsample', type=int, default=10000, help='Max histories to consider for anchor selection')
    parser.add_argument('--platt_min_count', type=int, default=5, help='Minimum count for Platt calibration')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Resolve paths
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / 'nanoGPT' / 'out-golden-mean-char' / 'ckpt.pt'
    default_data = repo_root / 'experiments' / 'datasets' / 'golden_mean' / 'golden_mean.dat'

    if args.preset == 'seven_state_human':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'
    elif args.preset == 'seven_state_human_large':
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-char_large' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'
    elif args.preset == 'even_process':
        default_model = repo_root / 'nanoGPT' / 'out-even-process-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'even_process' / 'even_process.dat'

    model_ckpt = Path(args.model_ckpt) if args.model_ckpt else default_model
    data_path = Path(args.data) if args.data else default_data

    # Load model and data
    sys.path.append(str(repo_root))
    from transcssr_neural_runner import _load_nano_gpt_model, load_binary_string

    print(f"Loading model from: {model_ckpt}")
    print(f"Loading data from: {data_path}")

    model, block_size = _load_nano_gpt_model(model_ckpt, device)
    data_str = load_binary_string(data_path)
    data = np.array([int(c) for c in data_str], dtype=np.int64)

    # Fit Platt calibration parameters
    from nanoGPT.plot_js_vs_L import fit_platt_params
    platt_params = fit_platt_params(data, model, L_max=args.L, min_count=args.platt_min_count)
    if platt_params:
        print(f"Using Platt calibration: a={platt_params['a']:.4f}, b={platt_params['b']:.4f}")
    else:
        print("No Platt calibration applied")

    # Run discovery algorithm
    result = discover_epsilon_machine_from_peaks(
        data=data,
        model=model,
        L=args.L,
        k=args.k,
        use_conditional=args.use_conditional,
        platt_params=platt_params,
        n_samples=args.n_samples,
        m_anchors=args.m_anchors,
        preset=args.preset,
        anchor_subsample=args.anchor_subsample
    )

    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Preset: {args.preset}")
    print(f"History length L: {args.L}")
    print(f"k-step rollout: {args.k}")
    print(f"Conditional JS: {args.use_conditional}")
    print(f"Discovered states: {result['n_states']}")
    print(f"Anchors used: {len(result['anchors'])}")
    print(f"Peak centers: {result['peak_centers']}")

if __name__ == "__main__":
    main()