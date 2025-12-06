"""
Analyze pairwise JS divergences between ground truth belief states
as predicted by the trained model. This helps us choose an optimal tolerance.
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent / 'nanoGPT'))
from model import GPT, GPTConfig

sys.path.insert(0, str(Path(__file__).parent / 'cssr_discovery'))
from js_metrics import js_divergence, get_next_token_distribution
from calibration import fit_platt_params


def load_model(ckpt_path, device='cuda'):
    """Load nanoGPT model."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def load_data_and_labels(data_path):
    """Load binary sequence and ground truth state labels."""
    # Load binary data
    with open(data_path, 'r') as f:
        raw = f.read()
    data = ''.join(ch for ch in raw if ch in ('0', '1'))
    data_arr = np.array([int(c) for c in data], dtype=np.uint8)

    # Load state labels
    base = Path(data_path).parent / Path(data_path).stem
    state_ids_file = base.parent / f"{base.name}.state_ids.dat"

    with open(state_ids_file, 'r') as f:
        content = f.read()
    # States are space-separated on potentially multiple lines
    state_labels = [s.strip() for s in content.split() if s.strip()]

    return data_arr, state_labels


def collect_histories_by_belief_state(data, state_labels, L, max_per_state=100):
    """Collect example histories for each belief state."""
    state_histories = defaultdict(list)

    for i in range(L, min(len(data), len(state_labels))):
        if len(data) - i < 1:
            break

        belief_state = state_labels[i]
        history = data[i-L:i]

        # Limit samples per state
        if len(state_histories[belief_state]) < max_per_state:
            state_histories[belief_state].append(history)

    return state_histories


def compute_belief_state_distributions(model, state_histories, platt_params, k_values=[1], device='cuda'):
    """Compute average k-step distributions for each belief state."""
    from js_metrics import get_kstep_distribution

    belief_distributions = {}

    for belief_state, histories in state_histories.items():
        if not histories:
            continue

        k_dists = {}

        for k in k_values:
            # Get k-step predictions for all histories in this belief state
            all_probs = []
            for hist in histories:
                if k == 1:
                    probs = get_next_token_distribution(model, hist, platt_params)
                else:
                    probs = get_kstep_distribution(model, hist, k, platt_params)
                all_probs.append(probs)

            # Average across all histories
            avg_probs = np.mean(all_probs, axis=0)
            std_probs = np.std(all_probs, axis=0)

            k_dists[k] = {
                'mean': avg_probs,
                'std': std_probs,
            }

        belief_distributions[belief_state] = {
            'k_distributions': k_dists,
            'n_samples': len(histories),
        }

    return belief_distributions


def compute_pairwise_divergences(belief_distributions, k_values=[1]):
    """Compute pairwise JS divergences between all belief states for each k."""
    states = sorted(belief_distributions.keys())
    n = len(states)

    divergence_matrices = {}

    for k in k_values:
        divergence_matrix = np.zeros((n, n))

        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                if i == j:
                    divergence_matrix[i, j] = 0.0
                else:
                    p = belief_distributions[state_i]['k_distributions'][k]['mean']
                    q = belief_distributions[state_j]['k_distributions'][k]['mean']
                    div = js_divergence(p, q)
                    divergence_matrix[i, j] = div

        divergence_matrices[k] = divergence_matrix

    return divergence_matrices, states


def plot_divergence_heatmap(divergence_matrix, state_labels, output_path):
    """Plot heatmap of JS divergences."""
    # Convert to bits
    div_bits = divergence_matrix / np.log(2)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(div_bits,
                xticklabels=state_labels,
                yticklabels=state_labels,
                annot=True,
                fmt='.4f',
                cmap='YlOrRd',
                cbar_kws={'label': 'JS Divergence (bits)'},
                square=True)

    plt.title('Pairwise JS Divergences Between Belief States\n(As predicted by the model)')
    plt.xlabel('Belief State')
    plt.ylabel('Belief State')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")


def analyze_divergences(divergence_matrix, state_labels):
    """Print statistics about the divergence distribution."""
    # Flatten upper triangle (no diagonal)
    n = len(state_labels)
    upper_tri = []
    for i in range(n):
        for j in range(i+1, n):
            upper_tri.append(divergence_matrix[i, j])

    upper_tri = np.array(upper_tri)
    upper_tri_bits = upper_tri / np.log(2)

    print("\n" + "="*80)
    print("DIVERGENCE STATISTICS")
    print("="*80)
    print(f"\nTotal belief states: {n}")
    print(f"Pairwise comparisons: {len(upper_tri)}")
    print()
    print("JS Divergence Distribution (in bits):")
    print(f"  Min:     {np.min(upper_tri_bits):.6f} bits")
    print(f"  Q1:      {np.percentile(upper_tri_bits, 25):.6f} bits")
    print(f"  Median:  {np.median(upper_tri_bits):.6f} bits")
    print(f"  Q3:      {np.percentile(upper_tri_bits, 75):.6f} bits")
    print(f"  Max:     {np.max(upper_tri_bits):.6f} bits")
    print(f"  Mean:    {np.mean(upper_tri_bits):.6f} bits")
    print(f"  Std:     {np.std(upper_tri_bits):.6f} bits")

    # Find closest pairs
    print("\n" + "="*80)
    print("CLOSEST BELIEF STATE PAIRS")
    print("="*80)
    print("\nTop 5 most similar pairs (smallest JS divergence):")

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((divergence_matrix[i, j], state_labels[i], state_labels[j]))

    pairs.sort()
    for div, state_i, state_j in pairs[:5]:
        div_bits = div / np.log(2)
        print(f"  {state_i:30} <-> {state_j:30}: {div_bits:.6f} bits")

    # Suggest tolerance
    print("\n" + "="*80)
    print("TOLERANCE RECOMMENDATIONS")
    print("="*80)
    print()

    min_div_bits = np.min(upper_tri_bits)

    print(f"Minimum divergence between different states: {min_div_bits:.6f} bits")
    print()
    print("Recommended tolerances:")
    print(f"  Very conservative:  {min_div_bits/10:.6f} bits  (merge if JS < {min_div_bits/10:.6f})")
    print(f"  Conservative:       {min_div_bits/3:.6f} bits   (merge if JS < {min_div_bits/3:.6f})")
    print(f"  Moderate:           {min_div_bits/2:.6f} bits   (merge if JS < {min_div_bits/2:.6f})")
    print(f"  Aggressive:         {min_div_bits*0.9:.6f} bits  (merge if JS < {min_div_bits*0.9:.6f})")
    print()
    print(f"Current CSSR setting: 0.001 bits")
    print(f"Ratio to minimum divergence: {0.001/min_div_bits:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Analyze belief state JS divergences')
    parser.add_argument('--data', type=str,
                        default='experiments/datasets/gm_seven_union/combined.dat',
                        help='Path to dataset')
    parser.add_argument('--model_ckpt', type=str,
                        default='nanoGPT/out-gm-seven-union-char/ckpt.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--L', type=int, default=6,
                        help='History length')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Max histories per belief state')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 3],
                        help='k-step horizons to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--output_heatmap', type=str,
                        default='results/belief_state_divergences_k{k}.png',
                        help='Output heatmap image (use {k} for k value)')
    parser.add_argument('--output_json', type=str,
                        default='results/belief_state_divergences.json',
                        help='Output JSON with divergence matrix')

    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model_ckpt, args.device)

    print("Loading data and labels...")
    data, state_labels = load_data_and_labels(args.data)

    print("Fitting Platt calibration...")
    platt_params = fit_platt_params(data, model, L_max=args.L)
    print(f"  Platt parameters: a={platt_params['a']:.3f}, b={platt_params['b']:.3f}")

    print(f"\nCollecting histories for each belief state (L={args.L})...")
    state_histories = collect_histories_by_belief_state(
        data, state_labels, args.L, max_per_state=args.max_samples
    )

    print(f"Found {len(state_histories)} unique belief states")
    for state, hists in sorted(state_histories.items()):
        print(f"  {state:30}: {len(hists)} histories")

    print(f"\nComputing average k-step distributions per belief state (k={args.k_values})...")
    belief_distributions = compute_belief_state_distributions(
        model, state_histories, platt_params, k_values=args.k_values, device=args.device
    )

    print("\nBelief state emission probabilities (k=1):")
    for state in sorted(belief_distributions.keys()):
        dist = belief_distributions[state]['k_distributions'][1]
        mean_p0 = dist['mean'][0]
        std_p0 = dist['std'][0]
        n = belief_distributions[state]['n_samples']
        print(f"  {state:30}: P(0)={mean_p0:.4f}±{std_p0:.4f}  (n={n})")

    print("\nComputing pairwise divergences...")
    divergence_matrices, state_names = compute_pairwise_divergences(
        belief_distributions, k_values=args.k_values
    )

    # Analyze and plot for each k
    import os
    os.makedirs(Path(args.output_heatmap).parent, exist_ok=True)

    for k in args.k_values:
        print(f"\n{'='*80}")
        print(f"ANALYSIS FOR k={k}")
        print(f"{'='*80}")

        divergence_matrix = divergence_matrices[k]

        # Analyze divergences
        analyze_divergences(divergence_matrix, state_names)

        # Plot heatmap
        heatmap_path = args.output_heatmap.replace('{k}', str(k))
        plot_divergence_heatmap(divergence_matrix, state_names, heatmap_path)

    # Save to JSON
    output_data = {
        'belief_states': state_names,
        'k_values': args.k_values,
        'divergence_matrices': {
            f'k{k}': {
                'nats': divergence_matrices[k].tolist(),
                'bits': (divergence_matrices[k] / np.log(2)).tolist(),
            }
            for k in args.k_values
        },
        'belief_distributions': {
            state: {
                'k_distributions': {
                    k: {
                        'mean': dist['k_distributions'][k]['mean'].tolist(),
                        'std': dist['k_distributions'][k]['std'].tolist(),
                    }
                    for k in args.k_values
                },
                'n_samples': dist['n_samples'],
            }
            for state, dist in belief_distributions.items()
        },
    }

    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved results to {args.output_json}")
    print(f"Saved heatmaps to {args.output_heatmap.replace('{k}', '*')}")


if __name__ == '__main__':
    main()
