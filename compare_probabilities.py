#!/usr/bin/env python3
"""
Compare empirical vs neural probabilities for CSSR analysis.
Analyzes how different probability estimation methods affect causal state discovery.
"""

import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import sys
import itertools

# Add transCSSR to path
sys.path.append('transCSSR')
from transCSSR import estimate_predictive_distributions

# Import our neural utilities
from neural_cssr_utils import NeuralCSSRProbabilityProvider

def _retemper_probs(p0: float, p1: float, temperature: float) -> tuple[float, float]:
    if temperature and temperature != 1.0:
        import math
        # operate in log-space to avoid underflow, then softmax back
        log0 = math.log(max(p0, 1e-9)) / temperature
        log1 = math.log(max(p1, 1e-9)) / temperature
        m = max(log0, log1)
        e0 = math.exp(log0 - m)
        e1 = math.exp(log1 - m)
        s = e0 + e1
        return e0 / s, e1 / s
    return p0, p1
from experiments.ebm.models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM

def load_model(checkpoint_path: Path, device='cpu'):
    """Load EBM or AR model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    model_type = config.get('model_type', 'ebm_binary')
    
    # Map config parameter names to model constructor names
    model_config = {}
    
    # Common mappings
    model_config['d_model'] = config.get('d_model', 128)
    model_config['dropout'] = config.get('dropout', 0.1)
    model_config['vocab_size'] = config.get('vocab_size', 3)
    model_config['output_vocab_size'] = config.get('output_vocab_size', 2)
    
    if model_type == 'ebm_binary':
        # EBM-specific parameter names
        model_config['nhead'] = config.get('heads', config.get('nhead', 8))
        model_config['num_layers'] = config.get('layers', config.get('num_layers', 4))
        model_config['max_len'] = config.get('context_window', config.get('max_len', 64))
    elif model_type == 'ar_binary':
        # AR-specific parameter names
        model_config['nhead'] = config.get('heads', config.get('nhead', 8))
        model_config['num_layers'] = config.get('layers', config.get('num_layers', 4))
        model_config['max_len'] = config.get('context_window', config.get('max_len', 64))
    
    if model_type == 'ebm_binary':
        model = EnergyBasedBinaryLM(**model_config)
    elif model_type == 'ar_binary':
        model = AutoRegressiveBinaryLM(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model

def get_empirical_probabilities(data: str, L_max: int = 6):
    """Get empirical conditional probabilities P(y_next | history) from data."""
    empirical_probs = {}
    
    for t in range(len(data)):
        for L in range(0, min(L_max + 1, t + 1)):
            if t >= L:
                history = data[max(0, t-L):t] if L > 0 else ""
                if t < len(data):
                    y_next = data[t]
                    
                    if history not in empirical_probs:
                        empirical_probs[history] = {'0': 0, '1': 0, 'total': 0}
                    
                    empirical_probs[history][y_next] += 1
                    empirical_probs[history]['total'] += 1
    
    # Convert counts to probabilities
    for history in empirical_probs:
        total = empirical_probs[history]['total']
        if total > 0:
            empirical_probs[history]['p0'] = empirical_probs[history]['0'] / total
            empirical_probs[history]['p1'] = empirical_probs[history]['1'] / total
        else:
            empirical_probs[history]['p0'] = 0.5
            empirical_probs[history]['p1'] = 0.5
    
    return empirical_probs

def compare_probability_methods(data_path: Path, model_path: Path, L_max: int = 6, temperature: float = 1.0):
    """Compare empirical vs neural probability estimates."""
    
    # Load data
    data = data_path.read_text().strip()
    print(f"Loaded dataset: {len(data)} tokens")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    provider = NeuralCSSRProbabilityProvider(model, device, context_window=64)
    
    # Get empirical probabilities
    print("Computing empirical probabilities...")
    emp_probs = get_empirical_probabilities(data, L_max)
    
    # Get neural probabilities for same histories
    print("Computing neural probabilities...")
    neural_probs = {}
    for history in emp_probs:
        context = [int(c) for c in history] if history else []
        dist = provider.predict_next_distribution(history)
        p0, p1 = _retemper_probs(dist['0'], dist['1'], temperature)
        neural_probs[history] = {'p0': p0, 'p1': p1}
    
    # Analysis
    print(f"\nFound {len(emp_probs)} unique histories (max length {L_max})")
    
    # Compare probabilities
    emp_p0_vals = []
    neural_p0_vals = []
    emp_p1_vals = []
    neural_p1_vals = []
    history_lengths = []
    sample_counts = []
    
    for history in emp_probs:
        if emp_probs[history]['total'] >= 5:  # Only histories with sufficient samples
            emp_p0_vals.append(emp_probs[history]['p0'])
            neural_p0_vals.append(neural_probs[history]['p0'])
            emp_p1_vals.append(emp_probs[history]['p1'])
            neural_p1_vals.append(neural_probs[history]['p1'])
            history_lengths.append(len(history))
            sample_counts.append(emp_probs[history]['total'])
    
    emp_p0_vals = np.array(emp_p0_vals)
    neural_p0_vals = np.array(neural_p0_vals)
    emp_p1_vals = np.array(emp_p1_vals)
    neural_p1_vals = np.array(neural_p1_vals)
    history_lengths = np.array(history_lengths)
    sample_counts = np.array(sample_counts)
    
    print(f"Analyzing {len(emp_p0_vals)} histories with ≥5 samples")
    
    # Statistics
    mae_p0 = np.mean(np.abs(emp_p0_vals - neural_p0_vals))
    mae_p1 = np.mean(np.abs(emp_p1_vals - neural_p1_vals))
    mse_p0 = np.mean((emp_p0_vals - neural_p0_vals) ** 2)
    mse_p1 = np.mean((emp_p1_vals - neural_p1_vals) ** 2)
    
    print(f"\nProbability Comparison Statistics:")
    print(f"  P(0|history) - MAE: {mae_p0:.4f}, MSE: {mse_p0:.4f}")
    print(f"  P(1|history) - MAE: {mae_p1:.4f}, MSE: {mse_p1:.4f}")
    
    # Sample size analysis
    low_samples = sample_counts <= 10
    med_samples = (sample_counts > 10) & (sample_counts <= 50)
    high_samples = sample_counts > 50
    
    print(f"\nSample Size Analysis:")
    print(f"  Low samples (≤10): {np.sum(low_samples)} histories, MAE: {np.mean(np.abs(emp_p0_vals[low_samples] - neural_p0_vals[low_samples])):.4f}")
    print(f"  Med samples (11-50): {np.sum(med_samples)} histories, MAE: {np.mean(np.abs(emp_p0_vals[med_samples] - neural_p0_vals[med_samples])):.4f}")
    print(f"  High samples (>50): {np.sum(high_samples)} histories, MAE: {np.mean(np.abs(emp_p0_vals[high_samples] - neural_p0_vals[high_samples])):.4f}")
    
    # History length analysis
    for L in range(0, L_max + 1):
        mask = history_lengths == L
        if np.sum(mask) > 0:
            mae_L = np.mean(np.abs(emp_p0_vals[mask] - neural_p0_vals[mask]))
            print(f"  Length {L}: {np.sum(mask)} histories, MAE: {mae_L:.4f}, Avg samples: {np.mean(sample_counts[mask]):.1f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot: Empirical vs Neural P(0)
    axes[0, 0].scatter(emp_p0_vals, neural_p0_vals, alpha=0.6, c=sample_counts, cmap='viridis')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Empirical P(0|history)')
    axes[0, 0].set_ylabel('Neural P(0|history)')
    axes[0, 0].set_title(f'P(0) Comparison (MAE: {mae_p0:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Probability differences vs sample count
    diff_p0 = np.abs(emp_p0_vals - neural_p0_vals)
    axes[0, 1].scatter(sample_counts, diff_p0, alpha=0.6)
    axes[0, 1].set_xlabel('Empirical Sample Count')
    axes[0, 1].set_ylabel('|Empirical - Neural| P(0)')
    axes[0, 1].set_title('Prediction Error vs Sample Size')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error by history length
    length_errors = []
    lengths = []
    for L in range(0, L_max + 1):
        mask = history_lengths == L
        if np.sum(mask) > 0:
            lengths.append(L)
            length_errors.append(np.mean(np.abs(emp_p0_vals[mask] - neural_p0_vals[mask])))
    
    axes[1, 0].bar(lengths, length_errors)
    axes[1, 0].set_xlabel('History Length')
    axes[1, 0].set_ylabel('Mean Absolute Error P(0)')
    axes[1, 0].set_title('Prediction Error vs History Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of errors
    axes[1, 1].hist(diff_p0, bins=30, alpha=0.7, density=True)
    axes[1, 1].set_xlabel('|Empirical - Neural| P(0)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save analysis
    output_dir = Path('probability_analysis')
    output_dir.mkdir(exist_ok=True)
    
    model_name = model_path.stem
    dataset_name = data_path.stem
    
    plt.savefig(output_dir / f'{dataset_name}_{model_name}_prob_comparison.png', dpi=150)
    print(f"\nSaved analysis plots to: {output_dir / f'{dataset_name}_{model_name}_prob_comparison.png'}")
    
    # Save detailed results
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'dataset_size': len(data),
        'unique_histories': len(emp_probs),
        'analyzed_histories': len(emp_p0_vals),
        'mae_p0': mae_p0,
        'mae_p1': mae_p1,
        'mse_p0': mse_p0,
        'mse_p1': mse_p1
    }
    
    return results, emp_probs, neural_probs

def dataset_size_analysis(data_path: Path, model_path: Path, sizes=[1000, 2000, 5000, 10000, 20000, 50000], temperature: float = 1.0):
    """Analyze how dataset size affects empirical probability estimation."""
    
    # Load full data
    full_data = data_path.read_text().strip()
    print(f"Full dataset: {len(full_data)} tokens")
    
    # Load model for neural probabilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    provider = NeuralCSSRProbabilityProvider(model, device, context_window=64)
    
    results_by_size = {}
    
    for size in sizes:
        if size > len(full_data):
            continue
            
        print(f"\nAnalyzing dataset size: {size}")
        data_subset = full_data[:size]
        
        # Get empirical probabilities for this subset
        emp_probs = get_empirical_probabilities(data_subset, L_max=3)  # Use smaller L_max for speed
        
        # Compare with neural probabilities
        mae_values = []
        sample_counts = []
        
        for history in emp_probs:
            if emp_probs[history]['total'] >= 3:  # Minimum samples
                context = [int(c) for c in history] if history else []
                neural_dist = provider.predict_next_distribution(history)
                p0, p1 = _retemper_probs(neural_dist['0'], neural_dist['1'], temperature)
                mae = abs(emp_probs[history]['p0'] - p0)
                mae_values.append(mae)
                sample_counts.append(emp_probs[history]['total'])
        
        if mae_values:
            avg_mae = np.mean(mae_values)
            avg_samples = np.mean(sample_counts)
            num_histories = len(mae_values)
        else:
            avg_mae = float('nan')
            avg_samples = 0
            num_histories = 0
        
        results_by_size[size] = {
            'avg_mae': avg_mae,
            'avg_samples_per_history': avg_samples,
            'num_histories': num_histories,
            'total_histories': len(emp_probs)
        }
        
        print(f"  Unique histories: {len(emp_probs)}, Analyzed: {num_histories}")
        print(f"  Average MAE: {avg_mae:.4f}, Average samples/history: {avg_samples:.1f}")
    
    # Plot results
    sizes_actual = list(results_by_size.keys())
    maes = [results_by_size[s]['avg_mae'] for s in sizes_actual]
    avg_samples = [results_by_size[s]['avg_samples_per_history'] for s in sizes_actual]
    num_histories = [results_by_size[s]['num_histories'] for s in sizes_actual]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE vs dataset size
    axes[0].plot(sizes_actual, maes, 'bo-')
    axes[0].set_xlabel('Dataset Size (tokens)')
    axes[0].set_ylabel('Average MAE P(0)')
    axes[0].set_title('Empirical vs Neural MAE by Dataset Size')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # Average samples per history
    axes[1].plot(sizes_actual, avg_samples, 'ro-')
    axes[1].set_xlabel('Dataset Size (tokens)')
    axes[1].set_ylabel('Average Samples per History')
    axes[1].set_title('Empirical Sample Density')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    # Number of histories
    axes[2].plot(sizes_actual, num_histories, 'go-')
    axes[2].set_xlabel('Dataset Size (tokens)')
    axes[2].set_ylabel('Number of Histories (≥3 samples)')
    axes[2].set_title('Analyzable History Count')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    
    plt.tight_layout()
    
    output_dir = Path('probability_analysis')
    output_dir.mkdir(exist_ok=True)
    
    dataset_name = data_path.stem
    model_name = model_path.stem
    
    plt.savefig(output_dir / f'{dataset_name}_{model_name}_size_analysis.png', dpi=150)
    print(f"\nSaved size analysis plots to: {output_dir / f'{dataset_name}_{model_name}_size_analysis.png'}")
    
    return results_by_size

def main():
    parser = argparse.ArgumentParser(description='Compare empirical vs neural probabilities')
    parser.add_argument('--data', type=Path, required=True, help='Path to dataset (.dat file)')
    parser.add_argument('--model', type=Path, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--L_max', type=int, default=6, help='Maximum history length')
    parser.add_argument('--size_analysis', action='store_true', help='Run dataset size analysis')
    parser.add_argument('--temperature', type=float, default=1.0, help='Retemper neural probabilities before comparison')
    
    args = parser.parse_args()
    
    print("=== Probability Comparison Analysis ===")
    
    # Main comparison
    results, emp_probs, neural_probs = compare_probability_methods(args.data, args.model, args.L_max, temperature=args.temperature)
    
    # Dataset size analysis
    if args.size_analysis:
        print("\n=== Dataset Size Analysis ===")
    size_results = dataset_size_analysis(args.data, args.model, temperature=args.temperature)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()