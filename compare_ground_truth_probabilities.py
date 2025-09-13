#!/usr/bin/env python3
"""
Compare neural probabilities against GROUND TRUTH machine probabilities.
This provides the definitive accuracy assessment since we know the true generative process.
"""

import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import sys

# Import our models and utilities
from neural_cssr_utils import NeuralCSSRProbabilityProvider

def _retemper_probs(p0: float, p1: float, temperature: float) -> tuple[float, float]:
    if temperature and temperature != 1.0:
        import math
        log0 = math.log(max(p0, 1e-9)) / temperature
        log1 = math.log(max(p1, 1e-9)) / temperature
        m = max(log0, log1)
        e0 = math.exp(log0 - m)
        e1 = math.exp(log1 - m)
        s = e0 + e1
        return e0 / s, e1 / s
    return p0, p1
from experiments.ebm.models import EnergyBasedBinaryLM, AutoRegressiveBinaryLM

# Import ground truth machines
from pysm_generator import create_machine

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

def get_ground_truth_probabilities(machine_type: str, histories: list, seed: int = 42):
    """Get ground truth probabilities from the actual generative machine."""
    
    # Create ground truth machine
    machine = create_machine(machine_type, seed=seed)
    
    gt_probs = {}
    
    for history in histories:
        # For each history, we need to simulate the machine to that state
        # and get the true emission probabilities
        
        if machine_type == 'golden_mean':
            # Golden Mean: A state (50% 0, 50% 1), B state (0% 0, 100% 1)
            gt_probs[history] = get_golden_mean_prob(history)
            
        elif machine_type == 'even_process':
            # Even Process: E state (50% 0, 50% 1), O state (0% 0, 100% 1)
            gt_probs[history] = get_even_process_prob(history)
            
        else:
            # For other machines, use simulation approach
            gt_probs[history] = simulate_machine_probability(machine, history, seed)
    
    return gt_probs

def get_golden_mean_prob(history: str):
    """Get exact Golden Mean probabilities for a given history."""
    if len(history) == 0:
        # Starting state A: 50% 0, 50% 1
        return {'p0': 0.5, 'p1': 0.5}
    
    # Trace through the Golden Mean states
    state = 'A'  # Start in state A
    
    for symbol in history:
        if state == 'A':
            if symbol == '0':
                state = 'B'  # A --0--> B
            else:  # symbol == '1'
                state = 'A'  # A --1--> A
        elif state == 'B':
            if symbol == '1':
                state = 'A'  # B --1--> A
            # Note: B --0--> is impossible in Golden Mean
    
    # Return probabilities based on final state
    if state == 'A':
        return {'p0': 0.5, 'p1': 0.5}  # State A: 50% 0, 50% 1
    elif state == 'B':
        return {'p0': 0.0, 'p1': 1.0}  # State B: 0% 0, 100% 1
    else:
        return {'p0': 0.5, 'p1': 0.5}  # Fallback

def get_even_process_prob(history: str):
    """Get exact Even Process probabilities for a given history."""
    if len(history) == 0:
        # Starting state E: 50% 0, 50% 1
        return {'p0': 0.5, 'p1': 0.5}
    
    # Count number of 1s to determine parity
    # Even Process: runs of 1s must have even length
    # E state: even number of 1s in current run, can emit 0 or 1
    # O state: odd number of 1s in current run, must emit 1
    
    ones_in_current_run = 0
    for i in range(len(history) - 1, -1, -1):
        if history[i] == '1':
            ones_in_current_run += 1
        else:
            break
    
    # Determine state based on parity of current run of 1s
    if ones_in_current_run % 2 == 0:
        # Even parity -> E state
        return {'p0': 0.5, 'p1': 0.5}  # E: 50% 0, 50% 1
    else:
        # Odd parity -> O state  
        return {'p0': 0.0, 'p1': 1.0}  # O: 0% 0, 100% 1

def simulate_machine_probability(machine, history: str, seed: int, num_samples: int = 10000):
    """Simulate machine to estimate probabilities at a given history."""
    # This is a fallback for complex machines
    # Reset machine and simulate the history sequence many times
    
    counts = {'0': 0, '1': 0}
    
    for _ in range(num_samples):
        # Reset machine
        test_machine = create_machine(machine.__class__.__name__.replace('Machine', '').lower(), seed=seed + _)
        
        # Play through the history
        for symbol in history:
            test_machine.step()
        
        # Get next emission
        next_symbol = test_machine.step()
        counts[next_symbol] += 1
    
    total = sum(counts.values())
    return {'p0': counts['0'] / total, 'p1': counts['1'] / total}

def compare_against_ground_truth(machine_type: str, model_path: Path, data_path: Path = None, L_max: int = 6, temperature: float = 1.0):
    """Compare neural model against ground truth machine probabilities."""
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    provider = NeuralCSSRProbabilityProvider(model, device, context_window=64)
    
    print(f"Comparing {model_path.stem} against ground truth {machine_type}")
    
    # Generate test histories - we'll use systematic enumeration for short histories
    # and some data-driven longer histories if available
    test_histories = []
    
    # Short systematic histories (all possible combinations up to length 3)
    for length in range(0, min(4, L_max + 1)):
        if length == 0:
            test_histories.append('')
        else:
            for i in range(2**length):
                binary_str = format(i, f'0{length}b')
                test_histories.append(binary_str)
    
    # If data is provided, add some empirically observed longer histories
    if data_path and data_path.exists():
        data = data_path.read_text().strip()
        history_counts = Counter()
        
        # Extract histories of various lengths from data
        for t in range(len(data)):
            for L in range(4, min(L_max + 1, t + 1)):
                history = data[t-L:t]
                history_counts[history] += 1
        
        # Add most common longer histories
        common_histories = [h for h, count in history_counts.most_common(20) if len(h) >= 4]
        test_histories.extend(common_histories[:10])  # Top 10 longer histories
    
    print(f"Testing {len(test_histories)} histories")
    
    # Get ground truth probabilities
    print("Computing ground truth probabilities...")
    gt_probs = get_ground_truth_probabilities(machine_type, test_histories)
    
    # Get neural probabilities
    print("Computing neural probabilities...")
    neural_probs = {}
    for history in test_histories:
        context = [int(c) for c in history] if history else []
        dist = provider.predict_next_distribution(history)
        p0, p1 = _retemper_probs(dist['0'], dist['1'], temperature)
        neural_probs[history] = {'p0': p0, 'p1': p1}
    
    # Compare results
    gt_p0_vals = []
    neural_p0_vals = []
    gt_p1_vals = []
    neural_p1_vals = []
    history_lengths = []
    history_list = []
    
    for history in test_histories:
        gt_p0_vals.append(gt_probs[history]['p0'])
        neural_p0_vals.append(neural_probs[history]['p0'])
        gt_p1_vals.append(gt_probs[history]['p1'])
        neural_p1_vals.append(neural_probs[history]['p1'])
        history_lengths.append(len(history))
        history_list.append(history if history else 'ε')  # ε for empty string
    
    gt_p0_vals = np.array(gt_p0_vals)
    neural_p0_vals = np.array(neural_p0_vals)
    gt_p1_vals = np.array(gt_p1_vals)
    neural_p1_vals = np.array(neural_p1_vals)
    history_lengths = np.array(history_lengths)
    
    # Compute errors
    mae_p0 = np.mean(np.abs(gt_p0_vals - neural_p0_vals))
    mae_p1 = np.mean(np.abs(gt_p1_vals - neural_p1_vals))
    mse_p0 = np.mean((gt_p0_vals - neural_p0_vals) ** 2)
    mse_p1 = np.mean((gt_p1_vals - neural_p1_vals) ** 2)
    
    # Maximum errors
    max_error_p0 = np.max(np.abs(gt_p0_vals - neural_p0_vals))
    max_error_p1 = np.max(np.abs(gt_p1_vals - neural_p1_vals))
    
    print(f"\n=== Ground Truth vs Neural Comparison ===")
    print(f"Model: {model_path.stem}")
    print(f"Machine: {machine_type}")
    print(f"Histories tested: {len(test_histories)}")
    print(f"\nAccuracy Metrics:")
    print(f"  P(0|history) - MAE: {mae_p0:.4f}, MSE: {mse_p0:.4f}, Max Error: {max_error_p0:.4f}")
    print(f"  P(1|history) - MAE: {mae_p1:.4f}, MSE: {mse_p1:.4f}, Max Error: {max_error_p1:.4f}")
    print(f"  Overall MAE: {(mae_p0 + mae_p1)/2:.4f}")
    
    # Find worst predictions
    errors_p0 = np.abs(gt_p0_vals - neural_p0_vals)
    worst_idx = np.argmax(errors_p0)
    print(f"\nWorst P(0) prediction:")
    print(f"  History: '{history_list[worst_idx]}'")
    print(f"  Ground Truth: {gt_p0_vals[worst_idx]:.4f}, Neural: {neural_p0_vals[worst_idx]:.4f}")
    print(f"  Error: {errors_p0[worst_idx]:.4f}")
    
    # Analysis by history length
    print(f"\nAccuracy by History Length:")
    for L in range(0, max(history_lengths) + 1):
        mask = history_lengths == L
        if np.sum(mask) > 0:
            mae_L = np.mean(np.abs(gt_p0_vals[mask] - neural_p0_vals[mask]))
            count_L = np.sum(mask)
            print(f"  Length {L}: {count_L} histories, MAE: {mae_L:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot: Ground Truth vs Neural P(0)
    axes[0, 0].scatter(gt_p0_vals, neural_p0_vals, alpha=0.7, c=history_lengths, cmap='viridis')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Agreement')
    axes[0, 0].set_xlabel('Ground Truth P(0|history)')
    axes[0, 0].set_ylabel('Neural P(0|history)')
    axes[0, 0].set_title(f'{machine_type.title()} - P(0) Comparison\nMAE: {mae_p0:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Error vs Ground Truth probability
    axes[0, 1].scatter(gt_p0_vals, errors_p0, alpha=0.7, c=history_lengths, cmap='viridis')
    axes[0, 1].set_xlabel('Ground Truth P(0|history)')
    axes[0, 1].set_ylabel('|Ground Truth - Neural| P(0)')
    axes[0, 1].set_title('Prediction Error vs Ground Truth')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error by history length
    length_errors = []
    lengths = []
    for L in range(0, max(history_lengths) + 1):
        mask = history_lengths == L
        if np.sum(mask) > 0:
            lengths.append(L)
            length_errors.append(np.mean(np.abs(gt_p0_vals[mask] - neural_p0_vals[mask])))
    
    axes[1, 0].bar(lengths, length_errors, alpha=0.7)
    axes[1, 0].set_xlabel('History Length')
    axes[1, 0].set_ylabel('Mean Absolute Error P(0)')
    axes[1, 0].set_title('Prediction Error vs History Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distribution of errors
    axes[1, 1].hist(errors_p0, bins=20, alpha=0.7, density=True, edgecolor='black')
    axes[1, 1].axvline(mae_p0, color='red', linestyle='--', label=f'Mean: {mae_p0:.3f}')
    axes[1, 1].set_xlabel('|Ground Truth - Neural| P(0)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path('ground_truth_analysis')
    output_dir.mkdir(exist_ok=True)
    
    model_name = model_path.stem
    
    plt.savefig(output_dir / f'{machine_type}_{model_name}_gt_comparison.png', dpi=150)
    print(f"\nSaved analysis plots to: {output_dir / f'{machine_type}_{model_name}_gt_comparison.png'}")
    
    # Save detailed results
    results = {
        'machine_type': machine_type,
        'model': model_name,
        'histories_tested': len(test_histories),
        'mae_p0': float(mae_p0),
        'mae_p1': float(mae_p1),
        'mse_p0': float(mse_p0),
        'mse_p1': float(mse_p1),
        'max_error_p0': float(max_error_p0),
        'max_error_p1': float(max_error_p1),
        'overall_mae': float((mae_p0 + mae_p1)/2),
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare neural probabilities against ground truth')
    parser.add_argument('--machine', required=True, choices=['golden_mean', 'even_process'], 
                       help='Machine type')
    parser.add_argument('--model', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=Path, help='Optional path to data file for longer histories')
    parser.add_argument('--L_max', type=int, default=6, help='Maximum history length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Retemper neural probabilities before GT comparison')
    
    args = parser.parse_args()
    
    print("=== Ground Truth vs Neural Probability Analysis ===")
    
    results = compare_against_ground_truth(args.machine, args.model, args.data, args.L_max, temperature=args.temperature)
    
    print(f"\nFinal Results Summary:")
    print(f"  Machine: {results['machine_type']}")
    print(f"  Model: {results['model']}")
    print(f"  Overall MAE: {results['overall_mae']:.4f} ({results['overall_mae']*100:.2f}%)")
    print(f"  Max Error: {max(results['max_error_p0'], results['max_error_p1']):.4f}")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()