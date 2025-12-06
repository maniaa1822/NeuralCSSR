"""
Evaluate next-token prediction quality for nanoGPT model on gm_seven_union dataset.
"""
import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Import nanoGPT model
import sys
sys.path.insert(0, str(Path(__file__).parent / 'nanoGPT'))
from model import GPTConfig, GPT


def load_model(ckpt_path, device='cuda'):
    """Load nanoGPT model from checkpoint."""
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Create model
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)

    # Load state dict (handle compiled models)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print(f"Model config: {checkpoint['model_args']}")
    if 'iter_num' in checkpoint:
        print(f"Checkpoint iteration: {checkpoint['iter_num']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f} nats = {checkpoint['best_val_loss'] / np.log(2):.4f} bits")

    return model, checkpoint


def load_dataset(data_dir):
    """Load dataset and ground truth labels."""
    data_dir = Path(data_dir)

    # Load binary sequence
    dat_file = data_dir / 'combined.dat'
    with open(dat_file, 'r') as f:
        raw = f.read()
    data = ''.join(ch for ch in raw if ch in ('0', '1'))

    print(f"Dataset length: {len(data):,} symbols")

    # Load metadata
    meta_file = data_dir / 'combined.meta.json'
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    # Load machine IDs (which machine generated each symbol)
    machine_ids_file = data_dir / 'combined.machine_ids.dat'
    with open(machine_ids_file, 'r') as f:
        machine_ids_raw = f.read()

    # Parse machine IDs (format: "golden_mean\ngolden_mean\n...")
    machine_ids = []
    for line in machine_ids_raw.strip().split('\n'):
        if line.strip():
            machine_ids.append(line.strip())

    print(f"Loaded {len(machine_ids):,} machine ID labels")

    return data, machine_ids, meta


def evaluate_predictions(model, data, context_window=128, n_samples=2000, device='cuda'):
    """
    Evaluate next-token prediction quality on sampled positions.

    Returns dict with:
        - losses: array of per-position cross-entropy losses (nats)
        - correct: array of per-position correctness (0 or 1)
        - probs: array of per-position predicted probability on correct token
    """
    # Encode data
    stoi = {'0': 0, '1': 1}
    tokens = torch.tensor([stoi[c] for c in data], dtype=torch.long)

    n = len(tokens)

    # Sample representative positions (skip first context_window to ensure we have context)
    np.random.seed(42)
    sample_positions = np.random.choice(
        range(context_window, n),
        size=min(n_samples, n - context_window),
        replace=False
    )
    sample_positions = sorted(sample_positions)

    print(f"  Evaluating {len(sample_positions):,} sampled positions (out of {n:,} total)...")

    # Storage for results
    losses = []
    correct = []
    probs = []

    # Evaluate in batches
    model.eval()
    with torch.no_grad():
        for i, pos in enumerate(sample_positions):
            # Get context (up to context_window tokens before current position)
            context_start = max(0, pos - context_window)
            context = tokens[context_start:pos].unsqueeze(0).to(device)  # [1, seq_len]

            # Get target
            target = tokens[pos].item()

            # Forward pass
            logits, _, _ = model(context)  # [1, 1, vocab_size]

            # Get prediction for last position in context
            last_logits = logits[0, 0, :]  # [vocab_size]

            # Compute loss
            loss = F.cross_entropy(last_logits.unsqueeze(0), torch.tensor([target], device=device))
            losses.append(loss.item())

            # Compute accuracy
            pred = last_logits.argmax().item()
            correct.append(1 if pred == target else 0)

            # Get probability of correct token
            probs_dist = F.softmax(last_logits, dim=0)
            probs.append(probs_dist[target].item())

            # Progress
            if (i + 1) % 500 == 0:
                print(f"    {i + 1:,} / {len(sample_positions):,} samples...")

    return {
        'losses': np.array(losses),
        'correct': np.array(correct),
        'probs': np.array(probs),
        'sample_positions': sample_positions,
    }


def compute_metrics(losses, correct, probs):
    """Compute summary metrics from prediction arrays."""
    return {
        'loss_nats': float(np.mean(losses)),
        'loss_bits': float(np.mean(losses) / np.log(2)),
        'accuracy': float(np.mean(correct)),
        'mean_prob_correct': float(np.mean(probs)),
        'num_predictions': len(losses),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate next-token prediction quality')
    parser.add_argument('--model_ckpt', type=str,
                        default='nanoGPT/out-gm-seven-union-char/ckpt.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                        default='experiments/datasets/gm_seven_union',
                        help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_json', type=str,
                        default='results/gm_seven_union_prediction.json',
                        help='Output JSON file')
    parser.add_argument('--n_samples', type=int, default=2000,
                        help='Number of positions to sample for evaluation')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # Load model
    model, checkpoint = load_model(args.model_ckpt, args.device)
    context_window = checkpoint['model_args']['block_size']

    # Load dataset
    data, machine_ids, meta = load_dataset(args.data_dir)

    print("\n" + "="*80)
    print("EVALUATING NEXT-TOKEN PREDICTIONS")
    print("="*80)

    # Overall evaluation
    print("\nEvaluating overall predictions...")
    results = evaluate_predictions(model, data, context_window=context_window,
                                   n_samples=args.n_samples, device=args.device)

    overall_metrics = compute_metrics(results['losses'], results['correct'], results['probs'])

    # Per-machine evaluation
    print("\nBreaking down by machine...")
    machine_metrics = {}

    # Get machine boundaries from metadata
    for segment in meta['segments']:
        machine_name = segment['machine']
        start = segment['start']
        end = segment['end']

        # Find which sampled positions fall in this segment
        segment_mask = [(start <= pos < end) for pos in results['sample_positions']]

        if not any(segment_mask):
            print(f"\n{machine_name}: No samples in this segment")
            continue

        segment_losses = results['losses'][segment_mask]
        segment_correct = results['correct'][segment_mask]
        segment_probs = results['probs'][segment_mask]

        machine_metrics[machine_name] = compute_metrics(segment_losses, segment_correct, segment_probs)

        print(f"\n{machine_name}:")
        print(f"  Loss: {machine_metrics[machine_name]['loss_bits']:.4f} bits")
        print(f"  Accuracy: {machine_metrics[machine_name]['accuracy']*100:.2f}%")
        print(f"  Samples: {machine_metrics[machine_name]['num_predictions']}")

    # Collect baselines
    baselines = {
        'theoretical_optimal': {
            'loss_bits': meta['expected_optimal_loss']['bits'],
            'source': 'Expected optimal (entropy of belief machine)',
        },
        'markov_L8': {
            'loss_bits': meta['markov_baseline']['avg_loss_per_symbol_bits'],
            'source': 'Markov-L=8 baseline',
        },
        'uniform_random': {
            'loss_bits': 1.0,
            'source': 'Uniform random (log2(2) = 1 bit)',
        },
    }

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOverall Performance:")
    print(f"  Loss: {overall_metrics['loss_bits']:.4f} bits")
    print(f"  Accuracy: {overall_metrics['accuracy']*100:.2f}%")
    print(f"  Mean probability on correct token: {overall_metrics['mean_prob_correct']:.4f}")

    print(f"\nBaseline Comparison:")
    print(f"  Theoretical optimal: {baselines['theoretical_optimal']['loss_bits']:.4f} bits")
    print(f"  Markov-L=8 baseline: {baselines['markov_L8']['loss_bits']:.4f} bits")
    print(f"  Uniform random: {baselines['uniform_random']['loss_bits']:.4f} bits")

    gap_to_optimal = overall_metrics['loss_bits'] - baselines['theoretical_optimal']['loss_bits']
    print(f"\nGap to optimal: {gap_to_optimal:.4f} bits ({gap_to_optimal/baselines['theoretical_optimal']['loss_bits']*100:.1f}% excess)")

    # Save results
    output = {
        'model_checkpoint': args.model_ckpt,
        'dataset': args.data_dir,
        'overall': overall_metrics,
        'per_machine': machine_metrics,
        'baselines': baselines,
        'gap_to_optimal_bits': float(gap_to_optimal),
    }

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output_json}")


if __name__ == '__main__':
    main()
