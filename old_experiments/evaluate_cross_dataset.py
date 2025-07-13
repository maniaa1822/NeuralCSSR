#!/usr/bin/env python3
"""
Evaluate a trained model on a different dataset to test cross-dataset generalization.
"""

import argparse
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path to import neural_cssr
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import from the main training script
from time_delay_transformer import (
    SequenceDataset, 
    BinaryTransformer, 
    collate_fn_ar, 
    collate_fn_time_delay,
    evaluate,
    show_predictions
)


def main():
    p = argparse.ArgumentParser(description="Evaluate trained model on different dataset")
    p.add_argument('--model', type=Path, required=True, help='Path to saved model (.pt file)')
    p.add_argument('--test_data', type=Path, required=True, help='Path to test dataset (.pt file)')
    p.add_argument('--mode', choices=['ar', 'td'], default='ar', help='Model mode (autoregressive or time-delay)')
    p.add_argument('--delay', type=int, default=10, help='k for time-delay mode')
    p.add_argument('--d_model', type=int, default=16, help='Model dimension')
    p.add_argument('--layers', type=int, default=1, help='Number of layers')
    p.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    p.add_argument('--batch', type=int, default=128, help='Batch size')
    p.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CROSS-DATASET EVALUATION")
    print("=" * 70)
    print(f"Model file: {args.model}")
    print(f"Test dataset: {args.test_data}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'td':
        print(f"Time delay (k): {args.delay}")
    print(f"Model config: d_model={args.d_model}, layers={args.layers}, heads={args.heads}")
    print(f"Device: {device}")
    print()

    # Load test dataset
    print("Loading test dataset...")
    test_ds = SequenceDataset(args.test_data)
    print(f"Test sequences: {len(test_ds)}")
    
    # Show sequence statistics
    test_lengths = [len(seq) for seq in test_ds.data[:1000]]  # Sample first 1000
    print(f"Test seq lengths - min: {min(test_lengths)}, max: {max(test_lengths)}, "
          f"avg: {sum(test_lengths)/len(test_lengths):.1f}")
    
    # Analyze token distribution
    total_tokens = sum(len(seq) for seq in test_ds.data)
    total_zeros = sum((seq == 0).sum().item() for seq in test_ds.data)
    total_ones = sum((seq == 1).sum().item() for seq in test_ds.data)
    zero_percent = (total_zeros / total_tokens * 100) if total_tokens > 0 else 0
    one_percent = (total_ones / total_tokens * 100) if total_tokens > 0 else 0
    
    print(f"Token distribution - Zeros: {total_zeros:,} ({zero_percent:.1f}%), "
          f"Ones: {total_ones:,} ({one_percent:.1f}%)")
    print(f"Zero bias ratio: {total_zeros/total_ones:.2f}:1" if total_ones > 0 else "N/A")
    print()

    # Setup data loader
    if args.mode == 'td':
        collate = lambda batch: collate_fn_time_delay(batch, args.delay)
    else:
        collate = collate_fn_ar
    
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    print(f"Test batches: {len(test_loader)}")
    print()

    # Initialize model architecture (same as training)
    print("Initializing model architecture...")
    model = BinaryTransformer(
        vocab_size=3, 
        d_model=args.d_model, 
        n_layers=args.layers,
        n_heads=args.heads,
        max_delay=args.delay if args.mode == 'td' else None
    )
    
    # Load trained weights
    print(f"Loading trained weights from {args.model}...")
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Evaluate
    print("Evaluating model...")
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding token
    
    model.eval()
    with torch.no_grad():
        metrics = evaluate(model, test_loader, device, loss_fn)
    
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print()
    print("Per-Token Results:")
    print(f"  Token '0': {metrics['token_0_acc']:.4f} ({metrics['token_0_acc']*100:.2f}%) "
          f"[n={metrics['token_0_count']:,}]")
    print(f"  Token '1': {metrics['token_1_acc']:.4f} ({metrics['token_1_acc']*100:.2f}%) "
          f"[n={metrics['token_1_count']:,}]")
    print(f"  Total tokens: {metrics['total_tokens']:,}")
    print()
    
    # Show some prediction examples
    print("Sample Predictions:")
    print("-" * 50)
    show_predictions(model, test_loader, device, num_examples=5)
    
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
