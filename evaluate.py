#!/usr/bin/env python3
"""
Evaluation script for trained transformer models.

This script handles model evaluation with support for:
- Loading checkpoints and evaluating on test data
- Generating sequences from trained models
- Computing detailed metrics and analysis
- Batch evaluation across multiple datasets

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --test domain_machines/golden_mean/golden_mean.dat
    python evaluate.py --checkpoint checkpoints/best.pt --generate --length 1000
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn

from models.transformer_models import create_model
from models.data_utils import create_dataloaders, load_sequence_file


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    model_config = checkpoint['model_config'].copy()
    
    # Map full class names to short names for create_model
    model_type_mapping = {
        'StreamlinedTransformer': 'streamlined',
        'TimeDelayTransformer': 'time_delay',
        'SlidingWindowTransformer': 'sliding_window'
    }
    
    if model_config['model_type'] in model_type_mapping:
        model_config['model_type'] = model_type_mapping[model_config['model_type']]
    
    model = create_model(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model: {model_config['model_type']}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, checkpoint


def evaluate_on_dataset(model, test_path: Path, device: torch.device, 
                       batch_size: int = 32, chunk_size: int = 25) -> Dict[str, float]:
    """Evaluate model on a test dataset."""
    model.eval()
    
    # Create test dataloader
    _, test_loader = create_dataloaders(
        test_path, test_path,  # Use same file for train/test to get full dataset
        batch_size=batch_size,
        chunk_size=chunk_size,
        test_split=0.0  # Use full dataset for evaluation
    )
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in test_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == target_ids).sum().item()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'num_batches': len(test_loader)
    }


def generate_sequence(model, device: torch.device, length: int = 1000, 
                     seed_sequence: Optional[str] = None, temperature: float = 1.0) -> str:
    """Generate a sequence from the model."""
    model.eval()
    
    # Initialize with seed or random start
    if seed_sequence:
        tokens = [int(c) for c in seed_sequence if c in '01']
    else:
        # Use a more interesting seed from the dataset
        tokens = [0, 1, 1, 1, 0, 0, 0, 1]  # From actual seven-state data
    
    generated = []
    
    with torch.no_grad():
        for _ in range(length):
            # Convert to tensor and add batch dimension
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Get model predictions
            logits = model(input_ids)[:, -1, :]  # Last position predictions
            
            # Filter out padding token (only consider 0 and 1)
            logits_binary = logits[:, :2]  # Only tokens 0 and 1
            
            # Apply temperature and sample
            if temperature > 0:
                logits_binary = logits_binary / temperature
                probs = torch.softmax(logits_binary, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits_binary.argmax(dim=-1).item()
            
            # Add to generated sequence
            generated.append(str(next_token))
            tokens.append(next_token)
            
            # Keep a sliding context roughly matching training window
            window_size = getattr(getattr(model, 'window_size', None), 'value', None)
            if window_size is None:
                window_size = getattr(model, 'window_size', 25)
            if len(tokens) > max(2 * window_size, 50):
                tokens = tokens[-window_size:]
    
    return ''.join(generated)


def analyze_sequence_patterns(sequence: str) -> Dict[str, Any]:
    """Analyze patterns in a binary sequence."""
    if not sequence:
        return {}
    
    # Basic statistics
    total_length = len(sequence)
    ones_count = sequence.count('1')
    zeros_count = sequence.count('0')
    
    # Transition counts
    transitions = {'00': 0, '01': 0, '10': 0, '11': 0}
    for i in range(len(sequence) - 1):
        bigram = sequence[i:i+2]
        if bigram in transitions:
            transitions[bigram] += 1
    
    # Pattern detection
    patterns = {}
    for pattern_len in [2, 3, 4]:
        pattern_counts = {}
        for i in range(len(sequence) - pattern_len + 1):
            pattern = sequence[i:i+pattern_len]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Top 5 most common patterns
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        patterns[f'top_{pattern_len}grams'] = top_patterns
    
    return {
        'length': total_length,
        'symbol_counts': {'0': zeros_count, '1': ones_count},
        'symbol_probs': {'0': zeros_count/total_length, '1': ones_count/total_length},
        'transitions': transitions,
        'transition_probs': {k: v/sum(transitions.values()) for k, v in transitions.items()},
        'patterns': patterns
    }


def batch_evaluate(checkpoint_path: Path, test_datasets: List[Path], 
                  device: torch.device) -> Dict[str, Dict[str, float]]:
    """Evaluate model on multiple datasets."""
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    
    results = {}
    
    for test_path in test_datasets:
        print(f"\nEvaluating on {test_path.name}...")
        
        try:
            metrics = evaluate_on_dataset(model, test_path, device)
            results[test_path.name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Perplexity: {metrics['perplexity']:.3f}")
            print(f"  Tokens: {metrics['total_tokens']:,}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[test_path.name] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained transformer models')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint to evaluate')
    
    # Evaluation modes
    parser.add_argument('--test', type=Path, help='Test dataset (.dat file)')
    parser.add_argument('--generate', action='store_true', help='Generate sequences from model')
    parser.add_argument('--batch_eval', nargs='+', type=Path, help='Evaluate on multiple datasets')
    
    # Generation parameters
    parser.add_argument('--length', type=int, default=1000, help='Length of generated sequence')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--seed', type=str, help='Seed sequence for generation')
    
    # Other parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--chunk_size', type=int, default=25, help='Sequence chunk size')
    parser.add_argument('--output', type=Path, help='Output file for results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("📊 NEURAL CSSR MODEL EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print()
    
    results = {}
    
    # Single dataset evaluation
    if args.test:
        print(f"Evaluating on {args.test}...")
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        
        metrics = evaluate_on_dataset(model, args.test, device, args.batch_size, args.chunk_size)
        results['evaluation'] = metrics
        
        print(f"Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Perplexity: {metrics['perplexity']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Tokens evaluated: {metrics['total_tokens']:,}")
    
    # Batch evaluation
    if args.batch_eval:
        print("Batch evaluation on multiple datasets...")
        batch_results = batch_evaluate(args.checkpoint, args.batch_eval, device)
        results['batch_evaluation'] = batch_results
    
    # Sequence generation
    if args.generate:
        print(f"Generating sequence of length {args.length}...")
        
        if 'model' not in locals():
            model, checkpoint = load_checkpoint(args.checkpoint, device)
        
        generated = generate_sequence(
            model, device, args.length, args.seed, args.temperature
        )
        
        # Analyze generated sequence
        analysis = analyze_sequence_patterns(generated)
        
        results['generation'] = {
            'sequence': generated,
            'analysis': analysis,
            'parameters': {
                'length': args.length,
                'temperature': args.temperature,
                'seed': args.seed
            }
        }
        
        print(f"Generated sequence (first 100 chars): {generated[:100]}...")
        print(f"Symbol distribution: 0={analysis['symbol_probs']['0']:.3f}, 1={analysis['symbol_probs']['1']:.3f}")
        print(f"Top patterns: {analysis['patterns']['top_3grams'][:3]}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()