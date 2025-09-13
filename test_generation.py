#!/usr/bin/env python3
"""
Test sequence generation from trained models to see what they actually produce.
"""

import torch
import argparse
from pathlib import Path
import numpy as np
from collections import Counter

# Import our models
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
    return model, model_type

def generate_sequence(model, model_type, length=1000, context_window=64, temperature=1.0, device='cpu'):
    """Generate a sequence from the model."""
    model.to(device)
    
    # Start with empty context (pad token = 2)
    sequence = [2]  # Start token
    
    with torch.no_grad():
        for _ in range(length):
            # Get current context (last context_window tokens)
            context = sequence[-context_window:]
            context_tensor = torch.tensor([context], dtype=torch.long, device=device)
            
            # Use class-provided probability helper to avoid sign mistakes
            probs = model.generate_probabilities(context_tensor)[0]
            if temperature and temperature != 1.0:
                logits = torch.log(probs + 1e-9) / temperature
                probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            sequence.append(next_token)
    
    # Convert to binary string (remove start token, map 0/1 to '0'/'1')
    binary_sequence = ''.join(str(token) for token in sequence[1:] if token in [0, 1])
    return binary_sequence, sequence

def analyze_golden_mean_violations(sequence: str):
    """Check for Golden Mean violations (consecutive 0s)."""
    violations = []
    for i in range(len(sequence) - 1):
        if sequence[i] == '0' and sequence[i+1] == '0':
            violations.append(i)
    return violations

def analyze_even_process_violations(sequence: str):
    """Check for Even Process violations (0 after odd number of 1s)."""
    violations = []
    ones_count = 0
    
    for i, char in enumerate(sequence):
        if char == '1':
            ones_count += 1
        elif char == '0':
            if ones_count % 2 == 1:  # Odd number of 1s before this 0
                violations.append(i)
            ones_count = 0  # Reset count after 0
            
    return violations

def analyze_sequence(sequence: str, machine_type: str):
    """Analyze generated sequence for machine-specific properties."""
    print(f"\n=== Sequence Analysis ({machine_type}) ===")
    print(f"Length: {len(sequence)}")
    print(f"First 100 chars: {sequence[:100]}")
    print(f"Last 100 chars: {sequence[-100:]}")
    
    # Basic statistics
    count_0 = sequence.count('0')
    count_1 = sequence.count('1')
    print(f"\nSymbol counts:")
    print(f"  '0': {count_0} ({count_0/len(sequence)*100:.1f}%)")
    print(f"  '1': {count_1} ({count_1/len(sequence)*100:.1f}%)")
    
    # Pattern analysis
    patterns = {}
    for length in [1, 2, 3, 4]:
        pattern_counts = Counter()
        for i in range(len(sequence) - length + 1):
            pattern = sequence[i:i+length]
            pattern_counts[pattern] += 1
        
        print(f"\nLength-{length} patterns:")
        for pattern, count in pattern_counts.most_common(10):
            prob = count / (len(sequence) - length + 1)
            print(f"  '{pattern}': {count} ({prob:.3f})")
        patterns[length] = pattern_counts
    
    # Machine-specific violation analysis
    if machine_type == 'golden_mean':
        violations = analyze_golden_mean_violations(sequence)
        print(f"\nGolden Mean violations (consecutive 0s): {len(violations)}")
        if violations:
            print(f"  First 10 violation positions: {violations[:10]}")
            violation_rate = len(violations) / len(sequence)
            print(f"  Violation rate: {violation_rate:.4f}")
        else:
            print("  No violations found!")
            
    elif machine_type == 'even_process':
        violations = analyze_even_process_violations(sequence)
        print(f"\nEven Process violations (0 after odd 1s): {len(violations)}")
        if violations:
            print(f"  First 10 violation positions: {violations[:10]}")
            violation_rate = len(violations) / len(sequence)
            print(f"  Violation rate: {violation_rate:.4f}")
        else:
            print("  No violations found!")
    
    # Check specific critical contexts
    print(f"\nCritical context analysis:")
    if machine_type == 'golden_mean':
        # Count what happens after '0' (should always be '1')
        after_0_count = {'0': 0, '1': 0}
        for i in range(len(sequence) - 1):
            if sequence[i] == '0':
                after_0_count[sequence[i+1]] += 1
        
        total_after_0 = sum(after_0_count.values())
        if total_after_0 > 0:
            print(f"  After '0': {after_0_count}")
            print(f"  P(1|0) = {after_0_count['1']/total_after_0:.3f} (should be 1.0)")
            print(f"  P(0|0) = {after_0_count['0']/total_after_0:.3f} (should be 0.0)")
        
    return patterns

def main():
    parser = argparse.ArgumentParser(description='Test sequence generation from trained models')
    parser.add_argument('--model', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--machine', required=True, choices=['golden_mean', 'even_process'], 
                       help='Machine type for violation analysis')
    parser.add_argument('--length', type=int, default=10000, help='Length of sequence to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"=== Sequence Generation Test ===")
    print(f"Model: {args.model}")
    print(f"Machine: {args.machine}")
    print(f"Length: {args.length}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {device}")
    
    # Load model
    model, model_type = load_model(args.model, device)
    
    # Generate sequence
    print(f"\nGenerating sequence...")
    sequence, raw_sequence = generate_sequence(
        model, model_type, args.length, temperature=args.temperature, device=device
    )
    
    print(f"Generated {len(sequence)} binary symbols")
    
    # Analyze sequence
    patterns = analyze_sequence(sequence, args.machine)
    
    # Save sequence for further analysis
    output_file = f"generated_{args.machine}_{args.model.stem}_{args.length}.dat"
    with open(output_file, 'w') as f:
        f.write(sequence)
    print(f"\nSaved generated sequence to: {output_file}")

if __name__ == '__main__':
    main()