#!/usr/bin/env python3
"""
Test sliding window generation with existing trained model.

This tests whether maintaining a consistent context window during generation
improves long sequence quality without retraining the model.
"""

import torch
from pathlib import Path
from models.transformer_models import create_model

def sliding_window_generate(model, device, length=1000, window_size=25, 
                           seed_sequence=None, temperature=1.0):
    """
    Generate using sliding window approach with existing model.
    
    Key insight: Always provide the model with context lengths it saw during training.
    """
    model.eval()
    
    # Initialize with seed
    if seed_sequence:
        tokens = [int(c) for c in seed_sequence if c in '01']
    else:
        # Use pattern from actual training data
        tokens = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    
    generated = []
    
    print(f"🎯 Sliding Window Generation (window={window_size})")
    print(f"Initial context: {''.join(map(str, tokens))}")
    
    with torch.no_grad():
        for step in range(length):
            # Key insight: Always use exactly (window_size-1) tokens as context
            # This matches the training setup where we predict token at position window_size-1
            if len(tokens) >= window_size:
                current_context = tokens[-(window_size-1):]
            else:
                current_context = tokens
            
            # Convert to tensor
            input_ids = torch.tensor([current_context], dtype=torch.long, device=device)
            
            # Get predictions
            logits = model(input_ids)[:, -1, :2]  # Last position, only binary tokens
            
            # Sample next token
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = logits.argmax(dim=-1).item()
            
            # Add to generated sequence
            generated.append(str(next_token))
            tokens.append(next_token)
            
            # Progress indicator
            if (step + 1) % 200 == 0:
                print(f"  Generated {step + 1}/{length} tokens...")
                # Show current pattern
                recent = ''.join(generated[-50:]) if len(generated) >= 50 else ''.join(generated)
                print(f"  Recent: ...{recent}")
    
    return ''.join(generated)

def compare_generation_methods(checkpoint_path, test_cases):
    """Compare different generation approaches."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint['model_config'].copy()
    
    if model_config['model_type'] == 'StreamlinedTransformer':
        model_config['model_type'] = 'streamlined'
    
    model = create_model(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("🔍 Comparing Generation Methods")
    print("=" * 70)
    
    for test_name, params in test_cases.items():
        print(f"\n📊 Test Case: {test_name}")
        print("-" * 50)
        
        # Generate with sliding window
        generated = sliding_window_generate(
            model, device, 
            length=params['length'],
            window_size=params.get('window_size', 25),
            seed_sequence=params.get('seed'),
            temperature=params.get('temperature', 1.0)
        )
        
        # Analyze results
        zeros = generated.count('0')
        ones = generated.count('1')
        total = len(generated)
        
        print(f"\n📈 Results:")
        print(f"  Length: {total}")
        print(f"  Distribution: 0={zeros} ({zeros/total:.3f}), 1={ones} ({ones/total:.3f})")
        print(f"  First 100: {generated[:100]}")
        print(f"  Last 100:  {generated[-100:]}")
        
        # Pattern analysis
        transitions = {'00': 0, '01': 0, '10': 0, '11': 0}
        for i in range(len(generated) - 1):
            bigram = generated[i:i+2]
            if bigram in transitions:
                transitions[bigram] += 1
        
        total_trans = sum(transitions.values())
        if total_trans > 0:
            print(f"  Transitions: ", end="")
            for bigram, count in transitions.items():
                print(f"{bigram}={count/total_trans:.3f} ", end="")
            print()
        
        # Check for repetitive patterns
        has_long_runs = False
        current_symbol = generated[0] if generated else ''
        run_length = 1
        max_run = 1
        
        for i in range(1, len(generated)):
            if generated[i] == current_symbol:
                run_length += 1
                max_run = max(max_run, run_length)
            else:
                if run_length > 20:
                    has_long_runs = True
                current_symbol = generated[i]
                run_length = 1
        
        print(f"  Max run length: {max_run}")
        print(f"  Has long runs (>20): {has_long_runs}")

if __name__ == '__main__':
    checkpoint_path = Path("checkpoints/streamlined_seven_state/best.pt")
    
    test_cases = {
        "Short Sequence": {
            "length": 200,
            "window_size": 25,
            "temperature": 0.8,
            "seed": "01110001"
        },
        "Medium Sequence": {
            "length": 500,
            "window_size": 25,
            "temperature": 0.8,
            "seed": "01110001"
        },
        "Long Sequence": {
            "length": 1000,
            "window_size": 25,
            "temperature": 0.8,
            "seed": "01110001"
        },
        "Very Long Sequence": {
            "length": 2000,
            "window_size": 25,
            "temperature": 1.0,
            "seed": "01110001110110001"
        }
    }
    
    compare_generation_methods(checkpoint_path, test_cases)