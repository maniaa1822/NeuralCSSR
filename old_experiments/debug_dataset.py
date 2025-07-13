#!/usr/bin/env python3

import sys
from pathlib import Path
import torch

# Add src to path to import neural_cssr
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_dataset(pt_file):
    print(f"Loading dataset from: {pt_file}")
    
    # Load with weights_only=False to handle custom classes
    sequences = torch.load(pt_file, weights_only=False)
    
    print(f"Dataset type: {type(sequences)}")
    
    if hasattr(sequences, 'examples'):
        print(f"Number of examples: {len(sequences.examples)}")
        
        # Examine first few examples
        for i, example in enumerate(sequences.examples[:5]):
            print(f"\nExample {i}:")
            print(f"  Keys: {example.keys()}")
            input_ids = example['input_ids']
            print(f"  Input IDs: {input_ids}")
            print(f"  Input IDs type: {type(input_ids)}")
            print(f"  Length: {len(input_ids)}")
            print(f"  Unique values: {set(input_ids)}")
            
            # Convert to sequence
            sequence = [token for token in input_ids if token != 2]
            print(f"  After removing padding: {sequence}")
            print(f"  Unique values after filtering: {set(sequence)}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    debug_dataset("datasets/small_exp/neural_format/train_dataset.pt")
    debug_dataset("datasets/small_exp/neural_format/val_dataset.pt")
