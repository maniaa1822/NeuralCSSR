#!/usr/bin/env python3
"""Debug script to understand model vocabulary and training setup."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_delay_transformer import BinaryTransformer, SequenceDataset

def debug_model_vocab():
    print("=" * 60)
    print("MODEL VOCABULARY DEBUGGING")
    print("=" * 60)
    
    # Load model
    model = BinaryTransformer(vocab_size=3, d_model=10, n_layers=1, n_heads=1, max_delay=None)
    state_dict = torch.load('checkpoints/best_model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model vocab_size: {model.proj.out_features}")
    print(f"Model embedding num_embeddings: {model.embed.num_embeddings}")
    
    # Load dataset to understand the data format
    dataset = SequenceDataset(Path('datasets/biased_exp/neural_format/train_dataset.pt'))
    print(f"Dataset size: {len(dataset.data)}")
    
    # Check first few sequences
    for i in range(3):
        seq = dataset.data[i]
        print(f"Sequence {i}: {seq[:10].tolist()} (length: {len(seq)})")
        print(f"  Unique values: {set(seq.tolist())}")
    
    print()
    print("Testing model on simple inputs...")
    
    # Test simple inputs
    test_inputs = [
        [0],
        [1], 
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0]
    ]
    
    with torch.no_grad():
        for inp in test_inputs:
            inp_tensor = torch.tensor(inp, dtype=torch.long).unsqueeze(0)
            logits = model(inp_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            print(f"Input: {inp}")
            print(f"  Logits: {logits[0, -1, :].numpy()}")
            print(f"  Probs: {probs[0, -1, :].numpy()}")
            print(f"  Max prob class: {torch.argmax(probs[0, -1, :]).item()}")
            print()

if __name__ == '__main__':
    debug_model_vocab()