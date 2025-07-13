#!/usr/bin/env python3
"""Debug script to understand exactly how the transformer is trained."""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_delay_transformer import BinaryTransformer, SequenceDataset, collate_fn_ar

def debug_training_process():
    print("=" * 70)
    print("TRANSFORMER TRAINING PROCESS ANALYSIS")
    print("=" * 70)
    
    # 1. Load the original neural dataset to see the raw format
    print("1. ORIGINAL NEURAL DATASET FORMAT:")
    print("-" * 40)
    
    sequences = torch.load('datasets/biased_exp/neural_format/train_dataset.pt', weights_only=False)
    print(f"Dataset type: {type(sequences)}")
    
    if hasattr(sequences, 'examples'):
        print(f"Number of examples: {len(sequences.examples)}")
        
        # Show first few examples
        for i in range(3):
            example = sequences.examples[i]
            print(f"\nExample {i}:")
            print(f"  Keys: {list(example.keys())}")
            print(f"  input_ids: {example['input_ids'][:15]}... (length: {len(example['input_ids'])})")
            print(f"  target_ids: {example['target_ids']} (type: {type(example['target_ids'])})")
            print(f"  input_ids unique: {set(example['input_ids'])}")
            
            # Check if there are other keys
            for key in example.keys():
                if key not in ['input_ids', 'target_ids']:
                    print(f"  {key}: {example[key]}")
    
    print("\n" + "=" * 70)
    print("2. SEQUENCE DATASET PROCESSING:")
    print("-" * 40)
    
    # 2. See how SequenceDataset processes the data
    dataset = SequenceDataset(Path('datasets/biased_exp/neural_format/train_dataset.pt'))
    print(f"Processed dataset size: {len(dataset.data)}")
    
    for i in range(3):
        seq = dataset.data[i]
        print(f"Processed sequence {i}: {seq[:15].tolist()}... (length: {len(seq)})")
        print(f"  Unique values: {set(seq.tolist())}")
    
    print("\n" + "=" * 70)
    print("3. COLLATE FUNCTION (TRAINING FORMAT):")
    print("-" * 40)
    
    # 3. See how the collate function creates training batches
    from torch.utils.data import DataLoader
    
    # Create small dataloader to see batch format
    small_dataset = [dataset.data[i] for i in range(3)]
    loader = DataLoader(small_dataset, batch_size=2, collate_fn=collate_fn_ar)
    
    batch = next(iter(loader))
    inputs, targets = batch
    
    print(f"Batch inputs shape: {inputs.shape}")
    print(f"Batch targets shape: {targets.shape}")
    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}")
    print(f"Inputs unique: {set(inputs.flatten().tolist())}")
    print(f"Targets unique: {set(targets.flatten().tolist())}")
    
    # Key insight: Let's trace the mapping
    print("\n" + "=" * 50)
    print("CRITICAL MAPPING ANALYSIS:")
    print("-" * 50)
    
    print("From SequenceDataset processing:")
    print("- Original neural format has input_ids with values {0, 2, 3}")
    print("- SequenceDataset maps: 2→0, 3→1, removes 0 (PAD)")
    print("- Result: binary sequences with values {0, 1}")
    print()
    
    print("From collate_fn_ar:")
    print("- Takes binary sequences {0, 1}")
    print("- Creates autoregressive (input, target) pairs")
    print("- Padding token = 2")
    print("- So training data has tokens {0, 1, 2}")
    print()
    
    print("Model training:")
    print("- Input tokens: {0, 1, 2} where 2=PAD")
    print("- Target tokens: {0, 1, 2} where 2=PAD (ignored in loss)")
    print("- Model learns: P(0|context) and P(1|context)")
    print("- vocab_size=3 outputs: [class_0, class_1, class_2]")
    print("- class_0 = symbol '0', class_1 = symbol '1', class_2 = padding")
    
    print("\n" + "=" * 70)
    print("4. MODEL ARCHITECTURE:")
    print("-" * 40)
    
    # 4. Check model architecture
    model = BinaryTransformer(vocab_size=3, d_model=10, n_layers=1, n_heads=1, max_delay=None)
    print(f"Model vocab_size: {model.proj.out_features}")
    print(f"Embedding vocab_size: {model.embed.num_embeddings}")
    
    # Test what the model outputs for the training format
    print("\n" + "=" * 70)
    print("5. MODEL OUTPUTS ON TRAINING FORMAT:")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        # Use the same inputs from the batch
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Probs shape: {probs.shape}")
        
        # Show predictions for first sequence
        seq_0_logits = logits[0]
        seq_0_probs = probs[0]
        seq_0_targets = targets[0]
        
        print(f"\nFirst sequence predictions:")
        print(f"Input sequence: {inputs[0].tolist()}")
        print(f"Target sequence: {targets[0].tolist()}")
        print()
        
        for pos in range(min(5, len(seq_0_logits))):
            if seq_0_targets[pos] != 2:  # Not padding
                print(f"Position {pos}:")
                print(f"  Target: {seq_0_targets[pos].item()}")
                print(f"  Logits: {seq_0_logits[pos].numpy()}")
                print(f"  Probs: {seq_0_probs[pos].numpy()}")
                print(f"  Predicted: {torch.argmax(seq_0_probs[pos]).item()}")
                print(f"  Interpretation: P(0)={seq_0_probs[pos][0]:.3f}, P(1)={seq_0_probs[pos][1]:.3f}, P(PAD)={seq_0_probs[pos][2]:.3f}")
                print()

    print("\n" + "=" * 70)
    print("6. TRAINING LOSS COMPUTATION:")
    print("-" * 40)
    
    # 5. Show how training loss is computed
    import torch.nn as nn
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding
    
    # Reshape for loss computation (as done in training)
    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
    print(f"Training loss: {loss.item()}")
    
    # Show which positions contribute to loss
    print("\nPositions used for loss computation:")
    flat_targets = targets.view(-1)
    non_padding_count = 0
    for i, target in enumerate(flat_targets):
        if target != 2:  # Not padding
            non_padding_count += 1
            if non_padding_count <= 5:  # Show first 5
                print(f"  Position {i}: target={target.item()}")
    
    print(f"Total non-padding positions: {non_padding_count}")

if __name__ == '__main__':
    debug_training_process()