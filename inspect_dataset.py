#!/usr/bin/env python3
"""
Quick dataset inspection script to visualize the saved dataset.
"""

import sys
import torch
import json
from collections import defaultdict, Counter
sys.path.append('src')

def load_dataset(dataset_path):
    """Load saved dataset and metadata."""
    # Need to set weights_only=False for custom classes in PyTorch 2.6+
    dataset = torch.load(dataset_path, weights_only=False)
    
    with open('data/small_test/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return dataset, metadata

def inspect_dataset_overview(dataset, metadata, name="Dataset"):
    """Show high-level dataset statistics."""
    print(f"\n=== {name} Overview ===")
    print(f"Size: {len(dataset)} examples")
    print(f"Vocabulary: {metadata['alphabet']}")
    print(f"Max history length: {metadata['max_history_length']}")
    
    # Analyze machine distribution
    machine_counts = Counter()
    state_counts = Counter()
    causal_state_counts = Counter()
    
    for example in dataset.examples:
        machine_counts[example['machine_id']] += 1
        state_counts[example['num_states']] += 1
        causal_state_counts[example['causal_state']] += 1
    
    print(f"\nMachine distribution:")
    for machine_id, count in sorted(machine_counts.items()):
        print(f"  Machine {machine_id}: {count} examples")
    
    print(f"\nComplexity distribution:")
    for num_states, count in sorted(state_counts.items()):
        print(f"  {num_states} states: {count} examples")
    
    print(f"\nMost common causal states:")
    for state, count in causal_state_counts.most_common(5):
        print(f"  {state}: {count} examples")

def show_sample_examples(dataset, num_examples=10):
    """Show detailed examples from the dataset."""
    print(f"\n=== Sample Examples ===")
    
    for i in range(min(num_examples, len(dataset))):
        example = dataset.examples[i]
        
        print(f"\nExample {i+1}:")
        print(f"  History: {''.join(example['history'])}")
        print(f"  Target: {example['target']}")
        print(f"  Causal state: {example['causal_state']}")
        print(f"  Machine ID: {example['machine_id']} ({example['num_states']} states)")
        print(f"  Target probability: {example['target_prob']:.3f}")
        print(f"  Emission probs: {example['emission_probs']}")

def show_machine_specific_examples(dataset):
    """Show examples grouped by machine."""
    print(f"\n=== Examples by Machine ===")
    
    # Group examples by machine
    machine_examples = defaultdict(list)
    for example in dataset.examples:
        machine_examples[example['machine_id']].append(example)
    
    # Show a few examples from each machine
    for machine_id in sorted(machine_examples.keys())[:5]:  # Show first 5 machines
        examples = machine_examples[machine_id]
        print(f"\nMachine {machine_id} ({examples[0]['num_states']} states):")
        
        for i, example in enumerate(examples[:3]):  # Show first 3 examples
            history_str = ''.join(example['history'])
            if len(history_str) == 0:
                history_str = "(empty)"
            print(f"  {history_str} -> {example['target']} (state: {example['causal_state']})")

def show_causal_state_patterns(dataset):
    """Show patterns in causal state transitions."""
    print(f"\n=== Causal State Patterns ===")
    
    # Track state transitions
    state_transitions = defaultdict(list)
    
    for example in dataset.examples:
        machine_id = example['machine_id']
        causal_state = example['causal_state']
        target = example['target']
        
        key = f"M{machine_id}_{causal_state}"
        state_transitions[key].append(target)
    
    # Show most common patterns
    print("State -> Target patterns:")
    for state_key in sorted(state_transitions.keys())[:10]:
        targets = state_transitions[state_key]
        target_counts = Counter(targets)
        print(f"  {state_key}: {dict(target_counts)}")

def show_pytorch_batch_example(dataset):
    """Show what a PyTorch batch looks like."""
    print(f"\n=== PyTorch Batch Example ===")
    
    dataloader = dataset.get_dataloader(batch_size=3, shuffle=False)
    
    for batch in dataloader:
        print("Batch contents:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Input IDs:\n{batch['input_ids']}")
        print(f"  Target IDs: {batch['target_id']}")
        print(f"  Attention mask:\n{batch['attention_mask']}")
        print(f"  Machine IDs: {batch['machine_id']}")
        print(f"  Causal states: {batch['causal_state']}")
        
        # Convert back to readable format
        print("\nReadable format:")
        for i in range(len(batch['causal_state'])):
            input_ids = batch['input_ids'][i]
            target_id = batch['target_id'][i]
            attention_mask = batch['attention_mask'][i]
            
            # Get non-padded tokens
            valid_tokens = []
            for j, (token_id, mask) in enumerate(zip(input_ids, attention_mask)):
                if mask == 1:
                    valid_tokens.append(dataset.id_to_token[token_id.item()])
            
            target_token = dataset.id_to_token[target_id.item()]
            causal_state = batch['causal_state'][i]
            
            print(f"  {''.join(valid_tokens)} -> {target_token} (state: {causal_state})")
        
        break  # Just show first batch

def main():
    """Main inspection function."""
    print("Dataset Inspection Tool")
    print("=" * 50)
    
    try:
        # Load datasets
        train_dataset, metadata = load_dataset('data/small_test/train_dataset.pt')
        val_dataset, _ = load_dataset('data/small_test/val_dataset.pt')
        
        # Show overviews
        inspect_dataset_overview(train_dataset, metadata, "Training Dataset")
        inspect_dataset_overview(val_dataset, metadata, "Validation Dataset")
        
        # Show detailed examples
        show_sample_examples(train_dataset, num_examples=8)
        
        # Show machine-specific patterns
        show_machine_specific_examples(train_dataset)
        
        # Show causal state patterns
        show_causal_state_patterns(train_dataset)
        
        # Show PyTorch batch
        show_pytorch_batch_example(train_dataset)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()