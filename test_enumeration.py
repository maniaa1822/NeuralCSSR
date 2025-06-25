#!/usr/bin/env python3
"""
Test script to enumerate epsilon-machines and visualize their functionality.
Uses config file for parameters.
"""

import sys
import os
import yaml
sys.path.append('src')

from neural_cssr.enumeration.enumerate_machines import enumerate_machines_library
from neural_cssr.data.dataset_generation import create_train_val_datasets
import json
import torch


def load_config(config_path='config/large_test_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_datasets(train_dataset, val_dataset, save_path):
    """Save datasets to disk as raw data to avoid module path issues."""
    os.makedirs(save_path, exist_ok=True)
    
    # Save datasets as raw data structures instead of pickled objects
    def extract_dataset_data(dataset):
        """Extract raw data from dataset."""
        data = []
        for i in range(len(dataset)):
            item = dataset[i]  # Get tensor version
            example = dataset.examples[i]  # Get raw example
            
            data.append({
                'input_ids': item['input_ids'].tolist(),
                'attention_mask': item['attention_mask'].tolist(),
                'target_id': item['target_id'].item(),
                'machine_id': item['machine_id'].item(),
                'num_states': item['num_states'].item(),
                'causal_state': item['causal_state'],
                'target_prob': item['target_prob'].item(),
                'raw_history': example['history'],
                'raw_target': example['target']
            })
        return data
    
    # Extract and save data
    train_data = extract_dataset_data(train_dataset)
    val_data = extract_dataset_data(val_dataset)
    
    torch.save({
        'data': train_data,
        'metadata': {
            'vocab_size': train_dataset.vocab_size,
            'alphabet': train_dataset.alphabet,
            'token_to_id': train_dataset.token_to_id,
            'id_to_token': train_dataset.id_to_token,
            'max_history_length': train_dataset.max_history_length
        }
    }, os.path.join(save_path, 'train_dataset.pt'))
    
    torch.save({
        'data': val_data,
        'metadata': {
            'vocab_size': val_dataset.vocab_size,
            'alphabet': val_dataset.alphabet,
            'token_to_id': val_dataset.token_to_id,
            'id_to_token': val_dataset.id_to_token,
            'max_history_length': val_dataset.max_history_length
        }
    }, os.path.join(save_path, 'val_dataset.pt'))
    
    # Save metadata
    metadata = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'vocab_size': train_dataset.vocab_size,
        'alphabet': train_dataset.alphabet,
        'token_to_id': train_dataset.token_to_id,
        'id_to_token': train_dataset.id_to_token,
        'max_history_length': train_dataset.max_history_length,
        'total_machines': len(set(example['machine_id'] for example in train_dataset.examples))
    }
    
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def test_machine_enumeration(config):
    """Test basic machine enumeration."""
    print("=== Testing Machine Enumeration ===")
    
    enum_config = config['enumeration']
    
    print(f"Enumerating machines with up to {enum_config['max_states']} states...")
    print(f"Alphabet: {enum_config['alphabet']}")
    print(f"Max machines per size: {enum_config['max_machines_per_size']}")
    
    try:
        machine_library = enumerate_machines_library(
            max_states=enum_config['max_states'],
            alphabet=enum_config['alphabet'],
            max_machines_per_size=enum_config['max_machines_per_size']
        )
        
        print(f"✓ Successfully enumerated {len(machine_library)} machines")
        
        # Show breakdown by complexity
        complexity_counts = {}
        for machine_data in machine_library:
            num_states = machine_data['properties']['num_states']
            complexity_counts[num_states] = complexity_counts.get(num_states, 0) + 1
            
        print("\nMachines by complexity:")
        for states, count in sorted(complexity_counts.items()):
            print(f"  {states} states: {count} machines")
            
        return machine_library
        
    except Exception as e:
        print(f"✗ Error in enumeration: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_machines(machine_library, num_to_show=3):
    """Visualize a few example machines."""
    print(f"\n=== Visualizing {num_to_show} Example Machines ===")
    
    for i, machine_data in enumerate(machine_library[:num_to_show]):
        machine = machine_data['machine']
        properties = machine_data['properties']
        
        print(f"\n--- Machine {i+1} (ID: {machine_data['id']}) ---")
        print(f"States: {sorted(list(machine.states))}")
        print(f"Alphabet: {machine.alphabet}")
        print(f"Properties: {properties}")
        
        # Show transitions
        print("Transitions:")
        for (state, symbol), transitions in machine.transitions.items():
            for next_state, prob in transitions:
                print(f"  {state} --{symbol}--> {next_state} (p={prob:.3f})")
        
        # Generate sample sequence
        print("Sample sequence:")
        try:
            sequence = machine.generate_sequence(15)
            print(f"  {''.join(sequence)}")
            
            # Show causal states for first few positions
            print("Causal states:")
            for pos in range(1, min(6, len(sequence))):
                history = sequence[:pos]
                causal_state = machine.compute_causal_state(history)
                print(f"  {''.join(history)} -> {causal_state}")
                
        except Exception as e:
            print(f"  Error generating sequence: {e}")


def test_dataset_creation(config):
    """Test PyTorch dataset creation."""
    print("\n=== Testing Dataset Creation ===")
    
    enum_config = config['enumeration']
    dataset_config = config['dataset']
    pytorch_config = config['pytorch']
    
    try:
        # Create datasets using config
        train_dataset, val_dataset = create_train_val_datasets(
            max_states=enum_config['max_states'],
            alphabet=enum_config['alphabet'],
            sequences_per_machine=dataset_config['sequences_per_machine'],
            sequence_length=dataset_config['sequence_length'],
            max_history_length=dataset_config['max_history_length'],
            val_split=dataset_config['validation_split'],
            max_machines_per_size=enum_config['max_machines_per_size']
        )
        
        print(f"✓ Created train dataset with {len(train_dataset)} examples")
        print(f"✓ Created val dataset with {len(val_dataset)} examples")
        print(f"Vocabulary size: {train_dataset.vocab_size}")
        print(f"Alphabet: {train_dataset.alphabet}")
        
        # Save datasets if enabled
        if config['output']['save_results']:
            save_path = config['output']['save_path']
            save_datasets(train_dataset, val_dataset, save_path)
            print(f"✓ Saved datasets to {save_path}")
        
        # Test DataLoader
        train_loader = train_dataset.get_dataloader(
            batch_size=pytorch_config['batch_size'],
            shuffle=pytorch_config['shuffle']
        )
        
        print("\nSample batch:")
        for batch in train_loader:
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Target IDs shape: {batch['target_id'].shape}")
            print(f"  Machine IDs: {batch['machine_id'][:3].tolist()}...")
            print(f"  Num states: {batch['num_states'][:3].tolist()}...")
            print(f"  Causal states: {batch['causal_state'][:3]}")
            
            # Show actual tokens for first example
            example_input = batch['input_ids'][0]
            example_target = batch['target_id'][0]
            
            # Convert back to tokens (skip padding)
            input_tokens = []
            for id in example_input:
                if id.item() != 0:  # Skip padding
                    input_tokens.append(train_dataset.id_to_token[id.item()])
            target_token = train_dataset.id_to_token[example_target.item()]
            
            print(f"  Example: {''.join(input_tokens)} -> {target_token}")
            print(f"  Target probability: {batch['target_prob'][0]:.3f}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Error in dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Neural CSSR Machine Enumeration Test")
    print("=" * 40)
    
    # Load config
    try:
        config = load_config()
        print(f"✓ Loaded configuration")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return
    
    # Test enumeration
    machine_library = test_machine_enumeration(config)
    if machine_library is None:
        print("Enumeration failed, exiting.")
        return
    
    # Visualize machines
    if config['output']['verbose']:
        visualize_machines(machine_library)
    
    # Test dataset creation
    dataset_success = test_dataset_creation(config)
    
    print("\n" + "=" * 40)
    if dataset_success:
        print("✓ All tests passed! Machine enumeration and dataset creation working.")
    else:
        print("✗ Some tests failed. Check error messages above.")


if __name__ == "__main__":
    main()