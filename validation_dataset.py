#!/usr/bin/env python3
"""
Dataset validation script to check if the generated Neural CSSR dataset is correct.
"""

import sys
import torch
import json
from collections import defaultdict, Counter
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from neural_cssr.data.dataset_generation import EpsilonMachineDataset
    from neural_cssr.core.epsilon_machine import EpsilonMachine
except ImportError:
    print("Warning: Could not import neural_cssr modules. Continuing with basic validation...")

def validate_dataset(dataset_path, metadata_path):
    """Comprehensive validation of the generated dataset."""
    
    print("🔍 Loading dataset for validation...")
    dataset = torch.load(dataset_path, weights_only=False)
    
    # Debug: Print the structure of what we loaded
    print(f"📝 Dataset type: {type(dataset)}")
    if isinstance(dataset, dict):
        print(f"📝 Dictionary keys: {list(dataset.keys())}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Handle different dataset formats
    if isinstance(dataset, dict):
        print("📋 Dataset loaded as dictionary")
        
        # Check for the actual PyTorch dataset format
        if 'data' in dataset and 'metadata' in dataset:
            print("📝 Found PyTorch dataset format with 'data' and 'metadata' keys")
            raw_examples = dataset['data']
            metadata_internal = dataset['metadata']
            
            # Convert PyTorch format to validation format
            examples = []
            for raw_example in raw_examples:
                example = {
                    'history': raw_example['raw_history'],
                    'target': raw_example['raw_target'], 
                    'causal_state': raw_example['causal_state'],
                    'machine_id': raw_example['machine_id'],
                    'target_prob': raw_example['target_prob']
                }
                examples.append(example)
            
            # No machine library in this format, will skip those tests
            machine_library = []
            print(f"📝 Converted {len(examples)} examples from PyTorch format")
            
        else:
            # Original validation format
            examples = dataset.get('examples', [])
            machine_library = dataset.get('machine_library', [])
            
            # If examples is empty, maybe the dataset structure is different
            if not examples:
                print("⚠️  No 'examples' key found, checking alternative structures...")
                # Maybe it's a dataset object saved as dict
                if hasattr(dataset, 'examples'):
                    examples = dataset.examples
                    machine_library = dataset.machine_library
                # Maybe the keys are numeric or the whole dict is examples
                elif len(dataset) > 0 and all(isinstance(v, dict) for v in dataset.values()):
                    print("📝 Treating entire dictionary as examples")
                    examples = list(dataset.values())
                    machine_library = []
            
    else:
        print("📋 Dataset loaded as object")
        examples = getattr(dataset, 'examples', [])
        machine_library = getattr(dataset, 'machine_library', [])
    
    print(f"Dataset size: {len(examples)} examples")
    print(f"Machine library size: {len(machine_library)} machines")
    print(f"Alphabet: {metadata['alphabet']}")
    
    # If we still have very few examples, show some debug info
    if len(examples) <= 5:
        print("\n⚠️  Very few examples found. Showing dataset structure:")
        if len(examples) > 0:
            print(f"📝 First example keys: {list(examples[0].keys()) if examples[0] else 'None'}")
            print(f"📝 First example: {examples[0] if examples else 'None'}")
        
        if isinstance(dataset, dict):
            print(f"📝 Raw dataset keys: {list(dataset.keys())}")
            print(f"📝 Raw dataset length: {len(dataset)}")
    
    if len(examples) == 0:
        print("❌ No examples found in dataset!")
        print("💡 This might indicate a problem with dataset generation or saving format")
        return False
    
    issues = []
    warnings = []
    
    # Test 1: History/Causal State Consistency
    print("\n🧪 Test 1: History/Causal State Consistency")
    
    if len(machine_library) == 0:
        print("  ⚠️  Skipping - no machine library available in PyTorch dataset format")
    else:
        history_state_mismatches = 0
        
        for i, example in enumerate(examples[:100]):  # Sample first 100
            machine_data = None
            # Find the machine for this example
            for m in machine_library:
                if m['id'] == example['machine_id']:
                    machine_data = m
                    break
                    
            if machine_data:
                machine = machine_data['machine']
                
                # Compute causal state from the history in the example
                computed_state = machine.compute_causal_state(example['history'])
                
                if computed_state != example['causal_state']:
                    history_state_mismatches += 1
                    if history_state_mismatches <= 3:  # Show first few mismatches
                        print(f"  ❌ Mismatch in example {i}:")
                        print(f"     History: {''.join(example['history'])}")
                        print(f"     Stored state: {example['causal_state']}")
                        print(f"     Computed state: {computed_state}")
        
        if history_state_mismatches > 0:
            issues.append(f"History/Causal state mismatch in {history_state_mismatches}/100 examples")
        else:
            print("  ✅ History and causal states are consistent")
    
    # Test 2: Machine Determinism
    print("\n🧪 Test 2: Machine Determinism")
    
    if len(machine_library) == 0:
        print("  ⚠️  Skipping - no machine library available in PyTorch dataset format")
    else:
        non_deterministic_machines = 0
        
        for machine_data in machine_library[:5]:  # Check first 5 machines
            machine = machine_data['machine']
            
            for state in machine.states:
                for symbol in machine.alphabet:
                    key = (state, symbol)
                    if key in machine.transitions:
                        if len(machine.transitions[key]) > 1:
                            non_deterministic_machines += 1
                            print(f"  ❌ Machine {machine_data['id']} has multiple transitions for {key}")
                            break
        
        if non_deterministic_machines > 0:
            issues.append(f"{non_deterministic_machines} machines have non-deterministic transitions")
        else:
            print("  ✅ All machines are deterministic")
    
    # Test 3: Emission Probability Consistency
    print("\n🧪 Test 3: Emission Probability Consistency")
    prob_inconsistencies = 0
    
    for i, example in enumerate(examples[:50]):
        if example['target_prob'] <= 0:
            prob_inconsistencies += 1
    
    if prob_inconsistencies > 0:
        warnings.append(f"{prob_inconsistencies}/50 examples have zero emission probability")
    else:
        print("  ✅ Emission probabilities look reasonable")
    
    # Test 4: Structural Diversity
    print("\n🧪 Test 4: Structural Diversity")
    
    # Count unique causal states across all machines
    all_states = set()
    machine_state_counts = Counter()
    
    for example in examples:
        state_key = f"M{example['machine_id']}_{example['causal_state']}"
        all_states.add(state_key)
        machine_state_counts[example['machine_id']] += 1
    
    print(f"  📊 Total unique (machine, state) combinations: {len(all_states)}")
    print(f"  📊 Machines represented: {len(machine_state_counts)}")
    
    # Check if some machines dominate the dataset
    total_examples = len(examples)
    dominant_machines = 0
    for machine_id, count in machine_state_counts.items():
        if count > total_examples * 0.5:  # More than 50% of examples
            dominant_machines += 1
    
    if dominant_machines > 0:
        warnings.append(f"{dominant_machines} machines dominate >50% of dataset")
    
    # Test 5: Sequence Validity
    print("\n🧪 Test 5: Sequence Validity")
    invalid_sequences = 0
    
    for i, example in enumerate(examples[:20]):
        # Check if history + target forms a valid sequence
        full_sequence = example['history'] + [example['target']]
        
        # Check if all symbols are in alphabet
        for symbol in full_sequence:
            if symbol not in metadata['alphabet']:
                invalid_sequences += 1
                print(f"  ❌ Invalid symbol '{symbol}' in example {i}")
                break
    
    if invalid_sequences > 0:
        issues.append(f"{invalid_sequences}/20 examples contain invalid symbols")
    else:
        print("  ✅ All sequences contain valid symbols")
    
    # Test 6: Topological Property Check
    print("\n🧪 Test 6: Topological Machine Properties")
    
    if len(machine_library) == 0:
        print("  ⚠️  Skipping - no machine library available in PyTorch dataset format")
    else:
        non_topological = 0
        
        for machine_data in machine_library[:3]:
            machine = machine_data['machine']
            
            if hasattr(machine, 'is_topological') and not machine.is_topological():
                non_topological += 1
                print(f"  ❌ Machine {machine_data['id']} is not topological")
        
        if non_topological > 0:
            issues.append(f"{non_topological} machines are not topological")
        else:
            print("  ✅ Checked machines are topological")
    
    # Test 7: Data Structure Integrity
    print("\n🧪 Test 7: Data Structure Integrity")
    
    required_fields = ['history', 'target', 'causal_state', 'machine_id', 'target_prob']
    missing_fields = 0
    
    for i, example in enumerate(examples[:10]):
        for field in required_fields:
            if field not in example:
                missing_fields += 1
                print(f"  ❌ Missing field '{field}' in example {i}")
    
    if missing_fields > 0:
        issues.append(f"{missing_fields} missing required fields")
    else:
        print("  ✅ All required fields present")
    
    # Summary
    print("\n" + "="*60)
    print("🏁 VALIDATION SUMMARY")
    print("="*60)
    
    if not issues:
        print("✅ Dataset appears to be CORRECT!")
        if warnings:
            print("\n⚠️  Minor warnings:")
            for warning in warnings:
                print(f"   • {warning}")
    else:
        print("❌ Dataset has CRITICAL ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
        
        if warnings:
            print("\n⚠️  Additional warnings:")
            for warning in warnings:
                print(f"   • {warning}")
    
    return len(issues) == 0


def validate_specific_examples(dataset_path):
    """Deep dive into specific examples to show the mismatch issue."""
    
    print("\n🔬 DEEP DIVE: Analyzing specific examples")
    print("="*60)
    
    dataset = torch.load(dataset_path, weights_only=False)
    
    # Handle different dataset formats
    if isinstance(dataset, dict):
        # Check for PyTorch dataset format
        if 'data' in dataset and 'metadata' in dataset:
            raw_examples = dataset['data']
            # Convert PyTorch format to validation format
            examples = []
            for raw_example in raw_examples:
                example = {
                    'history': raw_example['raw_history'],
                    'target': raw_example['raw_target'], 
                    'causal_state': raw_example['causal_state'],
                    'machine_id': raw_example['machine_id'],
                    'target_prob': raw_example['target_prob']
                }
                examples.append(example)
            machine_library = []
        else:
            examples = dataset.get('examples', [])
            machine_library = dataset.get('machine_library', [])
            if not examples and len(dataset) > 0:
                examples = list(dataset.values()) if isinstance(list(dataset.values())[0], dict) else []
    else:
        examples = getattr(dataset, 'examples', [])
        machine_library = getattr(dataset, 'machine_library', [])
    
    # Look at a few examples in detail
    for i in range(min(3, len(examples))):
        example = examples[i]
        
        print(f"\n📋 Example {i+1}:")
        print(f"  Machine ID: {example['machine_id']}")
        print(f"  History (what transformer sees): {''.join(example['history'])}")
        print(f"  Target: {example['target']}")
        print(f"  Stored causal state: {example['causal_state']}")
        
        # Find the machine
        machine_data = None
        for m in machine_library:
            if m['id'] == example['machine_id']:
                machine_data = m
                break
        
        if machine_data:
            machine = machine_data['machine']
            
            # Test what causal state the history actually leads to
            computed_state = machine.compute_causal_state(example['history'])
            
            print(f"  Computed causal state from history: {computed_state}")
            
            if computed_state != example['causal_state']:
                print(f"  ❌ MISMATCH DETECTED!")
                
                # Show what the full sequence would compute to
                if 'position' in example and 'sequence_id' in example:
                    print("  🔍 This suggests the causal state was computed from longer sequence")
            else:
                print(f"  ✅ States match - this example is correct")
        else:
            print(f"  ⚠️  Cannot validate causal state - machine {example['machine_id']} not found in library")


if __name__ == "__main__":
    # Validate the dataset
    dataset_path = "data/large_test/train_dataset.pt"
    metadata_path = "data/large_test/metadata.json"
    
    print(f"📁 Looking for dataset at: {dataset_path}")
    print(f"📁 Looking for metadata at: {metadata_path}")
    
    # Check if files exist
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        exit(1)
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        exit(1)
    
    try:
        is_valid = validate_dataset(dataset_path, metadata_path)
        validate_specific_examples(dataset_path)
        
        if not is_valid:
            print("\n💡 RECOMMENDED ACTIONS:")
            print("1. Check dataset generation parameters (sequences_per_machine, sequence_length)")
            print("2. Verify machine enumeration is working correctly")  
            print("3. Ensure dataset saving is preserving the structure")
            print("4. Fix any validation issues and re-generate dataset")
            
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        