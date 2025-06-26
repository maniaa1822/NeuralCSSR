"""Test classical CSSR on the generated dataset."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import json
from src.neural_cssr.classical.cssr import ClassicalCSSR
from src.neural_cssr.data.dataset_generation import EpsilonMachineDataset


def main():
    print("Testing Classical CSSR on large dataset...")
    
    # Load the existing dataset
    print("\nLoading dataset...")
    try:
        # Load the raw data format
        try:
            dataset_dict = torch.load('/home/matteo/NeuralCSSR/data/large_test/train_dataset.pt', weights_only=False)
            if isinstance(dataset_dict, dict) and 'data' in dataset_dict:
                # New format - raw data
                train_data = dataset_dict['data']
                metadata = dataset_dict['metadata']
                print(f"Loaded training dataset with {len(train_data)} examples (new format)")
            else:
                # Old format - try to regenerate
                raise ModuleNotFoundError("Old format dataset")
        except (ModuleNotFoundError, KeyError):
            # Regenerate the dataset with new format
            print("Dataset has incompatible format. Regenerating...")
            import subprocess
            result = subprocess.run(['uv', 'run', 'python', 'test_enumeration.py'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to regenerate dataset: {result.stderr}")
                return
            
            dataset_dict = torch.load('data/large_test/train_dataset.pt', weights_only=False)
            train_data = dataset_dict['data']
            metadata = dataset_dict['metadata']
            print(f"Loaded training dataset with {len(train_data)} examples (regenerated)")
        
        print(f"Vocabulary: {metadata['alphabet']}")
        
        # Also load metadata for context
        with open('/home/matteo/NeuralCSSR/data/large_test/metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"Dataset covers {metadata['total_machines']} machines")
        print(f"Alphabet: {metadata['alphabet']}")
        
    except FileNotFoundError:
        print("Dataset not found. Please run test_enumeration.py first.")
        return
    
    # Initialize Classical CSSR with optimal parameters from small dataset testing
    print("\nInitializing Classical CSSR...")
    print("Using optimal parameters: KL divergence, min_count=2, threshold=0.01")
    cssr = ClassicalCSSR(
        significance_level=0.05,  # Not used for KL
        min_count=2,  # Lower threshold for better coverage
        test_type="kl_divergence"
    )
    cssr.statistical_tests.kl_threshold = 0.01  # Most sensitive setting
    
    # Load data from dataset
    print("\nLoading data into CSSR...")
    cssr.load_from_raw_data(train_data, metadata)
    
    # Run CSSR algorithm
    print("\nRunning CSSR algorithm...")
    converged = cssr.run_cssr(max_iterations=10)
    
    # Get results
    print("\n" + "="*50)
    print("CSSR RESULTS")
    print("="*50)
    
    results = cssr.get_results_summary()
    print(f"Converged: {converged}")
    print(f"Number of causal states discovered: {results['num_causal_states']}")
    print(f"Total histories observed: {results['total_histories_observed']}")
    print(f"Histories assigned to states: {results['total_histories_assigned']}")
    print(f"Coverage: {results['coverage']:.2%}")
    
    # Show details of discovered states
    print(f"\nDetailed breakdown of causal states:")
    for state_info in results['causal_states']:
        print(f"\nState {state_info['state_id']}:")
        print(f"  Histories: {state_info['num_histories']}")
        print(f"  Observations: {state_info['total_observations']}")
        print(f"  Next symbol distribution: {state_info['next_symbol_distribution']}")
        print(f"  Sample histories: {state_info['sample_histories']}")
    
    # Evaluate against ground truth
    print(f"\n" + "="*50)
    print("EVALUATION AGAINST GROUND TRUTH")
    print("="*50)
    
    evaluation = cssr.evaluate_against_raw_data(train_data)
    print(f"Prediction accuracy: {evaluation['accuracy']:.2%}")
    print(f"Predictions made: {evaluation['predictions_made']}")
    print(f"States discovered vs ground truth: {evaluation['num_states_discovered']} (discovered)")
    
    # Compare with ground truth from metadata
    if 'ground_truth_states' in metadata:
        gt_states = metadata['ground_truth_states']
        print(f"Ground truth total states: {gt_states}")
        print(f"State discovery ratio: {evaluation['num_states_discovered'] / gt_states:.2f}")
    
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    # Save results
    results_file = '/home/matteo/NeuralCSSR/data/large_test/classical_cssr_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'cssr_results': results,
            'evaluation': evaluation,
            'converged': converged,
            'parameters': {
                'significance_level': cssr.significance_level,
                'min_count': cssr.min_count,
                'test_type': cssr.test_type
            }
        }, f, indent=2)
    
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()