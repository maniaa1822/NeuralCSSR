"""Test CSSR with different parameters to see if we can recover more states."""

import torch
from src.neural_cssr.classical.cssr import ClassicalCSSR


def test_different_parameters():
    """Test CSSR with various parameter combinations."""
    
    # Load dataset
    dataset_dict = torch.load('data/small_test/train_dataset.pt', weights_only=False)
    train_data = dataset_dict['data']
    metadata = dataset_dict['metadata']
    
    print("Testing CSSR with different parameters...")
    print(f"Dataset: {len(train_data)} examples")
    print("Ground truth: 3 causal states (S0, S1, S2)")
    print()
    
    # Parameter combinations to test
    test_configs = [
        {"significance_level": 0.05, "min_count": 3, "test_type": "chi_square"},
        {"significance_level": 0.01, "min_count": 3, "test_type": "chi_square"},  # More sensitive
        {"significance_level": 0.10, "min_count": 3, "test_type": "chi_square"},  # Less sensitive
        {"significance_level": 0.05, "min_count": 2, "test_type": "chi_square"},  # Lower threshold
        {"significance_level": 0.05, "min_count": 5, "test_type": "chi_square"},  # Higher threshold
        {"significance_level": 0.05, "min_count": 3, "test_type": "kl_divergence"},  # Different test
        {"significance_level": 0.01, "min_count": 2, "test_type": "chi_square"},  # Most sensitive
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"--- Test {i+1}: {config} ---")
        
        cssr = ClassicalCSSR(**config)
        cssr.load_from_raw_data(train_data, metadata)
        
        print(f"  Loaded {len(cssr.history_data)} unique histories")
        print(f"  Sufficient histories: {len(cssr.get_sufficient_histories())}")
        
        # Run CSSR
        converged = cssr.run_cssr(max_iterations=10)
        
        # Get results
        summary = cssr.get_results_summary()
        evaluation = cssr.evaluate_against_raw_data(train_data)
        
        result = {
            'config': config,
            'converged': converged,
            'num_states': summary['num_causal_states'],
            'coverage': summary['coverage'],
            'accuracy': evaluation['accuracy'],
            'predictions': evaluation['predictions_made']
        }
        results.append(result)
        
        print(f"  Converged: {converged}")
        print(f"  States found: {summary['num_causal_states']}")
        print(f"  Coverage: {summary['coverage']:.2%}")
        print(f"  Accuracy: {evaluation['accuracy']:.2%}")
        print()
    
    # Summary
    print("="*60)
    print("PARAMETER TESTING SUMMARY")
    print("="*60)
    print(f"{'Test':<4} {'Sig':<6} {'MinCnt':<6} {'Test Type':<12} {'States':<6} {'Coverage':<8} {'Accuracy':<8}")
    print("-" * 60)
    
    for i, result in enumerate(results):
        config = result['config']
        print(f"{i+1:<4} {config['significance_level']:<6.2f} {config['min_count']:<6} "
              f"{config['test_type']:<12} {result['num_states']:<6} "
              f"{result['coverage']:<8.2%} {result['accuracy']:<8.2%}")
    
    # Find best result
    best_states = max(results, key=lambda x: x['num_states'])
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    
    print(f"\nBest state recovery: Test {results.index(best_states)+1} with {best_states['num_states']} states")
    print(f"Best accuracy: Test {results.index(best_accuracy)+1} with {best_accuracy['accuracy']:.2%} accuracy")


if __name__ == "__main__":
    test_different_parameters()