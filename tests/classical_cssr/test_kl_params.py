"""Test CSSR with KL divergence and different thresholds."""

import torch
from src.neural_cssr.classical.cssr import ClassicalCSSR


def test_kl_thresholds():
    """Test different KL divergence thresholds."""
    
    # Load dataset
    dataset_dict = torch.load('data/small_test/train_dataset.pt', weights_only=False)
    train_data = dataset_dict['data']
    metadata = dataset_dict['metadata']
    
    print("Testing CSSR with KL divergence and different thresholds...")
    print(f"Dataset: {len(train_data)} examples")
    print("Ground truth: 3 causal states (S0, S1, S2)")
    print()
    
    # KL divergence threshold values to test
    kl_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    min_counts = [2, 3, 4]
    
    results = []
    
    for min_count in min_counts:
        for threshold in kl_thresholds:
            print(f"--- KL threshold: {threshold}, min_count: {min_count} ---")
            
            # Note: For KL divergence, we need to modify the statistical tests
            # to accept the threshold parameter
            cssr = ClassicalCSSR(
                significance_level=0.05,  # Not used for KL
                min_count=min_count,
                test_type="kl_divergence"
            )
            
            # Set the KL threshold directly
            cssr.statistical_tests.kl_threshold = threshold
            
            cssr.load_from_raw_data(train_data, metadata)
            
            print(f"  Sufficient histories: {len(cssr.get_sufficient_histories())}")
            
            # Run CSSR
            converged = cssr.run_cssr(max_iterations=10)
            
            # Get results
            summary = cssr.get_results_summary()
            evaluation = cssr.evaluate_against_raw_data(train_data)
            
            result = {
                'min_count': min_count,
                'kl_threshold': threshold,
                'converged': converged,
                'num_states': summary['num_causal_states'],
                'coverage': summary['coverage'],
                'accuracy': evaluation['accuracy'],
                'predictions': evaluation['predictions_made']
            }
            results.append(result)
            
            print(f"  States found: {summary['num_causal_states']}")
            print(f"  Coverage: {summary['coverage']:.2%}")
            print(f"  Accuracy: {evaluation['accuracy']:.2%}")
            print()
    
    # Summary
    print("="*70)
    print("KL DIVERGENCE PARAMETER TESTING SUMMARY")
    print("="*70)
    print(f"{'MinCnt':<6} {'KL Thresh':<10} {'States':<6} {'Coverage':<8} {'Accuracy':<8} {'Converged':<9}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['min_count']:<6} {result['kl_threshold']:<10.2f} "
              f"{result['num_states']:<6} {result['coverage']:<8.2%} "
              f"{result['accuracy']:<8.2%} {result['converged']}")
    
    # Find best results
    # Best balance: close to 3 states with good accuracy
    best_states = min(results, key=lambda x: abs(x['num_states'] - 3))
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    
    print(f"\nClosest to 3 states: min_count={best_states['min_count']}, "
          f"threshold={best_states['kl_threshold']:.2f}, "
          f"states={best_states['num_states']}, accuracy={best_states['accuracy']:.2%}")
    
    print(f"Best accuracy: min_count={best_accuracy['min_count']}, "
          f"threshold={best_accuracy['kl_threshold']:.2f}, "
          f"states={best_accuracy['num_states']}, accuracy={best_accuracy['accuracy']:.2%}")


if __name__ == "__main__":
    test_kl_thresholds()