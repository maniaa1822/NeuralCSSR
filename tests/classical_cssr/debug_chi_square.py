"""Debug the chi-square test implementation."""

from src.neural_cssr.classical.statistical_tests import StatisticalTests


def test_chi_square_manual():
    """Test chi-square with known examples."""
    print("Testing Chi-square implementation...")
    
    tests = StatisticalTests(significance_level=0.05)
    
    # Test 1: Identical distributions - should NOT be different
    dist1 = {'0': 10, '1': 10}
    dist2 = {'0': 10, '1': 10}
    
    is_different, chi2, p_value = tests.chi_square_test(dist1, dist2)
    print(f"\nTest 1 - Identical distributions:")
    print(f"  dist1: {dist1}")
    print(f"  dist2: {dist2}")
    print(f"  Different: {is_different}, chi2: {chi2:.4f}, p-value: {p_value:.4f}")
    
    # Test 2: Very different distributions - should be different
    dist1 = {'0': 20, '1': 0}
    dist2 = {'0': 0, '1': 20}
    
    is_different, chi2, p_value = tests.chi_square_test(dist1, dist2)
    print(f"\nTest 2 - Very different distributions:")
    print(f"  dist1: {dist1}")
    print(f"  dist2: {dist2}")
    print(f"  Different: {is_different}, chi2: {chi2:.4f}, p-value: {p_value:.4f}")
    
    # Test 3: Slightly different - might be different
    dist1 = {'0': 15, '1': 5}
    dist2 = {'0': 5, '1': 15}
    
    is_different, chi2, p_value = tests.chi_square_test(dist1, dist2)
    print(f"\nTest 3 - Moderately different distributions:")
    print(f"  dist1: {dist1}")
    print(f"  dist2: {dist2}")
    print(f"  Different: {is_different}, chi2: {chi2:.4f}, p-value: {p_value:.4f}")
    
    # Test 4: Real examples from CSSR data
    # These are from the analysis output
    dist1 = {'1': 3, '0': 1}  # '010100'
    dist2 = {'0': 2, '1': 2}  # '0111110000'
    
    is_different, chi2, p_value = tests.chi_square_test(dist1, dist2)
    print(f"\nTest 4 - Real CSSR example:")
    print(f"  dist1: {dist1} (history '010100')")
    print(f"  dist2: {dist2} (history '0111110000')")
    print(f"  Different: {is_different}, chi2: {chi2:.4f}, p-value: {p_value:.4f}")
    
    # Test 5: Very small sample sizes
    dist1 = {'1': 2, '0': 1}
    dist2 = {'0': 1, '1': 2}
    
    is_different, chi2, p_value = tests.chi_square_test(dist1, dist2, min_count=1)
    print(f"\nTest 5 - Very small samples:")
    print(f"  dist1: {dist1}")
    print(f"  dist2: {dist2}")
    print(f"  Different: {is_different}, chi2: {chi2:.4f}, p-value: {p_value:.4f}")


def test_ground_truth_distributions():
    """Test chi-square on actual ground truth distributions."""
    print(f"\n" + "="*50)
    print("TESTING GROUND TRUTH DISTRIBUTIONS")
    print("="*50)
    
    import torch
    
    # Load dataset
    dataset_dict = torch.load('data/small_test/train_dataset.pt', weights_only=False)
    train_data = dataset_dict['data']
    
    # Group by ground truth causal state
    state_distributions = {}
    
    for item in train_data:
        gt_state = item['causal_state']
        target = item['raw_target']
        
        if gt_state not in state_distributions:
            state_distributions[gt_state] = {'0': 0, '1': 0}
        
        state_distributions[gt_state][target] += 1
    
    print("Ground truth next-symbol distributions:")
    for state, dist in state_distributions.items():
        total = sum(dist.values())
        prob_dist = {k: v/total for k, v in dist.items()}
        print(f"  {state}: {dist} -> {prob_dist}")
    
    # Test if these are significantly different
    tests = StatisticalTests(significance_level=0.05)
    
    states = list(state_distributions.keys())
    print(f"\nChi-square tests between ground truth states:")
    
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            state1, state2 = states[i], states[j]
            dist1 = state_distributions[state1]
            dist2 = state_distributions[state2]
            
            is_different, chi2, p_value = tests.chi_square_test(dist1, dist2)
            print(f"  {state1} vs {state2}: Different={is_different}, "
                  f"chi2={chi2:.4f}, p-value={p_value:.4f}")


if __name__ == "__main__":
    test_chi_square_manual()
    test_ground_truth_distributions()