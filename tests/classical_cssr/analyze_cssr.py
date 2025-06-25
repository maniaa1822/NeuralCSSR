"""Detailed analysis of Classical CSSR behavior."""

import torch
import json
from collections import defaultdict, Counter
from src.neural_cssr.classical.cssr import ClassicalCSSR


def analyze_history_distributions(cssr):
    """Analyze the distributions of different histories."""
    print("="*60)
    print("HISTORY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Group histories by length
    by_length = defaultdict(list)
    for hist in cssr.history_data.keys():
        by_length[len(hist)].append(hist)
    
    print(f"Histories by length:")
    for length in sorted(by_length.keys()):
        histories = by_length[length]
        total_count = sum(cssr.history_counts[h] for h in histories)
        sufficient_count = len([h for h in histories if cssr.history_counts[h] >= cssr.min_count])
        
        print(f"  Length {length}: {len(histories)} histories, "
              f"{sufficient_count} with ≥{cssr.min_count} observations, "
              f"total observations: {total_count}")
        
        # Show some examples
        if sufficient_count > 0:
            examples = [h for h in histories if cssr.history_counts[h] >= cssr.min_count][:5]
            for hist in examples:
                dist = cssr.history_data[hist]
                count = cssr.history_counts[hist]
                print(f"    '{hist}' ({count} obs): {dict(dist)}")


def analyze_statistical_tests(cssr):
    """Analyze what the statistical tests are doing."""
    print("\n" + "="*60)
    print("STATISTICAL TEST ANALYSIS")
    print("="*60)
    
    sufficient_histories = cssr.get_sufficient_histories()
    histories_list = list(sufficient_histories)
    
    print(f"Testing equivalence between {len(histories_list)} sufficient histories...")
    print(f"Significance level: {cssr.significance_level}")
    print(f"Minimum count threshold: {cssr.min_count}")
    
    # Test some pairs
    equivalent_pairs = []
    different_pairs = []
    
    for i in range(min(10, len(histories_list))):
        for j in range(i + 1, min(10, len(histories_list))):
            hist1, hist2 = histories_list[i], histories_list[j]
            
            are_equiv, test_results = cssr.histories_are_equivalent(hist1, hist2)
            
            if are_equiv:
                equivalent_pairs.append((hist1, hist2, test_results))
            else:
                different_pairs.append((hist1, hist2, test_results))
    
    print(f"\nFound {len(equivalent_pairs)} equivalent pairs and {len(different_pairs)} different pairs")
    
    print(f"\nSample equivalent pairs:")
    for hist1, hist2, test_results in equivalent_pairs[:3]:
        print(f"  '{hist1}' ≡ '{hist2}'")
        print(f"    {hist1}: {dict(cssr.history_data[hist1])}")
        print(f"    {hist2}: {dict(cssr.history_data[hist2])}")
        print(f"    Test: {test_results}")
    
    print(f"\nSample different pairs:")
    for hist1, hist2, test_results in different_pairs[:3]:
        print(f"  '{hist1}' ≠ '{hist2}'")
        print(f"    {hist1}: {dict(cssr.history_data[hist1])}")
        print(f"    {hist2}: {dict(cssr.history_data[hist2])}")
        print(f"    Test: {test_results}")


def analyze_state_formation(cssr):
    """Analyze how causal states were formed."""
    print("\n" + "="*60)
    print("CAUSAL STATE FORMATION ANALYSIS")
    print("="*60)
    
    for state in cssr.causal_states:
        print(f"\nState {state.state_id}:")
        print(f"  Total histories: {len(state.histories)}")
        print(f"  Total observations: {state.total_count}")
        print(f"  Next symbol distribution: {state.get_probability_distribution()}")
        
        # Group histories by length
        hist_by_length = defaultdict(list)
        for hist in state.histories:
            hist_by_length[len(hist)].append(hist)
        
        print(f"  Histories by length:")
        for length in sorted(hist_by_length.keys()):
            hists = hist_by_length[length]
            print(f"    Length {length}: {len(hists)} histories")
            
            # Show a few examples with their distributions
            for hist in hists[:3]:
                count = cssr.history_counts[hist]
                dist = cssr.history_data[hist]
                print(f"      '{hist}' ({count} obs): {dict(dist)}")


def analyze_unassigned_histories(cssr):
    """Analyze histories that weren't assigned to any state."""
    print("\n" + "="*60)
    print("UNASSIGNED HISTORIES ANALYSIS")
    print("="*60)
    
    assigned_histories = set()
    for state in cssr.causal_states:
        assigned_histories.update(state.histories)
    
    all_sufficient = cssr.get_sufficient_histories()
    unassigned = all_sufficient - assigned_histories
    
    print(f"Total sufficient histories: {len(all_sufficient)}")
    print(f"Assigned histories: {len(assigned_histories)}")
    print(f"Unassigned histories: {len(unassigned)}")
    
    if unassigned:
        print(f"\nSample unassigned histories:")
        for hist in list(unassigned)[:10]:
            count = cssr.history_counts[hist]
            dist = cssr.history_data[hist]
            print(f"  '{hist}' ({count} obs): {dict(dist)}")


def analyze_ground_truth_comparison(data_list):
    """Analyze ground truth causal states from the data."""
    print("\n" + "="*60)
    print("GROUND TRUTH ANALYSIS")
    print("="*60)
    
    # Count ground truth causal states
    gt_state_counts = Counter()
    machine_states = defaultdict(set)
    
    for item in data_list:
        machine_id = item['machine_id']
        causal_state = item['causal_state']
        gt_state_counts[causal_state] += 1
        machine_states[machine_id].add(causal_state)
    
    print(f"Ground truth causal state distribution:")
    for state, count in gt_state_counts.most_common():
        print(f"  {state}: {count} observations")
    
    print(f"\nStates per machine:")
    for machine_id in sorted(machine_states.keys()):
        states = machine_states[machine_id]
        print(f"  Machine {machine_id}: {len(states)} states: {sorted(states)}")


def main():
    print("Detailed Classical CSSR Analysis")
    print("="*60)
    
    # Load dataset
    dataset_dict = torch.load('data/small_test/train_dataset.pt', weights_only=False)
    train_data = dataset_dict['data']
    metadata = dataset_dict['metadata']
    
    print(f"Dataset: {len(train_data)} examples")
    print(f"Alphabet: {metadata['alphabet']}")
    
    # Initialize and run CSSR
    cssr = ClassicalCSSR(
        significance_level=0.05,
        min_count=3,
        test_type="chi_square"
    )
    
    cssr.load_from_raw_data(train_data, metadata)
    
    # Analyze before running CSSR
    analyze_history_distributions(cssr)
    analyze_ground_truth_comparison(train_data)
    
    # Run CSSR
    print(f"\n" + "="*60)
    print("RUNNING CSSR")
    print("="*60)
    cssr.run_cssr(max_iterations=10)
    
    # Analyze results
    analyze_statistical_tests(cssr)
    analyze_state_formation(cssr)
    analyze_unassigned_histories(cssr)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()