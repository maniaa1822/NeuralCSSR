#!/usr/bin/env python3
"""
Test script for new epsilon machine distance metrics.
Tests the information theoretic, causal equivalence, and optimality analysis metrics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_cssr.evaluation.metrics.information_theoretic import InformationTheoreticDistance
from neural_cssr.evaluation.metrics.causal_equivalence import CausalEquivalenceDistance
from neural_cssr.evaluation.metrics.optimality_analysis import OptimalityAnalysis
from neural_cssr.evaluation.utils.data_loading import load_experiment_data


def test_epsilon_metrics():
    """Test all three new epsilon machine metrics."""
    
    print("Testing Epsilon Machine Distance Metrics")
    print("=" * 50)
    
    try:
        # Load biased_exp data
        print("Loading biased_exp dataset...")
        data = load_experiment_data('biased_exp')
        cssr_results = data['cssr_results']
        ground_truth_machines = data['ground_truth_machines']
        
        print(f"✓ Loaded CSSR results")
        print(f"✓ Loaded {len(ground_truth_machines)} ground truth machines")
        
        # Test Information Theoretic Distance
        print("\n" + "-" * 30)
        print("Testing Information Theoretic Distance")
        print("-" * 30)
        
        info_metric = InformationTheoreticDistance()
        info_results = info_metric.compute(cssr_results, ground_truth_machines)
        
        print(f"Statistical Complexity Distance: {info_results['distances']['statistical_complexity_distance']:.4f}")
        print(f"Entropy Rate Distance: {info_results['distances']['entropy_rate_distance']:.4f}")
        print(f"Combined Information Distance: {info_results['combined_information_distance']:.4f}")
        print(f"Theoretical Optimality Ratio: {info_results['theoretical_optimality_ratio']:.4f}")
        print(f"Quality Level: {info_results['quality_assessment']['quality_level']}")
        
        # Test Causal Equivalence Distance
        print("\n" + "-" * 30)
        print("Testing Causal Equivalence Distance")
        print("-" * 30)
        
        causal_metric = CausalEquivalenceDistance()
        causal_results = causal_metric.compute(cssr_results, ground_truth_machines)
        
        print(f"Causal Equivalence Score: {causal_results['causal_equivalence_score']:.4f}")
        print(f"Prediction Consistency: {causal_results['future_prediction_consistency']['consistency_score']:.4f}")
        print(f"Refinement Type: {causal_results['state_refinement_analysis']['refinement_summary']['refinement_type']}")
        print(f"Quality Level: {causal_results['quality_assessment']['quality_level']}")
        
        # Show some equivalence mappings
        mappings = causal_results['equivalence_class_mapping'][:3]  # Show first 3
        print("\nTop Equivalence Mappings:")
        for mapping in mappings:
            best_match = mapping['best_match']
            if best_match:
                print(f"  {mapping['discovered_state']} → {best_match['ground_truth_state']} "
                      f"(strength: {best_match['equivalence_strength']:.3f})")
        
        # Test Optimality Analysis
        print("\n" + "-" * 30)
        print("Testing Optimality Analysis")
        print("-" * 30)
        
        optimality_metric = OptimalityAnalysis()
        optimality_results = optimality_metric.compute(cssr_results, ground_truth_machines)
        
        print(f"Minimality Score: {optimality_results['minimality_score']:.4f}")
        print(f"Unifilarity Score: {optimality_results['unifilarity_score']:.4f}")
        print(f"Causal Sufficiency Score: {optimality_results['causal_sufficiency_score']:.4f}")
        print(f"Prediction Optimality: {optimality_results['prediction_optimality']:.4f}")
        print(f"Overall Optimality Score: {optimality_results['overall_optimality_score']:.4f}")
        print(f"Assessment Level: {optimality_results['optimality_assessment']['assessment_level']}")
        
        # Show theoretical bounds
        bounds = optimality_results['theoretical_bounds']
        print(f"\nTheoretical Bounds:")
        print(f"  Minimum States: {bounds['minimum_possible_states']}")
        print(f"  Optimal Complexity: {bounds['optimal_statistical_complexity']:.4f}")
        print(f"  Optimal Entropy Rate: {bounds['optimal_entropy_rate']:.4f}")
        
        # Test Comprehensive Integration
        print("\n" + "-" * 30)
        print("Testing Comprehensive Integration")
        print("-" * 30)
        
        from neural_cssr.evaluation.machine_distance import MachineDistanceCalculator
        
        calculator = MachineDistanceCalculator()
        comprehensive_results = calculator.compute_comprehensive_analysis(cssr_results, ground_truth_machines)
        
        combined = comprehensive_results['combined_assessment']
        
        print(f"Theoretical Quality: {combined['overall_theoretical_quality']:.4f}")
        print(f"Empirical Quality: {combined['overall_empirical_quality']:.4f}")
        print(f"Consensus Quality: {combined['consensus_quality_score']:.4f}")
        print(f"Confidence Level: {combined['confidence_level']:.4f}")
        print(f"Agreement Level: {combined['methodological_agreement']['agreement_level']}")
        
        # Test backward compatibility
        print("\nTesting Backward Compatibility...")
        original_results = calculator.compute_all_distances(cssr_results, ground_truth_machines)
        print(f"✓ Original method works - Quality: {original_results['summary']['overall_quality_score']:.3f}")
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        
        # Enhanced Summary
        print(f"\nCOMPREHENSIVE SUMMARY:")
        print(f"Individual Metrics:")
        print(f"  Information Theoretic Quality: {info_results['quality_assessment']['quality_score']:.3f}")
        print(f"  Causal Equivalence Quality: {causal_results['quality_assessment']['quality_score']:.3f}")
        print(f"  Optimality Score: {optimality_results['overall_optimality_score']:.3f}")
        print(f"Integrated Analysis:")
        print(f"  Theoretical Quality: {combined['overall_theoretical_quality']:.3f}")
        print(f"  Empirical Quality: {combined['overall_empirical_quality']:.3f}")
        print(f"  Consensus Quality: {combined['consensus_quality_score']:.3f}")
        print(f"  Agreement: {combined['methodological_agreement']['agreement_level']}")
        
        print(f"\nResearch Interpretation:")
        print(f"  {combined['research_interpretation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_epsilon_metrics()
    sys.exit(0 if success else 1)