#!/usr/bin/env python3
"""Test script for machine distance analysis on biased_exp dataset."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_cssr.evaluation.machine_distance import MachineDistanceCalculator
from neural_cssr.evaluation.utils.data_loading import load_experiment_data
from neural_cssr.evaluation.utils.visualization import create_distance_visualizations


def main():
    """Run machine distance analysis on biased_exp dataset."""
    print("Testing Machine Distance Analysis on biased_exp dataset")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading data...")
        data = load_experiment_data('biased_exp')
        
        cssr_results = data['cssr_results']
        ground_truth_machines = data['ground_truth_machines']
        
        print(f"✓ Loaded CSSR results")
        print(f"✓ Loaded {len(ground_truth_machines)} ground truth machines")
        
        # Initialize calculator
        print("\nInitializing distance calculator...")
        calculator = MachineDistanceCalculator()
        
        # Compute distances
        print("Computing distance metrics...")
        results = calculator.compute_all_distances(cssr_results, ground_truth_machines)
        
        print("✓ Computed all distance metrics")
        
        # Print summary results
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        summary = results.get('summary', {})
        
        print(f"Overall Quality Score: {summary.get('overall_quality_score', 0.0):.3f}")
        print(f"Overall Distance Score: {summary.get('overall_distance_score', 0.0):.3f}")
        print(f"Confidence Level: {summary.get('confidence', 0.0):.3f}")
        print(f"Best Performing Metric: {summary.get('best_metric', 'Unknown')}")
        
        print(f"\nInterpretation:")
        interpretation = summary.get('interpretation', {})
        print(f"- {interpretation.get('overall_quality', 'No interpretation available')}")
        print(f"- {interpretation.get('confidence_level', '')}")
        print(f"- {interpretation.get('strongest_correspondence', '')}")
        
        # Print detailed metrics
        print(f"\n" + "-"*40)
        print("DETAILED METRICS")
        print("-"*40)
        
        # Symbol distribution
        symbol_dist = results.get('symbol_distribution_distance', {})
        print(f"\nSymbol Distribution Distance:")
        print(f"  Average JS Divergence: {symbol_dist.get('average_js_divergence', 0.0):.4f}")
        print(f"  Quality Score: {symbol_dist.get('quality_assessment', {}).get('overall_quality_score', 0.0):.3f}")
        coverage = symbol_dist.get('coverage_score', {})
        print(f"  Coverage: {coverage.get('coverage_fraction', 0.0):.1%} ({coverage.get('well_matched_states', 0)}/{coverage.get('total_ground_truth_states', 0)} states)")
        
        # State mapping
        state_mapping = results.get('state_mapping_distance', {})
        print(f"\nState Mapping Distance:")
        print(f"  Total Assignment Cost: {state_mapping.get('total_cost', 0.0):.4f}")
        print(f"  Average Cost per State: {state_mapping.get('average_cost', 0.0):.4f}")
        print(f"  Quality Score: {state_mapping.get('assignment_quality', {}).get('quality_score', 0.0):.3f}")
        print(f"  Unmatched Discovered: {len(state_mapping.get('unmatched_discovered_states', []))}")
        print(f"  Unmatched Ground Truth: {len(state_mapping.get('unmatched_true_states', []))}")
        
        # Transition structure
        transition_struct = results.get('transition_structure_distance', {})
        print(f"\nTransition Structure Distance:")
        print(f"  Graph Edit Distance: {transition_struct.get('graph_edit_distance', 0.0):.4f}")
        print(f"  Spectral Distance: {transition_struct.get('spectral_distance', 0.0):.4f}")
        print(f"  Connectivity Similarity: {transition_struct.get('connectivity_similarity', 0.0):.4f}")
        
        # Show best state mappings
        print(f"\n" + "-"*40)
        print("BEST STATE MAPPINGS")
        print("-"*40)
        
        mappings = symbol_dist.get('state_mappings', [])[:5]  # Top 5
        for i, mapping in enumerate(mappings, 1):
            disc_state = mapping.get('discovered_state', 'Unknown')
            best_match = mapping.get('best_match', {})
            js_div = mapping.get('js_divergence', 0.0)
            
            print(f"{i}. {disc_state} → {best_match.get('full_name', 'Unknown')} (JS: {js_div:.4f})")
        
        # Show recommendations
        print(f"\n" + "-"*40)
        print("RECOMMENDATIONS")
        print("-"*40)
        
        recommendations = summary.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Generate report
        print(f"\n" + "="*60)
        print("GENERATING REPORTS")
        print("="*60)
        
        # Create output directory
        output_dir = Path('results/machine_distance_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown report
        report_path = calculator.generate_report(results, str(output_dir / 'machine_distance_report.md'))
        print(f"✓ Generated markdown report: {report_path}")
        
        # Generate visualizations
        print("Creating visualizations...")
        try:
            viz_files = create_distance_visualizations(results, str(output_dir))
            print(f"✓ Generated {len(viz_files)} visualization files:")
            for viz_file in viz_files:
                print(f"  - {viz_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not generate visualizations: {e}")
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"All results saved to: {output_dir}")
        
        # Quick validation of expected results
        print(f"\n" + "-"*40)
        print("VALIDATION CHECK")
        print("-"*40)
        
        expected_validation(results)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def expected_validation(results):
    """Validate results match expected patterns from the plan."""
    print("Checking against expected patterns from analysis plan...")
    
    # Expected: State_1 (75.5% "0") should match Machine_10.S0 (80% "0")
    symbol_dist = results.get('symbol_distribution_distance', {})
    mappings = symbol_dist.get('state_mappings', [])
    
    state_1_mapping = None
    for mapping in mappings:
        if mapping.get('discovered_state') == 'State_1':
            state_1_mapping = mapping
            break
    
    if state_1_mapping:
        best_match = state_1_mapping.get('best_match', {})
        js_div = state_1_mapping.get('js_divergence', 1.0)
        machine_id = best_match.get('machine_id', '')
        state_id = best_match.get('state_id', '')
        
        print(f"State_1 mapping: {best_match.get('full_name', 'Unknown')} (JS: {js_div:.4f})")
        
        if machine_id == '10' and state_id == 'S0' and js_div < 0.1:
            print("✓ Expected State_1 → Machine_10.S0 mapping confirmed (low JS divergence)")
        elif machine_id == '10' and state_id == 'S0':
            print(f"✓ Expected State_1 → Machine_10.S0 mapping found (JS: {js_div:.4f})")
        else:
            print(f"⚠ Unexpected mapping for State_1: expected Machine_10.S0")
    
    # Check overall quality
    overall_quality = results.get('summary', {}).get('overall_quality_score', 0.0)
    if overall_quality > 0.5:
        print(f"✓ Good overall quality score: {overall_quality:.3f}")
    else:
        print(f"⚠ Low overall quality score: {overall_quality:.3f}")
    
    # Check if we found the expected number of states
    discovered_states = len(mappings)
    if 4 <= discovered_states <= 6:
        print(f"✓ Reasonable number of discovered states: {discovered_states}")
    else:
        print(f"⚠ Unexpected number of discovered states: {discovered_states}")


if __name__ == '__main__':
    sys.exit(main())