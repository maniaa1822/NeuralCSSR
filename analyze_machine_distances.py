#!/usr/bin/env python3
"""
Machine Distance Analysis Tool

Compare CSSR-discovered machines against ground truth using three distance metrics:
1. State Mapping Distance (Hungarian algorithm + Jensen-Shannon divergence)
2. Symbol Distribution Distance (JS divergence between emission distributions)  
3. Transition Structure Distance (Graph-based metrics)

Usage:
    python analyze_machine_distances.py <dataset_name> [--results-dir <path>] [--output-dir <path>]

Example:
    python analyze_machine_distances.py biased_exp
    python analyze_machine_distances.py biased_exp --output-dir results/analysis
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_cssr.evaluation.machine_distance import MachineDistanceCalculator
from neural_cssr.evaluation.utils.data_loading import load_experiment_data
from neural_cssr.evaluation.utils.visualization import create_distance_visualizations


def main():
    """Main entry point for machine distance analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze distance between CSSR-discovered and ground truth machines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('dataset_name', 
                       help='Name of the dataset to analyze (e.g., biased_exp)')
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing CSSR results (default: results)')
    parser.add_argument('--output-dir', 
                       help='Output directory for analysis results (default: results/machine_distance_analysis_<dataset>)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip generating visualization plots')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set default output directory
    if not args.output_dir:
        args.output_dir = f'results/machine_distance_analysis_{args.dataset_name}'
    
    if not args.quiet:
        print(f"Machine Distance Analysis: {args.dataset_name}")
        print("=" * 60)
    
    try:
        # Load data
        if not args.quiet:
            print("Loading data...")
        
        data = load_experiment_data(args.dataset_name, args.results_dir)
        cssr_results = data['cssr_results']
        ground_truth_machines = data['ground_truth_machines']
        
        if not args.quiet:
            print(f"✓ Loaded CSSR results")
            print(f"✓ Loaded {len(ground_truth_machines)} ground truth machines")
        
        # Initialize calculator and compute distances
        if not args.quiet:
            print("\nComputing distance metrics...")
        
        calculator = MachineDistanceCalculator()
        results = calculator.compute_all_distances(cssr_results, ground_truth_machines)
        
        if not args.quiet:
            print("✓ Computed all distance metrics")
        
        # Print results summary
        if not args.quiet:
            print_results_summary(results)
        
        # Generate reports
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.quiet:
            print(f"\nGenerating analysis reports...")
        
        # Markdown report
        report_path = calculator.generate_report(results, str(output_dir / 'machine_distance_report.md'))
        
        # Visualizations
        viz_files = []
        if not args.no_visualizations:
            try:
                viz_files = create_distance_visualizations(results, str(output_dir))
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")
        
        # Final summary
        if not args.quiet:
            print(f"\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"Results saved to: {output_dir}")
            print(f"- Report: {report_path}")
            print(f"- JSON data: {report_path.replace('.md', '.json')}")
            if viz_files:
                print(f"- Visualizations: {len(viz_files)} files")
        
        # Return summary for scripting
        summary = results.get('summary', {})
        print(f"\nSUMMARY: Quality={summary.get('overall_quality_score', 0.0):.3f}, "
              f"Distance={summary.get('overall_distance_score', 0.0):.3f}, "
              f"Confidence={summary.get('confidence', 0.0):.3f}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"Make sure {args.dataset_name} dataset exists and CSSR analysis has been run.")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


def print_results_summary(results):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    summary = results.get('summary', {})
    
    print(f"Overall Quality Score: {summary.get('overall_quality_score', 0.0):.3f}")
    print(f"Overall Distance Score: {summary.get('overall_distance_score', 0.0):.3f}")
    print(f"Confidence Level: {summary.get('confidence', 0.0):.3f}")
    print(f"Best Performing Metric: {summary.get('best_metric', 'Unknown')}")
    
    # Interpretation
    interpretation = summary.get('interpretation', {})
    print(f"\nInterpretation:")
    print(f"- {interpretation.get('overall_quality', 'No interpretation available')}")
    print(f"- {interpretation.get('confidence_level', '')}")
    print(f"- {interpretation.get('strongest_correspondence', '')}")
    
    # Key metrics
    print(f"\n" + "-"*40)
    print("KEY METRICS")
    print("-"*40)
    
    symbol_dist = results.get('symbol_distribution_distance', {})
    state_mapping = results.get('state_mapping_distance', {})
    
    print(f"Symbol Distribution:")
    print(f"  Average JS Divergence: {symbol_dist.get('average_js_divergence', 0.0):.4f}")
    coverage = symbol_dist.get('coverage_score', {})
    print(f"  Coverage: {coverage.get('coverage_fraction', 0.0):.1%}")
    
    print(f"State Mapping:")
    print(f"  Average Assignment Cost: {state_mapping.get('average_cost', 0.0):.4f}")
    print(f"  Quality Score: {state_mapping.get('assignment_quality', {}).get('quality_score', 0.0):.3f}")
    
    # Top mappings
    mappings = symbol_dist.get('state_mappings', [])[:3]
    if mappings:
        print(f"\nTop State Mappings:")
        for i, mapping in enumerate(mappings, 1):
            disc_state = mapping.get('discovered_state', 'Unknown')
            best_match = mapping.get('best_match', {})
            js_div = mapping.get('js_divergence', 0.0)
            print(f"  {i}. {disc_state} → {best_match.get('full_name', 'Unknown')} (JS: {js_div:.4f})")
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations[:3], 1):  # Top 3
            print(f"  {i}. {rec}")


if __name__ == '__main__':
    sys.exit(main())