#!/usr/bin/env python3
"""
Machine Distance Analysis Tool

Compare CSSR-discovered machines against ground truth using three distance metrics:
1. State Mapping Distance (Hungarian algorithm + Jensen-Shannon divergence)
2. Symbol Distribution Distance (JS divergence between emission distributions)  
3. Transition Structure Distance (Graph-based metrics)

Usage:
    python analyze_machine_distances.py <dataset_name> [--comprehensive] [--results-dir <path>] [--output-dir <path>]

Examples:
    python analyze_machine_distances.py biased_exp
    python analyze_machine_distances.py biased_exp --comprehensive
    python analyze_machine_distances.py biased_exp --comprehensive --output-dir results/analysis
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_cssr.evaluation.machine_distance import MachineDistanceCalculator
from neural_cssr.evaluation.utils.data_loading import load_experiment_data
from neural_cssr.evaluation.utils.visualization import create_distance_visualizations, create_comprehensive_visualizations


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
    parser.add_argument('--comprehensive', action='store_true',
                       help='Include theoretical ε-machine analysis (information-theoretic, causal equivalence, optimality)')
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
            analysis_type = "comprehensive ε-machine analysis" if args.comprehensive else "empirical distance metrics"
            print(f"\nComputing {analysis_type}...")
        
        calculator = MachineDistanceCalculator()
        
        if args.comprehensive:
            results = calculator.compute_comprehensive_analysis(cssr_results, ground_truth_machines)
        else:
            results = calculator.compute_all_distances(cssr_results, ground_truth_machines)
        
        if not args.quiet:
            metrics_computed = "theoretical + empirical metrics" if args.comprehensive else "empirical distance metrics"
            print(f"✓ Computed {metrics_computed}")
        
        # Print results summary
        if not args.quiet:
            print_results_summary(results, args.comprehensive)
        
        # Generate reports
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not args.quiet:
            print(f"\nGenerating analysis reports...")
        
        # Markdown report
        if args.comprehensive:
            # For comprehensive analysis, we need to extract the empirical part for legacy report generation
            # TODO: Implement comprehensive report generation
            empirical_results = results.get('empirical_analysis', {})
            # Add metadata from the comprehensive results
            empirical_results['metadata'] = results.get('metadata', {})
            # Add a summary that reflects the comprehensive analysis
            empirical_results['summary'] = {
                'overall_quality_score': results.get('combined_assessment', {}).get('consensus_quality_score', 0.0),
                'overall_distance_score': 1.0 - results.get('combined_assessment', {}).get('consensus_quality_score', 0.0),
                'confidence': results.get('combined_assessment', {}).get('confidence_level', 0.0),
                'best_metric': 'comprehensive_analysis',
                'interpretation': {
                    'overall_quality': results.get('combined_assessment', {}).get('research_interpretation', 'No interpretation available'),
                    'confidence_level': f"Agreement level: {results.get('combined_assessment', {}).get('methodological_agreement', {}).get('agreement_level', 'unknown')}",
                    'strongest_correspondence': f"Theoretical quality: {results.get('combined_assessment', {}).get('overall_theoretical_quality', 0.0):.3f}"
                },
                'recommendations': results.get('combined_assessment', {}).get('methodological_agreement', {}).get('recommendation', 'No recommendations available').split('. ')
            }
            report_path = calculator.generate_report(empirical_results, str(output_dir / 'machine_distance_report.md'))
        else:
            report_path = calculator.generate_report(results, str(output_dir / 'machine_distance_report.md'))
        
        # Visualizations
        viz_files = []
        if not args.no_visualizations:
            try:
                if args.comprehensive:
                    # Use comprehensive visualizations for theoretical + empirical metrics
                    viz_files = create_comprehensive_visualizations(results, str(output_dir))
                else:
                    # Use empirical-only visualizations
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
        if args.comprehensive:
            combined = results.get('combined_assessment', {})
            print(f"\nSUMMARY: Consensus={combined.get('consensus_quality_score', 0.0):.3f}, "
                  f"Theoretical={combined.get('overall_theoretical_quality', 0.0):.3f}, "
                  f"Empirical={combined.get('overall_empirical_quality', 0.0):.3f}, "
                  f"Confidence={combined.get('confidence_level', 0.0):.3f}")
        else:
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


def print_results_summary(results, is_comprehensive=False):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    if is_comprehensive:
        # Handle comprehensive analysis results
        combined = results.get('combined_assessment', {})
        theoretical = results.get('theoretical_analysis', {})
        empirical = results.get('empirical_analysis', {})
        
        print(f"Analysis Type: Comprehensive ε-machine Analysis")
        print(f"Consensus Quality Score: {combined.get('consensus_quality_score', 0.0):.3f}")
        print(f"Theoretical Quality: {combined.get('overall_theoretical_quality', 0.0):.3f}")
        print(f"Empirical Quality: {combined.get('overall_empirical_quality', 0.0):.3f}")
        print(f"Confidence Level: {combined.get('confidence_level', 0.0):.3f}")
        
        # Methodological agreement
        agreement = combined.get('methodological_agreement', {})
        print(f"Agreement Level: {agreement.get('agreement_level', 'unknown')}")
        
        # Research interpretation
        print(f"\nResearch Interpretation:")
        print(f"  {combined.get('research_interpretation', 'No interpretation available')}")
        
        # Theoretical metrics detail
        print(f"\n" + "-"*40)
        print("THEORETICAL METRICS")
        print("-"*40)
        
        info_theoretic = theoretical.get('information_theoretic_distance', {})
        causal_equiv = theoretical.get('causal_equivalence_distance', {})
        optimality = theoretical.get('optimality_analysis', {})
        
        print(f"Information Theoretic:")
        print(f"  Quality Score: {info_theoretic.get('quality_assessment', {}).get('quality_score', 0.0):.3f}")
        print(f"  Statistical Complexity Distance: {info_theoretic.get('distances', {}).get('statistical_complexity_distance', 0.0):.4f}")
        print(f"  Entropy Rate Distance: {info_theoretic.get('distances', {}).get('entropy_rate_distance', 0.0):.4f}")
        
        print(f"Causal Equivalence:")
        print(f"  Quality Score: {causal_equiv.get('quality_assessment', {}).get('quality_score', 0.0):.3f}")
        print(f"  Equivalence Score: {causal_equiv.get('causal_equivalence_score', 0.0):.3f}")
        print(f"  Refinement Type: {causal_equiv.get('state_refinement_analysis', {}).get('refinement_summary', {}).get('refinement_type', 'unknown')}")
        
        print(f"Optimality Analysis:")
        print(f"  Overall Score: {optimality.get('overall_optimality_score', 0.0):.3f}")
        print(f"  Assessment: {optimality.get('optimality_assessment', {}).get('assessment_level', 'unknown')}")
        
        # Empirical metrics (brief)
        print(f"\n" + "-"*40)
        print("EMPIRICAL METRICS")
        print("-"*40)
        
        symbol_dist = empirical.get('symbol_distribution_distance', {})
        state_mapping = empirical.get('state_mapping_distance', {})
        
        print(f"Symbol Distribution:")
        print(f"  Average JS Divergence: {symbol_dist.get('average_js_divergence', 0.0):.4f}")
        coverage = symbol_dist.get('coverage_score', {})
        print(f"  Coverage: {coverage.get('coverage_fraction', 0.0):.1%}")
        
        print(f"State Mapping:")
        print(f"  Average Assignment Cost: {state_mapping.get('average_cost', 0.0):.4f}")
        
    else:
        # Handle empirical-only results (backward compatibility)
        summary = results.get('summary', {})
        
        print(f"Analysis Type: Empirical Distance Analysis")
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
        
        # Key metrics for empirical-only
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
        
        # Top mappings for empirical-only
        mappings = symbol_dist.get('state_mappings', [])[:3]
        if mappings:
            print(f"\nTop State Mappings:")
            for i, mapping in enumerate(mappings, 1):
                disc_state = mapping.get('discovered_state', 'Unknown')
                best_match = mapping.get('best_match', {})
                js_div = mapping.get('js_divergence', 0.0)
                print(f"  {i}. {disc_state} → {best_match.get('full_name', 'Unknown')} (JS: {js_div:.4f})")
        
        # Recommendations for empirical-only
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations[:3], 1):  # Top 3
                print(f"  {i}. {rec}")
    
    # Add comprehensive-specific mappings
    if is_comprehensive:
        print(f"\n" + "-"*40)
        print("KEY STATE MAPPINGS")
        print("-"*40)
        
        # Get mappings from causal equivalence analysis
        causal_equiv = theoretical.get('causal_equivalence_distance', {})
        mappings = causal_equiv.get('equivalence_class_mapping', [])[:3]
        if mappings:
            print(f"Top Causal Equivalence Mappings:")
            for i, mapping in enumerate(mappings, 1):
                disc_state = mapping.get('discovered_state', 'Unknown')
                best_match = mapping.get('best_match', {})
                if best_match:
                    strength = best_match.get('equivalence_strength', 0.0)
                    print(f"  {i}. {disc_state} → {best_match.get('ground_truth_state', 'Unknown')} (strength: {strength:.3f})")


if __name__ == '__main__':
    sys.exit(main())