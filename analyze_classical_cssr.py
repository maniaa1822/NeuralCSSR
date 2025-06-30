#!/usr/bin/env python3
"""
Classical CSSR Analysis CLI

End-to-end classical CSSR analysis for unified datasets with
comprehensive evaluation and reporting.

Usage:
    python analyze_classical_cssr.py --dataset datasets/small_exp --output results/classical_analysis
    python analyze_classical_cssr.py --dataset datasets/medium_exp --output results/medium_analysis --parameter-sweep
    python analyze_classical_cssr.py --batch --datasets-dir datasets --output results/batch_analysis
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_cssr.analysis.classical_analyzer import ClassicalCSSRAnalyzer, BatchAnalyzer
from neural_cssr.analysis.dataset_loader import validate_dataset_for_analysis, DatasetValidationError


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classical CSSR Analysis for Unified Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset analysis
  python analyze_classical_cssr.py --dataset datasets/small_exp --output results/small_analysis
  
  # Parameter sweep analysis
  python analyze_classical_cssr.py --dataset datasets/medium_exp --output results/medium_analysis --parameter-sweep
  
  # Analysis with machine distance evaluation
  python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/biased_analysis --distance-analysis
  
  # Batch analysis across multiple datasets
  python analyze_classical_cssr.py --batch --datasets-dir datasets --output results/batch_analysis
  
  # Batch analysis with distance evaluation
  python analyze_classical_cssr.py --batch --datasets-dir datasets --output results/batch_analysis --distance-analysis
  
  # Quick single-parameter analysis
  python analyze_classical_cssr.py --dataset datasets/test_exp --output results/test --max-length 8 --significance 0.05 --no-sweep
        """
    )
    
    # Dataset specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, help='Path to single dataset directory')
    group.add_argument('--batch', action='store_true', help='Run batch analysis on multiple datasets')
    
    parser.add_argument('--datasets-dir', type=str, default='datasets',
                       help='Directory containing datasets for batch analysis (default: datasets)')
    parser.add_argument('--dataset-names', nargs='+', help='Specific dataset names for batch analysis')
    
    # Output configuration
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for analysis results')
    
    # CSSR parameters
    parser.add_argument('--max-length', type=int, default=10,
                       help='Maximum history length for CSSR (default: 10)')
    parser.add_argument('--significance', type=float, default=0.05,
                       help='Statistical significance level (default: 0.05)')
    
    # Analysis options
    parser.add_argument('--parameter-sweep', action='store_true',
                       help='Run parameter sweep analysis (default for single dataset)')
    parser.add_argument('--no-sweep', action='store_true',
                       help='Disable parameter sweep, use single parameter set')
    
    # Validation and dry run
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate datasets without running analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    # Machine distance analysis
    parser.add_argument('--distance-analysis', action='store_true',
                       help='Run machine distance analysis comparing CSSR results to ground truth')
    
    args = parser.parse_args()
    
    # Configure verbosity
    if args.verbose:
        print("Verbose mode enabled")
    
    try:
        if args.batch:
            run_batch_analysis(args)
        else:
            run_single_analysis(args)
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_single_analysis(args):
    """Run analysis on a single dataset."""
    
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    
    print(f"Classical CSSR Analysis")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    # Validate dataset
    print("Validating dataset...")
    try:
        validate_dataset_for_analysis(str(dataset_dir))
        print("✓ Dataset validation passed")
    except DatasetValidationError as e:
        print(f"✗ Dataset validation failed: {e}")
        if not args.validate_only:
            print("Use --validate-only to check dataset structure without running analysis")
        sys.exit(1)
    
    if args.validate_only:
        print("Validation complete. Exiting (--validate-only specified).")
        return
    
    # Determine parameter sweep setting
    parameter_sweep = True  # Default for single analysis
    if args.no_sweep:
        parameter_sweep = False
    elif args.parameter_sweep:
        parameter_sweep = True
    
    print(f"Parameter sweep: {'Enabled' if parameter_sweep else 'Disabled'}")
    if not parameter_sweep:
        print(f"Single parameters: max_length={args.max_length}, significance={args.significance}")
    
    print(f"Machine distance analysis: {'Enabled' if args.distance_analysis else 'Disabled'}")
    if args.distance_analysis:
        print("  Will compare CSSR results against ground truth using 3 distance metrics")
    
    # Create analyzer and run analysis
    analyzer = ClassicalCSSRAnalyzer(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir)
    )
    
    start_time = time.time()
    
    if parameter_sweep:
        results = analyzer.run_complete_analysis(parameter_sweep=True, distance_analysis=args.distance_analysis)
    else:
        results = analyzer.run_complete_analysis(
            max_length=args.max_length,
            significance_level=args.significance,
            parameter_sweep=False,
            distance_analysis=args.distance_analysis
        )
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total time: {total_time:.2f} seconds")
    
    # Extract and display key results
    display_results_summary(results)
    
    # Display output files
    report_files = results.get('report_files', {})
    print(f"\nOutput files:")
    for file_type, file_path in report_files.items():
        print(f"  {file_type}: {file_path}")


def run_batch_analysis(args):
    """Run batch analysis on multiple datasets."""
    
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output)
    
    print(f"Batch Classical CSSR Analysis")
    print(f"Datasets directory: {datasets_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    if not datasets_dir.exists():
        print(f"Error: Datasets directory does not exist: {datasets_dir}")
        sys.exit(1)
    
    # Create batch analyzer
    batch_analyzer = BatchAnalyzer(
        datasets_dir=str(datasets_dir),
        output_dir=str(output_dir)
    )
    
    # Run batch analysis
    start_time = time.time()
    
    results = batch_analyzer.analyze_all_datasets(args.dataset_names, distance_analysis=args.distance_analysis)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total time: {total_time:.2f} seconds")
    
    # Display batch results summary
    display_batch_results_summary(results)
    
    print(f"\nComparative analysis saved to: {output_dir / 'comparative_analysis.json'}")


def display_results_summary(results: dict):
    """Display summary of analysis results."""
    
    dataset_info = results.get('dataset_info', {})
    cssr_results = results.get('cssr_results', {})
    eval_metrics = results.get('evaluation_metrics', {})
    baseline_comp = results.get('baseline_comparison', {})
    
    print(f"\nDataset: {dataset_info.get('dataset_name', 'Unknown')}")
    print(f"Sequences: {dataset_info.get('num_sequences', 0):,}")
    
    # CSSR results
    best_params = cssr_results.get('best_parameters', {})
    if best_params:
        overall_best = best_params.get('overall_best', {})
        params = overall_best.get('parameters', {})
        if params:
            print(f"Best parameters: L={params.get('max_length', 'N/A')}, α={params.get('significance_level', 'N/A')}")
    
    # Get best result info
    if 'parameter_results' in cssr_results:
        param_key = best_params.get('overall_best', {}).get('parameter_key')
        if param_key and param_key in cssr_results['parameter_results']:
            best_result = cssr_results['parameter_results'][param_key]
        else:
            best_result = None
    else:
        best_result = cssr_results
    
    if best_result:
        discovered_states = best_result.get('discovered_structure', {}).get('num_states', 0)
        converged = best_result.get('execution_info', {}).get('converged', False)
        runtime = best_result.get('execution_info', {}).get('runtime_seconds', 0.0)
        
        print(f"Discovered states: {discovered_states}")
        print(f"Converged: {'Yes' if converged else 'No'}")
        print(f"Runtime: {runtime:.2f} seconds")
    
    # Evaluation metrics
    if eval_metrics:
        structure_recovery = eval_metrics.get('structure_recovery', {})
        if structure_recovery:
            structural_sim = structure_recovery.get('structural_similarity', 0.0)
            state_accuracy = structure_recovery.get('state_count_accuracy', 0.0)
            print(f"Structural similarity: {structural_sim:.3f}")
            print(f"State count accuracy: {state_accuracy:.3f}")
        
        pred_perf = eval_metrics.get('prediction_performance', {})
        if pred_perf:
            ce_ratio = pred_perf.get('cross_entropy_ratio', 0.0)
            if ce_ratio > 0:
                print(f"Cross-entropy ratio: {ce_ratio:.3f}")
    
    # Baseline comparison
    if baseline_comp:
        rel_perf = baseline_comp.get('relative_performance', {})
        if rel_perf:
            overall_score = rel_perf.get('overall_score', 0.0)
            category = rel_perf.get('performance_category', 'unknown')
            print(f"Overall performance: {overall_score:.3f} ({category})")


def display_batch_results_summary(results: dict):
    """Display summary of batch analysis results."""
    
    summary = results.get('summary', {})
    aggregate = results.get('aggregate_metrics', {})
    
    print(f"Total datasets: {summary.get('total_datasets', 0)}")
    print(f"Successful analyses: {summary.get('successful_analyses', 0)}")
    print(f"Failed analyses: {summary.get('failed_analyses', 0)}")
    
    if aggregate:
        # Structure recovery aggregate
        structure_agg = aggregate.get('structure_recovery', {})
        if structure_agg:
            print(f"\nStructure Recovery (across datasets):")
            print(f"  Mean accuracy: {structure_agg.get('mean', 0.0):.3f}")
            print(f"  Range: {structure_agg.get('min', 0.0):.3f} - {structure_agg.get('max', 0.0):.3f}")
        
        # Prediction performance aggregate
        pred_agg = aggregate.get('prediction_performance', {})
        if pred_agg:
            print(f"\nPrediction Performance (across datasets):")
            print(f"  Mean cross-entropy ratio: {pred_agg.get('mean_cross_entropy_ratio', 0.0):.3f}")
            print(f"  Best ratio: {pred_agg.get('best_cross_entropy_ratio', 0.0):.3f}")
    
    # Show top performing datasets
    dataset_comparison = results.get('dataset_comparison', {})
    if dataset_comparison:
        print(f"\nDataset Performance Summary:")
        for dataset, comparison in list(dataset_comparison.items())[:5]:  # Top 5
            structure_score = comparison.get('structure_recovery_score', 0.0)
            print(f"  {dataset}: {structure_score:.3f}")


if __name__ == '__main__':
    main()