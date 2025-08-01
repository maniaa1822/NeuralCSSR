#!/usr/bin/env python3
"""
Classical CSSR Analysis CLI

End-to-end classical CSSR analysis using the transCSSR wrapper with
comprehensive evaluation and reporting. Supports the complete workflow
from dataset generation through CSSR analysis.

Complete Workflow:
    1. Generate dataset: python generate_unified_dataset.py --preset biased --output datasets/biased_exp
    2. Convert format: python convert_to_transcssr.py --dataset-path datasets/biased_exp --output-file transcssr_data.dat
    3. Run analysis: python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/analysis

Usage Examples:
    # Unified dataset analysis with parameter sweep
    python analyze_classical_cssr.py --dataset datasets/small_exp --output results/classical_analysis
    
    # Direct transCSSR .dat file analysis  
    python analyze_classical_cssr.py --dat-file transcssr_data.dat --output results/dat_analysis
    
    # Batch analysis across multiple datasets
    python analyze_classical_cssr.py --batch --datasets-dir datasets --output results/batch_analysis
    
    # Analysis with machine distance evaluation
    python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/analysis --distance-analysis
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
from collections import Counter

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neural_cssr.analysis.classical_analyzer import ClassicalCSSRAnalyzer, BatchAnalyzer
from neural_cssr.analysis.dataset_loader import validate_dataset_for_analysis, DatasetValidationError
from neural_cssr.classical.transcssr_wrapper import TransCSSRWrapper


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classical CSSR Analysis for Unified Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # COMPLETE WORKFLOW: Dataset generation → CSSR analysis
  # 1. Generate synthetic dataset
  python generate_unified_dataset.py --preset biased --output datasets/biased_exp
  
  # 2. Analyze with unified dataset format (recommended)
  python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/biased_analysis --parameter-sweep
  
  # ALTERNATIVE: Convert and analyze .dat file
  # 2a. Convert to transCSSR format (optional)
  python convert_to_transcssr.py --dataset-path datasets/biased_exp --output-file data.dat --per-seq-burn 8
  
  # 2b. Analyze .dat file directly
  python analyze_classical_cssr.py --dat-file data.dat --output results/dat_analysis --parameter-sweep
  
  # BATCH ANALYSIS: Multiple datasets
  python analyze_classical_cssr.py --batch --datasets-dir datasets --output results/batch_analysis
  
  # ADVANCED OPTIONS
  # Analysis with machine distance evaluation (compares to ground truth)
  python analyze_classical_cssr.py --dataset datasets/biased_exp --output results/analysis --distance-analysis
  
  # Single parameter analysis (faster)
  python analyze_classical_cssr.py --dataset datasets/test_exp --output results/test --max-length 8 --significance 0.05 --no-sweep
  
  # Validation only (check dataset structure)
  python analyze_classical_cssr.py --dataset datasets/test_exp --output results/test --validate-only
        """
    )
    
    # Dataset specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', type=str, 
                      help='Path to single dataset directory (unified format from generate_unified_dataset.py)')
    group.add_argument('--dat-file', type=str, 
                      help='Path to single .dat file (transCSSR format from convert_to_transcssr.py)')
    group.add_argument('--batch', action='store_true', 
                      help='Run batch analysis on multiple datasets in unified format')
    
    parser.add_argument('--datasets-dir', type=str, default='datasets',
                       help='Directory containing datasets for batch analysis (default: datasets)')
    parser.add_argument('--dataset-names', nargs='+', 
                       help='Specific dataset names for batch analysis (e.g., biased_exp small_exp)')
    
    # Output configuration
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for analysis results')
    
    # CSSR parameters
    parser.add_argument('--max-length', type=int, default=10,
                       help='Maximum history length (L_max) for CSSR algorithm (default: 10)')
    parser.add_argument('--significance', type=float, default=0.05,
                       help='Statistical significance level (alpha) for chi-square tests (default: 0.05)')
    
    # Analysis options
    parser.add_argument('--parameter-sweep', action='store_true',
                       help='Run parameter sweep analysis across multiple L_max and alpha values')
    parser.add_argument('--no-sweep', action='store_true',
                       help='Disable parameter sweep, use single parameter set only')
    
    # Advanced options  
    parser.add_argument('--distance-analysis', action='store_true',
                       help='Run machine distance analysis comparing CSSR results to ground truth machines')
    
    # Conversion parameters (for .dat file workflow)
    parser.add_argument('--per-seq-burn', type=int, default=8,
                       help='Per-sequence burn-in for conversion (symbols to remove at start of each sequence)')
    parser.add_argument('--global-burn', type=int, default=0,
                       help='Global burn-in for conversion (symbols to remove after concatenation)')
    
    # Validation and debugging
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate datasets without running analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output for debugging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be analyzed without running CSSR')
    
    args = parser.parse_args()
    
    # Configure verbosity
    if args.verbose:
        print("Verbose mode enabled")
    
    try:
        if args.dry_run:
            run_dry_run_analysis(args)
        elif args.batch:
            run_batch_analysis(args)
        elif args.dat_file:
            run_dat_file_analysis(args)
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


def run_dry_run_analysis(args):
    """Show what would be analyzed without running CSSR."""
    print("DRY RUN MODE - No analysis will be performed")
    print("=" * 50)
    
    if args.batch:
        datasets_dir = Path(args.datasets_dir)
        if not datasets_dir.exists():
            print(f"✗ Datasets directory not found: {datasets_dir}")
            return
        
        dataset_names = args.dataset_names
        if dataset_names is None:
            dataset_names = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
        
        print(f"Would analyze {len(dataset_names)} datasets:")
        for dataset in dataset_names:
            dataset_path = datasets_dir / dataset
            if dataset_path.exists():
                print(f"  ✓ {dataset}")
            else:
                print(f"  ✗ {dataset} (not found)")
                
    elif args.dat_file:
        dat_file = Path(args.dat_file)
        print(f"Would analyze .dat file: {dat_file}")
        if dat_file.exists():
            print(f"  ✓ File exists ({dat_file.stat().st_size} bytes)")
        else:
            print(f"  ✗ File not found")
    else:
        dataset_dir = Path(args.dataset)
        print(f"Would analyze dataset: {dataset_dir}")
        if dataset_dir.exists():
            print(f"  ✓ Directory exists")
            try:
                from neural_cssr.analysis.dataset_loader import validate_dataset_for_analysis
                validate_dataset_for_analysis(str(dataset_dir))
                print("  ✓ Dataset structure valid")
            except Exception as e:
                print(f"  ✗ Dataset validation failed: {e}")
        else:
            print(f"  ✗ Directory not found")
    
    # Show parameter settings
    parameter_sweep = not args.no_sweep
    print(f"\nParameter configuration:")
    print(f"  Parameter sweep: {'Enabled' if parameter_sweep else 'Disabled'}")
    if not parameter_sweep:
        print(f"  Single parameters: L_max={args.max_length}, α={args.significance}")
    else:
        print(f"  Sweep ranges: L_max=[6,8,9,10,12], α=[0.001,0.01,0.05,0.1]")
    
    print(f"  Distance analysis: {'Enabled' if args.distance_analysis else 'Disabled'}")
    print(f"  Output directory: {args.output}")
    
    print("\nTo run actual analysis, remove --dry-run flag")


def run_dat_file_analysis(args):
    """Run analysis on a single .dat file (transCSSR format)."""
    
    dat_file = Path(args.dat_file)
    output_dir = Path(args.output)
    
    print(f"Classical CSSR Analysis - transCSSR .dat file format")
    print(f"Input file: {dat_file}")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    # Validate .dat file exists
    if not dat_file.exists():
        print(f"✗ .dat file not found: {dat_file}")
        print("\nTip: Generate .dat file from unified dataset:")
        print("  python convert_to_transcssr.py --dataset-path datasets/your_dataset --output-file your_file.dat")
        sys.exit(1)
    
    # Load .dat file
    try:
        with open(dat_file, 'r') as f:
            string_y = f.read().strip()
        
        if not string_y:
            print(f"✗ .dat file is empty: {dat_file}")
            sys.exit(1)
        
        # Validate content
        valid_symbols = set('01')
        file_symbols = set(string_y)
        if not file_symbols.issubset(valid_symbols):
            print(f"✗ Invalid symbols in .dat file: {file_symbols - valid_symbols}")
            print("Expected binary symbols: 0, 1")
            sys.exit(1)
        
        string_x = '0' * len(string_y)  # Input sequence for epsilon-machine (constant)
        print(f"✓ Loaded .dat file: {len(string_y):,} symbols")
        print(f"  Symbol distribution: {dict(Counter(string_y))}")
        
    except Exception as e:
        print(f"✗ Error reading .dat file: {e}")
        sys.exit(1)
    
    # Determine parameter sweep setting
    parameter_sweep = True  # Default for .dat file analysis
    if args.no_sweep:
        parameter_sweep = False
    elif args.parameter_sweep:
        parameter_sweep = True
    
    print(f"Analysis mode: {'Parameter sweep' if parameter_sweep else 'Single parameter'}")
    if not parameter_sweep:
        print(f"  Parameters: L_max={args.max_length}, α={args.significance}")
    else:
        print(f"  Sweep ranges: L_max=[6,8,9,10,12], α=[0.001,0.01,0.05,0.1]")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import transCSSR wrapper
    try:
        from neural_cssr.classical.transcssr_wrapper import TransCSSRWrapper
    except ImportError as e:
        print(f"✗ Could not import TransCSSRWrapper: {e}")
        print("Make sure the src/neural_cssr directory is in your Python path")
        sys.exit(1)
    
    start_time = time.time()
    
    # Run analysis
    wrapper = TransCSSRWrapper(significance_level=args.significance)
    
    if parameter_sweep:
        print("Running parameter sweep analysis...")
        
        # Define parameter ranges (standard ranges for comprehensive analysis)
        max_lengths = [6, 8, 9, 10, 12]  # Added L=9 based on reference CSSR validation
        significance_levels = [0.001, 0.01, 0.05, 0.1]
        
        if args.verbose:
            print(f"  Testing {len(max_lengths)} × {len(significance_levels)} = {len(max_lengths) * len(significance_levels)} parameter combinations")
        
        results = wrapper.run_parameter_sweep(string_x, string_y, max_lengths, significance_levels)
        
        # Get best result
        best_params = results['best_parameters']['overall_best']
        if best_params:
            best_result = results['parameter_results'][best_params['parameter_key']]
        else:
            print("✗ Parameter sweep failed to find any successful results")
            sys.exit(1)
        
    else:
        print(f"Running single analysis (L_max={args.max_length}, α={args.significance})")
        results = wrapper.run_cssr(string_x, string_y, args.max_length)
        best_result = results
        best_params = {
            'parameters': {
                'max_length': args.max_length,
                'significance_level': args.significance
            },
            'parameter_key': f"L{args.max_length}_alpha{args.significance}"
        }
    
    total_time = time.time() - start_time
    
    # Save results
    import json
    results_file = output_dir / "dat_file_cssr_results.json"
    
    # Create summary for .dat file analysis
    summary_results = {
        'input_file': str(dat_file),
        'dataset_info': {
            'file_name': dat_file.name,
            'sequence_length': len(string_y),
            'alphabet': list(set(string_y)),
            'first_100_symbols': string_y[:100]
        },
        'analysis_results': results if parameter_sweep else {'single_result': results},
        'best_result': best_result,
        'best_parameters': best_params,
        'analysis_info': {
            'total_time': total_time,
            'parameter_sweep': parameter_sweep,
            'transCSSR_version': 'reference'
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total time: {total_time:.2f} seconds")
    
    # Display results
    print(f"\nDataset: {dat_file.name}")
    print(f"Sequence length: {len(string_y):,} symbols")
    
    if parameter_sweep:
        params = best_params['parameters']
        print(f"Best parameters: L={params['max_length']}, α={params['significance_level']}")
    
    discovered_states = best_result['discovered_structure']['num_states']
    runtime = best_result['execution_info']['runtime_seconds']
    
    print(f"Discovered states: {discovered_states}")
    print(f"Runtime: {runtime:.2f} seconds")
    
    print(f"\nOutput files:")
    print(f"  results: {results_file}")
    
    if args.verbose:
        print(f"\nDetailed results:")
        if parameter_sweep:
            print(f"  Parameter combinations tested: {len(results.get('parameter_results', {}))}")
            print(f"  Best parameter combination: {best_params.get('parameter_key', 'N/A')}")
        print(f"  Algorithm: {best_result.get('execution_info', {}).get('algorithm', 'transCSSR')}")
        print(f"  Convergence: {best_result.get('execution_info', {}).get('converged', 'N/A')}")


def run_single_analysis(args):
    """Run analysis on a single dataset."""
    
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    
    print(f"Classical CSSR Analysis - Unified Dataset Format")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("-" * 50)
    
    # Check if dataset conversion might be needed
    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        print("\nTo generate a dataset, use:")
        print("  python generate_unified_dataset.py --preset biased --output datasets/biased_exp")
        sys.exit(1)
    
    # Validate dataset
    print("Validating dataset structure...")
    try:
        validate_dataset_for_analysis(str(dataset_dir))
        print("✓ Dataset validation passed")
        
        # Load dataset info for summary
        from neural_cssr.analysis.dataset_loader import UnifiedDatasetLoader
        loader = UnifiedDatasetLoader()
        summary = loader.get_dataset_summary(str(dataset_dir))
        
        print(f"  Dataset name: {summary.get('dataset_name', 'Unknown')}")
        seq_stats = summary.get('sequence_stats', {})
        if seq_stats:
            print(f"  Training sequences: {seq_stats.get('train_count', 0):,}")
            print(f"  Average sequence length: {seq_stats.get('avg_length', 0):.1f}")
        
    except DatasetValidationError as e:
        print(f"✗ Dataset validation failed: {e}")
        print("\nCommon issues:")
        print("  - Missing raw_sequences/ directory")
        print("  - Missing train_sequences.txt file")
        print("  - Dataset not in unified format")
        print("\nTo convert existing .dat file to unified format, or regenerate dataset:")
        print("  python generate_unified_dataset.py --preset small --output datasets/new_dataset")
        if not args.validate_only:
            print("Use --validate-only to check dataset structure without running analysis")
        sys.exit(1)
    
    if args.validate_only:
        print("✓ Validation complete. Exiting (--validate-only specified).")
        return
    
    # Determine parameter sweep setting
    parameter_sweep = True  # Default for single analysis
    if args.no_sweep:
        parameter_sweep = False
    elif args.parameter_sweep:
        parameter_sweep = True
    
    print(f"Analysis mode: {'Parameter sweep' if parameter_sweep else 'Single parameter'}")
    if not parameter_sweep:
        print(f"  Parameters: L_max={args.max_length}, α={args.significance}")
    else:
        print(f"  Sweep ranges: L_max=[6,8,9,10,12], α=[0.001,0.01,0.05,0.1]")

    print(f"Distance analysis: {'Enabled' if args.distance_analysis else 'Disabled'}")
    if args.distance_analysis:
        print("  Will compare CSSR results against ground truth using machine distance metrics")
    
    # Create analyzer and run analysis
    try:
        analyzer = ClassicalCSSRAnalyzer(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"✗ Failed to create analyzer: {e}")
        print("\nMake sure all required modules are available:")
        print("  - neural_cssr.analysis.classical_analyzer")
        print("  - neural_cssr.analysis.dataset_loader")
        print("  - neural_cssr.classical.transcssr_wrapper")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        if parameter_sweep:
            results = analyzer.run_complete_analysis(parameter_sweep=True, distance_analysis=args.distance_analysis)
        else:
            results = analyzer.run_complete_analysis(
                max_length=args.max_length,
                significance_level=args.significance,
                parameter_sweep=False,
                distance_analysis=args.distance_analysis
            )
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Check dataset format with --validate-only")
        print("  - Try single parameter analysis with --no-sweep")
        print("  - Use --verbose for detailed error information")
        sys.exit(1)
    
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
        print(f"✗ Datasets directory does not exist: {datasets_dir}")
        print("\nTo generate multiple datasets, use:")
        print("  python generate_unified_dataset.py --preset small --output datasets/small_exp")
        print("  python generate_unified_dataset.py --preset medium --output datasets/medium_exp")
        print("  python generate_unified_dataset.py --preset biased --output datasets/biased_exp")
        sys.exit(1)
    
    # Check available datasets
    available_datasets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
    if not available_datasets:
        print(f"✗ No datasets found in {datasets_dir}")
        print("Generate datasets first using generate_unified_dataset.py")
        sys.exit(1)
    
    print(f"Available datasets: {len(available_datasets)}")
    for dataset in sorted(available_datasets):
        print(f"  - {dataset}")
    
    # Determine which datasets to analyze
    if args.dataset_names:
        datasets_to_analyze = args.dataset_names
        missing = [name for name in datasets_to_analyze if name not in available_datasets]
        if missing:
            print(f"\n✗ Specified datasets not found: {missing}")
            sys.exit(1)
    else:
        datasets_to_analyze = available_datasets
    
    print(f"\nWill analyze {len(datasets_to_analyze)} datasets: {datasets_to_analyze}")
    print(f"Distance analysis: {'Enabled' if args.distance_analysis else 'Disabled'}")
    
    # Create batch analyzer
    try:
        batch_analyzer = BatchAnalyzer(
            datasets_dir=str(datasets_dir),
            output_dir=str(output_dir)
        )
    except Exception as e:
        print(f"✗ Failed to create batch analyzer: {e}")
        sys.exit(1)
    
    # Run batch analysis
    start_time = time.time()
    
    try:
        results = batch_analyzer.analyze_all_datasets(datasets_to_analyze, distance_analysis=args.distance_analysis)
    except Exception as e:
        print(f"✗ Batch analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
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
    print("=" * 60)
    print("Classical CSSR Analysis CLI")
    print("Neural CSSR Project - Complete Workflow Support")
    print("=" * 60)
    print()
    
    main()
    
    print()
    print("=" * 60)
    print("Analysis Complete")
    print("For more information, see: plans_and_guides/classical_cssr_analysis_framework.md")
    print("=" * 60)