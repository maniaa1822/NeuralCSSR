#!/usr/bin/env python3
"""
Convenience script to run JS analysis with automatic result organization.

This script runs plot_js_vs_L.py and automatically moves results to the
organized directory structure.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_js_analysis(preset: str, analysis_type: str, L_max: int = 4,
                   n_samples: int = 4000, k: int = 0, custom_model: str = None,
                   custom_data: str = None):
    """
    Run JS analysis and organize results.

    Args:
        preset: Model preset (golden_mean, seven_state_human, even_process)
        analysis_type: Type of analysis (random, instate, cross, both, all)
        L_max: Maximum history length
        n_samples: Number of sample pairs
        k: k-step rollout length (0 to disable)
        custom_model: Path to custom model checkpoint
        custom_data: Path to custom data file
    """
    # Get repository root
    repo_root = Path(__file__).parent.parent.parent

    # Change to nanoGPT directory
    nanogpt_dir = repo_root / 'nanoGPT'
    os.chdir(nanogpt_dir)

    # Build command
    cmd = [
        sys.executable, 'plot_js_vs_L.py',
        '--preset', preset,
        '--analysis_type', analysis_type,
        '--L_max', str(L_max),
        '--n_samples', str(n_samples)
    ]

    if k > 0:
        cmd.extend(['--k', str(k)])

    if custom_model:
        cmd.extend(['--model_ckpt', custom_model])

    if custom_data:
        cmd.extend(['--data', custom_data])

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {nanogpt_dir}")

    # Run the analysis
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Analysis completed successfully!")
        print(result.stdout)
        if result.stderr:
            print("Warnings/errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Analysis failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False

    # Organize results
    print("\nOrganizing results...")
    organize_script = repo_root / 'experiments' / 'js_analysis' / 'organize_results.py'

    # Determine likely output directories based on preset
    source_dirs = []
    if preset == 'golden_mean':
        source_dirs.append(str(nanogpt_dir / 'out-golden-mean-char'))
    elif preset == 'seven_state_human':
        source_dirs.append(str(nanogpt_dir / 'out-seven-state-char'))
    elif preset == 'even_process':
        source_dirs.append(str(nanogpt_dir / 'out-even-process-char'))

    if custom_model:
        # Add the directory containing the custom model
        model_dir = Path(custom_model).parent
        source_dirs.append(str(model_dir))

    if source_dirs:
        organize_cmd = [
            sys.executable, str(organize_script),
            '--source-dirs'] + source_dirs

        try:
            organize_result = subprocess.run(organize_cmd, check=True,
                                           capture_output=True, text=True,
                                           cwd=repo_root)
            print("Results organized successfully!")
            print(organize_result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Organization failed: {e}")
            print(e.stdout)
            print(e.stderr)

    return True


def main():
    parser = argparse.ArgumentParser(description='Run JS analysis with automatic organization')
    parser.add_argument('--preset', default='golden_mean',
                       choices=['golden_mean', 'seven_state_human', 'even_process'],
                       help='Model preset to analyze')
    parser.add_argument('--analysis_type', default='both',
                       choices=['random', 'instate', 'cross', 'both', 'all'],
                       help='Type of analysis to perform')
    parser.add_argument('--L_max', type=int, default=4,
                       help='Maximum history length')
    parser.add_argument('--n_samples', type=int, default=4000,
                       help='Number of sample pairs')
    parser.add_argument('--k', type=int, default=0,
                       help='k-step rollout length (0 to disable)')
    parser.add_argument('--custom_model', type=str,
                       help='Path to custom model checkpoint')
    parser.add_argument('--custom_data', type=str,
                       help='Path to custom data file')

    args = parser.parse_args()

    success = run_js_analysis(
        preset=args.preset,
        analysis_type=args.analysis_type,
        L_max=args.L_max,
        n_samples=args.n_samples,
        k=args.k,
        custom_model=args.custom_model,
        custom_data=args.custom_data
    )

    if success:
        print(f"\nResults are organized in: experiments/js_analysis/{args.preset}/")
    else:
        print("\nAnalysis failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()