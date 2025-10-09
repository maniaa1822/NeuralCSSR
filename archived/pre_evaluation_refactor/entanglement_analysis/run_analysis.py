#!/usr/bin/env python3
"""
Example script for running the entanglement analysis pipeline.

Usage:
    uv run python run_analysis.py --model_path /path/to/model --machine seven_state_human --output_dir ./results
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from machines import get_machine, generate_sequence_with_states
from entanglement_analysis.analysis_pipeline import EntanglementAnalysisPipeline


def main():
    parser = argparse.ArgumentParser(description='Run entanglement analysis on nanoGPT')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained nanoGPT checkpoint')
    parser.add_argument('--machine', type=str, default='seven_state_human',
                       help='Machine name (default: seven_state_human)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples for analysis')
    parser.add_argument('--n_formulas', type=int, default=100,
                       help='Number of composition formulas to test')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to pre-generated data (optional)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Load machine
    machine = get_machine(args.machine)
    print(f"Using machine: {machine.display_name}")

    # Generate or load data
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        # Load pre-generated data (simplified for now)
        # This would need to be implemented based on the actual data format
        raise NotImplementedError("Data loading from file not implemented yet")
    else:
        print("Generating data...")
        # Generate fresh data - scale sequences with n_samples for better statistics
        sequences = []
        n_sequences = max(50, args.n_samples // 10)  # At least 50, or scale with samples
        sequence_length = 64  # Use reasonable length

        for i in range(n_sequences):
            sequence, _ = generate_sequence_with_states(
                machine, sequence_length, seed=42 + i
            )
            # Convert string to numpy array of ints (0s and 1s)
            sequence_array = np.array([int(c) for c in sequence], dtype=int)
            sequences.append(sequence_array)

        print(f"Generated {len(sequences)} sequences of length {sequence_length}")

        # Convert to proper format for nanoGPT
        # The model expects sequences as torch tensors
        import torch
        sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    # Initialize pipeline
    pipeline = EntanglementAnalysisPipeline(
        machine=machine,
        model_path=args.model_path,
        output_dir=args.output_dir
    )

    # Run analysis
    results = pipeline.run_full_analysis(
        sequences=sequences,
        n_samples=args.n_samples,
        n_formulas=args.n_formulas
    )

    # Print summary
    fer_scores = results['fer_scores']
    best_layer = min(fer_scores, key=fer_scores.get)
    best_score = fer_scores[best_layer]

    print(f"\n{'='*50}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Best layer: {best_layer}")
    print(f"Best FER score: {best_score:.3f}")
    print(f"Interpretation: {results['fer_interpretations'][best_layer]}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
