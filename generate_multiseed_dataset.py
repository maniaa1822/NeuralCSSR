"""
Generate a multi-seed dataset by concatenating sequences from different seeds.
This ensures better coverage of state space and transition dynamics.

Usage:
    uv run python generate_multiseed_dataset.py \
        --machine seven_state_human \
        --n_seeds 10 \
        --length_per_seed 100000 \
        --output experiments/datasets
"""

import argparse
import numpy as np
from pathlib import Path
from machines import get_machine, generate_sequence_with_states, save_dataset


def generate_multiseed_dataset(
    machine_name: str,
    n_seeds: int,
    length_per_seed: int,
    output_root: Path,
    start_seed: int = 0
):
    """Generate dataset by concatenating sequences from multiple seeds."""

    machine = get_machine(machine_name)

    print(f"Generating multi-seed dataset for {machine_name}")
    print(f"  Seeds: {n_seeds} (starting from {start_seed})")
    print(f"  Length per seed: {length_per_seed:,}")
    print(f"  Total length: {n_seeds * length_per_seed:,}")

    all_sequences = []
    all_states = []

    for i in range(n_seeds):
        seed = start_seed + i
        print(f"\n  Generating seed {seed} ({i+1}/{n_seeds})...")

        sequence, states = generate_sequence_with_states(
            machine, length_per_seed, seed=seed
        )

        all_sequences.append(sequence)
        all_states.extend(states)

    # Concatenate all sequences
    full_sequence = ''.join(all_sequences)

    print(f"\n✓ Generated {len(full_sequence):,} tokens")
    print(f"  Total states: {len(all_states):,}")

    # Count state distribution
    from collections import Counter
    state_counts = Counter(all_states)
    print(f"\nState distribution:")
    for state in sorted(machine.states):
        count = state_counts.get(state, 0)
        pct = 100 * count / len(all_states)
        print(f"  {state:6s}: {count:7d} ({pct:5.2f}%)")

    # Save dataset
    output_root = Path(output_root)

    # Create machine-specific directory with suffix
    dataset_dir = output_root / f"{machine_name}_multiseed_{n_seeds}x{length_per_seed//1000}k"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Save files
    metadata = {
        'n_seeds': n_seeds,
        'length_per_seed': length_per_seed,
        'start_seed': start_seed,
        'seeds': list(range(start_seed, start_seed + n_seeds)),
    }

    result = save_dataset(full_sequence, all_states, machine, output_root, metadata)

    print(f"\n✓ Dataset saved to {dataset_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-seed dataset for training'
    )
    parser.add_argument('--machine', type=str, required=True,
                        help='Machine name (e.g., seven_state_human)')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of different seeds to use')
    parser.add_argument('--length_per_seed', type=int, default=100000,
                        help='Sequence length per seed')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='Starting seed value')
    parser.add_argument('--output', type=str, default='experiments/datasets',
                        help='Output directory')

    args = parser.parse_args()

    generate_multiseed_dataset(
        machine_name=args.machine,
        n_seeds=args.n_seeds,
        length_per_seed=args.length_per_seed,
        output_root=Path(args.output),
        start_seed=args.start_seed
    )


if __name__ == '__main__':
    main()
