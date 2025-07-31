#!/usr/bin/env python3
"""
Generate domain-specific machine datasets for neural CSSR training.

Creates a single long sequence from a domain-specific epsilon-machine
with aligned state trajectories for linear probe training.

Usage:
    python generate_domain_dataset.py --machine even_process --length 100000 --output data/even_process
    python generate_domain_dataset.py --machine golden_mean --length 50000 --output data/golden_mean --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_cssr.machines.domain_specific import (
    EvenProcessMachine, AlternatingMachine, IncompressibleCounterMachine,
    TrulyIncompressibleMachine, ContextSensitiveMachine, Period4Machine, 
    GoldenMeanMachine, SevenStateHumanSequenceMachine
)
from neural_cssr.core.epsilon_machine import EpsilonMachine


def generate_sequence_with_states(machine: EpsilonMachine, length: int, seed: int = None) -> Tuple[str, List[int], Dict[str, int]]:
    """
    Generate a sequence from an epsilon machine while tracking state trajectory.
    
    Args:
        machine: The epsilon machine to generate from
        length: Number of symbols to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sequence_string, state_trajectory_indices, state_name_to_index_mapping)
    """
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    
    # Create state name to index mapping
    state_names = sorted(list(machine.states))  # Sort for consistency
    state_to_index = {name: idx for idx, name in enumerate(state_names)}
    
    machine.reset()
    sequence = []
    state_indices = []
    
    for _ in range(length):
        # Record current state index before generating symbol
        state_idx = state_to_index[machine.current_state]
        state_indices.append(state_idx)
        
        # Generate next symbol and transition
        symbol = machine.generate_symbol()
        sequence.append(symbol)
    
    return ''.join(sequence), state_indices, state_to_index


def save_dataset(sequence: str, state_indices: List[int], machine: EpsilonMachine, 
                state_to_index: Dict[str, int], output_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Save the generated dataset in the required format.
    
    Args:
        sequence: Generated symbol sequence
        state_indices: Aligned state trajectory as numerical indices
        machine: The source epsilon machine
        state_to_index: Mapping from state names to indices
        output_path: Base output path (without extension)
        metadata: Additional metadata to save
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save sequence as .dat file
    dat_file = output_path.with_suffix('.dat')
    dat_file.write_text(sequence)
    
    # Save state trajectory as numerical indices (space-separated single line)
    states_file = output_path.with_suffix('.states')
    with open(states_file, 'w') as f:
        f.write(' '.join(map(str, state_indices)))
    
    # Save machine definition
    machine_file = output_path.with_suffix('.machine.json')
    machine_data = {
        'alphabet': machine.alphabet,
        'states': list(machine.states),
        'start_state': machine.start_state,
        'transitions': {}
    }
    
    # Convert transitions to serializable format
    for (from_state, symbol), transitions in machine.transitions.items():
        key = f"{from_state}|{symbol}"
        machine_data['transitions'][key] = [
            {'to_state': to_state, 'probability': prob}
            for to_state, prob in transitions
        ]
    
    with open(machine_file, 'w') as f:
        json.dump(machine_data, f, indent=2)
    
    # Save metadata with state mapping
    metadata_file = output_path.with_suffix('.meta.json')
    
    # Create reverse mapping for reference
    index_to_state = {idx: name for name, idx in state_to_index.items()}
    
    full_metadata = {
        'sequence_length': len(sequence),
        'num_states': len(machine.states),
        'alphabet_size': len(machine.alphabet),
        'machine_type': metadata.get('machine_type', 'unknown'),
        'generation_seed': metadata.get('seed'),
        'state_mapping': {
            'name_to_index': state_to_index,
            'index_to_name': index_to_state
        },
        'files': {
            'sequence': str(dat_file.name),
            'states': str(states_file.name),
            'machine': str(machine_file.name)
        }
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    print(f"Dataset saved:")
    print(f"  Sequence: {dat_file} ({len(sequence)} symbols)")
    print(f"  States: {states_file} ({len(state_indices)} state indices)")
    print(f"  Machine: {machine_file}")
    print(f"  Metadata: {metadata_file}")
    print(f"  State mapping: {state_to_index}")


def create_machine(machine_type: str, seed: int = None) -> Tuple[EpsilonMachine, str]:
    """
    Create a domain-specific epsilon machine.
    
    Args:
        machine_type: Type of machine to create
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (machine, machine_type_name)
    """
    machine_builders = {
        'even_process': EvenProcessMachine,
        'alternating': AlternatingMachine,
        'incompressible_counter': IncompressibleCounterMachine,
        'truly_incompressible': TrulyIncompressibleMachine,
        'context_sensitive': ContextSensitiveMachine,
        'period4': Period4Machine,
        'golden_mean': GoldenMeanMachine,
        'seven_state_human': SevenStateHumanSequenceMachine
    }
    
    if machine_type not in machine_builders:
        available = ', '.join(machine_builders.keys())
        raise ValueError(f"Unknown machine type: {machine_type}. Available: {available}")
    
    builder_class = machine_builders[machine_type]
    builder = builder_class(seed=seed)
    machine = builder.create_machine()
    
    return machine, machine_type


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate domain-specific machine datasets with state trajectories"
    )
    parser.add_argument(
        '--machine', 
        required=True,
        choices=['even_process', 'alternating', 'incompressible_counter', 'truly_incompressible', 
                'context_sensitive', 'period4', 'golden_mean', 'seven_state_human'],
        help="Type of domain-specific machine to generate from"
    )
    parser.add_argument(
        '--length', 
        type=int, 
        default=100000,
        help="Length of sequence to generate (default: 100000)"
    )
    parser.add_argument(
        '--output', 
        required=True,
        help="Base output directory (e.g., 'data' -> creates 'data/even_process/' subfolder)"
    )
    parser.add_argument(
        '--seed', 
        type=int,
        help="Random seed for reproducible generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create the machine
    machine, machine_type = create_machine(args.machine, args.seed)
    
    print(f"Generating {args.machine} dataset...")
    print(f"  Length: {args.length}")
    print(f"  Output: {args.output}/{machine_type}/")
    if args.seed:
        print(f"  Seed: {args.seed}")
    
    # Generate sequence and states
    sequence, state_indices, state_to_index = generate_sequence_with_states(machine, args.length, args.seed)
    
    # Prepare metadata
    metadata = {
        'machine_type': machine_type,
        'seed': args.seed
    }
    
    # Save the dataset with machine subfolder
    base_output_path = Path(args.output)
    machine_output_path = base_output_path / machine_type / machine_type  # e.g., data/even_process/even_process
    save_dataset(sequence, state_indices, machine, state_to_index, machine_output_path, metadata)
    
    print("\nDataset generation complete!")
    
    # Show state distribution with both indices and names
    unique_indices, counts = np.unique(state_indices, return_counts=True)
    index_to_name = {idx: name for name, idx in state_to_index.items()}
    state_dist = {f"{idx}({index_to_name[idx]})": count for idx, count in zip(unique_indices, counts)}
    print(f"State distribution: {state_dist}")


if __name__ == '__main__':
    import numpy as np
    main()