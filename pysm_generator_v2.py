#!/usr/bin/env python3
"""Streamlined dataset generator leveraging unified machines framework."""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from machines import (
    generate_sequence_with_states,
    get_machine,
    list_machines,
    save_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate datasets from unified epsilon machines.",
    )
    parser.add_argument(
        "--machine",
        help="Machine name (use --list to inspect available machines).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100_000,
        help="Number of symbols to generate (default: 100000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/datasets"),
        help="Output directory root (default: experiments/datasets).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available machines and exit.",
    )
    return parser


def list_available_machines() -> None:
    print("Available machines:")
    for name in list_machines():
        machine = get_machine(name)
        memory = f"{machine.memory_type}" if machine.memory_length is None else f"{machine.memory_type} (L={machine.memory_length})"
        print(f"  {name:25} → {machine.num_states} states, memory={memory}")


def generate_dataset(machine_name: str, length: int, output: Path, seed: Optional[int]) -> None:
    machine = get_machine(machine_name)
    sequence, states = generate_sequence_with_states(machine, length, seed)
    result = save_dataset(sequence, states, machine, output, metadata={"seed": seed})

    print(f"Generated sequence of length {len(sequence)} for '{machine_name}' → saved under {output / machine_name}")
    for key in ("sequence_path", "states_path", "states_dat_path", "machine_path", "metadata_path"):
        path_obj = result[key]
        print(f"  {key.replace('_', ' '):15}: {path_obj}")

    counts = Counter(states)
    total = len(states)
    mapping = result["state_mapping"]
    print("State distribution:")
    for name in sorted(counts.keys(), key=lambda s: mapping[s]):
        freq = counts[name]
        idx = mapping[name]
        print(f"  {idx:2d} ({name}): {freq} ({freq/total:.2%})")

    try:
        entropy = machine.compute_theoretical_entropy()
        print(f"Entropy: {entropy:.4f} nats ({entropy / math.log(2):.4f} bits)")
    except Exception:
        print("Entropy: unavailable")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        list_available_machines()
        return

    if not args.machine:
        parser.error("--machine is required unless --list is provided.")

    generate_dataset(args.machine, args.length, args.output, args.seed)


if __name__ == "__main__":
    main()
