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
from mixed_machine_regimes.mixed_dataset import (
    generate_switching_dataset,
    generate_union_dataset,
    resolve_machines,
    save_mixed_dataset,
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
    parser.add_argument(
        "--mixed_name",
        help="Name for a mixed-machine dataset (enables mixed-generation mode).",
    )
    parser.add_argument(
        "--mixed_mode",
        choices=("union", "switch"),
        default="union",
        help="Mixed dataset mode: 'union' concatenates per-machine sequences, 'switch' simulates regime switching.",
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        help="Machines to include in mixed mode.",
    )
    parser.add_argument(
        "--segment_lengths",
        nargs="+",
        type=int,
        help="Per-machine segment lengths for union mode (same order as --machines).",
    )
    parser.add_argument(
        "--switch_interval",
        type=int,
        default=50,
        help="Number of symbols before switching machines in switch mode.",
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
        if not args.mixed_name:
            parser.error("--machine is required unless --list or --mixed_name is provided.")
        handle_mixed_mode(args, parser)
        return

    if args.mixed_name:
        parser.error("--machine cannot be combined with --mixed_name. Pick one mode.")

    generate_dataset(args.machine, args.length, args.output, args.seed)


def handle_mixed_mode(args, parser: argparse.ArgumentParser) -> None:
    if not args.machines or len(args.machines) < 2:
        parser.error("--mixed_name requires --machines with at least two entries.")
    machines = resolve_machines(args.machines)
    metadata = {"seed": args.seed, "mixed_name": args.mixed_name}

    if args.mixed_mode == "union":
        if not args.segment_lengths or len(args.segment_lengths) != len(machines):
            parser.error("--segment_lengths must match --machines for union mode.")
        specs = list(zip(machines, args.segment_lengths))
        result = generate_union_dataset(specs, seed=args.seed)
    else:
        if args.length is None:
            parser.error("--length is required in switch mode.")
        result = generate_switching_dataset(
            machines,
            total_length=args.length,
            switch_interval=args.switch_interval,
            seed=args.seed,
        )
    paths = save_mixed_dataset(
        result,
        output_dir=args.output,
        dataset_name=args.mixed_name,
        metadata=metadata,
    )
    print(f"Mixed dataset '{args.mixed_name}' ({result.mode}) length={result.length}")
    print(f"  Machines: {', '.join(result.machines)}")
    for seg in result.segments:
        print(f"  Segment {seg.machine}: [{seg.start}, {seg.end}) len={seg.length}")
    for key, path in paths.items():
        print(f"  {key:10}: {path}")


if __name__ == "__main__":
    main()
