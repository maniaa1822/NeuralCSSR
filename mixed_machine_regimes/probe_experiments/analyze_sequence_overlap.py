import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import matplotlib.pyplot as plt


def load_sequence(sequence_path: Path) -> str:
    text = sequence_path.read_text().strip()
    if not text:
        raise ValueError(f"No data found in {sequence_path}")
    return text


def load_segments(meta_path: Path) -> Sequence[Dict]:
    with meta_path.open() as fh:
        meta = json.load(fh)
    if "segments" not in meta:
        raise ValueError(f"{meta_path} missing 'segments' entry")
    return meta["segments"]


def collect_substrings(sequence: str, max_length: int) -> List[Set[str]]:
    """Return a list of sets indexed by substring length."""
    holders: List[Set[str]] = [set() for _ in range(max_length + 1)]
    seq_len = len(sequence)
    cap = min(seq_len, max_length)
    for start in range(seq_len):
        max_L = min(cap, seq_len - start)
        for L in range(1, max_L + 1):
            holders[L].add(sequence[start : start + L])
    return holders


def compute_stats(
    segments: Iterable[Dict], sequence: str, max_length: int
) -> Dict[int, Dict[str, int]]:
    machines = {segment["machine"] for segment in segments}
    per_machine: Dict[str, List[Set[str]]] = {
        machine: [set() for _ in range(max_length + 1)] for machine in machines
    }

    for segment in segments:
        machine = segment["machine"]
        start = segment["start"]
        end = segment["end"]
        if start < 0 or end > len(sequence):
            raise ValueError(
                f"Segment {segment} outside of sequence bounds ({len(sequence)})"
            )
        segment_sets = collect_substrings(sequence[start:end], max_length)
        machine_sets = per_machine[machine]
        for L in range(1, min(max_length, end - start) + 1):
            machine_sets[L].update(segment_sets[L])

    stats: Dict[int, Dict[str, int]] = {}
    machines_list = sorted(per_machine.keys())
    for L in range(1, max_length + 1):
        length_stats: Dict[str, int] = {}
        machine_sets = {m: per_machine[m][L] for m in machines_list}
        shared = set.intersection(*machine_sets.values()) if machine_sets else set()
        length_stats["shared_count"] = len(shared)
        for m in machines_list:
            current = machine_sets[m]
            others = set().union(*(machine_sets[o] for o in machines_list if o != m))
            length_stats[f"{m}_total"] = len(current)
            length_stats[f"{m}_unique"] = len(current - others)
        stats[L] = length_stats
    return stats


def print_table(stats: Dict[int, Dict[str, int]]) -> None:
    if not stats:
        print("No stats available")
        return
    lengths = sorted(stats.keys())
    first_row = stats[lengths[0]]
    headers = ["L"] + list(first_row.keys())
    print("\t".join(headers))
    for L in lengths:
        row = [str(L)] + [str(stats[L][key]) for key in first_row.keys()]
        print("\t".join(row))


def plot_stats(
    stats: Dict[int, Dict[str, int]], output_path: Path, use_secondary_axis: bool
) -> None:
    lengths = sorted(stats.keys())
    if not lengths:
        raise ValueError("No stats available to plot")
    keys = list(stats[lengths[0]].keys())
    plt.figure(figsize=(10, 6))
    primary_ax = plt.gca()
    secondary_ax = None
    for key in keys:
        values = [stats[L][key] for L in lengths]
        target_ax = primary_ax
        style = "-"
        if key == "shared_count" and use_secondary_axis:
            if secondary_ax is None:
                secondary_ax = primary_ax.twinx()
                secondary_ax.set_ylabel("Shared count")
            target_ax = secondary_ax
            style = "--"
        target_ax.plot(lengths, values, label=key, linestyle=style)

    primary_ax.set_xlabel("Substring length (L)")
    primary_ax.set_ylabel("Unique/total counts")
    primary_ax.set_title("Substring overlap statistics")
    handles, labels = primary_ax.get_legend_handles_labels()
    if secondary_ax is not None:
        h2, l2 = secondary_ax.get_legend_handles_labels()
        handles += h2
        labels += l2
    primary_ax.legend(handles, labels)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def describe_shared_sequences(
    segments: Iterable[Dict], sequence: str, max_length: int
) -> None:
    machines = {segment["machine"] for segment in segments}
    per_machine: Dict[str, List[Set[str]]] = {
        machine: [set() for _ in range(max_length + 1)] for machine in machines
    }
    for segment in segments:
        machine = segment["machine"]
        segment_sets = collect_substrings(
            sequence[segment["start"] : segment["end"]], max_length
        )
        for L in range(1, min(max_length, segment["end"] - segment["start"]) + 1):
            per_machine[machine][L].update(segment_sets[L])

    machines_list = sorted(per_machine.keys())
    for L in range(1, max_length + 1):
        shared = set.intersection(*(per_machine[m][L] for m in machines_list))
        preview = ", ".join(sorted(shared)[:20])
        suffix = " ..." if len(shared) > 20 else ""
        print(f"L={L}, shared={len(shared)}: {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-machine unique and shared substring counts."
    )
    parser.add_argument(
        "--sequence-path",
        type=Path,
        required=True,
        help="Path to combined .dat sequence file.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        required=True,
        help="Path to combined .meta.json file with segment metadata.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum substring length to evaluate.",
    )
    parser.add_argument(
        "--show-shared-up-to",
        type=int,
        default=0,
        help="Optional maximum length for printing actual shared sequences.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path to save a line plot of the statistics.",
    )
    parser.add_argument(
        "--single-axis",
        action="store_true",
        help="Plot all series on a single axis (default uses secondary axis for shared count).",
    )
    args = parser.parse_args()

    sequence = load_sequence(args.sequence_path)
    segments = load_segments(args.meta_path)
    stats = compute_stats(segments, sequence, args.max_length)
    print_table(stats)

    if args.show_shared_up_to > 0:
        describe_shared_sequences(segments, sequence, args.show_shared_up_to)

    if args.plot_output:
        plot_stats(stats, args.plot_output, use_secondary_axis=not args.single_axis)


if __name__ == "__main__":
    main()
