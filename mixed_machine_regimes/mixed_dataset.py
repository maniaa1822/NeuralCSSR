"""Utilities for constructing mixed-machine datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from machines import Machine, get_machine
from machines.generator import MachineGenerator, generate_sequence_with_states


@dataclass
class MixedSegment:
    machine: str
    start: int
    end: int  # exclusive
    seed: Optional[int]

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class MixedDatasetResult:
    sequence: str
    machine_ids: List[str]
    state_labels: List[str]
    segments: List[MixedSegment]
    mode: str
    machines: List[str]
    extra_metadata: Dict[str, object]

    @property
    def length(self) -> int:
        return len(self.sequence)


def _draw_child_seed(parent_rng: np.random.Generator) -> int:
    return int(parent_rng.integers(0, 2**31 - 1))


def generate_union_dataset(
    machine_specs: Sequence[Tuple[Machine, int]],
    seed: Optional[int] = None,
) -> MixedDatasetResult:
    rng = np.random.default_rng(seed)
    sequence_parts: List[str] = []
    machine_ids: List[str] = []
    state_labels: List[str] = []
    segments: List[MixedSegment] = []
    offset = 0

    for machine, seg_length in machine_specs:
        child_seed = _draw_child_seed(rng)
        seq, states = generate_sequence_with_states(machine, seg_length, seed=child_seed)
        sequence_parts.append(seq)
        ids = [machine.name] * len(seq)
        machine_ids.extend(ids)
        state_labels.extend([f"{machine.name}:{state}" for state in states])
        segments.append(MixedSegment(machine=machine.name, start=offset, end=offset + len(seq), seed=child_seed))
        offset += len(seq)

    sequence = "".join(sequence_parts)
    return MixedDatasetResult(
        sequence=sequence,
        machine_ids=machine_ids,
        state_labels=state_labels,
        segments=segments,
        mode="union",
        machines=[machine.name for machine, _ in machine_specs],
        extra_metadata={"union_lengths": {machine.name: length for machine, length in machine_specs}},
    )


def generate_switching_dataset(
    machines: Sequence[Machine],
    total_length: int,
    switch_interval: int,
    seed: Optional[int] = None,
) -> MixedDatasetResult:
    if switch_interval <= 0:
        raise ValueError("switch_interval must be positive.")
    rng = np.random.default_rng(seed)
    generators: Dict[str, MachineGenerator] = {}
    for machine in machines:
        generators[machine.name] = MachineGenerator(machine, seed=_draw_child_seed(rng))

    sequence: List[str] = []
    machine_ids: List[str] = []
    state_labels: List[str] = []
    segments: List[MixedSegment] = []
    current_machine_idx = 0
    steps_in_segment = 0
    segment_start = 0
    active_machine = machines[current_machine_idx]

    for position in range(total_length):
        gen = generators[active_machine.name]
        state_labels.append(f"{active_machine.name}:{gen.current_state}")
        machine_ids.append(active_machine.name)
        sequence.append(gen.step())
        steps_in_segment += 1

        if steps_in_segment == switch_interval and position + 1 < total_length:
            segments.append(
                MixedSegment(
                    machine=active_machine.name,
                    start=segment_start,
                    end=position + 1,
                    seed=None,
                )
            )
            steps_in_segment = 0
            segment_start = position + 1
            current_machine_idx = (current_machine_idx + 1) % len(machines)
            active_machine = machines[current_machine_idx]

    segments.append(
        MixedSegment(
            machine=active_machine.name,
            start=segment_start,
            end=total_length,
            seed=None,
        )
    )

    return MixedDatasetResult(
        sequence="".join(sequence),
        machine_ids=machine_ids,
        state_labels=state_labels,
        segments=segments,
        mode="switch",
        machines=[machine.name for machine in machines],
        extra_metadata={"switch_interval": switch_interval},
    )


def save_mixed_dataset(
    result: MixedDatasetResult,
    output_dir: Path,
    dataset_name: str,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = output_dir / "combined.dat"
    machine_ids_path = output_dir / "combined.machine_ids.dat"
    state_ids_path = output_dir / "combined.state_ids.dat"
    meta_path = output_dir / "combined.meta.json"

    sequence_path.write_text(result.sequence)
    machine_ids_path.write_text(" ".join(result.machine_ids))
    state_ids_path.write_text(" ".join(result.state_labels))

    segments_payload = [
        {
            "machine": seg.machine,
            "start": seg.start,
            "end": seg.end,
            "length": seg.length,
            "seed": seg.seed,
        }
        for seg in result.segments
    ]
    machine_counts = Counter(result.machine_ids)
    stats, expected_entropy_nats, expected_entropy_bits = compute_machine_stats(
        result.machines, machine_counts
    )
    joint_stats, belief_entropy_nats, belief_entropy_bits = compute_joint_state_stats(
        result.state_labels
    )
    belief_machine = build_belief_machine(resolve_machines(result.machines))
    markov_baseline = compute_markov_baseline(result.sequence)

    meta: Dict[str, object] = {
        "length": result.length,
        "mode": result.mode,
        "machines": result.machines,
        "segments": segments_payload,
        "machine_counts": dict(machine_counts),
        "machine_stats": stats,
        "expected_optimal_loss": {
            "nats": expected_entropy_nats,
            "bits": expected_entropy_bits,
        },
        "joint_state_stats": joint_stats,
        "expected_belief_loss": {
            "nats": belief_entropy_nats,
            "bits": belief_entropy_bits,
        },
        "belief_machine": belief_machine,
        "markov_baseline": markov_baseline,
    }
    if metadata:
        meta.update(metadata)
    meta.update(result.extra_metadata)
    meta_path.write_text(json_dumps(meta))
    return {
        "sequence": sequence_path,
        "machine_ids": machine_ids_path,
        "state_ids": state_ids_path,
        "meta": meta_path,
    }


def json_dumps(payload: Dict[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2)


def resolve_machines(names: Sequence[str]) -> List[Machine]:
    return [get_machine(name) for name in names]


def compute_machine_stats(
    machine_names: Sequence[str], counts: Counter
) -> Tuple[Dict[str, Dict[str, object]], float, float]:
    """Compute entropy/loss stats for each machine and the expected joint process."""
    stats: Dict[str, Dict[str, object]] = {}
    total = sum(counts.values())
    total = max(total, 1)
    combined_entropy_nats = 0.0
    for name in machine_names:
        machine = get_machine(name)
        count = counts.get(name, 0)
        fraction = count / total
        entropy_nats = float(machine.compute_theoretical_entropy())
        entropy_bits = entropy_nats / math.log(2)
        combined_entropy_nats += fraction * entropy_nats
        stats[name] = {
            "count": count,
            "fraction": fraction,
            "entropy_nats": entropy_nats,
            "entropy_bits": entropy_bits,
            "num_states": machine.num_states,
            "memory_type": machine.memory_type,
            "memory_length": machine.memory_length,
        }
    combined_entropy_bits = combined_entropy_nats / math.log(2)
    return stats, combined_entropy_nats, combined_entropy_bits


def compute_joint_state_stats(
    state_labels: Sequence[str],
) -> Tuple[Dict[str, Dict[str, object]], float, float]:
    """Compute stats for each (machine, state) pair across the dataset."""
    counts = Counter(state_labels)
    total = sum(counts.values())
    total = max(total, 1)
    stats: Dict[str, Dict[str, object]] = {}
    cache: Dict[str, Machine] = {}
    combined_entropy_nats = 0.0

    for label, count in counts.items():
        if ":" not in label:
            continue
        machine_name, state_name = label.split(":", 1)
        if machine_name not in cache:
            cache[machine_name] = get_machine(machine_name)
        machine = cache[machine_name]
        if state_name not in machine.emissions:
            continue
        emission_probs = machine.emissions[state_name]
        state_entropy_nats = 0.0
        for prob in emission_probs.values():
            if prob > 0:
                state_entropy_nats -= prob * math.log(prob)
        state_entropy_bits = state_entropy_nats / math.log(2)
        fraction = count / total
        combined_entropy_nats += fraction * state_entropy_nats
        stats[label] = {
            "machine": machine_name,
            "state": state_name,
            "count": count,
            "fraction": fraction,
            "state_entropy_nats": state_entropy_nats,
            "state_entropy_bits": state_entropy_bits,
            "num_states": machine.num_states,
        }
    combined_entropy_bits = combined_entropy_nats / math.log(2)
    return stats, combined_entropy_nats, combined_entropy_bits


def compute_markov_baseline(sequence: str, L: int = 8) -> Dict[str, object]:
    """Compute empirical Markov-L optimal loss baseline directly from the sequence."""
    data = np.array([int(ch) for ch in sequence], dtype=np.int64)
    L = max(1, min(L, len(data) - 1))
    counts: Dict[Tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    for t in range(L, len(data)):
        hist = tuple(int(x) for x in data[t - L : t])
        nxt = int(data[t])
        if nxt not in (0, 1):
            continue
        counts[hist][nxt] += 1

    probs: Dict[Tuple[int, ...], np.ndarray] = {}
    for hist, c in counts.items():
        total = int(c.sum())
        if total == 0:
            continue
        probs[hist] = c.astype(np.float64) / float(total)

    total_loss = 0.0
    num_predictions = 0
    for t in range(L, len(data)):
        hist = tuple(int(x) for x in data[t - L : t])
        dist = probs.get(hist)
        if dist is None:
            continue
        nxt = int(data[t])
        prob = float(dist[nxt])
        if prob <= 1e-12:
            continue
        total_loss -= math.log(prob)
        num_predictions += 1

    if num_predictions == 0:
        return {
            "L": L,
            "avg_loss_per_symbol": float("inf"),
            "avg_loss_per_symbol_bits": float("inf"),
            "num_predictions": 0,
            "num_unique_histories": len(probs),
        }

    avg_loss = total_loss / num_predictions
    return {
        "L": L,
        "avg_loss_per_symbol": avg_loss,
        "avg_loss_per_symbol_bits": avg_loss / math.log(2),
        "num_predictions": num_predictions,
        "num_unique_histories": len(probs),
    }


def build_belief_machine(machines: Sequence[Machine]) -> Dict[str, object]:
    """Construct a joint machine over (machine, state) belief modes."""
    alphabet = set()
    states_meta: List[Dict[str, object]] = []
    emissions: Dict[str, Dict[str, float]] = {}
    transitions: Dict[str, Dict[str, str]] = {}

    for machine in machines:
        alphabet.update(machine.alphabet)
        for state in machine.states:
            state_key = f"{machine.name}:{state}"
            states_meta.append(
                {
                    "id": state_key,
                    "machine": machine.name,
                    "state": state,
                }
            )
            emissions[state_key] = dict(machine.emissions[state])
            state_transitions: Dict[str, str] = {}
            for symbol in machine.alphabet:
                if (state, symbol) in machine.transitions:
                    next_state = machine.transitions[(state, symbol)]
                    state_transitions[symbol] = f"{machine.name}:{next_state}"
            transitions[state_key] = state_transitions

    return {
        "alphabet": sorted(alphabet),
        "states": states_meta,
        "transitions": transitions,
        "emissions": emissions,
    }
