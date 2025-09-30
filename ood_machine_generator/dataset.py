from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .meta_distribution import sample_machine
from .sequence import sample_sequence


class MachineSample(NamedTuple):
    machine_id: str
    split: str
    seq: np.ndarray
    states: Optional[np.ndarray]
    length: int
    alphabet_size: int
    anchor_view: int
    pair_key: str
    emissions: np.ndarray
    transitions: np.ndarray
    stationary: np.ndarray
    canonical_perm: np.ndarray
    canonical_signature: str
    padded_emissions: np.ndarray
    padded_transitions: np.ndarray
    state_mask: np.ndarray
    meta: Dict[str, Any]


class MachineBundle(NamedTuple):
    machine_id: str
    split: str
    alphabet_size: int
    emissions: np.ndarray
    transitions: np.ndarray
    stationary: np.ndarray
    canonical_perm: np.ndarray
    canonical_signature: str
    padded_emissions: np.ndarray
    padded_transitions: np.ndarray
    state_mask: np.ndarray
    samples: Tuple[MachineSample, ...]
    meta: Dict[str, Any]


def _flatten_split(splits: Dict[str, str]) -> str:
    if any(value == "test" for value in splits.values()):
        return "test"
    if any(value == "val" for value in splits.values()):
        return "val"
    return "train"


def dataset_iter(
    n_machines: int,
    seqs_per_machine: int,
    lengths: Sequence[int],
    *,
    pad_K: int,
    rng: Optional[np.random.Generator | int] = None,
    include_states: bool = False,
    machine_kwargs: Optional[Dict[str, Any]] = None,
    bundle_per_machine: bool = False,
) -> Iterator[MachineSample | MachineBundle]:
    if n_machines <= 0:
        raise ValueError("n_machines must be positive")
    if seqs_per_machine <= 0:
        raise ValueError("seqs_per_machine must be positive")
    if not lengths:
        raise ValueError("lengths must be non-empty")
    if pad_K <= 0:
        raise ValueError("pad_K must be positive")

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    length_choices = np.asarray(lengths, dtype=np.int64)
    if np.any(length_choices <= 0):
        raise ValueError("sequence lengths must be positive")

    machine_kwargs = dict(machine_kwargs or {})
    machine_kwargs.setdefault("K_max", pad_K)

    for machine_idx in range(n_machines):
        machine_seed = int(generator.integers(0, 2**63 - 1))
        spec = sample_machine(rng=machine_seed, **machine_kwargs)
        if spec.num_states > pad_K:
            raise ValueError(f"sampled machine with K={spec.num_states} exceeds padding limit {pad_K}")

        machine_id = f"{spec.metadata['family']}_{machine_idx:06d}"
        padded_emissions = np.zeros(pad_K, dtype=np.float64)
        padded_emissions[: spec.num_states] = spec.emissions

        padded_transitions = np.zeros((pad_K, spec.alphabet_size), dtype=np.int64)
        padded_transitions[: spec.num_states] = spec.transitions

        state_mask = np.zeros(pad_K, dtype=bool)
        state_mask[: spec.num_states] = True

        samples: list[MachineSample] = []
        for sequence_idx in range(seqs_per_machine):
            seq_seed = int(generator.integers(0, 2**63 - 1))
            length = int(generator.choice(length_choices))
            sequence, states = sample_sequence(
                spec,
                length,
                rng=seq_seed,
                return_states=include_states,
            )

            meta = copy.deepcopy(spec.metadata)
            meta.update(
                {
                    "machine_id": machine_id,
                    "T_len": length,
                    "machine_index": machine_idx,
                    "sequence_index": sequence_idx,
                    "rng_seeds": {"machine": machine_seed, "sequence": seq_seed},
                }
            )

            split = _flatten_split(meta.get("splits", {}))

            sample = MachineSample(
                machine_id=machine_id,
                split=split,
                seq=sequence,
                states=states,
                length=length,
                alphabet_size=spec.alphabet_size,
                anchor_view=sequence_idx,
                pair_key=f"{machine_id}_view{sequence_idx}",
                emissions=spec.emissions.copy(),
                transitions=spec.transitions.copy(),
                stationary=spec.stationary.copy(),
                canonical_perm=spec.canonical_perm.copy(),
                canonical_signature=spec.canonical_signature,
                padded_emissions=padded_emissions.copy(),
                padded_transitions=padded_transitions.copy(),
                state_mask=state_mask.copy(),
                meta=meta,
            )
            if bundle_per_machine:
                samples.append(sample)
            else:
                yield sample

        if bundle_per_machine:
            machine_meta = copy.deepcopy(spec.metadata)
            machine_meta.update(
                {
                    "machine_id": machine_id,
                    "machine_index": machine_idx,
                    "rng_seed": machine_seed,
                    "num_sequences": seqs_per_machine,
                }
            )
            yield MachineBundle(
                machine_id=machine_id,
                split=_flatten_split(machine_meta.get("splits", {})),
                alphabet_size=spec.alphabet_size,
                emissions=spec.emissions.copy(),
                transitions=spec.transitions.copy(),
                stationary=spec.stationary.copy(),
                canonical_perm=spec.canonical_perm.copy(),
                canonical_signature=spec.canonical_signature,
                padded_emissions=padded_emissions.copy(),
                padded_transitions=padded_transitions.copy(),
                state_mask=state_mask.copy(),
                samples=tuple(samples),
                meta=machine_meta,
            )


def sample_to_payload(sample: MachineSample, *, include_states: bool = True) -> Dict[str, Any]:
    payload = {
        "machine_id": sample.machine_id,
        "split": sample.split,
        "seq": sample.seq.astype(np.uint8).tolist(),
        "length": sample.length,
        "alphabet_size": sample.alphabet_size,
        "anchor_view": sample.anchor_view,
        "pair_key": sample.pair_key,
        "machine": {
            "K": int(sample.emissions.shape[0]),
            "E": sample.emissions.tolist(),
            "T": sample.transitions.tolist(),
            "pi": sample.stationary.tolist(),
            "canonical_perm": sample.canonical_perm.tolist(),
            "canonical_signature": sample.canonical_signature,
        },
        "padded": {
            "K_max": int(sample.padded_emissions.shape[0]),
            "E": sample.padded_emissions.tolist(),
            "T": sample.padded_transitions.tolist(),
            "state_mask": sample.state_mask.astype(bool).tolist(),
            "pi": sample.stationary.tolist(),
        },
        "meta": copy.deepcopy(sample.meta),
    }
    if include_states:
        payload["states"] = sample.states.tolist() if sample.states is not None else None
    return payload


def bundle_to_payload(bundle: MachineBundle, *, include_states: bool = True) -> Dict[str, Any]:
    payload = {
        "machine_id": bundle.machine_id,
        "split": bundle.split,
        "alphabet_size": bundle.alphabet_size,
        "machine": {
            "K": int(bundle.emissions.shape[0]),
            "E": bundle.emissions.tolist(),
            "T": bundle.transitions.tolist(),
            "pi": bundle.stationary.tolist(),
            "canonical_perm": bundle.canonical_perm.tolist(),
            "canonical_signature": bundle.canonical_signature,
        },
        "padded": {
            "K_max": int(bundle.padded_emissions.shape[0]),
            "E": bundle.padded_emissions.tolist(),
            "T": bundle.padded_transitions.tolist(),
            "state_mask": bundle.state_mask.astype(bool).tolist(),
            "pi": bundle.stationary.tolist(),
        },
        "meta": copy.deepcopy(bundle.meta),
    }

    seq_entries = []
    for sample in bundle.samples:
        seq_entry = {
            "anchor_view": sample.anchor_view,
            "pair_key": sample.pair_key,
            "length": sample.length,
            "seq": sample.seq.astype(np.uint8).tolist(),
            "meta": copy.deepcopy(sample.meta),
        }
        if include_states:
            seq_entry["states"] = sample.states.tolist() if sample.states is not None else None
        seq_entries.append(seq_entry)
    payload["sequences"] = seq_entries
    return payload
