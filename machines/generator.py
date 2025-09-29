"""Utility adapters for generating sequences from Machine specifications."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Machine


class MachineGenerator:
    """Adapter that generates sequences from unified :class:`Machine` objects.

    Bridges machine specifications with sample generation and transition metadata.
    """

    def __init__(self, machine: Machine, seed: Optional[int] = None):
        """Initialize generator from a Machine object."""

        self.machine = machine
        self.rng = np.random.default_rng(seed)
        self.current_state = machine.start_state
        self.alphabet = machine.alphabet
        self.states = machine.states

    @property
    def start_state_name(self) -> str:
        """Return the initial state name."""

        return self.machine.start_state

    def get_transition_structure(self) -> Dict[str, Any]:
        """Return the machine's transition structure in JSON-friendly format."""

        transitions_json: Dict[str, Any] = {}
        for (state, symbol), next_state in self.machine.transitions.items():
            key = f"{state.upper()}|{symbol}"
            transitions_json[key] = [{
                "to_state": next_state.upper(),
                "probability": 1.0,
            }]
        return transitions_json

    def step(self) -> str:
        """Sample one symbol and advance to the next state."""

        emission_probs = self.machine.emissions[self.current_state]
        symbols = list(emission_probs.keys())
        probs = list(emission_probs.values())
        symbol = self.rng.choice(symbols, p=probs)

        next_state = self.machine.transitions[(self.current_state, symbol)]
        self.current_state = next_state

        return symbol


def generate_sequence_with_states(machine: Machine, length: int,
                                  seed: Optional[int] = None) -> Tuple[str, List[str]]:
    """Generate a symbol sequence and state history from a machine."""

    generator = MachineGenerator(machine, seed=seed)
    sequence: List[str] = []
    states: List[str] = []

    for _ in range(length):
        states.append(generator.current_state)
        sequence.append(generator.step())

    return ''.join(sequence), states


def save_dataset(sequence: str, states: List[str], machine: Machine,
                 output_root: Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist sequence, state traces, and metadata to disk."""

    metadata = metadata or {}
    output_root = Path(output_root)
    base_path = output_root / machine.name / machine.name
    base_path.parent.mkdir(parents=True, exist_ok=True)

    state_names = sorted(machine.states)
    state_to_index = {name: idx for idx, name in enumerate(state_names)}
    state_indices = [state_to_index[state] for state in states]

    dat_path = base_path.with_suffix('.dat')
    dat_path.write_text(sequence)

    states_path = base_path.with_suffix('.states')
    states_path.write_text(' '.join(map(str, state_indices)))

    index_to_char = {idx: chr(ord('A') + idx) for idx in range(len(state_names))}
    states_dat_path = base_path.with_suffix('.states.dat')
    states_dat_path.write_text(''.join(index_to_char[idx] for idx in state_indices))

    transitions_json: Dict[str, Any] = {}
    for (state, symbol), next_state in machine.transitions.items():
        key = f"{state.upper()}|{symbol}"
        transitions_json[key] = [{
            "to_state": next_state.upper(),
            "probability": 1.0,
        }]

    machine_json = {
        'alphabet': machine.alphabet,
        'states': state_names,
        'start_state': machine.start_state,
        'transitions': transitions_json,
    }

    machine_path = base_path.with_suffix('.machine.json')
    machine_path.write_text(json_dumps(machine_json))

    metadata_payload = {
        'sequence_length': len(sequence),
        'num_states': len(state_names),
        'alphabet_size': len(machine.alphabet),
        'machine_name': machine.name,
        'display_name': machine.display_name,
        'generation_seed': metadata.get('seed'),
        'state_mapping': {
            'name_to_index': state_to_index,
            'index_to_name': {idx: name for name, idx in state_to_index.items()}
        },
        'state_index_to_char': index_to_char,
        'files': {
            'sequence': dat_path.name,
            'states': states_path.name,
            'states_dat': states_dat_path.name,
            'machine': machine_path.name,
        },
    }
    metadata_payload.update({k: v for k, v in metadata.items() if v is not None})

    metadata_path = base_path.with_suffix('.meta.json')
    metadata_path.write_text(json_dumps(metadata_payload))

    return {
        'sequence_path': dat_path,
        'states_path': states_path,
        'states_dat_path': states_dat_path,
        'machine_path': machine_path,
        'metadata_path': metadata_path,
        'state_mapping': state_to_index,
        'index_to_char': index_to_char,
    }


def json_dumps(payload: Dict[str, Any]) -> str:
    """Serialize payload to formatted JSON."""

    import json

    return json.dumps(payload, indent=2)
