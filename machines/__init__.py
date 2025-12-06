"""
Unified machine specifications for the Neural CSSR pipeline.

This package provides a centralized Machine class that encapsulates all machine
properties and serves as a single source of truth across generation, training,
discovery, and evaluation.

Usage:
    from machines import get_machine, list_machines

    # Get a specific machine
    machine = get_machine("seven_state_human")

    # Access properties
    print(machine.states)           # ['bb', 'aaa', ...]
    print(machine.num_states)       # 7
    print(machine.memory_length)    # 4

    # Ground truth evaluation
    state = machine.get_gt_state(history)

    # Theoretical properties
    entropy = machine.compute_theoretical_entropy()

    # List all available machines
    machines = list_machines()
"""

from typing import Dict, List
from .base import Machine
from .generator import MachineGenerator, generate_sequence_with_states, save_dataset

# Global registry of all machines
MACHINE_REGISTRY: Dict[str, Machine] = {}


def register_machine(name: str):
    """Decorator to register a machine in the global registry.

    Usage:
        @register_machine("my_machine")
        class MyMachine(Machine):
            ...
    """
    def decorator(cls):
        instance = cls()
        MACHINE_REGISTRY[instance.name] = instance
        return cls
    return decorator


def get_machine(name: str) -> Machine:
    """Factory function to retrieve a machine by name.

    Args:
        name: Machine name (e.g., "seven_state_human")

    Returns:
        Machine instance

    Raises:
        ValueError: If machine name is not registered
    """
    if name not in MACHINE_REGISTRY:
        available = ', '.join(sorted(MACHINE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown machine: '{name}'. "
            f"Available machines: {available}"
        )
    return MACHINE_REGISTRY[name]


def list_machines() -> List[str]:
    """Get all registered machine names.

    Returns:
        Sorted list of machine names
    """
    return sorted(MACHINE_REGISTRY.keys())


# Import machine definitions to auto-register them
from . import seven_state_human
from . import golden_mean
from . import even_process
from . import phoneme_machine
from . import butterfly_machine


__all__ = [
    'Machine',
    'MachineGenerator',
    'generate_sequence_with_states',
    'save_dataset',
    'get_machine',
    'list_machines',
    'register_machine',
    'MACHINE_REGISTRY',
]