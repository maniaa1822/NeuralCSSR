"""
Ground truth state mapping functions using unified Machine objects.

This module provides wrapper functions that delegate to Machine.get_gt_state()
for backward compatibility. New code should use Machine objects directly.

Deprecated functions (use machines.get_machine() instead):
- get_seven_state_gt_state() -> use machines.get_machine("seven_state_human").get_gt_state()
- get_golden_mean_gt_state() -> use machines.get_machine("golden_mean").get_gt_state()
- get_even_process_gt_state() -> use machines.get_machine("even_process").get_gt_state()
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Import machines package
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from machines import get_machine


# ===== Backward Compatibility Wrappers =====

def get_seven_state_gt_state(history: np.ndarray) -> Optional[str]:
    """DEPRECATED: Use machines.get_machine("seven_state_human").get_gt_state() instead."""
    return get_machine("seven_state_human").get_gt_state(history)


def get_golden_mean_gt_state(history: np.ndarray) -> str:
    """DEPRECATED: Use machines.get_machine("golden_mean").get_gt_state() instead."""
    return get_machine("golden_mean").get_gt_state(history)


def get_even_process_gt_state(history: np.ndarray) -> str:
    """DEPRECATED: Use machines.get_machine("even_process").get_gt_state() instead."""
    return get_machine("even_process").get_gt_state(history)


def get_gt_state(history: np.ndarray, preset: str) -> Optional[str]:
    """Get ground truth state for a given preset using Machine objects.

    Args:
        history: Array of symbols
        preset: Machine name or preset alias

    Returns:
        State name, or None if state cannot be determined

    Raises:
        ValueError: If preset is not recognized
    """
    # Map preset aliases to machine names
    preset_to_machine = {
        'seven_state_human': 'seven_state_human',
        'seven_state_human_100k': 'seven_state_human',
        'seven_state_human_large': 'seven_state_human',
        'seven_state_human_char_large': 'seven_state_human',
        'sevestateold': 'seven_state_human',  # Uses same structure
        'golden_mean': 'golden_mean',
        'even_process': 'even_process',
    }

    machine_name = preset_to_machine.get(preset)
    if machine_name is None:
        raise ValueError(f"Ground truth state mapping not available for preset: {preset}")

    machine = get_machine(machine_name)
    return machine.get_gt_state(history)


def get_all_state_suffixes(preset: str = 'seven_state_human') -> Dict[str, List[str]]:
    """Get all possible suffix patterns that define each state.

    Args:
        preset: Machine name or preset alias

    Returns:
        Dict mapping state names to their suffix patterns
    """
    # Map preset aliases to machine names
    preset_to_machine = {
        'seven_state_human': 'seven_state_human',
        'seven_state_human_100k': 'seven_state_human',
        'seven_state_human_large': 'seven_state_human',
        'seven_state_human_char_large': 'seven_state_human',
        'sevestateold': 'seven_state_human',
        'golden_mean': 'golden_mean',
        'even_process': 'even_process',
    }

    machine_name = preset_to_machine.get(preset)
    if machine_name is None:
        raise ValueError(f"State suffixes not defined for preset: {preset}")

    machine = get_machine(machine_name)
    return machine.get_state_suffixes()


def collect_histories_by_state(data: np.ndarray, L: int, preset: str) -> Dict[str, List[np.ndarray]]:
    """Group all length-L histories by their ground-truth state."""
    histories_by_state: Dict[str, List[np.ndarray]] = {}
    for i in range(len(data) - L + 1):
        hist = data[i:i+L]
        state = get_gt_state(hist, preset)
        if state is None:
            continue
        histories_by_state.setdefault(state, []).append(hist)
    return histories_by_state


def find_histories_for_state(data: np.ndarray, state: str, suffix_patterns: List[str], L: int) -> List[np.ndarray]:
    """Find all L-length histories that end with patterns defining the given state."""
    if not suffix_patterns:
        return []  # No patterns defined

    histories = []

    # Find all positions where patterns for this state occur
    for i in range(len(data)):
        # Check if position i ends a pattern for this state
        for pattern in suffix_patterns:
            pattern_len = len(pattern)
            if i >= pattern_len - 1:
                # Check if data[i-pattern_len+1:i+1] matches the pattern
                candidate = data[i-pattern_len+1:i+1]
                candidate_str = ''.join(map(str, candidate))
                if candidate_str == pattern:
                    # Found pattern ending at position i
                    # Extract L-length history ending at i
                    start_pos = max(0, i - L + 1)
                    hist = data[start_pos:i+1]
                    if len(hist) == L:
                        histories.append(hist)
                    break  # Found a match for this position

    return histories


# ===== Legacy Test Functions (kept for backward compatibility) =====

def test_seven_state_mapping() -> None:
    """Test function to verify seven-state human machine state mapping."""
    machine = get_machine("seven_state_human")
    test_histories = [
        np.array([0]),
        np.array([1]),
        np.array([1, 0]),
        np.array([1, 1]),
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([1, 0, 1]),
        np.array([1, 1, 1]),
        np.array([1, 0, 1, 1]),
    ]

    print("Testing seven-state human machine state mapping:")
    for hist in test_histories:
        state = machine.get_gt_state(hist)
        hist_str = ''.join(map(str, hist))
        print(f"  History '{hist_str}' -> State {state}")
    print()


def test_state_mapping_simple() -> None:
    """Test the get_seven_state_gt_state function with some examples."""
    machine = get_machine("seven_state_human")
    test_histories = [
        np.array([0,1]),
        np.array([0,1,0]),
        np.array([0,0]),
        np.array([1,1]),
        np.array([1,1,1]),
        np.array([1,1,1,0]),
    ]

    print("Testing state mapping:")
    for hist in test_histories:
        state = machine.get_gt_state(hist)
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' -> {state}")
    print()