"""
Ground truth state mapping functions for different machine presets.

This module provides functions to map histories to their ground truth causal states
for supported machine types:
- seven_state_human: New corrected emission probabilities from Figure 3
- sevestateold: Legacy emission probabilities (preserved for compatibility)
- golden_mean, even_process: Other supported machines

Note: Both seven_state_human and sevestateold use identical transition structures
but have different emission probabilities.
"""

import numpy as np
from typing import List, Dict, Optional


def get_seven_state_gt_state(history: np.ndarray) -> Optional[str]:
    """Determine ground truth state for a history in the seven-state human machine.

    The seven states are defined by suffix patterns. We need to find the LONGEST
    suffix that matches a known state pattern, checking longer patterns first.

    State patterns (longest first):
    - AAAB: ends with 'AAAB' (4 chars) -> state 'AAAB'
    - BAAB: ends with 'BAAB' (4 chars) -> state 'BAAB'
    - BAB: ends with 'BAB' (3 chars) -> state 'BAB'
    - BAA: ends with 'BAA' (3 chars) -> state 'BAA'
    - AAA: ends with 'AAA' (3 chars) -> state 'AAA' (but NOT 'AAAB')
    - BB: ends with 'BB' (2 chars) -> state 'BB'
    - BA: ends with 'BA' (2 chars) -> state 'BA' (but NOT longer patterns)

    For histories shorter than the required pattern length, we cannot determine
    the exact state, so we return None.
    """
    history_str = ''.join(map(str, history))

    # Check for longest patterns first (4 characters, binary)
    if len(history) >= 4:
        if history_str.endswith('0001'):  # AAAB pattern
            return 'aaab'
        elif history_str.endswith('1001'):  # BAAB pattern
            return 'baab'

    # Check for 3-character patterns (binary)
    if len(history) >= 3:
        if history_str.endswith('101'):  # BAB pattern
            return 'bab'
        elif history_str.endswith('100'):  # BAA pattern
            return 'baa'
        elif history_str.endswith('000') and not history_str.endswith('0001'):  # AAA pattern (but not AAAB)
            return 'aaa'

    # Check for 2-character patterns (binary)
    if len(history) >= 2:
        if history_str.endswith('11'):  # BB pattern
            return 'bb'
        elif history_str.endswith('10') and not (len(history) >= 3 and (history_str.endswith('101') or history_str.endswith('100'))):  # BA pattern (but not longer patterns)
            return 'ba'

    # Cannot determine state for shorter histories
    return None


def get_golden_mean_gt_state(history: np.ndarray) -> str:
    """Determine ground truth causal state for a history in the golden mean process.

    Golden mean causal states are based on conditional futures:
    - State A: histories ending with 1 (P(next=0)=0.5, P(next=1)=0.5)
    - State B: histories ending with 0 (P(next=0)=0, P(next=1)=1)

    The empty history (length 0) is in State A.
    """
    if len(history) == 0:
        return 'A'

    last_symbol = int(history[-1])
    return 'A' if last_symbol == 1 else 'B'


def get_even_process_gt_state(history: np.ndarray) -> str:
    """Determine ground truth causal state for a history in the even process."""
    ones_run = 0
    for bit in history[::-1]:
        if int(bit) == 1:
            ones_run += 1
        else:
            break
    return 'E' if ones_run % 2 == 0 else 'O'


def get_gt_state(history: np.ndarray, preset: str) -> Optional[str]:
    """Generic function to get ground truth state for a given preset."""
    if preset in ['seven_state_human', 'seven_state_human_100k', 'seven_state_human_large', 'seven_state_human_char_large', 'sevestateold']:
        return get_seven_state_gt_state(history)
    elif preset == 'golden_mean':
        return get_golden_mean_gt_state(history)
    elif preset == 'even_process':
        return get_even_process_gt_state(history)
    else:
        raise ValueError(f"Ground truth state mapping not available for preset: {preset}")


def get_all_state_suffixes(preset: str = 'seven_state_human') -> Dict[str, List[str]]:
    """Get all possible suffix patterns that define each state."""
    if preset in ['seven_state_human', 'seven_state_human_100k', 'seven_state_human_large', 'seven_state_human_char_large', 'sevestateold']:
        return {
            'bb': ['11'],        # BB state: ends with 11
            'aaa': ['000'],      # AAA state: ends with 000
            'aaab': ['0001'],    # AAAB state: ends with 0001
            'ba': ['10'],        # BA state: ends with 10
            'bab': ['101'],      # BAB state: ends with 101
            'baab': ['1001'],    # BAAB state: ends with 1001
            'baa': ['100']       # BAA state: ends with 100
        }
    elif preset == 'golden_mean':
        return {
            'A': [],  # All histories ending with 1
            'B': []   # All histories ending with 0
        }
    elif preset == 'even_process':
        return {
            'E': [],  # Even parity of trailing 1s
            'O': []   # Odd parity of trailing 1s
        }
    else:
        raise ValueError(f"State suffixes not defined for preset: {preset}")


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


def debug_pattern_matching(data: np.ndarray, L: int = 8) -> None:
    """Debug function to check what patterns are actually found in the data."""
    patterns_to_check = {
        'bb': ['00'],
        'aaa': ['111'],
        'aaab': ['1110'],
        'ba': ['01'],
        'bab': ['010'],
        'baab': ['0110'],
        'baa': ['011']
    }

    print(f"Debug: Checking patterns in L={L} histories")
    for i in range(len(data) - L + 1):
        hist = data[i:i+L]
        hist_str = ''.join(map(str, hist))
        print(f"  History {i}: '{hist_str}'")

        for state, patterns in patterns_to_check.items():
            for pattern in patterns:
                if hist_str.endswith(pattern):
                    print(f"    -> Matches {state} pattern '{pattern}'")

    print()


# Test functions
def test_seven_state_mapping() -> None:
    """Test function to verify seven-state human machine state mapping."""
    test_histories = [
        np.array([0]),        # Should be BB
        np.array([1]),        # Should be BA
        np.array([1, 0]),     # Should be BB (ends with 0)
        np.array([1, 1]),     # Should be BA (ends with 1)
        np.array([0, 1]),     # Should be BA (ends with 1)
        np.array([0, 0]),     # Should be BB (ends with 0)
        np.array([1, 0, 1]),  # Should be BB (ends with 01)
        np.array([1, 1, 1]),  # Should be BA (ends with 1)
        np.array([1, 0, 1, 1]),  # Should be BB (ends with 011)
    ]

    print("Testing seven-state human machine state mapping:")
    for hist in test_histories:
        state = get_seven_state_gt_state(hist)
        hist_str = ''.join(map(str, hist))
        print(f"  History '{hist_str}' -> State {state}")
    print()


def test_pattern_extraction() -> None:
    """Test function to verify pattern extraction logic."""
    # Create test data with known patterns
    test_data = np.array([1,1,1,0,1,0,1,0,0,1,1,0,1,1,1])  # Contains 111, 010, 00, 110, 111
    test_data_str = ''.join(map(str, test_data))
    print(f"Test data: {test_data_str}")

    # Test finding histories for bab state (pattern '010')
    L = 8
    bab_histories = find_histories_for_state(test_data, 'bab', ['010'], L)
    print(f"\nBAB state histories (L={L}):")
    for hist in bab_histories:
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' (ends with '{hist_str[-3:]}')")

    # Test finding histories for bb state (pattern '00')
    bb_histories = find_histories_for_state(test_data, 'bb', ['00'], L)
    print(f"\nBB state histories (L={L}):")
    for hist in bb_histories:
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' (ends with '{hist_str[-2:]}')")

    print()


def test_state_mapping_simple() -> None:
    """Test the get_seven_state_gt_state function with some examples."""
    test_histories = [
        np.array([0,1]),      # '01' - should be BA
        np.array([0,1,0]),    # '010' - should be BAB
        np.array([0,0]),      # '00' - should be BB
        np.array([1,1]),      # '11' - should be None (too short)
        np.array([1,1,1]),    # '111' - should be AAA
        np.array([1,1,1,0]),  # '1110' - should be AAAB
    ]

    print("Testing state mapping:")
    for hist in test_histories:
        state = get_seven_state_gt_state(hist)
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' -> {state}")
    print()