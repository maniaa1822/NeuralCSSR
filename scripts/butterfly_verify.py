import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from machines import get_machine

def verify_butterfly():
    print("Verifying Butterfly Machine...")
    
    # 1. Load Machine
    try:
        m = get_machine("butterfly")
        print(f"Loaded machine: {m.display_name}")
    except Exception as e:
        print(f"Failed to load machine: {e}")
        return

    # 2. Check basics
    print(f"States: {m.states}")
    print(f"Alphabet: {m.alphabet}")
    print(f"Memory Length: {m.memory_length}")

    # 3. Simulate and Verify
    # We will manually simulate to track current state and verify updates
    current_state = m.start_state
    history = []
    
    steps = 2000
    visited_states = set()
    visited_states.add(current_state)
    
    violations = 0
    gt_mismatches = 0
    
    print(f"\nSimulating {steps} steps...")
    
    for t in range(steps):
        # Emit symbol
        probs = m.emissions[current_state]
        symbols = list(probs.keys())
        probabilities = list(probs.values())
        symbol = np.random.choice(symbols, p=probabilities)
        
        # Update history
        history.append(int(symbol))
        
        # Transition
        next_state = m.transitions[(current_state, symbol)]
        
        # Debug print for first few steps
        if t < 20:
            print(f"Step {t}: {current_state} --({symbol})--> {next_state}")
            
        current_state = next_state
        visited_states.add(current_state)
        
        # Verify Ground Truth Map
        if len(history) >= m.memory_length:
            hist_arr = np.array(history)
            gt_state = m.get_gt_state(hist_arr)
            
            if gt_state is None:
                # Should not happen if L >= 2, effectively
                # Wait, get_gt_state checks history length.
                # If history is exactly 2, it should work.
                print(f"WARNING: get_gt_state returned None at step {t} with history tail {history[-5:]}")
                gt_mismatches += 1
            elif gt_state != current_state:
                print(f"ERROR: GT Mismatch at step {t}!")
                print(f"  Actual: {current_state}")
                print(f"  Inferred: {gt_state}")
                print(f"  History Tail: {history[-5:]}")
                gt_mismatches += 1

    # 4. Report
    print("\nVerification Results:")
    print(f"Visited States: {sorted(list(visited_states))}")
    if len(visited_states) == len(m.states):
        print("PASS: All states visited.")
    else:
        print(f"FAIL: Missed states: {set(m.states) - visited_states}")

    if gt_mismatches == 0:
        print("PASS: Ground truth mapping perfect.")
    else:
        print(f"FAIL: {gt_mismatches} ground truth mapping errors.")

if __name__ == "__main__":
    verify_butterfly()
