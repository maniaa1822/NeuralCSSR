import sys
from pathlib import Path

def check_boundary_histories(data_path, boundary_idx=50000, L=6):
    with open(data_path, 'r') as f:
        raw = f.read()
    data = ''.join(ch for ch in raw if ch in ('0', '1'))
    
    print(f"Checking boundary at index {boundary_idx} with L={L}")
    
    # Histories that contain the boundary:
    # The boundary is between data[boundary_idx-1] and data[boundary_idx]
    # (assuming 0-indexed, 50000 is the first char of the second machine)
    
    # Histories ending at i (exclusive of next token): data[i-L : i]
    # We want histories that overlap the boundary.
    # Range of i such that [i-L, i) contains 50000.
    # i > 50000 and i-L <= 50000
    # i goes from 50001 to 50000+L
    
    mixed_histories = []
    for i in range(boundary_idx + 1, boundary_idx + L + 1):
        hist = data[i-L : i]
        mixed_histories.append((i, hist))
        print(f"End Index {i}: {hist} (Mixed)")
        
    return mixed_histories

if __name__ == '__main__':
    data_path = 'experiments/datasets/gm_seven_union/combined.dat'
    mixed = check_boundary_histories(data_path)
    
    # Singleton states from previous run (gm_seven_union_k3_no_platt.json)
    # State 1: 000100
    # State 9: 011011
    # State 10: 001111
    
    singletons = {
        '000100': 'State 1',
        '011011': 'State 9',
        '001111': 'State 10'
    }
    
    print("\nComparing with singleton states:")
    found = False
    for i, hist in mixed:
        if hist in singletons:
            print(f"MATCH! Mixed history at {i} ({hist}) matches {singletons[hist]}")
            found = True
            
    if not found:
        print("No matches found. The singleton states are NOT boundary artifacts.")
