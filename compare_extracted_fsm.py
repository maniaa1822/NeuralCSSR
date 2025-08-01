#!/usr/bin/env python3
"""
Compare extracted FSM against ground truth seven-state machine.

This script performs detailed analysis of how well our trajectory dynamics
extraction recovered the true FSM structure.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def load_ground_truth_fsm(machine_path: Path) -> Dict:
    """Load ground truth FSM structure."""
    with open(machine_path, 'r') as f:
        gt_machine = json.load(f)
    
    print("🎯 Ground Truth FSM Structure:")
    print(f"   States: {gt_machine['states']}")
    print(f"   Start state: {gt_machine['start_state']}")
    
    # Convert to transition matrix format
    states = gt_machine['states']
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    # Initialize transition matrices
    trans_0 = np.zeros((len(states), len(states)))
    trans_1 = np.zeros((len(states), len(states)))
    
    # Parse transitions
    for transition_key, transitions in gt_machine['transitions'].items():
        if '|' in transition_key:
            from_state, token = transition_key.split('|')
            from_idx = state_to_idx[from_state]
            
            for trans in transitions:
                to_state = trans['to_state']
                to_idx = state_to_idx[to_state]
                prob = trans['probability']
                
                if token == '0':
                    trans_0[from_idx, to_idx] = prob
                elif token == '1':
                    trans_1[from_idx, to_idx] = prob
    
    return {
        'states': states,
        'state_to_idx': state_to_idx,
        'transition_0': trans_0,
        'transition_1': trans_1,
        'raw': gt_machine
    }


def load_extracted_fsm(results_path: Path) -> Dict:
    """Load extracted FSM results from CSSR-enhanced method."""
    with open(results_path, 'r') as f:
        extracted = json.load(f)
    
    fsm_structure = extracted['epsilon_machine']
    
    print("🔍 Extracted FSM Structure:")
    print(f"   States: {fsm_structure['states']}")
    print(f"   Extraction method: {extracted['extraction_method']}")
    
    # Convert to transition matrix format
    states = fsm_structure['states']
    state_to_idx = {state: i for i, state in enumerate(states)}
    num_states = len(states)
    
    trans_0 = np.zeros((num_states, num_states))
    trans_1 = np.zeros((num_states, num_states))
    
    for state_name, transitions in fsm_structure['transitions'].items():
        from_idx = state_to_idx[state_name]
        
        if '0' in transitions:
            for target_name, prob in transitions['0'].items():
                if target_name in state_to_idx:
                    to_idx = state_to_idx[target_name]
                    trans_0[from_idx, to_idx] = prob
        
        if '1' in transitions:
            for target_name, prob in transitions['1'].items():
                if target_name in state_to_idx:
                    to_idx = state_to_idx[target_name]
                    trans_1[from_idx, to_idx] = prob
    
    return {
        'states': states,
        'state_to_idx': state_to_idx,
        'transition_0': trans_0,
        'transition_1': trans_1,
        'raw': extracted
    }


def compute_transition_matrix_distance(gt_matrix: np.ndarray, ext_matrix: np.ndarray) -> Dict:
    """Compute various distance metrics between transition matrices."""
    
    # Frobenius norm (L2 distance)
    frobenius_dist = np.linalg.norm(gt_matrix - ext_matrix, 'fro')
    
    # Element-wise L1 distance  
    l1_dist = np.sum(np.abs(gt_matrix - ext_matrix))
    
    # KL divergence (for rows with non-zero entries)
    kl_divs = []
    for i in range(gt_matrix.shape[0]):
        gt_row = gt_matrix[i]
        ext_row = ext_matrix[i]
        
        if np.sum(gt_row) > 0 and np.sum(ext_row) > 0:
            # Add small epsilon to avoid log(0)
            gt_row_norm = gt_row / np.sum(gt_row)
            ext_row_norm = ext_row / np.sum(ext_row)
            
            epsilon = 1e-10
            gt_row_norm = gt_row_norm + epsilon
            ext_row_norm = ext_row_norm + epsilon
            
            kl_div = np.sum(gt_row_norm * np.log(gt_row_norm / ext_row_norm))
            kl_divs.append(kl_div)
    
    mean_kl_div = np.mean(kl_divs) if kl_divs else float('inf')
    
    return {
        'frobenius_distance': frobenius_dist,
        'l1_distance': l1_dist, 
        'mean_kl_divergence': mean_kl_div,
        'max_element_diff': np.max(np.abs(gt_matrix - ext_matrix))
    }


def find_best_state_alignment(gt_fsm: Dict, ext_fsm: Dict) -> Tuple[Dict, float]:
    """Find best alignment between ground truth and extracted states."""
    
    gt_trans_0 = gt_fsm['transition_0']
    gt_trans_1 = gt_fsm['transition_1']
    ext_trans_0 = ext_fsm['transition_0'] 
    ext_trans_1 = ext_fsm['transition_1']
    
    # Try all possible permutations of extracted states
    from itertools import permutations
    
    best_alignment = None
    best_score = float('inf')
    
    num_states = min(len(gt_fsm['states']), len(ext_fsm['states']))
    
    import math
    print(f"🔄 Testing {math.factorial(num_states)} possible state alignments...")
    
    for perm in permutations(range(num_states)):
        # Create permuted extracted matrices
        perm_ext_0 = ext_trans_0[np.ix_(perm, perm)]
        perm_ext_1 = ext_trans_1[np.ix_(perm, perm)]
        
        # Compute total distance
        dist_0 = compute_transition_matrix_distance(gt_trans_0, perm_ext_0)
        dist_1 = compute_transition_matrix_distance(gt_trans_1, perm_ext_1)
        
        total_score = dist_0['frobenius_distance'] + dist_1['frobenius_distance']
        
        if total_score < best_score:
            best_score = total_score
            best_alignment = {
                'permutation': perm,
                'score': total_score,
                'distances_0': dist_0,
                'distances_1': dist_1,
                'aligned_matrices_0': perm_ext_0,
                'aligned_matrices_1': perm_ext_1
            }
    
    print(f"✅ Best alignment found with score: {best_score:.4f}")
    
    return best_alignment, best_score


def analyze_probability_accuracy(gt_fsm: Dict, ext_fsm: Dict, alignment: Dict) -> Dict:
    """Analyze how well specific transition probabilities were recovered."""
    
    perm = alignment['permutation']
    aligned_0 = alignment['aligned_matrices_0']
    aligned_1 = alignment['aligned_matrices_1']
    
    gt_0 = gt_fsm['transition_0']
    gt_1 = gt_fsm['transition_1']
    
    # Find significant transitions (probability > 0.1)
    threshold = 0.1
    significant_transitions = []
    
    for i in range(gt_0.shape[0]):
        for j in range(gt_0.shape[1]):
            if gt_0[i, j] > threshold:
                error = abs(gt_0[i, j] - aligned_0[i, j])
                significant_transitions.append({
                    'from_state': gt_fsm['states'][i],
                    'to_state': gt_fsm['states'][j], 
                    'token': '0',
                    'gt_prob': gt_0[i, j],
                    'extracted_prob': aligned_0[i, j],
                    'error': error
                })
            
            if gt_1[i, j] > threshold:
                error = abs(gt_1[i, j] - aligned_1[i, j])
                significant_transitions.append({
                    'from_state': gt_fsm['states'][i],
                    'to_state': gt_fsm['states'][j],
                    'token': '1', 
                    'gt_prob': gt_1[i, j],
                    'extracted_prob': aligned_1[i, j],
                    'error': error
                })
    
    # Sort by error
    significant_transitions.sort(key=lambda x: x['error'])
    
    return {
        'significant_transitions': significant_transitions,
        'mean_error': np.mean([t['error'] for t in significant_transitions]),
        'max_error': np.max([t['error'] for t in significant_transitions]) if significant_transitions else 0,
        'num_transitions': len(significant_transitions)
    }


def create_comparison_visualizations(gt_fsm: Dict, ext_fsm: Dict, alignment: Dict, output_dir: Path):
    """Create visualizations comparing ground truth vs extracted FSMs."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot transition matrices comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ground truth matrices
    gt_0 = gt_fsm['transition_0']
    gt_1 = gt_fsm['transition_1']
    aligned_0 = alignment['aligned_matrices_0']
    aligned_1 = alignment['aligned_matrices_1']
    
    state_labels = [s if len(s) <= 4 else s[:4] for s in gt_fsm['states']]
    
    # Token 0 transitions
    sns.heatmap(gt_0, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=state_labels, yticklabels=state_labels,
                ax=axes[0, 0])
    axes[0, 0].set_title('Ground Truth: Token "0" Transitions')
    
    sns.heatmap(aligned_0, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=state_labels, yticklabels=state_labels, 
                ax=axes[0, 1])
    axes[0, 1].set_title('Extracted: Token "0" Transitions')
    
    # Difference matrix for token 0
    diff_0 = np.abs(gt_0 - aligned_0)
    sns.heatmap(diff_0, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=state_labels, yticklabels=state_labels,
                ax=axes[0, 2])
    axes[0, 2].set_title('Absolute Difference: Token "0"')
    
    # Token 1 transitions  
    sns.heatmap(gt_1, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=state_labels, yticklabels=state_labels,
                ax=axes[1, 0])
    axes[1, 0].set_title('Ground Truth: Token "1" Transitions')
    
    sns.heatmap(aligned_1, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=state_labels, yticklabels=state_labels,
                ax=axes[1, 1]) 
    axes[1, 1].set_title('Extracted: Token "1" Transitions')
    
    # Difference matrix for token 1
    diff_1 = np.abs(gt_1 - aligned_1)
    sns.heatmap(diff_1, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=state_labels, yticklabels=state_labels,
                ax=axes[1, 2])
    axes[1, 2].set_title('Absolute Difference: Token "1"')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fsm_comparison_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}")


def main():
    # Paths
    gt_machine_path = Path("domain_machines/seven_state_human/seven_state_human/seven_state_human.machine.json")
    extracted_results_path = Path("results/cssr_enhanced_extraction/cssr_enhanced_results.json")
    output_dir = Path("results/fsm_comparison_analysis")
    
    print("🔍 FSM EXTRACTION QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load FSMs
    gt_fsm = load_ground_truth_fsm(gt_machine_path)
    ext_fsm = load_extracted_fsm(extracted_results_path)
    
    print("\n" + "=" * 60)
    
    # Find best alignment
    alignment, best_score = find_best_state_alignment(gt_fsm, ext_fsm)
    
    print(f"\n🎯 ALIGNMENT RESULTS:")
    print(f"   Best permutation: {alignment['permutation']}")
    print(f"   Total distance score: {best_score:.4f}")
    print(f"   Token '0' Frobenius distance: {alignment['distances_0']['frobenius_distance']:.4f}")
    print(f"   Token '1' Frobenius distance: {alignment['distances_1']['frobenius_distance']:.4f}")
    
    # Analyze probability accuracy
    prob_analysis = analyze_probability_accuracy(gt_fsm, ext_fsm, alignment)
    
    print(f"\n📊 PROBABILITY ACCURACY:")
    print(f"   Significant transitions analyzed: {prob_analysis['num_transitions']}")
    print(f"   Mean absolute error: {prob_analysis['mean_error']:.4f}")
    print(f"   Max absolute error: {prob_analysis['max_error']:.4f}")
    
    print(f"\n🎯 TOP 10 BEST RECOVERED TRANSITIONS:")
    for i, trans in enumerate(prob_analysis['significant_transitions'][:10]):
        print(f"   {i+1:2d}. {trans['from_state']:>4} --{trans['token']}--> {trans['to_state']:>4}: "
              f"GT={trans['gt_prob']:.4f}, Ext={trans['extracted_prob']:.4f}, "
              f"Error={trans['error']:.4f}")
    
    print(f"\n❌ TOP 5 WORST RECOVERED TRANSITIONS:")
    for i, trans in enumerate(prob_analysis['significant_transitions'][-5:]):
        print(f"   {i+1:2d}. {trans['from_state']:>4} --{trans['token']}--> {trans['to_state']:>4}: "
              f"GT={trans['gt_prob']:.4f}, Ext={trans['extracted_prob']:.4f}, "
              f"Error={trans['error']:.4f}")
    
    # Create visualizations
    create_comparison_visualizations(gt_fsm, ext_fsm, alignment, output_dir)
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if prob_analysis['mean_error'] < 0.1:
        print("   ✅ EXCELLENT: Mean error < 0.1")
    elif prob_analysis['mean_error'] < 0.2:
        print("   ✅ GOOD: Mean error < 0.2")
    elif prob_analysis['mean_error'] < 0.3:
        print("   ⚠️  FAIR: Mean error < 0.3")
    else:
        print("   ❌ POOR: Mean error >= 0.3")
    
    print("=" * 60)
    
    # Save detailed analysis
    analysis_results = {
        'alignment': {
            'permutation': alignment['permutation'],
            'total_score': float(best_score),
            'distances_token_0': {k: float(v) for k, v in alignment['distances_0'].items()},
            'distances_token_1': {k: float(v) for k, v in alignment['distances_1'].items()}
        },
        'probability_analysis': {
            'mean_error': float(prob_analysis['mean_error']),
            'max_error': float(prob_analysis['max_error']),
            'num_significant_transitions': prob_analysis['num_transitions'],
            'significant_transitions': prob_analysis['significant_transitions']
        },
        'ground_truth_states': gt_fsm['states'],
        'extraction_method': ext_fsm['raw']['extraction_method']
    }
    
    with open(output_dir / 'fsm_comparison_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"📁 Detailed analysis saved to {output_dir}")

if __name__ == '__main__':
    main()
