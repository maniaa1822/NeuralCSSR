#!/usr/bin/env python3
"""
Extract and analyze the finite state machine structure from transformer internal states.
Determines if the transition matrices represent actual extracted FSMs.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_transition_analysis(results_path: Path) -> dict:
    """Load the transition analysis results."""
    with open(results_path) as f:
        data = json.load(f)
    return data['transition_analysis']

def analyze_fsm_properties(transition_matrix: np.ndarray, layer_name: str) -> dict:
    """Analyze whether a transition matrix represents a valid FSM."""
    
    # Convert to numpy array
    trans_matrix = np.array(transition_matrix)
    n_states = trans_matrix.shape[0]
    
    # Normalize to get probabilities
    row_sums = trans_matrix.sum(axis=1)
    prob_matrix = np.divide(
        trans_matrix, 
        row_sums[:, np.newaxis], 
        out=np.zeros_like(trans_matrix), 
        where=row_sums[:, np.newaxis] != 0
    )
    
    # FSM properties analysis
    properties = {
        'layer': layer_name,
        'num_states': n_states,
        'total_transitions': trans_matrix.sum(),
        'raw_matrix': trans_matrix,
        'probability_matrix': prob_matrix
    }
    
    # Check if it's a valid stochastic matrix (rows sum to ~1)
    row_sums_prob = prob_matrix.sum(axis=1)
    valid_stochastic = np.allclose(row_sums_prob[row_sums_prob > 0], 1.0, atol=1e-6)
    properties['is_valid_stochastic'] = valid_stochastic
    
    # Check which states are actually used (have outgoing transitions)
    active_states = row_sums > 0
    properties['active_states'] = np.where(active_states)[0].tolist()
    properties['num_active_states'] = active_states.sum()
    
    # Analyze connectivity
    # States that can reach other states
    outgoing_connections = (prob_matrix > 0).sum(axis=1)
    # States that can be reached from other states  
    incoming_connections = (prob_matrix > 0).sum(axis=0)
    
    properties['outgoing_connections'] = outgoing_connections.tolist()
    properties['incoming_connections'] = incoming_connections.tolist()
    
    # Check for self-loops (diagonal elements)
    self_loops = np.diag(prob_matrix)
    properties['self_loops'] = self_loops.tolist()
    properties['strong_self_loops'] = (self_loops > 0.5).sum()
    
    # Check if it's deterministic (each row has at most one 1.0)
    max_probs = prob_matrix.max(axis=1)
    is_deterministic = np.allclose(max_probs[max_probs > 0], 1.0, atol=1e-6)
    properties['is_deterministic'] = is_deterministic
    
    # Find dominant transitions (probability > 0.1)
    dominant_transitions = []
    for i in range(n_states):
        for j in range(n_states):
            if prob_matrix[i, j] > 0.1:
                dominant_transitions.append({
                    'from': i,
                    'to': j, 
                    'probability': prob_matrix[i, j],
                    'count': trans_matrix[i, j]
                })
    properties['dominant_transitions'] = dominant_transitions
    
    return properties

def create_fsm_visualization(properties: dict, output_dir: Path):
    """Create FSM visualization showing the extracted state machine."""
    layer = properties['layer']
    prob_matrix = properties['probability_matrix']
    
    # Create a cleaner FSM visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Heatmap of transition probabilities
    mask = prob_matrix == 0
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='Blues',
                mask=mask, ax=ax1, cbar_kws={'label': 'Transition Probability'})
    ax1.set_title(f'Layer {layer} - FSM Transition Probabilities')
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')
    
    # 2. Graph-like visualization of significant transitions
    ax2.set_xlim(-1, properties['num_states'])
    ax2.set_ylim(-1, properties['num_states'])
    ax2.set_aspect('equal')
    
    # Draw states as circles
    state_positions = {}
    for i in range(properties['num_states']):
        angle = 2 * np.pi * i / properties['num_states']
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        state_positions[i] = (x, y)
        
        # Color active vs inactive states
        color = 'lightblue' if i in properties['active_states'] else 'lightgray'
        circle = plt.Circle((x, y), 0.3, color=color, ec='black')
        ax2.add_patch(circle)
        ax2.text(x, y, str(i), ha='center', va='center', fontweight='bold')
    
    # Draw significant transitions (probability > 0.1)
    for trans in properties['dominant_transitions']:
        from_state, to_state = trans['from'], trans['to']
        prob = trans['probability']
        
        if from_state in state_positions and to_state in state_positions:
            x1, y1 = state_positions[from_state]
            x2, y2 = state_positions[to_state]
            
            # Line thickness proportional to probability
            linewidth = max(0.5, prob * 5)
            
            if from_state == to_state:
                # Self-loop
                circle = plt.Circle((x1 + 0.5, y1), 0.2, fill=False, 
                                  linewidth=linewidth, color='red')
                ax2.add_patch(circle)
                ax2.text(x1 + 0.7, y1, f'{prob:.2f}', fontsize=8)
            else:
                # Regular transition
                ax2.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', lw=linewidth, color='blue'))
                # Add probability label
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax2.text(mid_x, mid_y, f'{prob:.2f}', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    ax2.set_title(f'Layer {layer} - FSM Graph (transitions > 0.1)')
    ax2.set_xlabel('State Transitions')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'extracted_fsm_layer_{layer}.png', dpi=150, bbox_inches='tight')
    plt.close()

def compare_with_ground_truth(all_properties: dict, dataset_path: Path) -> dict:
    """Compare extracted FSMs with ground truth."""
    # Load ground truth
    gt_path = dataset_path / "ground_truth" / "machine_properties.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    
    comparison = {
        'ground_truth_summary': {},
        'extracted_summary': {},
        'analysis': []
    }
    
    # Summarize ground truth
    for machine_id, machine_data in ground_truth.items():
        comparison['ground_truth_summary'][machine_id] = {
            'num_states': machine_data['num_states'],
            'is_topological': machine_data.get('is_topological', True),
            'entropy_rate': machine_data.get('entropy_rate', 'unknown')
        }
    
    total_gt_states = sum(m['num_states'] for m in ground_truth.values())
    
    # Summarize extracted FSMs
    for layer, props in all_properties.items():
        comparison['extracted_summary'][layer] = {
            'num_states': props['num_states'],
            'num_active_states': props['num_active_states'],
            'is_deterministic': props['is_deterministic'],
            'strong_self_loops': props['strong_self_loops'],
            'total_transitions': props['total_transitions']
        }
    
    # Analysis
    comparison['analysis'].append(f"Ground truth: {len(ground_truth)} machines with {total_gt_states} total states")
    comparison['analysis'].append(f"Extracted: {len(all_properties)} layers with {all_properties[list(all_properties.keys())[0]]['num_states']} states each")
    
    # Check if extracted states match expected count
    expected_states = total_gt_states
    for layer, props in all_properties.items():
        active_states = props['num_active_states']
        if active_states >= expected_states:
            comparison['analysis'].append(f"Layer {layer}: {active_states} active states (≥ {expected_states} expected) ✅")
        else:
            comparison['analysis'].append(f"Layer {layer}: {active_states} active states (< {expected_states} expected) ⚠️")
    
    return comparison

def generate_fsm_extraction_report(all_properties: dict, comparison: dict) -> str:
    """Generate report on extracted FSM analysis."""
    
    report = []
    report.append("=" * 80)
    report.append("EXTRACTED FINITE STATE MACHINE ANALYSIS")
    report.append("=" * 80)
    report.append("")
    
    # Answer the key question
    report.append("🔍 KEY QUESTION: Are the transition PNGs actual FSMs?")
    report.append("-" * 50)
    report.append("")
    
    # Determine if these are valid FSMs
    valid_fsms = []
    for layer, props in all_properties.items():
        is_valid = (
            props['is_valid_stochastic'] and 
            props['num_active_states'] > 1 and
            len(props['dominant_transitions']) > 0
        )
        valid_fsms.append(is_valid)
        
        if is_valid:
            report.append(f"✅ Layer {layer}: VALID FSM STRUCTURE")
        else:
            report.append(f"❌ Layer {layer}: NOT A COMPLETE FSM")
    
    if all(valid_fsms):
        report.append("")
        report.append("🎯 ANSWER: YES - The transition matrices represent extracted FSMs!")
    elif any(valid_fsms):
        report.append("")
        report.append("⚠️  ANSWER: PARTIAL - Some layers show FSM-like structure")
    else:
        report.append("")
        report.append("❌ ANSWER: NO - These are not complete FSM structures")
    
    report.append("")
    
    # Detailed analysis for each layer
    report.append("📊 DETAILED LAYER ANALYSIS")
    report.append("-" * 40)
    
    for layer, props in all_properties.items():
        report.append(f"")
        report.append(f"Layer {layer}:")
        report.append(f"  • Total states defined: {props['num_states']}")
        report.append(f"  • Active states (with transitions): {props['num_active_states']}")
        report.append(f"  • Valid stochastic matrix: {'Yes' if props['is_valid_stochastic'] else 'No'}")
        report.append(f"  • Deterministic transitions: {'Yes' if props['is_deterministic'] else 'No'}")
        report.append(f"  • States with strong self-loops: {props['strong_self_loops']}")
        report.append(f"  • Total transition observations: {props['total_transitions']}")
        
        # Show key transitions
        if props['dominant_transitions']:
            report.append(f"  • Key transitions (>10% probability):")
            for trans in props['dominant_transitions'][:5]:  # Show top 5
                report.append(f"    State {trans['from']} → State {trans['to']}: {trans['probability']:.3f} ({trans['count']} times)")
        else:
            report.append(f"  • No dominant transitions found")
    
    report.append("")
    
    # Ground truth comparison
    report.append("🎰 COMPARISON WITH GROUND TRUTH")
    report.append("-" * 40)
    
    for analysis_line in comparison['analysis']:
        report.append(f"  {analysis_line}")
    
    report.append("")
    
    # Key insights
    report.append("💡 KEY INSIGHTS")
    report.append("-" * 40)
    
    # Check if transformer learned more complex states than ground truth
    gt_states = sum(m['num_states'] for m in comparison['ground_truth_summary'].values())
    extracted_active = [props['num_active_states'] for props in all_properties.values()]
    
    if any(states > gt_states for states in extracted_active):
        report.append("✅ ENHANCED STATE SPACE: Transformer learned more states than ground truth")
        report.append("   This suggests the model discovered additional context-dependent states")
    
    # Check for layer-wise progression
    if len(extracted_active) > 1:
        if extracted_active[-1] != extracted_active[0]:
            report.append("✅ LAYER SPECIALIZATION: Different layers learned different state structures")
        else:
            report.append("⚠️  Consistent state count across layers")
    
    # Check transition complexity
    total_transitions = sum(len(props['dominant_transitions']) for props in all_properties.values())
    if total_transitions > gt_states * 2:  # Heuristic for complex connectivity
        report.append("✅ RICH CONNECTIVITY: Extracted FSMs show complex transition patterns")
    
    report.append("")
    
    # Final verdict
    report.append("🏆 FINAL VERDICT")
    report.append("-" * 40)
    
    if all(valid_fsms) and any(states >= gt_states for states in extracted_active):
        report.append("🎯 SUCCESSFUL FSM EXTRACTION: The transformer learned interpretable FSM structures!")
        report.append("   • Transition matrices represent actual finite state machines")
        report.append("   • Model discovered states matching or exceeding ground truth complexity")
        report.append("   • Internal representations encode structured state transition dynamics")
    elif any(valid_fsms):
        report.append("⚠️  PARTIAL FSM EXTRACTION: Some FSM-like structures detected")
        report.append("   • Transition patterns show some state-like organization")
        report.append("   • May represent distributed or hierarchical state encoding")
    else:
        report.append("❌ LIMITED FSM EXTRACTION: Transition matrices don't represent clear FSMs")
        report.append("   • Patterns may represent other forms of internal organization")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    # Paths
    results_path = Path("results/internal_analysis/internal_analysis_results.json")
    dataset_path = Path("datasets/biased_exp")
    output_dir = Path("results/internal_analysis")
    
    print("Loading transition analysis results...")
    transition_data = load_transition_analysis(results_path)
    
    print("Analyzing FSM properties for each layer...")
    all_properties = {}
    
    for layer_id, layer_data in transition_data.items():
        props = analyze_fsm_properties(
            layer_data['normalized_transition_matrix'], 
            layer_id
        )
        all_properties[layer_id] = props
        
        print(f"Layer {layer_id}: {props['num_active_states']}/{props['num_states']} active states, "
              f"{len(props['dominant_transitions'])} dominant transitions")
        
        # Create FSM visualization
        create_fsm_visualization(props, output_dir)
    
    print("Comparing with ground truth...")
    comparison = compare_with_ground_truth(all_properties, dataset_path)
    
    print("Generating analysis report...")
    report = generate_fsm_extraction_report(all_properties, comparison)
    
    print(report)
    
    # Save results
    with open(output_dir / "fsm_extraction_analysis.txt", 'w') as f:
        f.write(report)
    
    # Save detailed FSM data
    fsm_data = {
        'fsm_properties': {k: {
            'layer': v['layer'],
            'num_states': v['num_states'],
            'num_active_states': v['num_active_states'],
            'is_valid_stochastic': v['is_valid_stochastic'],
            'is_deterministic': v['is_deterministic'],
            'dominant_transitions': v['dominant_transitions'],
            'probability_matrix': v['probability_matrix'].tolist()
        } for k, v in all_properties.items()},
        'comparison': comparison
    }
    
    with open(output_dir / "extracted_fsm_data.json", 'w') as f:
        json.dump(fsm_data, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()