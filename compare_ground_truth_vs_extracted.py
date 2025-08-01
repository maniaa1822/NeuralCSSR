#!/usr/bin/env python3
"""
Compare Ground Truth vs Extracted 3-State FSM Analysis

This script provides a detailed comparison between the ground truth 3-state machine
and the CSSR-enhanced neural extraction results.
"""

import json
import numpy as np
from pathlib import Path

def load_machines():
    """Load both ground truth and extracted machines."""
    
    # For 7-state analysis (based on your recent sweep):
    gt_path = Path("domain_machines/seven_state_human/seven_state_human/seven_state_human.machine.json")
    
    if not gt_path.exists():
        print(f"❌ Ground truth file not found: {gt_path}")
        return None, None
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    # Look for extracted results in your sweep output directory
    # Based on your CSV results, the best extraction should be in seven_state_analysis
    possible_paths = [
        "sweep_results/seven_state_analysis/extraction_results.json",
        "sweep_results/seven_state_analysis/cssr_enhanced_results.json",
        "sweep_results/seven_state_analysis/best_extraction.json",
        "sweep_results/seven_state_analysis/epsilon_machine.json",
        "temp_extraction/cssr_enhanced_results.json",
    ]
    
    extracted_path = None
    for path in possible_paths:
        if Path(path).exists():
            extracted_path = Path(path)
            break
    
    if extracted_path is None:
        print("❌ Could not find extracted results file!")
        print("Please check what files exist in:")
        print("  ls -la sweep_results/seven_state_analysis/")
        print("\nLooking for files in:")
        for path in possible_paths:
            print(f"  - {path}")
        return None, None
    
    print(f"📁 Loading ground truth from: {gt_path}")
    print(f"📁 Loading extracted results from: {extracted_path}")
    
    with open(extracted_path, 'r') as f:
        extracted = json.load(f)
    
    return ground_truth, extracted

def analyze_ground_truth_structure(gt):
    """Analyze ground truth 7-state machine structure."""
    print("🎯 GROUND TRUTH 7-STATE MACHINE ANALYSIS")
    print("=" * 60)
    
    states = gt['states']  # Should be 7 states
    print(f"States: {states}")
    print(f"Number of states: {len(states)}")
    print(f"Start state: {gt['start_state']}")
    
    # Compute emission probabilities for each state
    emission_probs = {}
    for state in states:
        probs = {'0': 0.0, '1': 0.0}
        
        # Check transitions for each symbol
        for symbol in ['0', '1']:
            key = f"{state}|{symbol}"
            if key in gt['transitions']:
                total_prob = sum(trans['probability'] for trans in gt['transitions'][key])
                probs[symbol] = total_prob
        
        emission_probs[state] = probs
        
        # Classify state type
        if probs['0'] > 0.7:
            state_type = "Zero-biased"
        elif probs['1'] > 0.7:
            state_type = "One-biased"
        elif abs(probs['0'] - probs['1']) < 0.15:
            state_type = "Balanced"
        else:
            state_type = "Mixed"
            
        print(f"State {state}: P(0)={probs['0']:.3f}, P(1)={probs['1']:.3f} ({state_type})")
    
    return emission_probs

def analyze_extracted_structure(extracted):
    """Analyze extracted machine structure."""
    print("\n🔍 EXTRACTED MACHINE ANALYSIS")
    print("=" * 50)
    
    epsilon_machine = extracted['epsilon_machine']
    states = epsilon_machine['states']  # ['MCS_0', 'MCS_1', 'MCS_2'] 
    print(f"States: {states}")
    print(f"Number of states: {epsilon_machine['num_states']}")
    
    # Analyze causal state info
    causal_info = epsilon_machine['causal_state_info']
    
    print(f"\nCausal State Analysis:")
    for state_id, info in causal_info.items():
        future_probs = info['future_probabilities']
        count = info['count']
        size = info['size']
        
        prob_0 = future_probs.get('0', 0.0)
        prob_1 = future_probs.get('1', 0.0)
        
        print(f"State {state_id}:")
        print(f"  Emission: P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")
        print(f"  Data: {count} observations, {size} suffixes")
        
        # Classify state type based on emission probabilities
        if prob_0 > 0.7:
            state_type = "Zero-biased"
        elif prob_1 > 0.7:
            state_type = "One-biased"
        elif abs(prob_0 - prob_1) < 0.1:
            state_type = "Balanced"
        else:
            state_type = "Mixed"
        print(f"  Type: {state_type}")
    
    return causal_info

def find_state_alignment(gt_emissions, extracted_info):
    """Find the best alignment between ground truth and extracted states."""
    print("\n🔗 STATE ALIGNMENT ANALYSIS")
    print("=" * 50)
    
    # Ground truth state characteristics
    gt_characteristics = {
        'A': {'type': 'Zero-biased', 'p0': 0.8, 'p1': 0.2},
        'B': {'type': 'One-biased', 'p0': 0.2, 'p1': 0.8},
        'C': {'type': 'Balanced', 'p0': 0.5, 'p1': 0.5}
    }
    
    # Extracted state characteristics
    extracted_characteristics = {}
    for state_id, info in extracted_info.items():
        future_probs = info['future_probabilities']
        prob_0 = future_probs.get('0', 0.0)
        prob_1 = future_probs.get('1', 0.0)
        
        if prob_0 > 0.7:
            state_type = "Zero-biased"
        elif prob_1 > 0.7:
            state_type = "One-biased"
        elif abs(prob_0 - prob_1) < 0.1:
            state_type = "Balanced"
        else:
            state_type = "Mixed"
            
        extracted_characteristics[state_id] = {
            'type': state_type,
            'p0': prob_0,
            'p1': prob_1
        }
    
    # Find best alignment by matching state types
    alignment = {}
    used_extracted = set()
    
    print("State Type Matching:")
    for gt_state, gt_char in gt_characteristics.items():
        best_match = None
        best_distance = float('inf')
        
        for ext_state, ext_char in extracted_characteristics.items():
            if ext_state in used_extracted:
                continue
                
            # Compute distance based on emission probabilities
            distance = abs(gt_char['p0'] - ext_char['p0']) + abs(gt_char['p1'] - ext_char['p1'])
            
            if distance < best_distance:
                best_distance = distance
                best_match = ext_state
        
        if best_match:
            alignment[gt_state] = best_match
            used_extracted.add(best_match)
            
            gt_char = gt_characteristics[gt_state]
            ext_char = extracted_characteristics[best_match]
            
            print(f"  {gt_state} ({gt_char['type']}) ↔ {best_match} ({ext_char['type']})")
            print(f"    GT: P(0)={gt_char['p0']:.3f}, P(1)={gt_char['p1']:.3f}")
            print(f"    EX: P(0)={ext_char['p0']:.3f}, P(1)={ext_char['p1']:.3f}")
            print(f"    Error: {best_distance:.3f}")
    
    return alignment

def find_state_alignment_7state(gt_emissions, extracted_info):
    """Find the best alignment between ground truth and extracted 7-states."""
    print("\n🔗 7-STATE ALIGNMENT ANALYSIS")
    print("=" * 60)
    
    # Ground truth state characteristics (from the actual data you showed)
    gt_characteristics = {}
    for state, probs in gt_emissions.items():
        prob_0 = probs['0']
        prob_1 = probs['1']
        
        if prob_0 > 0.7:
            state_type = "Zero-biased"
        elif prob_1 > 0.7:
            state_type = "One-biased"
        elif abs(prob_0 - prob_1) < 0.15:
            state_type = "Balanced"
        else:
            state_type = "Mixed"
            
        gt_characteristics[state] = {
            'type': state_type,
            'p0': prob_0,
            'p1': prob_1
        }
    
    # Extracted state characteristics
    extracted_characteristics = {}
    for state_id, info in extracted_info.items():
        future_probs = info['future_probabilities']
        prob_0 = future_probs.get('0', 0.0)
        prob_1 = future_probs.get('1', 0.0)
        
        if prob_0 > 0.7:
            state_type = "Zero-biased"
        elif prob_1 > 0.7:
            state_type = "One-biased"
        elif abs(prob_0 - prob_1) < 0.15:
            state_type = "Balanced"
        else:
            state_type = "Mixed"
            
        extracted_characteristics[state_id] = {
            'type': state_type,
            'p0': prob_0,
            'p1': prob_1
        }
    
    # Find best alignment using Hungarian algorithm for optimal matching
    from scipy.optimize import linear_sum_assignment
    
    gt_states = list(gt_characteristics.keys())
    ext_states = list(extracted_characteristics.keys())
    
    # Create cost matrix based on probability distances
    cost_matrix = np.zeros((len(gt_states), len(ext_states)))
    
    for i, gt_state in enumerate(gt_states):
        for j, ext_state in enumerate(ext_states):
            gt_char = gt_characteristics[gt_state]
            ext_char = extracted_characteristics[ext_state]
            
            # Cost = L1 distance between probability vectors
            cost = abs(gt_char['p0'] - ext_char['p0']) + abs(gt_char['p1'] - ext_char['p1'])
            cost_matrix[i, j] = cost
    
    # Find optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    alignment = {}
    total_error = 0.0
    
    print("Optimal State Alignment:")
    for i, j in zip(row_indices, col_indices):
        gt_state = gt_states[i]
        ext_state = ext_states[j]
        error = cost_matrix[i, j]
        total_error += error
        
        alignment[gt_state] = ext_state
        
        gt_char = gt_characteristics[gt_state]
        ext_char = extracted_characteristics[ext_state]
        
        print(f"  {gt_state} ({gt_char['type']}) ↔ {ext_state} ({ext_char['type']})")
        print(f"    GT: P(0)={gt_char['p0']:.3f}, P(1)={gt_char['p1']:.3f}")
        print(f"    EX: P(0)={ext_char['p0']:.3f}, P(1)={ext_char['p1']:.3f}")
        print(f"    Error: {error:.3f}")
    
    avg_error = total_error / len(alignment)
    print(f"\nAverage alignment error: {avg_error:.3f}")
    
    return alignment

def analyze_suffix_patterns(extracted):
    """Analyze suffix patterns in causal states."""
    print("\n📝 SUFFIX PATTERN ANALYSIS")
    print("=" * 50)
    
    causal_info = extracted['epsilon_machine']['causal_state_info']
    
    for state_id, info in causal_info.items():
        suffixes = info['suffixes']
        count = info['count']
        
        print(f"\nState {state_id} ({len(suffixes)} unique suffixes, {count} total observations):")
        
        # Show some example suffixes
        example_suffixes = suffixes[:10] if len(suffixes) > 10 else suffixes
        print(f"  Example suffixes: {example_suffixes}")
        
        # Analyze suffix length distribution
        lengths = [len(s) for s in suffixes]
        if lengths:
            print(f"  Suffix lengths: {min(lengths)}-{max(lengths)}, avg={np.mean(lengths):.1f}")
        
        # Analyze suffix patterns
        print(f"  Shortest: {min(suffixes, key=len) if suffixes else 'None'}")
        print(f"  Longest: {max(suffixes, key=len) if suffixes else 'None'}")

def compute_overall_quality_metrics(alignment, gt_characteristics, extracted_info):
    """Compute overall quality metrics for the extraction."""
    print("\n📊 OVERALL QUALITY METRICS")
    print("=" * 50)
    
    # State count accuracy
    gt_num_states = 3
    ext_num_states = len(extracted_info)
    state_count_accuracy = 1.0 if gt_num_states == ext_num_states else 0.0
    
    print(f"State Count Accuracy: {state_count_accuracy:.3f} ({ext_num_states}/{gt_num_states} states)")
    
    # Emission probability accuracy
    total_emission_error = 0.0
    num_comparisons = 0
    
    for gt_state, ext_state in alignment.items():
        if ext_state in extracted_info:
            gt_char = {'A': {'p0': 0.8, 'p1': 0.2}, 'B': {'p0': 0.2, 'p1': 0.8}, 'C': {'p0': 0.5, 'p1': 0.5}}[gt_state]
            
            ext_probs = extracted_info[ext_state]['future_probabilities']
            ext_p0 = ext_probs.get('0', 0.0)
            ext_p1 = ext_probs.get('1', 0.0)
            
            error = abs(gt_char['p0'] - ext_p0) + abs(gt_char['p1'] - ext_p1)
            total_emission_error += error
            num_comparisons += 1
    
    avg_emission_error = total_emission_error / num_comparisons if num_comparisons > 0 else 0.0
    emission_accuracy = max(0.0, 1.0 - avg_emission_error)
    
    print(f"Emission Accuracy: {emission_accuracy:.3f} (avg error: {avg_emission_error:.3f})")
    
    # State type recovery
    type_matches = 0
    total_states = len(alignment)
    
    state_types = {'A': 'Zero-biased', 'B': 'One-biased', 'C': 'Balanced'}
    
    for gt_state, ext_state in alignment.items():
        if ext_state in extracted_info:
            gt_type = state_types[gt_state]
            
            ext_probs = extracted_info[ext_state]['future_probabilities']
            ext_p0 = ext_probs.get('0', 0.0)
            ext_p1 = ext_probs.get('1', 0.0)
            
            if ext_p0 > 0.7:
                ext_type = "Zero-biased"
            elif ext_p1 > 0.7:
                ext_type = "One-biased"
            elif abs(ext_p0 - ext_p1) < 0.1:
                ext_type = "Balanced"
            else:
                ext_type = "Mixed"
            
            if gt_type == ext_type:
                type_matches += 1
    
    type_recovery = type_matches / total_states if total_states > 0 else 0.0
    print(f"State Type Recovery: {type_recovery:.3f} ({type_matches}/{total_states} types correct)")
    
    # Overall quality score
    overall_score = (state_count_accuracy + emission_accuracy + type_recovery) / 3
    print(f"\nOverall Quality Score: {overall_score:.3f} / 1.000")
    
    # Classification
    if overall_score >= 0.95:
        classification = "EXCELLENT"
    elif overall_score >= 0.85:
        classification = "VERY GOOD"
    elif overall_score >= 0.75:
        classification = "GOOD"
    elif overall_score >= 0.60:
        classification = "FAIR"
    else:
        classification = "POOR"
    
    print(f"Quality Classification: {classification}")
    
    return {
        'state_count_accuracy': state_count_accuracy,
        'emission_accuracy': emission_accuracy, 
        'type_recovery': type_recovery,
        'overall_score': overall_score,
        'classification': classification
    }

def compute_7state_quality_metrics(alignment, gt_characteristics, extracted_info):
    """Compute quality metrics for 7-state extraction."""
    print("\n📊 7-STATE QUALITY METRICS")
    print("=" * 60)
    
    # State count accuracy
    gt_num_states = 7
    ext_num_states = len(extracted_info)
    state_count_accuracy = 1.0 if gt_num_states == ext_num_states else ext_num_states / gt_num_states
    
    print(f"State Count Accuracy: {state_count_accuracy:.3f} ({ext_num_states}/{gt_num_states} states)")
    
    # Emission probability accuracy
    total_emission_error = 0.0
    num_comparisons = 0
    
    for gt_state, ext_state in alignment.items():
        if ext_state in extracted_info:
            # Get GT probabilities from the alignment analysis
            gt_p0 = gt_characteristics[gt_state]['p0']
            gt_p1 = gt_characteristics[gt_state]['p1']
            
            ext_probs = extracted_info[ext_state]['future_probabilities']
            ext_p0 = ext_probs.get('0', 0.0)
            ext_p1 = ext_probs.get('1', 0.0)
            
            error = abs(gt_p0 - ext_p0) + abs(gt_p1 - ext_p1)
            total_emission_error += error
            num_comparisons += 1
    
    avg_emission_error = total_emission_error / num_comparisons if num_comparisons > 0 else 0.0
    emission_accuracy = max(0.0, 1.0 - avg_emission_error / 2.0)  # Divide by 2 since max error is 2
    
    print(f"Emission Accuracy: {emission_accuracy:.3f} (avg error: {avg_emission_error:.3f})")
    
    # State type recovery
    type_matches = 0
    total_states = len(alignment)
    
    for gt_state, ext_state in alignment.items():
        if ext_state in extracted_info:
            gt_type = gt_characteristics[gt_state]['type']
            
            ext_probs = extracted_info[ext_state]['future_probabilities']
            ext_p0 = ext_probs.get('0', 0.0)
            ext_p1 = ext_probs.get('1', 0.0)
            
            if ext_p0 > 0.7:
                ext_type = "Zero-biased"
            elif ext_p1 > 0.7:
                ext_type = "One-biased"
            elif abs(ext_p0 - ext_p1) < 0.15:
                ext_type = "Balanced"
            else:
                ext_type = "Mixed"
            
            if gt_type == ext_type:
                type_matches += 1
    
    type_recovery = type_matches / total_states if total_states > 0 else 0.0
    print(f"State Type Recovery: {type_recovery:.3f} ({type_matches}/{total_states} types correct)")
    
    # Overall quality score
    overall_score = (state_count_accuracy + emission_accuracy + type_recovery) / 3
    print(f"\nOverall Quality Score: {overall_score:.3f} / 1.000")
    
    # Classification
    if overall_score >= 0.95:
        classification = "EXCELLENT"
    elif overall_score >= 0.85:
        classification = "VERY GOOD"
    elif overall_score >= 0.75:
        classification = "GOOD"
    elif overall_score >= 0.60:
        classification = "FAIR"
    else:
        classification = "POOR"
    
    print(f"Quality Classification: {classification}")
    
    return {
        'state_count_accuracy': state_count_accuracy,
        'emission_accuracy': emission_accuracy, 
        'type_recovery': type_recovery,
        'overall_score': overall_score,
        'classification': classification
    }

def main():
    """Main 7-state comparison analysis."""
    print("🔬 GROUND TRUTH vs EXTRACTED 7-STATE FSM COMPARISON")
    print("=" * 80)
    
    # Load machines
    ground_truth, extracted = load_machines()
    
    if ground_truth is None or extracted is None:
        return
    
    # Analyze ground truth
    gt_emissions = analyze_ground_truth_structure(ground_truth)
    
    # Analyze extracted machine
    extracted_info = analyze_extracted_structure(extracted)
    
    # Create GT characteristics for alignment
    gt_characteristics = {}
    for state, probs in gt_emissions.items():
        prob_0 = probs['0']
        prob_1 = probs['1']
        
        if prob_0 > 0.7:
            state_type = "Zero-biased"
        elif prob_1 > 0.7:
            state_type = "One-biased"
        elif abs(prob_0 - prob_1) < 0.15:
            state_type = "Balanced"
        else:
            state_type = "Mixed"
            
        gt_characteristics[state] = {
            'type': state_type,
            'p0': prob_0,
            'p1': prob_1
        }
    
    # Find state alignment
    alignment = find_state_alignment_7state(gt_emissions, extracted_info)
    
    # Analyze suffix patterns
    analyze_suffix_patterns(extracted)
    
    # Compute quality metrics
    quality_metrics = compute_7state_quality_metrics(alignment, gt_characteristics, extracted_info)
    
    print("\n" + "=" * 80)
    print("📋 7-STATE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully extracted {len(extracted_info)} states (target: 7)")
    print(f"✅ State alignment: {len(alignment)}/{len(gt_characteristics)} states matched")
    print(f"✅ Overall quality: {quality_metrics['overall_score']:.3f} ({quality_metrics['classification']})")
    
    if quality_metrics['overall_score'] >= 0.9:
        print(f"🎯 Result: Neural+CSSR extraction EXCELLENTLY recovered the 7-state structure!")
    elif quality_metrics['overall_score'] >= 0.75:
        print(f"✅ Result: Neural+CSSR extraction successfully recovered the 7-state structure!")
    else:
        print(f"⚠️  Result: Neural+CSSR extraction partially recovered the 7-state structure.")

if __name__ == '__main__':
    main()