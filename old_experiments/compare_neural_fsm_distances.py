#!/usr/bin/env python3
"""
Compare extracted neural FSMs with ground truth using machine distance analysis.
Applies the same distance metrics we used for classical CSSR to neural FSM extraction.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neural_cssr.evaluation.machine_distance import MachineDistanceCalculator
from neural_cssr.evaluation.utils.data_loading import load_ground_truth

class NeuralFSMConverter:
    """Convert extracted neural FSM data to format compatible with distance analysis."""
    
    def __init__(self, fsm_data: Dict):
        self.fsm_data = fsm_data
    
    def convert_to_cssr_format(self, layer_id: str) -> Dict:
        """Convert neural FSM to CSSR-like format for distance analysis."""
        layer_fsm = self.fsm_data['fsm_properties'][layer_id]
        
        # Extract transition matrix
        prob_matrix = np.array(layer_fsm['probability_matrix'])
        n_states = layer_fsm['num_states']
        
        # Create CSSR-like format
        cssr_format = {
            'states': {},
            'transitions': {},
            'num_states': n_states,
            'alphabet': ['0', '1']
        }
        
        # Create states
        for i in range(n_states):
            state_name = f"S{i}"
            cssr_format['states'][state_name] = {
                'name': state_name,
                'index': i
            }
        
        # Create transitions with symbol distributions
        for from_idx in range(n_states):
            from_state = f"S{from_idx}"
            cssr_format['transitions'][from_state] = {}
            
            # For neural FSM, we need to infer symbol distributions from transitions
            # Since we only have state-to-state transitions, we'll approximate
            # symbol distributions based on transition patterns
            
            # Simple heuristic: map state transitions to symbol emissions
            # High-index target states might correspond to '1', low-index to '0'
            symbol_0_prob = 0.0
            symbol_1_prob = 0.0
            
            for to_idx in range(n_states):
                trans_prob = prob_matrix[from_idx, to_idx]
                if trans_prob > 0:
                    # Heuristic: lower state indices more likely for '0', higher for '1'
                    if to_idx < n_states // 2:
                        symbol_0_prob += trans_prob * 0.7  # Bias toward '0'
                        symbol_1_prob += trans_prob * 0.3
                    else:
                        symbol_0_prob += trans_prob * 0.3
                        symbol_1_prob += trans_prob * 0.7  # Bias toward '1'
            
            # Normalize
            total = symbol_0_prob + symbol_1_prob
            if total > 0:
                symbol_0_prob /= total
                symbol_1_prob /= total
            else:
                symbol_0_prob = symbol_1_prob = 0.5
            
            cssr_format['transitions'][from_state] = {
                '0': symbol_0_prob,
                '1': symbol_1_prob
            }
        
        return cssr_format

def analyze_neural_fsm_extraction_quality(neural_data: Dict, ground_truth: Dict) -> Dict:
    """Analyze the quality of neural FSM extraction using our existing metrics."""
    
    converter = NeuralFSMConverter(neural_data)
    results = {}
    
    # Analyze each layer
    for layer_id in neural_data['fsm_properties'].keys():
        print(f"Analyzing Layer {layer_id} FSM...")
        
        # Convert to CSSR format
        neural_fsm = converter.convert_to_cssr_format(layer_id)
        
        # Create distance calculator
        calculator = MachineDistanceCalculator(
            discovered_machine=neural_fsm,
            ground_truth_machines=ground_truth['machines'],
            discovered_transitions=neural_fsm['transitions'],
            ground_truth_transitions=ground_truth.get('transitions', {}),
            dataset_name=f"neural_layer_{layer_id}"
        )
        
        # Calculate distances
        try:
            layer_results = calculator.calculate_all_distances()
            results[f"layer_{layer_id}"] = layer_results
            
            print(f"  Layer {layer_id} - Overall Quality: {layer_results.get('overall_quality_score', 'N/A'):.3f}")
        except Exception as e:
            print(f"  Error analyzing Layer {layer_id}: {e}")
            results[f"layer_{layer_id}"] = {"error": str(e)}
    
    return results

def compare_extraction_methods(neural_results: Dict, classical_results_path: Path) -> Dict:
    """Compare neural FSM extraction with classical CSSR results."""
    
    comparison = {
        'neural_fsm_analysis': neural_results,
        'extraction_method_comparison': {}
    }
    
    # Load classical CSSR results if available
    classical_results = None
    if classical_results_path.exists():
        with open(classical_results_path) as f:
            classical_results = json.load(f)
        
        comparison['classical_cssr_analysis'] = classical_results
    
    # Compare key metrics
    for layer_id, layer_results in neural_results.items():
        if 'error' in layer_results:
            continue
            
        layer_name = layer_id.replace('_', ' ').title()
        
        neural_quality = layer_results.get('overall_quality_score', 0)
        neural_states = layer_results.get('state_mapping_distance', {}).get('discovered_states', 0)
        
        method_comparison = {
            'neural_quality_score': neural_quality,
            'neural_states_discovered': neural_states,
        }
        
        if classical_results:
            classical_quality = classical_results.get('overall_quality_score', 0)
            classical_states = classical_results.get('state_mapping_distance', {}).get('discovered_states', 0)
            
            method_comparison.update({
                'classical_quality_score': classical_quality,
                'classical_states_discovered': classical_states,
                'quality_difference': neural_quality - classical_quality,
                'state_difference': neural_states - classical_states
            })
        
        comparison['extraction_method_comparison'][layer_name] = method_comparison
    
    return comparison

def generate_neural_fsm_comparison_report(comparison_results: Dict, 
                                         neural_fsm_data: Dict) -> str:
    """Generate report comparing neural FSM extraction with classical methods."""
    
    report = []
    report.append("=" * 80)
    report.append("NEURAL FSM EXTRACTION vs CLASSICAL CSSR COMPARISON")
    report.append("=" * 80)
    report.append("")
    
    # Extraction method summary
    report.append("🔍 EXTRACTION METHOD COMPARISON")
    report.append("-" * 50)
    report.append("")
    report.append("CLASSICAL CSSR:")
    report.append("  • Method: Statistical hypothesis testing on raw sequences")
    report.append("  • Input: Symbol sequences (0,1,0,1,...)")
    report.append("  • Algorithm: Causal State Splitting Reconstruction")
    report.append("  • Output: Statistically equivalent causal states")
    report.append("")
    report.append("NEURAL FSM EXTRACTION:")
    report.append("  • Method: Clustering of learned internal representations")
    report.append("  • Input: Transformer hidden states during prediction")
    report.append("  • Algorithm: K-means clustering + transition analysis")
    report.append("  • Output: Functional states from neural representations")
    report.append("")
    
    # Analysis results for each layer
    report.append("📊 NEURAL FSM LAYER ANALYSIS")
    report.append("-" * 50)
    
    neural_analysis = comparison_results['neural_fsm_analysis']
    
    for layer_id, layer_results in neural_analysis.items():
        if 'error' in layer_results:
            report.append(f"{layer_id.title()}: Analysis failed - {layer_results['error']}")
            continue
            
        layer_name = layer_id.replace('_', ' ').title()
        
        # Extract key metrics
        overall_quality = layer_results.get('overall_quality_score', 0)
        state_mapping = layer_results.get('state_mapping_distance', {})
        symbol_dist = layer_results.get('symbol_distribution_distance', {})
        
        discovered_states = state_mapping.get('discovered_states', 'N/A')
        quality_score = state_mapping.get('quality_score', 0)
        avg_js_divergence = symbol_dist.get('average_js_divergence', 0)
        
        report.append(f"")
        report.append(f"{layer_name}:")
        report.append(f"  • Overall Quality Score: {overall_quality:.3f}")
        report.append(f"  • States Discovered: {discovered_states}")
        report.append(f"  • State Mapping Quality: {quality_score:.3f}")
        report.append(f"  • Symbol Distribution Divergence: {avg_js_divergence:.3f}")
        
        # Interpretation
        if overall_quality > 0.7:
            report.append(f"  ✅ HIGH QUALITY extraction")
        elif overall_quality > 0.5:
            report.append(f"  ⚠️  MODERATE QUALITY extraction")
        else:
            report.append(f"  ❌ LOW QUALITY extraction")
    
    report.append("")
    
    # Method comparison if classical results available
    if 'classical_cssr_analysis' in comparison_results:
        report.append("⚖️  CLASSICAL vs NEURAL COMPARISON")
        report.append("-" * 50)
        
        classical = comparison_results['classical_cssr_analysis']
        classical_quality = classical.get('overall_quality_score', 0)
        
        report.append(f"Classical CSSR Quality: {classical_quality:.3f}")
        report.append("")
        
        for layer_name, comparison in comparison_results['extraction_method_comparison'].items():
            neural_qual = comparison['neural_quality_score']
            quality_diff = comparison.get('quality_difference', 0)
            
            report.append(f"{layer_name}:")
            report.append(f"  • Neural Quality: {neural_qual:.3f}")
            if quality_diff > 0.1:
                report.append(f"  ✅ Neural BETTER by {quality_diff:.3f}")
            elif quality_diff < -0.1:
                report.append(f"  ❌ Classical BETTER by {abs(quality_diff):.3f}")
            else:
                report.append(f"  ⚠️  Similar quality (diff: {quality_diff:.3f})")
    
    report.append("")
    
    # Key insights about the extraction process
    report.append("💡 KEY INSIGHTS ABOUT FSM EXTRACTION")
    report.append("-" * 50)
    
    report.append("1. EXTRACTION VALIDITY:")
    neural_qualities = [r.get('overall_quality_score', 0) for r in neural_analysis.values() 
                       if 'error' not in r]
    if neural_qualities and max(neural_qualities) > 0.5:
        report.append("   ✅ Neural FSM extraction produces interpretable state machines")
    else:
        report.append("   ⚠️  Neural FSM extraction may not capture clear state structure")
    
    report.append("")
    report.append("2. CLUSTERING APPROACH:")
    report.append("   • K-means clustering converts continuous representations to discrete states")
    report.append("   • Transition analysis reveals state-to-state dynamics")
    report.append("   • This approximates but doesn't guarantee causal state structure")
    
    report.append("")
    report.append("3. INTERPRETATION CAVEATS:")
    report.append("   • Neural states are functionally defined, not statistically validated")
    report.append("   • Clustering parameters affect the extracted FSM structure")
    report.append("   • Symbol distributions are inferred heuristically from state transitions")
    
    report.append("")
    
    # Final assessment
    report.append("🏆 FINAL ASSESSMENT")
    report.append("-" * 50)
    
    if neural_qualities and max(neural_qualities) > 0.7:
        report.append("🎯 SUCCESSFUL NEURAL FSM EXTRACTION:")
        report.append("   • Transformer internal states show clear FSM-like organization")
        report.append("   • Clustering reveals interpretable state transition structure")
        report.append("   • Quality scores suggest meaningful correspondence with ground truth")
        report.append("   • Neural approach complements classical CSSR with implementation insights")
    elif neural_qualities and max(neural_qualities) > 0.5:
        report.append("⚠️  MODERATE NEURAL FSM EXTRACTION:")
        report.append("   • Some FSM-like structure detected but with limitations")
        report.append("   • May reflect distributed rather than discrete state representations")
        report.append("   • Clustering approach has inherent approximations")
    else:
        report.append("❌ LIMITED NEURAL FSM EXTRACTION:")
        report.append("   • Unclear if clustering reveals true FSM structure")
        report.append("   • Internal representations may be too distributed for discrete FSM analysis")
        report.append("   • Alternative analysis approaches may be needed")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    # Paths
    neural_fsm_path = Path("results/internal_analysis/extracted_fsm_data.json")
    classical_results_path = Path("results/fixed_theoretical_viz/machine_distance_report.json")
    dataset_path = Path("datasets/biased_exp")
    output_dir = Path("results/neural_fsm_distance_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("Loading neural FSM extraction results...")
    with open(neural_fsm_path) as f:
        neural_fsm_data = json.load(f)
    
    print("Loading ground truth data...")
    ground_truth = load_ground_truth(str(dataset_path))
    
    print("Analyzing neural FSM extraction quality...")
    neural_analysis_results = analyze_neural_fsm_extraction_quality(neural_fsm_data, ground_truth)
    
    print("Comparing with classical CSSR results...")
    comparison_results = compare_extraction_methods(neural_analysis_results, classical_results_path)
    
    print("Generating comparison report...")
    report = generate_neural_fsm_comparison_report(comparison_results, neural_fsm_data)
    
    print(report)
    
    # Save results
    with open(output_dir / "neural_fsm_distance_analysis.txt", 'w') as f:
        f.write(report)
    
    # Save detailed results
    with open(output_dir / "neural_fsm_comparison_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()