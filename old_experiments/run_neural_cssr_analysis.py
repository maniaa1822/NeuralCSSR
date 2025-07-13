#!/usr/bin/env python3
"""
Run CSSR analysis using neural oracle probabilities and compare with classical CSSR.

This script:
1. Loads a trained transformer model as a neural oracle
2. Runs CSSR analysis using neural probabilities 
3. Runs classical CSSR analysis using empirical probabilities
4. Compares both approaches using machine distance metrics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neural_oracle import NeuralProbabilityOracle
from src.neural_cssr.classical.cssr import ClassicalCSSR
from src.neural_cssr.analysis.dataset_loader import UnifiedDatasetLoader
from src.neural_cssr.evaluation.machine_distance import MachineDistanceCalculator


class NeuralCSSRProbabilityProvider:
    """Adapter to connect NeuralProbabilityOracle to Classical CSSR."""
    
    def __init__(self, oracle: NeuralProbabilityOracle):
        self.oracle = oracle
        # Set required attributes for CSSR compatibility
        self.id_to_token = {0: '0', 1: '1'}  # Binary alphabet
        self.token_to_id = {'0': 0, '1': 1}
        
    def get_probabilities(self, history: str) -> Dict[str, float]:
        """Convert string history to list and get probabilities from oracle."""
        context = [int(symbol) for symbol in history]
        return self.oracle.get_probabilities(context)
    
    def get_empirical_distribution(self, history: str) -> Dict[str, int]:
        """Convert neural probabilities to pseudo-empirical counts for CSSR."""
        probs = self.get_probabilities(history)
        
        # Convert probabilities to pseudo-counts (scale by 1000 for precision)
        scale_factor = 1000
        counts = {}
        for symbol, prob in probs.items():
            counts[symbol] = int(prob * scale_factor)
        
        return counts


def run_neural_cssr_analysis(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    d_model: int = 10,
    layers: int = 1, 
    heads: int = 1,
    L: int = 10,
    alpha: float = 0.001
) -> Dict[str, Any]:
    """Run complete neural CSSR analysis pipeline."""
    
    print("=" * 70)
    print("NEURAL CSSR ANALYSIS")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Parameters: L={L}, α={alpha}")
    print()
    
    # Load dataset
    loader = UnifiedDatasetLoader()
    train_sequences = loader.load_sequences(dataset_path, 'train')
    
    # Create neural oracle
    print("Loading neural oracle...")
    oracle = NeuralProbabilityOracle(
        Path(model_path), d_model=d_model, layers=layers, heads=heads
    )
    neural_provider = NeuralCSSRProbabilityProvider(oracle)
    
    # Convert sequences to CSSR format
    print("Converting sequences to CSSR format...")
    data_list = []
    for seq in train_sequences:
        for i in range(len(seq) - 1):
            history = list(seq[:i+1])  # Everything up to position i
            target = seq[i+1]          # Next symbol
            data_list.append({
                'raw_history': history,
                'raw_target': target
            })
    
    print(f"Generated {len(data_list)} history-target pairs from {len(train_sequences)} sequences")
    
    # Run Neural CSSR
    print("Running Neural CSSR...")
    neural_cssr = ClassicalCSSR(
        significance_level=alpha,
        min_count=5,
        test_type="kl_divergence",
        neural_probability_provider=neural_provider
    )
    
    neural_cssr.load_from_raw_data(data_list, metadata={})
    neural_converged = neural_cssr.run_cssr(
        max_iterations=20,
        max_history_length=L
    )
    
    # Run Classical CSSR for comparison
    print("Running Classical CSSR (empirical probabilities)...")
    classical_cssr = ClassicalCSSR(
        significance_level=alpha,
        min_count=5,
        test_type="kl_divergence"
    )
    
    classical_cssr.load_from_raw_data(data_list, metadata={})
    classical_converged = classical_cssr.run_cssr(
        max_iterations=20,
        max_history_length=L
    )
    
    # Extract results in proper format
    neural_results = neural_cssr.causal_states
    classical_results = classical_cssr.causal_states
    
    # Load ground truth for comparison  
    ground_truth = loader.load_ground_truth(dataset_path)
    
    # Compile results (skip distance calculation for now to get basic functionality working)
    results = {
        'parameters': {'L': L, 'alpha': alpha, 'd_model': d_model, 'layers': layers, 'heads': heads},
        'execution_info': {
            'neural_converged': neural_converged,
            'classical_converged': classical_converged
        },
        'neural_cssr': {
            'num_states': len(neural_results),
            'states': {str(i): {'histories': list(state.histories), 
                               'distribution': state.get_probability_distribution()}
                      for i, state in enumerate(neural_results)}
        },
        'classical_cssr': {
            'num_states': len(classical_results),
            'states': {str(i): {'histories': list(state.histories), 
                               'distribution': state.get_probability_distribution()}
                      for i, state in enumerate(classical_results)}
        },
        'ground_truth_loaded': bool(ground_truth)
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'neural_cssr_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"Neural CSSR: {results['neural_cssr']['num_states']} states (converged: {neural_converged})")
    print(f"Classical CSSR: {results['classical_cssr']['num_states']} states (converged: {classical_converged})")
    print(f"Ground truth data loaded: {results['ground_truth_loaded']}")
    print()
    
    # Show sample neural CSSR states
    print("Neural CSSR States (sample):")
    for i, (state_id, state_info) in enumerate(list(results['neural_cssr']['states'].items())[:3]):
        print(f"  State {state_id}: {len(state_info['histories'])} histories")
        print(f"    Sample histories: {list(state_info['histories'])[:3]}")
        print(f"    Distribution: {state_info['distribution']}")
    
    print("\nClassical CSSR States (sample):")
    for i, (state_id, state_info) in enumerate(list(results['classical_cssr']['states'].items())[:3]):
        print(f"  State {state_id}: {len(state_info['histories'])} histories")
        print(f"    Sample histories: {list(state_info['histories'])[:3]}")
        print(f"    Distribution: {state_info['distribution']}")
    
    print(f"\nResults saved to: {output_path}/neural_cssr_comparison.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Neural CSSR Analysis")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained transformer model')
    parser.add_argument('--output', type=str, default='results/neural_cssr_analysis', help='Output directory')
    parser.add_argument('--d_model', type=int, default=10, help='Model dimension')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--L', type=int, default=10, help='Maximum context length')
    parser.add_argument('--alpha', type=float, default=0.001, help='Significance level')
    
    args = parser.parse_args()
    
    run_neural_cssr_analysis(
        dataset_path=args.dataset,
        model_path=args.model, 
        output_dir=args.output,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        L=args.L,
        alpha=args.alpha
    )


if __name__ == '__main__':
    main()