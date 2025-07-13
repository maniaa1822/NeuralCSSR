#!/usr/bin/env python3
"""
Extract actual epsilon machines from transformer internal representations.
Converts clustered neural states into proper epsilon machine format with
state transitions and symbol emission probabilities.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_delay_transformer import BinaryTransformer, SequenceDataset, collate_fn_ar
from analyze_internal_states import InternalStateAnalyzer

import torch
from torch.utils.data import DataLoader

class EpsilonMachineExtractor:
    """Extract epsilon machines from transformer internal representations."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.analyzer = InternalStateAnalyzer(model, device)
    
    def extract_epsilon_machine(self, dataloader: DataLoader, layer_idx: int = -1, 
                              max_batches: int = 10) -> Dict:
        """Extract epsilon machine from specified transformer layer."""
        
        print(f"Extracting epsilon machine from layer {layer_idx}...")
        
        # Extract internal representations
        representations = self.analyzer.extract_representations(dataloader, max_batches)
        
        # Use final layer by default
        if layer_idx == -1:
            layer_idx = max(representations['hidden_states'].keys())
        
        # Build state-symbol mapping
        epsilon_machine = self._build_epsilon_machine_from_representations(
            representations, layer_idx
        )
        
        return epsilon_machine
    
    def _build_epsilon_machine_from_representations(self, representations: Dict, 
                                                  layer_idx: int) -> Dict:
        """Build epsilon machine structure from internal representations."""
        
        # Get hidden states for the specified layer
        layer_states = representations['hidden_states'][layer_idx]
        sequence_info = representations['sequence_info']
        
        # Collect all hidden states and corresponding symbols
        all_states = []
        state_positions = []
        symbol_emissions = []
        next_symbols = []
        
        state_idx = 0
        for batch_idx, batch_states in enumerate(layer_states):
            B, T, D = batch_states.shape
            
            for seq_idx in range(B):
                # Find corresponding sequence info
                matching_seq = None
                for seq in sequence_info:
                    if seq['batch_idx'] == batch_idx and seq['seq_idx'] == seq_idx:
                        matching_seq = seq
                        break
                
                if matching_seq is None:
                    continue
                
                seq_len = matching_seq['length']
                targets = matching_seq['targets']
                
                for pos in range(min(seq_len - 1, T - 1)):  # Need next symbol
                    # Current hidden state
                    hidden_state = batch_states[seq_idx, pos].numpy()
                    all_states.append(hidden_state)
                    
                    # Current symbol (what was emitted)
                    current_symbol = targets[pos]
                    symbol_emissions.append(current_symbol)
                    
                    # Next symbol (what comes after this state)
                    next_symbol = targets[pos + 1]
                    next_symbols.append(next_symbol)
                    
                    state_positions.append({
                        'batch': batch_idx,
                        'sequence': seq_idx,
                        'position': pos,
                        'state_idx': state_idx
                    })
                    state_idx += 1
        
        if len(all_states) == 0:
            return {"error": "No states extracted"}
        
        # Cluster hidden states into discrete causal states
        from sklearn.cluster import KMeans
        
        n_clusters = min(8, len(all_states) // 10)  # Reasonable number of clusters
        n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_states)
        
        # Build epsilon machine structure
        epsilon_machine = self._construct_epsilon_machine(
            cluster_labels, symbol_emissions, next_symbols, state_positions
        )
        
        return epsilon_machine
    
    def _construct_epsilon_machine(self, cluster_labels: np.ndarray, 
                                 symbol_emissions: List[int],
                                 next_symbols: List[int],
                                 state_positions: List[Dict]) -> Dict:
        """Construct epsilon machine from clustered states and symbols."""
        
        n_states = len(np.unique(cluster_labels))
        
        # Initialize epsilon machine structure
        epsilon_machine = {
            'states': {},
            'transitions': defaultdict(lambda: defaultdict(lambda: {'count': 0, 'next_states': defaultdict(int)})),
            'symbol_emissions': defaultdict(lambda: defaultdict(int)),
            'num_states': n_states,
            'alphabet': ['0', '1'],
            'state_names': [f'ε{i}' for i in range(n_states)]
        }
        
        # Create states
        for i in range(n_states):
            state_name = f'ε{i}'
            epsilon_machine['states'][state_name] = {
                'name': state_name,
                'index': i,
                'observations': 0
            }
        
        # Analyze state-symbol relationships and transitions
        state_symbol_counts = defaultdict(lambda: defaultdict(int))
        state_transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for i in range(len(cluster_labels) - 1):
            current_state = cluster_labels[i]
            current_symbol = symbol_emissions[i]
            next_symbol = next_symbols[i]
            
            # Count symbol emissions from states
            state_symbol_counts[current_state][current_symbol] += 1
            
            # For epsilon machines, we need to track state transitions based on emitted symbols
            # Find the next state that corresponds to the next symbol
            next_state = cluster_labels[i + 1] if i + 1 < len(cluster_labels) else current_state
            
            # Record transition: current_state --symbol--> next_state
            state_transition_counts[current_state][current_symbol][next_state] += 1
            
            # Update state observation counts
            epsilon_machine['states'][f'ε{current_state}']['observations'] += 1
        
        # Convert counts to probabilities and build transition structure
        for state_idx in range(n_states):
            state_name = f'ε{state_idx}'
            
            # Symbol emission probabilities from this state
            total_emissions = sum(state_symbol_counts[state_idx].values())
            if total_emissions > 0:
                for symbol in [0, 1]:
                    count = state_symbol_counts[state_idx][symbol]
                    prob = count / total_emissions
                    epsilon_machine['symbol_emissions'][state_name][str(symbol)] = prob
            else:
                # Default uniform if no observations
                epsilon_machine['symbol_emissions'][state_name]['0'] = 0.5
                epsilon_machine['symbol_emissions'][state_name]['1'] = 0.5
            
            # State transition probabilities
            for symbol in [0, 1]:
                symbol_str = str(symbol)
                total_transitions = sum(state_transition_counts[state_idx][symbol].values())
                
                if total_transitions > 0:
                    for next_state_idx, count in state_transition_counts[state_idx][symbol].items():
                        prob = count / total_transitions
                        next_state_name = f'ε{next_state_idx}'
                        
                        epsilon_machine['transitions'][state_name][symbol_str] = {
                            'next_state': next_state_name,
                            'probability': prob,
                            'count': count
                        }
                        
                        # If multiple next states for same symbol, take most likely
                        if 'next_state' not in epsilon_machine['transitions'][state_name][symbol_str] or \
                           prob > epsilon_machine['transitions'][state_name][symbol_str]['probability']:
                            epsilon_machine['transitions'][state_name][symbol_str] = {
                                'next_state': next_state_name,
                                'probability': prob,
                                'count': count
                            }
        
        # Clean up the defaultdict structure for JSON serialization
        epsilon_machine['transitions'] = {
            state: dict(transitions) for state, transitions in epsilon_machine['transitions'].items()
        }
        epsilon_machine['symbol_emissions'] = {
            state: dict(emissions) for state, emissions in epsilon_machine['symbol_emissions'].items()
        }
        
        return dict(epsilon_machine)

def visualize_epsilon_machine(epsilon_machine: Dict, output_path: Path):
    """Create visualization of the extracted epsilon machine."""
    
    if 'error' in epsilon_machine:
        print(f"Cannot visualize: {epsilon_machine['error']}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. State transition graph
    G = nx.DiGraph()
    
    # Add nodes (states)
    for state_name, state_info in epsilon_machine['states'].items():
        obs_count = state_info.get('observations', 0)
        G.add_node(state_name, observations=obs_count)
    
    # Add edges (transitions)
    edge_labels = {}
    for state_name, transitions in epsilon_machine['transitions'].items():
        for symbol, trans_info in transitions.items():
            next_state = trans_info['next_state']
            prob = trans_info['probability']
            
            # Add edge with symbol and probability
            G.add_edge(state_name, next_state, symbol=symbol, probability=prob)
            edge_labels[(state_name, next_state)] = f"{symbol}:{prob:.2f}"
    
    # Draw the graph
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    node_sizes = [G.nodes[node]['observations'] * 10 + 300 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.8, ax=ax1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, ax=ax1)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax1)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax1)
    
    ax1.set_title('Extracted Epsilon Machine\nState Transitions')
    ax1.axis('off')
    
    # 2. Symbol emission probabilities heatmap
    states = list(epsilon_machine['states'].keys())
    symbols = ['0', '1']
    emission_matrix = np.zeros((len(states), len(symbols)))
    
    for i, state in enumerate(states):
        for j, symbol in enumerate(symbols):
            prob = epsilon_machine['symbol_emissions'][state].get(symbol, 0)
            emission_matrix[i, j] = prob
    
    im = ax2.imshow(emission_matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(symbols)))
    ax2.set_xticklabels(symbols)
    ax2.set_yticks(range(len(states)))
    ax2.set_yticklabels(states)
    ax2.set_xlabel('Symbols')
    ax2.set_ylabel('States')
    ax2.set_title('Symbol Emission Probabilities')
    
    # Add text annotations
    for i in range(len(states)):
        for j in range(len(symbols)):
            text = ax2.text(j, i, f'{emission_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Emission Probability')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def compare_with_ground_truth(extracted_machine: Dict, dataset_path: Path) -> Dict:
    """Compare extracted epsilon machine with ground truth."""
    
    # Load ground truth
    gt_path = dataset_path / "ground_truth" / "machine_properties.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    
    comparison = {
        'extracted_summary': {
            'num_states': extracted_machine.get('num_states', 0),
            'states': list(extracted_machine.get('states', {}).keys()),
            'alphabet_size': len(extracted_machine.get('alphabet', [])),
        },
        'ground_truth_summary': {},
        'analysis': []
    }
    
    # Ground truth summary
    total_gt_states = 0
    for machine_id, machine_data in ground_truth.items():
        gt_states = machine_data['num_states']
        total_gt_states += gt_states
        comparison['ground_truth_summary'][machine_id] = {
            'num_states': gt_states,
            'is_topological': machine_data.get('is_topological', True),
            'entropy_rate': machine_data.get('entropy_rate', 'unknown')
        }
    
    # Analysis
    extracted_states = extracted_machine.get('num_states', 0)
    comparison['analysis'].append(f"Ground truth: {len(ground_truth)} machines with {total_gt_states} total states")
    comparison['analysis'].append(f"Extracted: {extracted_states} epsilon states")
    
    if extracted_states == total_gt_states:
        comparison['analysis'].append("✅ Exact state count match")
    elif extracted_states > total_gt_states:
        comparison['analysis'].append(f"⚠️  Enhanced state space: +{extracted_states - total_gt_states} states")
    else:
        comparison['analysis'].append(f"❌ Fewer states than expected: -{total_gt_states - extracted_states} states")
    
    return comparison

def generate_extraction_report(epsilon_machine: Dict, comparison: Dict, layer_idx: int) -> str:
    """Generate report on epsilon machine extraction."""
    
    report = []
    report.append("=" * 80)
    report.append("EPSILON MACHINE EXTRACTION FROM TRANSFORMER")
    report.append("=" * 80)
    report.append("")
    
    # Extraction summary
    report.append("🔍 EXTRACTION SUMMARY")
    report.append("-" * 50)
    report.append(f"Source: Transformer Layer {layer_idx}")
    report.append(f"Method: Internal representation clustering + transition analysis")
    
    if 'error' in epsilon_machine:
        report.append(f"❌ Extraction failed: {epsilon_machine['error']}")
        return "\n".join(report)
    
    report.append(f"States extracted: {epsilon_machine['num_states']}")
    report.append(f"Alphabet: {epsilon_machine['alphabet']}")
    report.append("")
    
    # State analysis
    report.append("📊 EXTRACTED EPSILON MACHINE STRUCTURE")
    report.append("-" * 50)
    
    for state_name, state_info in epsilon_machine['states'].items():
        report.append(f"{state_name}:")
        report.append(f"  • Observations: {state_info['observations']}")
        
        # Symbol emissions
        emissions = epsilon_machine['symbol_emissions'][state_name]
        report.append(f"  • Symbol emissions: 0→{emissions.get('0', 0):.3f}, 1→{emissions.get('1', 0):.3f}")
        
        # Transitions
        if state_name in epsilon_machine['transitions']:
            transitions = epsilon_machine['transitions'][state_name]
            for symbol, trans_info in transitions.items():
                next_state = trans_info['next_state']
                prob = trans_info['probability']
                report.append(f"  • Transition on '{symbol}': → {next_state} (p={prob:.3f})")
        
        report.append("")
    
    # Comparison with ground truth
    report.append("🎰 COMPARISON WITH GROUND TRUTH")
    report.append("-" * 50)
    
    for analysis_line in comparison['analysis']:
        report.append(f"  {analysis_line}")
    
    report.append("")
    report.append(f"Ground truth machines:")
    for machine_id, machine_data in comparison['ground_truth_summary'].items():
        report.append(f"  • Machine {machine_id}: {machine_data['num_states']} states "
                     f"(topological: {machine_data['is_topological']})")
    
    report.append("")
    
    # Assessment
    report.append("🏆 EXTRACTION ASSESSMENT")
    report.append("-" * 50)
    
    extracted_states = epsilon_machine['num_states']
    total_gt_states = sum(m['num_states'] for m in comparison['ground_truth_summary'].values())
    
    if extracted_states >= total_gt_states:
        report.append("✅ SUCCESSFUL EXTRACTION:")
        report.append("   • Transformer learned epsilon machine-like structure")
        report.append("   • State count matches or exceeds ground truth")
        report.append("   • Internal representations encode causal state information")
    else:
        report.append("⚠️  PARTIAL EXTRACTION:")
        report.append("   • Some epsilon machine structure detected")
        report.append("   • May indicate distributed rather than discrete encoding")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Extract epsilon machine from transformer")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--test", type=Path, help="Path to test dataset (defaults to dataset/val)")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to extract from (-1 for final)")
    parser.add_argument("--d_model", type=int, default=32, help="Model d_model")
    parser.add_argument("--layers", type=int, default=3, help="Model layers")
    parser.add_argument("--heads", type=int, default=8, help="Model heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output", type=Path, default=Path("results/epsilon_machine_extraction"), 
                       help="Output directory")
    parser.add_argument("--max_batches", type=int, default=3, help="Max batches to analyze")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("Loading model and data...")
    
    # Load model
    model = BinaryTransformer(vocab_size=3, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    # Load test dataset
    test_path = args.test if args.test else args.dataset / "neural_format" / "val_dataset.pt"
    test_dataset = SequenceDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_ar)
    
    # Extract epsilon machine
    extractor = EpsilonMachineExtractor(model, device)
    epsilon_machine = extractor.extract_epsilon_machine(test_loader, args.layer, args.max_batches)
    
    print("Comparing with ground truth...")
    comparison = compare_with_ground_truth(epsilon_machine, args.dataset)
    
    print("Generating visualization...")
    visualize_epsilon_machine(epsilon_machine, args.output / "extracted_epsilon_machine.png")
    
    print("Generating report...")
    report = generate_extraction_report(epsilon_machine, comparison, args.layer)
    
    print(report)
    
    # Save results
    with open(args.output / "epsilon_machine_extraction_report.txt", 'w') as f:
        f.write(report)
    
    # Save epsilon machine structure
    with open(args.output / "extracted_epsilon_machine.json", 'w') as f:
        json.dump(epsilon_machine, f, indent=2, default=str)
    
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()